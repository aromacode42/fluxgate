package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"strings"
	"time"

	"github.com/jacek/fluxgate/internal/adapter"
	"github.com/jacek/fluxgate/internal/balancer"
	"github.com/jacek/fluxgate/internal/limiter"
	"github.com/jacek/fluxgate/internal/modelrouter"
	"github.com/jacek/fluxgate/internal/provider"
	"github.com/jacek/fluxgate/internal/proxy"
	"github.com/jacek/fluxgate/internal/retry"
	pkgconfig "github.com/jacek/fluxgate/pkg/config"
)

// httpError indicates the backend returned a definitive HTTP status (4xx/5xx).
// These are retried a limited number of times at the gateway level.
type httpError struct {
	inner error
}

func (e *httpError) Error() string { return e.inner.Error() }
func (e *httpError) Unwrap() error { return e.inner }

// hopByHopHeaders should not be forwarded between proxies.
var hopByHopHeaders = map[string]bool{
	"Connection":          true,
	"Keep-Alive":          true,
	"Proxy-Authenticate":  true,
	"Proxy-Authorization": true,
	"TE":                  true,
	"Trailer":             true,
	"Transfer-Encoding":   true,
	"Upgrade":             true,
}

type Gateway struct {
	registry      *provider.Registry
	balancer      balancer.Balancer
	modelRouter   *modelrouter.Router
	limiter       *limiter.RateLimiter
	retryer       *retry.Retryer
	config        pkgconfig.GatewayConfig
	proxyRegistry *proxy.Registry
	logger        *slog.Logger
}

func New(
	registry *provider.Registry,
	bal balancer.Balancer,
	lim *limiter.RateLimiter,
	cfg pkgconfig.GatewayConfig,
	proxyRegistry *proxy.Registry,
	modelRouter *modelrouter.Router,
) *Gateway {
	return &Gateway{
		registry:      registry,
		balancer:      bal,
		modelRouter:   modelRouter,
		limiter:       lim,
		retryer: retry.New(retry.Config{
			Max:     cfg.RetryMax,
			WaitMin: cfg.RetryWaitMin,
			WaitMax: cfg.RetryWaitMax,
		}),
		config:        cfg,
		proxyRegistry: proxyRegistry,
		logger:        slog.Default(),
	}
}

func (g *Gateway) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	start := time.Now()

	if !g.limiter.Allow() {
		g.logger.Warn("rate limit exceeded", "path", r.URL.Path, "remote", r.RemoteAddr)
		writeError(w, http.StatusTooManyRequests, "rate limit exceeded")
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeError(w, http.StatusBadRequest, "failed to read request body")
		return
	}
	r.Body.Close()

	// Validate JSON body
	if !json.Valid(body) {
		writeError(w, http.StatusBadRequest, "invalid JSON request body")
		return
	}

	// Detect what format the client expects (entry format)
	entryFormat := g.detectEntryFormat(r.URL.Path)
	model := extractModel(body)

	// Gateway-level retry: keep retrying while the client is still connected.
	maxAttempts := g.config.GatewayRetryMax
	if maxAttempts <= 0 {
		maxAttempts = 3 // default
	}

	var lastErr error
	for attempt := 0; attempt < maxAttempts; attempt++ {
		if err := r.Context().Err(); err != nil {
			g.logger.Debug("client disconnected, stopping retry", "path", r.URL.Path)
			return
		}

		if attempt > 0 {
			wait := g.gatewayBackoff(attempt)
			g.logger.Info("gateway retry",
				"path", r.URL.Path,
				"model", model,
				"attempt", attempt,
				"wait", wait,
				"last_error", lastErr)
			select {
			case <-r.Context().Done():
				return
			case <-time.After(wait):
			}
		}

		resp, backendFormat, err := g.forwardRequest(r.Context(), r, body, entryFormat)
		if err != nil {
			// If all backends returned HTTP errors (not network/timeout),
			// retry a limited number of times then give up — the client
			// (e.g. Claude Code) handles its own retry logic.
			var httpErr *httpError
			if errors.As(err, &httpErr) {
				// HTTP errors: retry up to gateway_retry_max_http times (default 3)
				httpRetryMax := g.config.GatewayRetryMax
				if httpRetryMax <= 0 {
					httpRetryMax = 3
				}
				if attempt >= httpRetryMax {
					g.logger.Warn("request failed, HTTP retries exhausted, returning to client",
						"error", err,
						"path", r.URL.Path,
						"model", model,
						"attempts", attempt+1)
					writeError(w, http.StatusBadGateway, fmt.Sprintf("gateway error: %v", err))
					return
				}
			}
			lastErr = err
			g.logger.Warn("request failed, will retry",
				"error", err,
				"path", r.URL.Path,
				"model", model,
				"attempt", attempt)
			continue
		}

		// Copy response headers, skipping hop-by-hop
		for k, vs := range resp.Header {
			if hopByHopHeaders[k] {
				continue
			}
			for _, v := range vs {
				w.Header().Add(k, v)
			}
		}
		w.WriteHeader(resp.StatusCode)

		// Stream response body, translating SSE if formats differ
		g.streamResponse(resp, w, entryFormat, backendFormat, model)

		g.logger.Info("request completed",
			"path", r.URL.Path,
			"status", resp.StatusCode,
			"attempts", attempt+1,
			"duration", time.Since(start).Round(time.Millisecond),
		)
		return
	}

	// All retries exhausted
	g.logger.Error("all gateway retries exhausted",
		"error", lastErr,
		"path", r.URL.Path,
		"model", model)
	writeError(w, http.StatusBadGateway, fmt.Sprintf("gateway error: %v", lastErr))
}

// detectEntryFormat returns the format the client is using based on the request path.
func (g *Gateway) detectEntryFormat(path string) string {
	if strings.HasPrefix(path, "/v1/messages") {
		return "anthropic"
	}
	if strings.HasPrefix(path, "/anthropic/") {
		return "anthropic"
	}
	if strings.HasPrefix(path, "/gemini/") {
		return "gemini"
	}
	return "openai"
}

// streamResponse handles SSE passthrough, SSE translation, or raw copy.
func (g *Gateway) streamResponse(resp *http.Response, w http.ResponseWriter, entryFormat, backendFormat, model string) {
	contentType := resp.Header.Get("Content-Type")
	if isSSE(contentType) {
		if entryFormat != "" && backendFormat != "" && entryFormat != backendFormat {
			sseTranslate(resp.Body, w, entryFormat, backendFormat, model)
		} else {
			ssePassthrough(resp.Body, w)
		}
	} else {
		rawPassthrough(resp.Body, w)
	}
}

func (g *Gateway) forwardRequest(ctx context.Context, r *http.Request, body []byte, entryFormat string) (*http.Response, string, error) {
	model := extractModel(body)

	if g.modelRouter != nil && model != "" {
		if entry := g.modelRouter.Resolve(model); entry != nil {
			backends := g.modelRouter.HealthyBackends(model)
			if len(backends) == 0 {
				return nil, "", fmt.Errorf("all backends down for model %q", model)
			}
			return g.forwardWithBackends(ctx, r, body, backends, model, entryFormat)
		}

		candidates := g.modelRouter.PassthroughCandidates(model, "")
		if len(candidates) > 0 {
			candidates = g.filterByFormat(candidates, entryFormat)
			if len(candidates) > 0 {
				return g.forwardPassthrough(ctx, r, body, candidates, entryFormat)
			}
		}
	}

	candidates := g.registry.Healthy()
	candidates = g.filterByFormat(candidates, entryFormat)
	if len(candidates) == 0 {
		candidates = g.registry.Healthy()
	}
	return g.forwardPassthrough(ctx, r, body, candidates, entryFormat)
}

func (g *Gateway) filterByFormat(providers []*provider.Provider, format string) []*provider.Provider {
	if format == "" {
		return providers
	}
	var matched []*provider.Provider
	for _, p := range providers {
		if p.Type == format {
			matched = append(matched, p)
		}
	}
	if len(matched) > 0 {
		return matched
	}
	return providers
}

func (g *Gateway) forwardWithBackends(
	ctx context.Context,
	r *http.Request,
	body []byte,
	backends []modelrouter.BackendRef,
	virtualModel string,
	entryFormat string,
) (*http.Response, string, error) {
	var lastErr error

	for i, backend := range backends {
		if err := ctx.Err(); err != nil {
			return nil, "", err
		}

		if i > 0 {
			wait := g.retryer.Backoff(i)
			select {
			case <-ctx.Done():
				return nil, "", ctx.Err()
			case <-time.After(wait):
			}
		}

		fwdBody := body
		if backend.RealModel != virtualModel {
			fwdBody = rewriteModel(body, backend.RealModel)
		}

		backendFormat := backend.Provider.Type
		if entryFormat != backendFormat {
			translated, err := g.translateRequest(fwdBody, entryFormat, backendFormat)
			if err != nil {
				g.logger.Warn("request translation failed, trying next",
					"provider", backend.ProviderName, "error", err)
				lastErr = err
				continue
			}
			fwdBody = translated
		}

		resp, err := g.sendToProvider(ctx, r, fwdBody, backend.Provider, backendFormat)
		if err != nil {
			lastErr = err
			g.logger.Warn("backend failed, trying next",
				"provider", backend.ProviderName, "model", backend.RealModel, "error", err)
			continue
		}

		if g.retryer.ShouldRetry(resp.StatusCode) {
			resp.Body.Close()
			lastErr = &httpError{inner: fmt.Errorf("backend %s returned %d", backend.ProviderName, resp.StatusCode)}
			g.logger.Warn("backend returned error, trying next",
				"provider", backend.ProviderName, "status", resp.StatusCode)
			continue
		}

		g.logger.Info("backend succeeded",
			"virtual", virtualModel,
			"provider", backend.ProviderName,
			"real_model", backend.RealModel)
		return resp, backendFormat, nil
	}

	if lastErr != nil {
		return nil, "", fmt.Errorf("all backends failed for model %q: %w", virtualModel, lastErr)
	}
	return nil, "", fmt.Errorf("all backends failed for model %q", virtualModel)
}

func (g *Gateway) forwardPassthrough(
	ctx context.Context,
	r *http.Request,
	body []byte,
	candidates []*provider.Provider,
	entryFormat string,
) (*http.Response, string, error) {
	if len(candidates) == 0 {
		return nil, "", fmt.Errorf("no healthy providers available")
	}

	var lastBackendFormat string
	resp, err := g.retryer.Do(ctx, func() (*http.Response, error) {
		p, err := g.balancer.Next(candidates)
		if err != nil {
			return nil, err
		}

		fwdBody := body
		backendFormat := p.Type
		lastBackendFormat = backendFormat
		if entryFormat != backendFormat {
			translated, err := g.translateRequest(body, entryFormat, backendFormat)
			if err != nil {
				return nil, err
			}
			fwdBody = translated
		}

		return g.sendToProvider(ctx, r, fwdBody, p, backendFormat)
	})
	return resp, lastBackendFormat, err
}

func (g *Gateway) sendToProvider(
	ctx context.Context,
	r *http.Request,
	body []byte,
	p *provider.Provider,
	backendFormat string,
) (*http.Response, error) {
	client, cb, err := g.proxyRegistry.Get(p.ProxyName)
	if err != nil {
		// Circuit breaker is open
		if cb != nil {
			cb.RecordFailure()
		}
		return nil, fmt.Errorf("proxy error: %w", err)
	}

	targetURL := g.buildTargetURL(p, r.URL.Path, backendFormat)
	req, err := http.NewRequestWithContext(ctx, r.Method, targetURL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}

	// Copy headers, skip hop-by-hop
	for k, vs := range r.Header {
		if hopByHopHeaders[k] {
			continue
		}
		for _, v := range vs {
			req.Header.Add(k, v)
		}
	}

	// Ensure correct Content-Type
	req.Header.Set("Content-Type", "application/json")

	g.logger.Debug("forwarding request",
		"provider", p.Name,
		"proxy", p.ProxyName,
		"target", targetURL)

	resp, err := g.doRequestWithAuthFallback(ctx, client, req, body, p)
	if err != nil {
		if cb != nil {
			cb.RecordFailure()
		}
		return nil, err
	}

	// Record success in circuit breaker
	if cb != nil && resp.StatusCode < 500 {
		cb.RecordSuccess()
	}

	return resp, nil
}

func (g *Gateway) doRequestWithAuthFallback(ctx context.Context, client *http.Client, req *http.Request, body []byte, p *provider.Provider) (*http.Response, error) {
	key := p.NextKey()

	// First attempt with provider-type default auth
	g.setAuthHeader(req, p, key)
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}

	// If unauthorized and provider is anthropic-type, retry with Bearer auth
	// This handles providers like Longcat that expect Bearer instead of x-api-key
	if resp.StatusCode == http.StatusUnauthorized && p.Type == "anthropic" {
		resp.Body.Close()
		req.Header.Set("Authorization", "Bearer "+key)
		return client.Do(req)
	}

	return resp, nil
}

func (g *Gateway) translateRequest(body []byte, from, to string) ([]byte, error) {
	if from == to {
		return body, nil
	}
	switch {
	case from == "anthropic" && to == "openai":
		return adapter.AnthropicToOpenAI(body)
	case from == "openai" && to == "anthropic":
		return adapter.OpenAIToAnthropic(body)
	case from == "gemini" && to == "openai":
		return adapter.GeminiToOpenAI(body, "")
	case from == "openai" && to == "gemini":
		return adapter.OpenAIToGemini(body)
	case from == "anthropic" && to == "gemini":
		// First convert Anthropic to OpenAI, then OpenAI to Gemini
		openAIBody, err := adapter.AnthropicToOpenAI(body)
		if err != nil {
			return nil, err
		}
		return adapter.OpenAIToGemini(openAIBody)
	case from == "gemini" && to == "anthropic":
		// First convert Gemini to OpenAI, then OpenAI to Anthropic
		openAIBody, err := adapter.GeminiToOpenAI(body, "")
		if err != nil {
			return nil, err
		}
		return adapter.OpenAIToAnthropic(openAIBody)
	default:
		return body, nil
	}
}

func (g *Gateway) buildTargetURL(p *provider.Provider, path string, backendFormat string) string {
	cleanPath := path

	// Strip format prefixes
	if strings.HasPrefix(path, "/anthropic/") {
		cleanPath = "/" + strings.TrimPrefix(path, "/anthropic/")
	}
	if strings.HasPrefix(path, "/openai/") {
		cleanPath = "/" + strings.TrimPrefix(path, "/openai/")
	}
	if strings.HasPrefix(path, "/gemini/") {
		cleanPath = "/" + strings.TrimPrefix(path, "/gemini/")
	}

	// For Gemini, use different path structure
	if backendFormat == "gemini" {
		// Gemini API uses /v1beta/models/{model}:generateContent
		if !strings.Contains(cleanPath, "/models/") {
			// Already has model in path
		}
		return strings.TrimRight(p.BaseURL, "/") + cleanPath
	}

	// Ensure /v1/ prefix for OpenAI/Anthropic formats
	if !strings.HasPrefix(cleanPath, "/v1/") {
		cleanPath = "/v1" + cleanPath
	}

	return strings.TrimRight(p.BaseURL, "/") + cleanPath
}

func (g *Gateway) setAuthHeader(req *http.Request, p *provider.Provider, key string) {
	switch p.Type {
	case "openai", "gemini":
		req.Header.Set("Authorization", "Bearer "+key)
	case "anthropic":
		req.Header.Set("x-api-key", key)
		req.Header.Set("anthropic-version", "2023-06-01")
	}
}

func (g *Gateway) HealthCheck(ctx context.Context) {
	for _, p := range g.registry.All() {
		if p.Disabled {
			continue
		}
		transport := g.proxyRegistry.GetTransport(p.ProxyName)
		err := p.CheckHealth(ctx, g.config.HealthTimeout, transport)
		if err != nil {
			g.logger.Warn("health check failed", "provider", p.Name, "error", err)
		} else {
			g.logger.Debug("health check passed", "provider", p.Name)
		}
	}
}

func writeError(w http.ResponseWriter, status int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"error": map[string]string{
			"message": msg,
			"type":    "gateway_error",
		},
	})
}

// gatewayBackoff returns a duration for gateway-level retry attempts.
// Uses exponential backoff: 2s, 4s, 8s, 16s, 30s, 30s, ...
// This is independent of backend-level retry backoff.
func (g *Gateway) gatewayBackoff(attempt int) time.Duration {
	seconds := math.Pow(2, float64(attempt))
	if seconds > 30 {
		seconds = 30
	}
	return time.Duration(seconds) * time.Second
}
