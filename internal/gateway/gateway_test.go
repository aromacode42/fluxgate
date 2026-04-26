package gateway

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/jacek/fluxgate/internal/balancer"
	"github.com/jacek/fluxgate/internal/limiter"
	"github.com/jacek/fluxgate/internal/modelrouter"
	"github.com/jacek/fluxgate/internal/provider"
	"github.com/jacek/fluxgate/internal/proxy"
	pkgconfig "github.com/jacek/fluxgate/pkg/config"
)

func defaultGWConfig() pkgconfig.GatewayConfig {
	return pkgconfig.GatewayConfig{
		BalancerType:    "round_robin",
		RetryMax:        2,
		RetryWaitMin:    10 * time.Millisecond,
		RetryWaitMax:    50 * time.Millisecond,
		GatewayRetryMax: 2,
		HealthCheck:     false,
		HealthInterval:  30 * time.Second,
		HealthTimeout:   5 * time.Second,
		RequestTimeout:  10 * time.Second,
	}
}

func setupGateway(upstream http.Handler) (*Gateway, *httptest.Server) {
	upstreamServer := httptest.NewServer(upstream)

	registry := provider.NewRegistry()
	registry.Add(provider.New("test-openai", "openai", upstreamServer.URL, []string{"test-key-1", "test-key-2"}, 1))
	registry.Add(provider.New("test-anthropic", "anthropic", upstreamServer.URL, []string{"test-key-a"}, 1))

	gw := New(
		registry,
		balancer.NewRoundRobin(),
		limiter.NewRateLimiter(1000, 2000),
		defaultGWConfig(),
		proxy.NewRegistry(nil, proxy.CircuitBreakerConfig{}),
		nil,
	)

	return gw, upstreamServer
}

func TestGateway_OpenAI_Proxy(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("expected path /v1/chat/completions, got %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-key-1" && r.Header.Get("Authorization") != "Bearer test-key-2" {
			t.Errorf("expected Bearer token, got %s", r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"id":      "chatcmpl-test",
			"object":  "chat.completion",
			"created": 1234567890,
			"choices": []map[string]interface{}{
				{"index": 0, "message": map[string]string{"role": "assistant", "content": "Hello!"}},
			},
		})
	})

	gw, upstream := setupGateway(handler)
	defer upstream.Close()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4","messages":[]}`))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	gw.ServeHTTP(w, req)

	resp := w.Result()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		t.Errorf("expected 200, got %d, body: %s", resp.StatusCode, string(body))
	}
}

func TestGateway_Anthropic_Proxy(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("x-api-key") != "test-key-a" {
			t.Errorf("expected x-api-key header, got %s", r.Header.Get("x-api-key"))
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"id":   "msg-test",
			"type": "message",
		})
	})

	gw, upstream := setupGateway(handler)
	defer upstream.Close()

	req := httptest.NewRequest(http.MethodPost, "/anthropic/v1/messages", strings.NewReader(`{"model":"claude","messages":[]}`))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	gw.ServeHTTP(w, req)

	if w.Result().StatusCode != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Result().StatusCode)
	}
}

func TestGateway_RateLimiting(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	registry := provider.NewRegistry()
	upstream := httptest.NewServer(handler)
	defer upstream.Close()
	registry.Add(provider.New("test", "openai", upstream.URL, []string{"key"}, 1))

	gw := New(
		registry, balancer.NewRoundRobin(), limiter.NewRateLimiter(1, 2),
		defaultGWConfig(), proxy.NewRegistry(nil, proxy.CircuitBreakerConfig{}), nil,
	)

	for i := 0; i < 5; i++ {
		req := httptest.NewRequest(http.MethodPost, "/v1/test", strings.NewReader(`{"model":"x"}`))
		w := httptest.NewRecorder()
		gw.ServeHTTP(w, req)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/test", strings.NewReader(`{"model":"x"}`))
	w := httptest.NewRecorder()
	gw.ServeHTTP(w, req)

	if w.Result().StatusCode != http.StatusTooManyRequests {
		t.Errorf("expected 429, got %d", w.Result().StatusCode)
	}
}

func TestGateway_NoHealthyProviders(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {})
	upstream := httptest.NewServer(handler)
	defer upstream.Close()

	registry := provider.NewRegistry()
	p := provider.New("test", "openai", upstream.URL, []string{"key"}, 1)
	p.SetHealthy(false)
	registry.Add(p)

	gw := New(
		registry, balancer.NewRoundRobin(), limiter.NewRateLimiter(1000, 2000),
		defaultGWConfig(), proxy.NewRegistry(nil, proxy.CircuitBreakerConfig{}), nil,
	)

	req := httptest.NewRequest(http.MethodPost, "/v1/test", strings.NewReader(`{"model":"x"}`))
	w := httptest.NewRecorder()
	gw.ServeHTTP(w, req)

	if w.Result().StatusCode != http.StatusBadGateway {
		t.Errorf("expected 502, got %d", w.Result().StatusCode)
	}
}

func TestGateway_RetryOn5xx(t *testing.T) {
	callCount := 0
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		if callCount < 3 {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	gw, upstream := setupGateway(handler)
	defer upstream.Close()

	req := httptest.NewRequest(http.MethodPost, "/v1/test", strings.NewReader(`{"model":"x"}`))
	w := httptest.NewRecorder()
	gw.ServeHTTP(w, req)

	if w.Result().StatusCode != http.StatusOK {
		t.Errorf("expected 200 after retries, got %d", w.Result().StatusCode)
	}
}

func TestGateway_HealthCheck(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	gw, upstream := setupGateway(handler)
	defer upstream.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	gw.HealthCheck(ctx)

	if len(gw.registry.Healthy()) == 0 {
		t.Error("expected healthy providers")
	}
}

func TestGateway_ModelRouting_BackendFailover(t *testing.T) {
	callCount := 0
	var receivedModel string

	backend1 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		body, _ := io.ReadAll(r.Body)
		var m map[string]interface{}
		json.Unmarshal(body, &m)
		receivedModel = m["model"].(string)
		w.WriteHeader(http.StatusInternalServerError) // fail
	}))
	defer backend1.Close()

	backend2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		body, _ := io.ReadAll(r.Body)
		var m map[string]interface{}
		json.Unmarshal(body, &m)
		receivedModel = m["model"].(string)
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]string{"status": "ok", "model": receivedModel})
	}))
	defer backend2.Close()

	registry := provider.NewRegistry()
	registry.Add(provider.New("backend1", "openai", backend1.URL, []string{"key1"}, 1))
	registry.Add(provider.New("backend2", "openai", backend2.URL, []string{"key2"}, 1))

	modelRouter, _ := modelrouter.NewRouter([]pkgconfig.ModelConfig{
		{Name: "gpt-5", Backends: []string{"backend1/gpt-4o", "backend2/gpt-4o"}},
	}, registry)

	gw := New(
		registry, balancer.NewRoundRobin(), limiter.NewRateLimiter(1000, 2000),
		defaultGWConfig(), proxy.NewRegistry(nil, proxy.CircuitBreakerConfig{}), modelRouter,
	)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-5","messages":[]}`))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	gw.ServeHTTP(w, req)

	if w.Result().StatusCode != http.StatusOK {
		t.Errorf("expected 200 after failover, got %d", w.Result().StatusCode)
	}
	if callCount != 2 {
		t.Errorf("expected 2 attempts, got %d", callCount)
	}
	if receivedModel != "gpt-4o" {
		t.Errorf("expected model rewritten to gpt-4o, got %s", receivedModel)
	}
}

func TestGateway_ModelRouting_AllBackendsDown(t *testing.T) {
	backend1 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer backend1.Close()

	backend2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
	}))
	defer backend2.Close()

	registry := provider.NewRegistry()
	registry.Add(provider.New("b1", "openai", backend1.URL, []string{"key1"}, 1))
	registry.Add(provider.New("b2", "openai", backend2.URL, []string{"key2"}, 1))

	modelRouter, _ := modelrouter.NewRouter([]pkgconfig.ModelConfig{
		{Name: "gpt-5", Backends: []string{"b1/gpt-4o", "b2/gpt-4o"}},
	}, registry)

	gw := New(
		registry, balancer.NewRoundRobin(), limiter.NewRateLimiter(1000, 2000),
		defaultGWConfig(), proxy.NewRegistry(nil, proxy.CircuitBreakerConfig{}), modelRouter,
	)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-5","messages":[]}`))
	w := httptest.NewRecorder()
	gw.ServeHTTP(w, req)

	if w.Result().StatusCode != http.StatusBadGateway {
		t.Errorf("expected 502 when all backends fail, got %d", w.Result().StatusCode)
	}
}

func TestGateway_ModelRouting_ModelRewrite(t *testing.T) {
	var receivedModel string

	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var m map[string]interface{}
		json.Unmarshal(body, &m)
		receivedModel = m["model"].(string)
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	}))
	defer backend.Close()

	registry := provider.NewRegistry()
	registry.Add(provider.New("prov", "openai", backend.URL, []string{"key"}, 1))

	modelRouter, _ := modelrouter.NewRouter([]pkgconfig.ModelConfig{
		{Name: "my-custom-model", Backends: []string{"prov/gpt-4o"}},
	}, registry)

	gw := New(
		registry, balancer.NewRoundRobin(), limiter.NewRateLimiter(1000, 2000),
		defaultGWConfig(), proxy.NewRegistry(nil, proxy.CircuitBreakerConfig{}), modelRouter,
	)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"my-custom-model","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	gw.ServeHTTP(w, req)

	if w.Result().StatusCode != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Result().StatusCode)
	}
	if receivedModel != "gpt-4o" {
		t.Errorf("expected model rewritten to gpt-4o, got %s", receivedModel)
	}
}

func TestGateway_ModelRouting_Passthrough(t *testing.T) {
	// Model not in router → falls through to passthrough with type matching
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	gw, upstream := setupGateway(handler)
	defer upstream.Close()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-4-turbo","messages":[]}`))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	gw.ServeHTTP(w, req)

	if w.Result().StatusCode != http.StatusOK {
		t.Errorf("expected 200 for passthrough, got %d", w.Result().StatusCode)
	}
}

func TestGateway_DetectEntryFormat(t *testing.T) {
	gw := &Gateway{}
	tests := []struct{ path, expected string }{
		{"/v1/chat/completions", "openai"},
		{"/v1/models", "openai"},
		{"/v1/messages", "anthropic"},
		{"/anthropic/v1/messages", "anthropic"},
		{"/openai/v1/chat/completions", "openai"},
	}
	for _, tc := range tests {
		if got := gw.detectEntryFormat(tc.path); got != tc.expected {
			t.Errorf("path %s: expected %q, got %q", tc.path, tc.expected, got)
		}
	}
}

func TestGateway_BuildTargetURL(t *testing.T) {
	gw := &Gateway{}
	tests := []struct {
		pt, base, path, backendFormat, expected string
	}{
		{"openai", "https://api.openai.com", "/v1/chat/completions", "openai", "https://api.openai.com/v1/chat/completions"},
		{"openai", "https://api.openai.com", "/v1/chat/completions", "openai", "https://api.openai.com/v1/chat/completions"},
		{"anthropic", "https://api.anthropic.com", "/v1/messages", "anthropic", "https://api.anthropic.com/v1/messages"},
		{"anthropic", "https://api.anthropic.com", "/anthropic/v1/messages", "anthropic", "https://api.anthropic.com/v1/messages"},
	}
	for _, tc := range tests {
		p := provider.New("test", tc.pt, tc.base, []string{"key"}, 1)
		if got := gw.buildTargetURL(p, tc.path, tc.backendFormat); got != tc.expected {
			t.Errorf("type=%s path=%s: expected %s, got %s", tc.pt, tc.path, tc.expected, got)
		}
	}
}

func TestGateway_GatewayRetry_SucceedsOnRetry(t *testing.T) {
	callCount := 0
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		if callCount < 3 {
			w.WriteHeader(http.StatusBadGateway)
			return
		}
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	gw, upstream := setupGateway(handler)
	defer upstream.Close()

	req := httptest.NewRequest(http.MethodPost, "/v1/test", strings.NewReader(`{"model":"x"}`))
	w := httptest.NewRecorder()
	gw.ServeHTTP(w, req)

	if w.Result().StatusCode != http.StatusOK {
		t.Errorf("expected 200 after gateway retries, got %d", w.Result().StatusCode)
	}
	if callCount < 3 {
		t.Errorf("expected at least 3 attempts, got %d", callCount)
	}
}

func TestGateway_GatewayRetry_Exhausted(t *testing.T) {
	callCount := 0
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		w.WriteHeader(http.StatusBadGateway)
	})

	gw, upstream := setupGateway(handler)
	defer upstream.Close()

	req := httptest.NewRequest(http.MethodPost, "/v1/test", strings.NewReader(`{"model":"x"}`))
	w := httptest.NewRecorder()
	gw.ServeHTTP(w, req)

	if w.Result().StatusCode != http.StatusBadGateway {
		t.Errorf("expected 502 after all retries exhausted, got %d", w.Result().StatusCode)
	}
}

func TestGateway_GatewayRetry_StopsOnClientDisconnect(t *testing.T) {
	callCount := 0
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		w.WriteHeader(http.StatusBadGateway)
	})

	cfg := defaultGWConfig()
	cfg.GatewayRetryMax = 0 // infinite

	upstream := httptest.NewServer(handler)
	defer upstream.Close()

	registry := provider.NewRegistry()
	registry.Add(provider.New("test", "openai", upstream.URL, []string{"key"}, 1))

	gw := New(
		registry, balancer.NewRoundRobin(), limiter.NewRateLimiter(1000, 2000),
		cfg, proxy.NewRegistry(nil, proxy.CircuitBreakerConfig{}), nil,
	)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	req := httptest.NewRequest(http.MethodPost, "/v1/test", strings.NewReader(`{"model":"x"}`)).WithContext(ctx)
	w := httptest.NewRecorder()
	gw.ServeHTTP(w, req)

	// Should stop after context expires, not hang forever
	// With 2s,4s backoff, only 1 retry should fit in 2s timeout
	if callCount > 5 {
		t.Errorf("expected bounded retries with backoff, got %d calls", callCount)
	}
}

func TestGateway_GatewayBackoff(t *testing.T) {
	gw := &Gateway{}

	tests := []struct {
		attempt  int
		expected time.Duration
	}{
		{1, 2 * time.Second},
		{2, 4 * time.Second},
		{3, 8 * time.Second},
		{4, 16 * time.Second},
		{5, 30 * time.Second},
		{10, 30 * time.Second},
		{100, 30 * time.Second},
	}

	for _, tc := range tests {
		got := gw.gatewayBackoff(tc.attempt)
		if got != tc.expected {
			t.Errorf("attempt %d: expected %v, got %v", tc.attempt, tc.expected, got)
		}
	}
}
