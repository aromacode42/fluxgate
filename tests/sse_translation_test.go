package tests

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/jacek/fluxgate/internal/balancer"
	"github.com/jacek/fluxgate/internal/gateway"
	"github.com/jacek/fluxgate/internal/limiter"
	"github.com/jacek/fluxgate/internal/modelrouter"
	"github.com/jacek/fluxgate/internal/provider"
	"github.com/jacek/fluxgate/internal/proxy"
	pkgconfig "github.com/jacek/fluxgate/pkg/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestIntegration_SSETranslation_OpenAIToAnthropic(t *testing.T) {
	// Simulates OpenAI backend returning SSE, client expects Anthropic format
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		fmt.Fprintf(w, "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}\n\n")
		fmt.Fprintf(w, "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi!\"}}]}\n\n")
		fmt.Fprintf(w, "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
		fmt.Fprintf(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()

	registry := provider.NewRegistry()
	registry.Add(provider.New("openai-backend", "openai", upstream.URL, []string{"test-key"}, 1))

	modelRouter, _ := modelrouter.NewRouter([]pkgconfig.ModelConfig{
		{Name: "my-model", Backends: []string{"openai-backend/gpt-4"}},
	}, registry)

	cfg := defaultGatewayConfig()
	gw := gateway.New(
		registry, balancer.NewRoundRobin(), limiter.NewRateLimiter(1000, 2000),
		cfg, proxy.NewRegistry(nil, proxy.CircuitBreakerConfig{}), modelRouter,
	)

	// Client sends Anthropic format request
	req := httptest.NewRequest(http.MethodPost, "/v1/messages",
		strings.NewReader(`{"model":"my-model","messages":[{"role":"user","content":"hi"}],"max_tokens":1024,"stream":true}`))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	gw.ServeHTTP(w, req)

	resp := w.Result()
	body, _ := io.ReadAll(resp.Body)
	result := string(body)

	assert.Equal(t, http.StatusOK, resp.StatusCode)
	assert.Contains(t, result, "event: message_start")
	assert.Contains(t, result, "event: content_block_start")
	assert.Contains(t, result, `"type":"text_delta"`)
	assert.Contains(t, result, `"text":"Hi!"`)
	assert.Contains(t, result, "event: message_stop")
}

func TestIntegration_SSEPassthrough_SameFormat(t *testing.T) {
	// Same format = no translation, direct passthrough
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"}}]}\n\n")
		fmt.Fprintf(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()

	registry := provider.NewRegistry()
	registry.Add(provider.New("openai-backend", "openai", upstream.URL, []string{"test-key"}, 1))

	cfg := defaultGatewayConfig()
	gw := gateway.New(
		registry, balancer.NewRoundRobin(), limiter.NewRateLimiter(1000, 2000),
		cfg, proxy.NewRegistry(nil, proxy.CircuitBreakerConfig{}), nil,
	)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"stream":true}`))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	gw.ServeHTTP(w, req)

	resp := w.Result()
	body, _ := io.ReadAll(resp.Body)
	result := string(body)

	assert.Equal(t, http.StatusOK, resp.StatusCode)
	// Should be raw passthrough, no translation
	assert.Contains(t, result, `"content":"Hello"`)
	assert.Contains(t, result, "data: [DONE]")
}

func TestIntegration_GatewayRetry_TimeoutThenSuccess(t *testing.T) {
	callCount := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		if callCount <= 1 {
			// First call: simulate timeout by not responding
			time.Sleep(5 * time.Second)
		}
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	}))
	defer upstream.Close()

	registry := provider.NewRegistry()
	registry.Add(provider.New("test", "openai", upstream.URL, []string{"key"}, 1))

	cfg := defaultGatewayConfig()
	cfg.RequestTimeout = 1 * time.Second // Short timeout to trigger failure fast
	cfg.GatewayRetryMax = 3

	gw := gateway.New(
		registry, balancer.NewRoundRobin(), limiter.NewRateLimiter(1000, 2000),
		cfg, proxy.NewRegistry(nil, proxy.CircuitBreakerConfig{}), nil,
	)

	req := httptest.NewRequest(http.MethodPost, "/v1/test",
		strings.NewReader(`{"model":"x","messages":[]}`))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	gw.ServeHTTP(w, req)

	// Should eventually succeed on retry
	assert.Equal(t, http.StatusOK, w.Result().StatusCode)
}

func TestIntegration_RateLimiting_UnderLoad(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	}))
	defer upstream.Close()

	registry := provider.NewRegistry()
	registry.Add(provider.New("test", "openai", upstream.URL, []string{"key"}, 1))

	cfg := defaultGatewayConfig()
	gw := gateway.New(
		registry, balancer.NewRoundRobin(), limiter.NewRateLimiter(10, 20),
		cfg, proxy.NewRegistry(nil, proxy.CircuitBreakerConfig{}), nil,
	)

	var success, rateLimited int
	for i := 0; i < 50; i++ {
		req := httptest.NewRequest(http.MethodPost, "/v1/test",
			strings.NewReader(`{"model":"x","messages":[]}`))
		w := httptest.NewRecorder()
		gw.ServeHTTP(w, req)
		if w.Result().StatusCode == http.StatusOK {
			success++
		} else if w.Result().StatusCode == http.StatusTooManyRequests {
			rateLimited++
		}
	}

	assert.True(t, success > 0, "some requests should succeed")
	assert.True(t, rateLimited > 0, "some requests should be rate limited")
}

func TestIntegration_MultiKeyRotation(t *testing.T) {
	var receivedKeys []string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedKeys = append(receivedKeys, r.Header.Get("Authorization"))
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	}))
	defer upstream.Close()

	registry := provider.NewRegistry()
	registry.Add(provider.New("test", "openai", upstream.URL, []string{"key-a", "key-b"}, 1))

	cfg := defaultGatewayConfig()
	gw := gateway.New(
		registry, balancer.NewRoundRobin(), limiter.NewRateLimiter(1000, 2000),
		cfg, proxy.NewRegistry(nil, proxy.CircuitBreakerConfig{}), nil,
	)

	for i := 0; i < 4; i++ {
		req := httptest.NewRequest(http.MethodPost, "/v1/test",
			strings.NewReader(`{"model":"x","messages":[]}`))
		w := httptest.NewRecorder()
		gw.ServeHTTP(w, req)
		require.Equal(t, http.StatusOK, w.Result().StatusCode)
	}

	assert.Contains(t, receivedKeys, "Bearer key-a")
	assert.Contains(t, receivedKeys, "Bearer key-b")
}

func TestIntegration_ModelRewrite(t *testing.T) {
	var receivedModel string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var m map[string]interface{}
		json.Unmarshal(body, &m)
		receivedModel = m["model"].(string)
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	}))
	defer upstream.Close()

	registry := provider.NewRegistry()
	registry.Add(provider.New("backend", "openai", upstream.URL, []string{"key"}, 1))

	modelRouter, _ := modelrouter.NewRouter([]pkgconfig.ModelConfig{
		{Name: "my-custom-model", Backends: []string{"backend/gpt-4o"}},
	}, registry)

	cfg := defaultGatewayConfig()
	gw := gateway.New(
		registry, balancer.NewRoundRobin(), limiter.NewRateLimiter(1000, 2000),
		cfg, proxy.NewRegistry(nil, proxy.CircuitBreakerConfig{}), modelRouter,
	)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"my-custom-model","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	gw.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Result().StatusCode)
	assert.Equal(t, "gpt-4o", receivedModel, "model should be rewritten from virtual to real")
}
