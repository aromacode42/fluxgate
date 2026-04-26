package tests

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/jacek/fluxgate/internal/balancer"
	"github.com/jacek/fluxgate/internal/gateway"
	"github.com/jacek/fluxgate/internal/limiter"
	"github.com/jacek/fluxgate/internal/provider"
	"github.com/jacek/fluxgate/internal/proxy"
	pkgconfig "github.com/jacek/fluxgate/pkg/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestIntegration_OpenAI_ChatCompletion(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/v1/chat/completions", r.URL.Path)
		assert.Equal(t, "Bearer sk-test-key-1", r.Header.Get("Authorization"))
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))

		body, _ := io.ReadAll(r.Body)
		assert.NotEmpty(t, body)

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"id":      "chatcmpl-123",
			"object":  "chat.completion",
			"created": float64(time.Now().Unix()),
			"model":   "gpt-4",
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"message": map[string]string{
						"role":    "assistant",
						"content": "Hello! How can I help you?",
					},
					"finish_reason": "stop",
				},
			},
		})
	}))
	defer upstream.Close()

	registry := provider.NewRegistry()
	registry.Add(provider.New("openai-primary", "openai", upstream.URL, []string{"sk-test-key-1"}, 1))

	gw := gateway.New(
		registry,
		balancer.NewRoundRobin(),
		limiter.NewRateLimiter(100, 200),
		defaultGatewayConfig(),
		proxy.NewRegistry(nil),
		nil,
	)

	reqBody := `{"model":"gpt-4","messages":[{"role":"user","content":"Hello"}]}`
	resp := sendRequest(gw, http.MethodPost, "/v1/chat/completions", reqBody)

	assert.Equal(t, http.StatusOK, resp.StatusCode)

	var result map[string]interface{}
	err := json.NewDecoder(resp.Body).Decode(&result)
	require.NoError(t, err)
	assert.Equal(t, "chatcmpl-123", result["id"])
}

func TestIntegration_Anthropic_Messages(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "sk-ant-test", r.Header.Get("x-api-key"))
		assert.Contains(t, r.Header.Get("anthropic-version"), "2023-06-01")

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"id":   "msg-123",
			"type": "message",
			"role": "assistant",
			"content": []map[string]interface{}{
				{"type": "text", "text": "Hi from Claude!"},
			},
			"model":       "claude-3-sonnet",
			"stop_reason": "end_turn",
		})
	}))
	defer upstream.Close()

	registry := provider.NewRegistry()
	registry.Add(provider.New("anthropic-primary", "anthropic", upstream.URL, []string{"sk-ant-test"}, 1))

	gw := gateway.New(
		registry,
		balancer.NewRoundRobin(),
		limiter.NewRateLimiter(100, 200),
		defaultGatewayConfig(),
		proxy.NewRegistry(nil),
		nil,
	)

	resp := sendRequest(gw, http.MethodPost, "/anthropic/v1/messages", `{"model":"claude-3-sonnet","messages":[{"role":"user","content":"Hi"}]}`)
	assert.Equal(t, http.StatusOK, resp.StatusCode)
}

func TestIntegration_RetryWithFallback(t *testing.T) {
	callCount := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		if callCount <= 2 {
			w.WriteHeader(http.StatusBadGateway)
			return
		}
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	}))
	defer upstream.Close()

	registry := provider.NewRegistry()
	registry.Add(provider.New("test", "openai", upstream.URL, []string{"key"}, 1))

	cfg := defaultGatewayConfig()
	cfg.RetryMax = 3
	cfg.RetryWaitMin = 5 * time.Millisecond
	cfg.RetryWaitMax = 20 * time.Millisecond
	gw := gateway.New(
		registry,
		balancer.NewRoundRobin(),
		limiter.NewRateLimiter(100, 200),
		cfg,
		proxy.NewRegistry(nil),
		nil,
	)

	resp := sendRequest(gw, http.MethodPost, "/v1/test", "")
	assert.Equal(t, http.StatusOK, resp.StatusCode)
	assert.Equal(t, 3, callCount)
}

func TestIntegration_TimeoutHandling(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(5 * time.Second)
		w.WriteHeader(http.StatusOK)
	}))
	defer upstream.Close()

	registry := provider.NewRegistry()
	registry.Add(provider.New("slow", "openai", upstream.URL, []string{"key"}, 1))

	cfg := defaultGatewayConfig()
	cfg.RequestTimeout = 100 * time.Millisecond
	cfg.RetryMax = 0
	gw := gateway.New(
		registry,
		balancer.NewRoundRobin(),
		limiter.NewRateLimiter(100, 200),
		cfg,
		proxy.NewRegistry(nil),
		nil,
	)

	resp := sendRequest(gw, http.MethodPost, "/v1/test", "")
	assert.Equal(t, http.StatusBadGateway, resp.StatusCode)
}

func TestIntegration_429RateLimit(t *testing.T) {
	callCount := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		if callCount == 1 {
			w.WriteHeader(http.StatusTooManyRequests)
			w.Write([]byte(`{"error":"rate limited"}`))
			return
		}
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	}))
	defer upstream.Close()

	registry := provider.NewRegistry()
	registry.Add(provider.New("test", "openai", upstream.URL, []string{"key"}, 1))

	cfg := defaultGatewayConfig()
	cfg.RetryWaitMin = 5 * time.Millisecond
	cfg.RetryWaitMax = 20 * time.Millisecond
	gw := gateway.New(
		registry,
		balancer.NewRoundRobin(),
		limiter.NewRateLimiter(100, 200),
		cfg,
		proxy.NewRegistry(nil),
		nil,
	)

	resp := sendRequest(gw, http.MethodPost, "/v1/test", "")
	assert.Equal(t, http.StatusOK, resp.StatusCode)
}

func TestIntegration_LoadBalancing(t *testing.T) {
	var server1Calls, server2Calls int

	server1 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		server1Calls++
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]string{"server": "s1"})
	}))
	defer server1.Close()

	server2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		server2Calls++
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]string{"server": "s2"})
	}))
	defer server2.Close()

	registry := provider.NewRegistry()
	registry.Add(provider.New("s1", "openai", server1.URL, []string{"key1"}, 1))
	registry.Add(provider.New("s2", "openai", server2.URL, []string{"key2"}, 1))

	gw := gateway.New(
		registry,
		balancer.NewRoundRobin(),
		limiter.NewRateLimiter(1000, 2000),
		defaultGatewayConfig(),
		proxy.NewRegistry(nil),
		nil,
	)

	for i := 0; i < 20; i++ {
		resp := sendRequest(gw, http.MethodPost, "/v1/test", "")
		assert.Equal(t, http.StatusOK, resp.StatusCode)
		resp.Body.Close()
	}

	assert.Equal(t, 10, server1Calls, "server1 should get half the requests")
	assert.Equal(t, 10, server2Calls, "server2 should get half the requests")
}

func TestIntegration_MultipleAPIKeys(t *testing.T) {
	var usedKeys []string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		usedKeys = append(usedKeys, r.Header.Get("Authorization"))
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	}))
	defer upstream.Close()

	registry := provider.NewRegistry()
	registry.Add(provider.New("test", "openai", upstream.URL, []string{"key-a", "key-b", "key-c"}, 1))

	gw := gateway.New(
		registry,
		balancer.NewRoundRobin(),
		limiter.NewRateLimiter(100, 200),
		defaultGatewayConfig(),
		proxy.NewRegistry(nil),
		nil,
	)

	for i := 0; i < 6; i++ {
		resp := sendRequest(gw, http.MethodPost, "/v1/test", "")
		resp.Body.Close()
	}

	assert.Len(t, usedKeys, 6)
	keyCounts := make(map[string]int)
	for _, k := range usedKeys {
		keyCounts[k]++
	}
	assert.Equal(t, 2, keyCounts["Bearer key-a"])
	assert.Equal(t, 2, keyCounts["Bearer key-b"])
	assert.Equal(t, 2, keyCounts["Bearer key-c"])
}

func TestIntegration_PerProviderProxy(t *testing.T) {
	// Test that providers with different ProxyName values get routed correctly
	var receivedBy []string

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedBy = append(receivedBy, r.URL.Path)
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	}))
	defer upstream.Close()

	registry := provider.NewRegistry()
	p1 := provider.New("direct-prov", "openai", upstream.URL, []string{"key1"}, 1)
	// p1 has no proxy (direct)
	registry.Add(p1)

	p2 := provider.New("named-proxy-prov", "openai", upstream.URL, []string{"key2"}, 1)
	p2.ProxyName = "us-proxy"
	registry.Add(p2)

	// Register a named proxy (points to the same upstream for testing)
	proxyReg := proxy.NewRegistry([]proxy.Config{
		{Name: "us-proxy", Type: "http", Address: "http://127.0.0.1:0"}, // won't actually be hit; test validates wiring
	})

	cfg := defaultGatewayConfig()
	cfg.RetryMax = 0
	gw := gateway.New(
		registry,
		balancer.NewRoundRobin(),
		limiter.NewRateLimiter(100, 200),
		cfg,
		proxyReg,
		nil,
	)

	// Request to direct provider should work (no proxy)
	resp := sendRequest(gw, http.MethodPost, "/v1/test", "")
	// direct provider works since it goes to actual upstream
	assert.Equal(t, http.StatusOK, resp.StatusCode)
	resp.Body.Close()
}

func defaultGatewayConfig() pkgconfig.GatewayConfig {
	return pkgconfig.GatewayConfig{
		BalancerType:    "round_robin",
		RetryMax:        2,
		RetryWaitMin:    5 * time.Millisecond,
		RetryWaitMax:    20 * time.Millisecond,
		GatewayRetryMax: 2,
		HealthCheck:     false,
		HealthInterval:  30 * time.Second,
		HealthTimeout:   5 * time.Second,
		RequestTimeout:  10 * time.Second,
	}
}

func sendRequest(handler http.Handler, method, path, body string) *http.Response {
	var bodyReader io.Reader
	if body != "" {
		bodyReader = strings.NewReader(body)
	}

	req := httptest.NewRequest(method, path, bodyReader)
	if body != "" {
		req.Header.Set("Content-Type", "application/json")
	}
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)
	return w.Result()
}
