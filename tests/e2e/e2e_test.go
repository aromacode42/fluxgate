package e2e

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/jacek/fluxgate/internal/balancer"
	"github.com/jacek/fluxgate/internal/gateway"
	"github.com/jacek/fluxgate/internal/limiter"
	"github.com/jacek/fluxgate/internal/provider"
	"github.com/jacek/fluxgate/internal/proxy"
	"github.com/jacek/fluxgate/pkg/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestE2E_ServerStartup(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"id":      "chatcmpl-e2e",
			"object":  "chat.completion",
			"created": float64(time.Now().Unix()),
			"model":   "gpt-4",
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"message": map[string]string{
						"role":    "assistant",
						"content": "E2E test response!",
					},
					"finish_reason": "stop",
				},
			},
		})
	}))
	defer upstream.Close()

	registry := provider.NewRegistry()
	registry.Add(provider.New("openai-mock", "openai", upstream.URL, []string{"sk-e2e-test-key"}, 1))

	gwCfg := config.GatewayConfig{
		BalancerType:    "round_robin",
		RetryMax:        1,
		RetryWaitMin:    5 * time.Millisecond,
		RetryWaitMax:    20 * time.Millisecond,
		GatewayRetryMax: 2,
		HealthCheck:     false,
		HealthInterval:  30 * time.Second,
		HealthTimeout:   5 * time.Second,
		RequestTimeout:  10 * time.Second,
	}

	gw := gateway.New(
		registry,
		balancer.NewRoundRobin(),
		limiter.NewRateLimiter(100, 200),
		gwCfg,
		proxy.NewRegistry(nil),
		nil,
	)

	mux := http.NewServeMux()
	mux.Handle("/", gw)
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status":"ok"}`))
	})

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)

	srv := &http.Server{Handler: mux}
	go srv.Serve(listener)
	defer srv.Shutdown(context.Background())

	addr := listener.Addr().String()

	resp, err := http.Get(fmt.Sprintf("http://%s/health", addr))
	require.NoError(t, err)
	defer resp.Body.Close()

	assert.Equal(t, http.StatusOK, resp.StatusCode)
	body, _ := io.ReadAll(resp.Body)
	assert.Contains(t, string(body), "ok")

	reqBody := `{"model":"gpt-4","messages":[{"role":"user","content":"Hello"}]}`
	resp2, err := http.Post(
		fmt.Sprintf("http://%s/v1/chat/completions", addr),
		"application/json",
		strings.NewReader(reqBody),
	)
	require.NoError(t, err)
	defer resp2.Body.Close()

	assert.Equal(t, http.StatusOK, resp2.StatusCode)
	var result map[string]interface{}
	err = json.NewDecoder(resp2.Body).Decode(&result)
	require.NoError(t, err)
	assert.Equal(t, "chatcmpl-e2e", result["id"])
}

func TestE2E_ConfigValidation(t *testing.T) {
	t.Run("no providers", func(t *testing.T) {
		cfg := &config.Config{
			Server: config.ServerConfig{Port: 8080},
		}
		err := cfg.Validate()
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "no providers")
	})

	t.Run("invalid port", func(t *testing.T) {
		cfg := &config.Config{
			Server: config.ServerConfig{Port: 0},
			Providers: []config.ProviderConfig{
				{Name: "test", Type: "openai", BaseURL: "https://api.openai.com", APIKeys: []string{"key"}},
			},
		}
		err := cfg.Validate()
		assert.Error(t, err)
	})

	t.Run("valid config", func(t *testing.T) {
		cfg := &config.Config{
			Server: config.ServerConfig{Port: 8080},
			Providers: []config.ProviderConfig{
				{Name: "openai", Type: "openai", BaseURL: "https://api.openai.com", APIKeys: []string{"sk-key"}, Weight: 1},
			},
		}
		err := cfg.Validate()
		assert.NoError(t, err)
	})

	t.Run("unknown proxy reference", func(t *testing.T) {
		cfg := &config.Config{
			Server: config.ServerConfig{Port: 8080},
			Providers: []config.ProviderConfig{
				{Name: "openai", Type: "openai", BaseURL: "https://api.openai.com", APIKeys: []string{"sk-key"}, Proxy: "nonexistent"},
			},
		}
		err := cfg.Validate()
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "unknown proxy")
	})

	t.Run("duplicate proxy name", func(t *testing.T) {
		cfg := &config.Config{
			Server: config.ServerConfig{Port: 8080},
			Proxies: []config.ProxyConfig{
				{Name: "my-proxy", Type: "http", Address: "http://a:7890"},
				{Name: "my-proxy", Type: "socks5", Address: "127.0.0.1:1080"},
			},
			Providers: []config.ProviderConfig{
				{Name: "openai", Type: "openai", BaseURL: "https://api.openai.com", APIKeys: []string{"sk-key"}},
			},
		}
		err := cfg.Validate()
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "duplicate proxy")
	})

	t.Run("invalid proxy type", func(t *testing.T) {
		cfg := &config.Config{
			Server: config.ServerConfig{Port: 8080},
			Proxies: []config.ProxyConfig{
				{Name: "bad", Type: "ftp", Address: "ftp://bad"},
			},
			Providers: []config.ProviderConfig{
				{Name: "openai", Type: "openai", BaseURL: "https://api.openai.com", APIKeys: []string{"sk-key"}},
			},
		}
		err := cfg.Validate()
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "type must be")
	})
}

func TestE2E_DefaultConfig(t *testing.T) {
	cfg := config.DefaultConfig()
	assert.Equal(t, 8080, cfg.Server.Port)
	assert.Equal(t, "round_robin", cfg.Gateway.BalancerType)
	assert.True(t, cfg.Gateway.HealthCheck)
}

func TestE2E_ConfigFromYAML(t *testing.T) {
	yamlContent := `
server:
  host: "127.0.0.1"
  port: 9090

proxies:
  - name: "us-proxy"
    type: "http"
    address: "http://127.0.0.1:7890"
  - name: "jp-socks"
    type: "socks5"
    address: "127.0.0.1:1080"
    username: "user"
    password: "pass"

providers:
  - name: "openai"
    type: "openai"
    base_url: "https://api.openai.com"
    api_keys:
      - "sk-test-1"
      - "sk-test-2"
    proxy: "us-proxy"
    weight: 2
  - name: "anthropic"
    type: "anthropic"
    base_url: "https://api.anthropic.com"
    api_keys:
      - "sk-ant-test"
    proxy: "jp-socks"

gateway:
  balancer_type: "weighted"
  retry_max: 5
  health_check: false
  request_timeout: 30s
`
	tmpFile, err := os.CreateTemp("", "fluxgate-test-*.yaml")
	require.NoError(t, err)
	defer os.Remove(tmpFile.Name())

	_, err = tmpFile.WriteString(yamlContent)
	require.NoError(t, err)
	tmpFile.Close()

	cfg, err := config.Load(tmpFile.Name())
	require.NoError(t, err)

	assert.Equal(t, 9090, cfg.Server.Port)
	assert.Len(t, cfg.Proxies, 2)
	assert.Len(t, cfg.Providers, 2)
	assert.Equal(t, "openai", cfg.Providers[0].Name)
	assert.Equal(t, "us-proxy", cfg.Providers[0].Proxy)
	assert.Equal(t, "jp-socks", cfg.Providers[1].Proxy)
	assert.Equal(t, "weighted", cfg.Gateway.BalancerType)
	assert.Equal(t, 5, cfg.Gateway.RetryMax)
	assert.Equal(t, []string{"sk-test-1", "sk-test-2"}, cfg.Providers[0].APIKeys)

	// Validate resolves correctly
	require.NoError(t, cfg.Validate())
	p := cfg.ResolveProxy("us-proxy")
	require.NotNil(t, p)
	assert.Equal(t, "http", p.Type)
	assert.Equal(t, "http://127.0.0.1:7890", p.Address)

	p2 := cfg.ResolveProxy("jp-socks")
	require.NotNil(t, p2)
	assert.Equal(t, "socks5", p2.Type)
	assert.Equal(t, "user", p2.Username)

	// Total key count
	assert.Equal(t, 3, cfg.TotalKeyCount())
}
