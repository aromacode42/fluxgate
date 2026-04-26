package provider

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// ============== DetectProviderType tests ==============

func TestDetectProviderType_OpenAI(t *testing.T) {
	tests := []struct {
		url      string
		expected string
	}{
		{"https://api.openai.com", "openai"},
		{"https://api.openai.com/v1", "openai"},
		{"https://openai.com/api", "openai"},
		{"HTTP://OPENAI.COM", "openai"}, // case insensitive
	}
	for _, tc := range tests {
		t.Run(tc.url, func(t *testing.T) {
			result := DetectProviderType(tc.url)
			assert.Equal(t, tc.expected, result)
		})
	}
}

func TestDetectProviderType_Anthropic(t *testing.T) {
	tests := []struct {
		url      string
		expected string
	}{
		{"https://api.anthropic.com", "anthropic"},
		{"https://console.anthropic.com", "anthropic"},
		{"HTTP://API.ANTHROPIC.COM", "anthropic"},
	}
	for _, tc := range tests {
		t.Run(tc.url, func(t *testing.T) {
			result := DetectProviderType(tc.url)
			assert.Equal(t, tc.expected, result)
		})
	}
}

func TestDetectProviderType_Gemini(t *testing.T) {
	tests := []struct {
		url      string
		expected string
	}{
		{"https://generativelanguage.googleapis.com", "gemini"},
		{"https://googleapis.com", "gemini"},
		{"https://gemini.google.com", "gemini"},
	}
	for _, tc := range tests {
		t.Run(tc.url, func(t *testing.T) {
			result := DetectProviderType(tc.url)
			assert.Equal(t, tc.expected, result)
		})
	}
}

func TestDetectProviderType_Default(t *testing.T) {
	// Unknown URLs default to openai
	result := DetectProviderType("https://custom.api.com")
	assert.Equal(t, "openai", result)
}

// ============== Provider model support tests ==============

func TestSupportsModel_EmptyList(t *testing.T) {
	p := New("test", "openai", "https://api.com", []string{"key1"}, 1)
	// Empty models list means all models supported
	assert.True(t, p.SupportsModel("gpt-4"))
	assert.True(t, p.SupportsModel("any-model"))
	assert.True(t, p.SupportsModel(""))
}

func TestSupportsModel_WithList(t *testing.T) {
	p := New("test", "openai", "https://api.com", []string{"key1"}, 1)
	p.Models = []string{"gpt-4", "gpt-3.5"}

	assert.True(t, p.SupportsModel("gpt-4"))
	assert.True(t, p.SupportsModel("gpt-3.5"))
	assert.False(t, p.SupportsModel("claude-3"))
	assert.False(t, p.SupportsModel("unknown"))
}

// ============== Provider key rotation tests ==============

func TestNextKey_SingleKey(t *testing.T) {
	p := New("test", "openai", "https://api.com", []string{"only-key"}, 1)
	// Same key should be returned every time
	for i := 0; i < 5; i++ {
		assert.Equal(t, "only-key", p.NextKey())
	}
}

func TestNextKey_TwoKeys(t *testing.T) {
	p := New("test", "openai", "https://api.com", []string{"key-a", "key-b"}, 1)
	keys := make(map[string]int)
	for i := 0; i < 100; i++ {
		keys[p.NextKey()]++
	}
	// Should alternate roughly equally
	assert.Equal(t, 50, keys["key-a"])
	assert.Equal(t, 50, keys["key-b"])
}

func TestNextKey_ThreeKeys(t *testing.T) {
	p := New("test", "openai", "https://api.com", []string{"a", "b", "c"}, 1)
	// With 3 keys and 30 requests, each should appear ~10 times
	keys := make(map[string]int)
	for i := 0; i < 30; i++ {
		keys[p.NextKey()]++
	}
	assert.Equal(t, 10, keys["a"])
	assert.Equal(t, 10, keys["b"])
	assert.Equal(t, 10, keys["c"])
}

// ============== Health check tests ==============

func TestCheckHealth_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify auth header
		auth := r.Header.Get("Authorization")
		assert.Equal(t, "Bearer test-key", auth)
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"data": []map[string]interface{}{
				{"id": "gpt-4", "object": "model"},
			},
		})
	}))
	defer server.Close()

	p := New("test", "openai", server.URL, []string{"test-key"}, 1)
	transport := http.DefaultTransport.(*http.Transport).Clone()
	err := p.CheckHealth(context.Background(), 5*time.Second, transport)
	assert.NoError(t, err)
	assert.True(t, p.IsHealthy())
}

func TestCheckHealth_AnthropicSuccess(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify anthropic headers
		assert.Equal(t, "test-key", r.Header.Get("x-api-key"))
		assert.Equal(t, "2023-06-01", r.Header.Get("anthropic-version"))
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	p := New("test", "anthropic", server.URL, []string{"test-key"}, 1)
	transport := http.DefaultTransport.(*http.Transport).Clone()
	err := p.CheckHealth(context.Background(), 5*time.Second, transport)
	assert.NoError(t, err)
	assert.True(t, p.IsHealthy())
}

func TestCheckHealth_AnthropicBearerFallback(t *testing.T) {
	first := true
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if first {
			// First request: x-api-key returns 401
			w.WriteHeader(http.StatusUnauthorized)
			first = false
			return
		}
		// Second request: Bearer auth works
		auth := r.Header.Get("Authorization")
		assert.Equal(t, "Bearer test-key", auth)
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	p := New("test", "anthropic", server.URL, []string{"test-key"}, 1)
	transport := http.DefaultTransport.(*http.Transport).Clone()
	err := p.CheckHealth(context.Background(), 5*time.Second, transport)
	assert.NoError(t, err)
	assert.True(t, p.IsHealthy())
}

func TestCheckHealth_ServerError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	p := New("test", "openai", server.URL, []string{"test-key"}, 1)
	transport := http.DefaultTransport.(*http.Transport).Clone()
	err := p.CheckHealth(context.Background(), 5*time.Second, transport)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "500")
	assert.False(t, p.IsHealthy())
}

func TestCheckHealth_ContextCanceled(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(10 * time.Second)
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	p := New("test", "openai", server.URL, []string{"test-key"}, 1)
	transport := http.DefaultTransport.(*http.Transport).Clone()
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	err := p.CheckHealth(ctx, 5*time.Second, transport)
	assert.Error(t, err)
	assert.False(t, p.IsHealthy())
}

func TestCheckHealth_TransportError(t *testing.T) {
	p := New("test", "openai", "http://localhost:99999", []string{"test-key"}, 1)
	transport := http.DefaultTransport.(*http.Transport).Clone()
	err := p.CheckHealth(context.Background(), 100*time.Millisecond, transport)
	assert.Error(t, err)
	assert.False(t, p.IsHealthy())
}

func TestCheckHealth_HealthURL_Gemini(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Contains(t, r.URL.Path, "/v1beta/models")
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	p := New("test", "gemini", server.URL, []string{"test-key"}, 1)
	transport := http.DefaultTransport.(*http.Transport).Clone()
	err := p.CheckHealth(context.Background(), 5*time.Second, transport)
	assert.NoError(t, err)
}

// ============== Registry edge cases ==============

func TestRegistry_Empty(t *testing.T) {
	reg := NewRegistry()
	assert.Empty(t, reg.All())
	assert.Empty(t, reg.Healthy())
	assert.Nil(t, reg.GetByName("nonexistent"))
}

func TestRegistry_AddMultiple(t *testing.T) {
	reg := NewRegistry()
	for i := 0; i < 5; i++ {
		p := New("p"+string(rune('0'+i)), "openai", "https://api.com", []string{"key"}, 1)
		reg.Add(p)
	}
	assert.Len(t, reg.All(), 5)
	assert.Len(t, reg.Healthy(), 5)
}

func TestRegistry_HealthyFiltering(t *testing.T) {
	reg := NewRegistry()
	providers := make([]*Provider, 5)
	for i := 0; i < 5; i++ {
		p := New("p"+string(rune('0'+i)), "openai", "https://api.com", []string{"key"}, 1)
		if i%2 == 0 {
			p.SetHealthy(false)
		}
		providers[i] = p
		reg.Add(p)
	}

	healthy := reg.Healthy()
	assert.Len(t, healthy, 2) // p0 and p2 are unhealthy
}

func TestRegistry_GetByName_NotFound(t *testing.T) {
	reg := NewRegistry()
	p := New("existing", "openai", "https://api.com", []string{"key"}, 1)
	reg.Add(p)

	result := reg.GetByName("nonexistent")
	assert.Nil(t, result)
}

func TestRegistry_All_ReturnsCopy(t *testing.T) {
	reg := NewRegistry()
	p := New("test", "openai", "https://api.com", []string{"key"}, 1)
	reg.Add(p)

	all1 := reg.All()
	all2 := reg.All()
	assert.Equal(t, all1, all2)
	// The slice is a copy but elements point to same objects
	assert.Same(t, all1[0], all2[0])
}

// ============== Provider struct edge cases ==============

func TestProvider_ProxyName(t *testing.T) {
	p := New("test", "openai", "https://api.com", []string{"key"}, 1)
	assert.Equal(t, "", p.ProxyName)
	p.ProxyName = "us-proxy"
	assert.Equal(t, "us-proxy", p.ProxyName)
}

func TestProvider_DisabledVsHealthy(t *testing.T) {
	p := New("test", "openai", "https://api.com", []string{"key"}, 1)

	// Healthy by default
	assert.True(t, p.IsHealthy())

	// Disabled takes precedence even when healthy
	p.Disabled = true
	assert.False(t, p.IsHealthy())

	// Can still set healthy=true but disabled=true overrides
	p.SetHealthy(true)
	assert.False(t, p.IsHealthy())
}

// ============== CheckHealth edge cases ==============

func TestCheckHealth_InvalidURL(t *testing.T) {
	p := New("test", "openai", "://invalid-url", []string{"key"}, 1)
	transport := http.DefaultTransport.(*http.Transport).Clone()
	err := p.CheckHealth(context.Background(), 5*time.Second, transport)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "creating health check request")
}
