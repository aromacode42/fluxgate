package proxy

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestRegistry_EmptyName(t *testing.T) {
	r := NewRegistry(nil, CircuitBreakerConfig{})
	client, cb, err := r.Get("", 0)
	assert.NoError(t, err)
	assert.NotNil(t, client)
	assert.Nil(t, cb)
}

func TestRegistry_NotFound(t *testing.T) {
	r := NewRegistry(nil, CircuitBreakerConfig{})
	_, _, err := r.Get("nonexistent", 0)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not found")
}

func TestRegistry_CacheHit(t *testing.T) {
	r := NewRegistry([]Config{
		{Name: "test", Type: "http", Address: "http://127.0.0.1:7890"},
	}, CircuitBreakerConfig{})

	client1, _, err := r.Get("test", 0)
	assert.NoError(t, err)

	client2, _, err := r.Get("test", 0)
	assert.NoError(t, err)

	// Same cached client
	assert.Same(t, client1, client2)
}

func TestNewTransport_Empty(t *testing.T) {
	// Direct connection when type is empty
	r := NewRegistry([]Config{}, CircuitBreakerConfig{})
	client, _, err := r.Get("", 0)
	assert.NoError(t, err)
	assert.NotNil(t, client)
}

func TestNewTransport_UnsupportedType(t *testing.T) {
	_, err := newTransport(Config{Type: "ftp", Address: "ftp://bad"})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "unsupported")
}

func TestNewTransport_HTTP(t *testing.T) {
	tr, err := newTransport(Config{Type: "http", Address: "http://127.0.0.1:7890"})
	assert.NoError(t, err)
	assert.NotNil(t, tr)
	assert.NotNil(t, tr.Proxy)
}

func TestNewTransport_HTTP_WithAuth(t *testing.T) {
	tr, err := newTransport(Config{
		Type:     "http",
		Address:  "http://127.0.0.1:7890",
		Username: "user",
		Password: "pass",
	})
	assert.NoError(t, err)
	assert.NotNil(t, tr)
}

func TestNewTransport_InvalidURL(t *testing.T) {
	_, err := newTransport(Config{Type: "http", Address: "://invalid"})
	assert.Error(t, err)
}

func TestBuildProxyURL(t *testing.T) {
	u, err := buildProxyURL(Config{Address: "http://proxy.example.com:8080"})
	assert.NoError(t, err)
	assert.Equal(t, "proxy.example.com:8080", u.Host)
}

func TestBuildProxyURL_WithAuth(t *testing.T) {
	u, err := buildProxyURL(Config{
		Address:  "http://proxy.example.com:8080",
		Username: "admin",
		Password: "secret",
	})
	assert.NoError(t, err)
	assert.Equal(t, "admin", u.User.Username())
	pw, _ := u.User.Password()
	assert.Equal(t, "secret", pw)
}

func TestRegistry_MultipleProxies(t *testing.T) {
	r := NewRegistry([]Config{
		{Name: "us-http", Type: "http", Address: "http://us.proxy:7890"},
		{Name: "jp-socks", Type: "socks5", Address: "jp.proxy:1080"},
	}, CircuitBreakerConfig{})

	client1, _, err := r.Get("us-http", 0)
	assert.NoError(t, err)
	assert.NotNil(t, client1)

	client2, _, err := r.Get("jp-socks", 0)
	assert.NoError(t, err)
	assert.NotNil(t, client2)

	// Different clients for different proxies
	assert.NotSame(t, client1, client2)
}

func TestRegistry_Get_EmptyThenCached(t *testing.T) {
	r := NewRegistry([]Config{
		{Name: "test", Type: "http", Address: "http://127.0.0.1:7890"},
	}, CircuitBreakerConfig{})

	// Get direct (empty name) — fresh client each time
	client1, _, _ := r.Get("", 0)
	client2, _, _ := r.Get("", 0)
	assert.NotSame(t, client1, client2) // fresh client for direct connections

	// Named proxy is cached
	client3, _, _ := r.Get("test", 0)
	client4, _, _ := r.Get("test", 0)
	assert.Same(t, client3, client4)
}

// Compile-time interface check
func TestTransportImplementsRoundTripper(t *testing.T) {
	var _ http.RoundTripper = (*http.Transport)(nil)
}
