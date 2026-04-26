package proxy

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestRegistry_EmptyName(t *testing.T) {
	r := NewRegistry(nil)
	tr, err := r.Get("")
	assert.NoError(t, err)
	assert.NotNil(t, tr)
}

func TestRegistry_NotFound(t *testing.T) {
	r := NewRegistry(nil)
	_, err := r.Get("nonexistent")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not found")
}

func TestRegistry_CacheHit(t *testing.T) {
	r := NewRegistry([]Config{
		{Name: "test", Type: "http", Address: "http://127.0.0.1:7890"},
	})

	tr1, err := r.Get("test")
	assert.NoError(t, err)

	tr2, err := r.Get("test")
	assert.NoError(t, err)

	// Same cached transport
	assert.Same(t, tr1, tr2)
}

func TestNewTransport_Empty(t *testing.T) {
	// Direct connection when type is empty
	r := NewRegistry([]Config{})
	tr, err := r.Get("")
	assert.NoError(t, err)
	assert.NotNil(t, tr)
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
	})

	tr1, err := r.Get("us-http")
	assert.NoError(t, err)
	assert.NotNil(t, tr1)

	tr2, err := r.Get("jp-socks")
	assert.NoError(t, err)
	assert.NotNil(t, tr2)

	// Different transports for different proxies
	assert.NotSame(t, tr1, tr2)
}

func TestRegistry_Get_EmptyThenCached(t *testing.T) {
	r := NewRegistry([]Config{
		{Name: "test", Type: "http", Address: "http://127.0.0.1:7890"},
	})

	// Get direct (empty name) — should return a clone each time
	tr1, _ := r.Get("")
	tr2, _ := r.Get("")
	assert.NotSame(t, tr1, tr2) // direct connections are cloned, not cached

	// Named proxy is cached
	tr3, _ := r.Get("test")
	tr4, _ := r.Get("test")
	assert.Same(t, tr3, tr4)
}

// Compile-time interface check
func TestTransportImplementsRoundTripper(t *testing.T) {
	var _ http.RoundTripper = (*http.Transport)(nil)
}
