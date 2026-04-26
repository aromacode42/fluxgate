package proxy

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"sync"
	"time"

	"golang.org/x/net/proxy"
)

type Config struct {
	Name            string
	Type            string // "http", "https", or "socks5"
	Address         string
	Username        string
	Password        string
	MaxIdleConns    int
	MaxConnsPerHost int
	IdleConnTimeout time.Duration
}

// TransportConfig holds default transport tuning parameters.
type TransportConfig struct {
	MaxIdleConns        int
	MaxIdleConnsPerHost int
	IdleConnTimeout     time.Duration
}

// Default transport config with sensible production values.
var DefaultTransportConfig = TransportConfig{
	MaxIdleConns:        100,
	MaxIdleConnsPerHost: 10,
	IdleConnTimeout:     90 * time.Second,
}

// Registry holds named proxy configs with cached HTTP clients.
type Registry struct {
	configs   map[string]Config
	clients   sync.Map // name -> *http.Client
	cbRegistry *CircuitBreakerRegistry
}

// NewRegistry creates a registry from a list of proxy configs.
func NewRegistry(configs []Config, cbConfig CircuitBreakerConfig) *Registry {
	r := &Registry{
		configs:   make(map[string]Config),
		cbRegistry: NewCircuitBreakerRegistry(cbConfig),
	}
	for _, cfg := range configs {
		r.configs[cfg.Name] = cfg
	}
	return r
}

// Get returns the cached *http.Client for the named proxy, or a direct client if name is empty.
// Returns the client and circuit breaker for the proxy.
// timeout is applied to direct connections.
func (r *Registry) Get(name string, timeout time.Duration) (*http.Client, *CircuitBreaker, error) {
	// Direct connection (no proxy)
	if name == "" {
		return r.getDirectClient(timeout), nil, nil
	}

	cb := r.cbRegistry.Get(name)

	// Check circuit breaker
	if !cb.Allow() {
		return nil, cb, fmt.Errorf("circuit breaker open for proxy %q", name)
	}

	// Check cache first
	if cached, ok := r.clients.Load(name); ok {
		return cached.(*http.Client), cb, nil
	}

	cfg, found := r.configs[name]
	if !found {
		return nil, cb, fmt.Errorf("proxy %q not found", name)
	}

	client, err := newClient(cfg)
	if err != nil {
		cb.RecordFailure()
		return nil, cb, fmt.Errorf("creating proxy client: %w", err)
	}

	// Cache it
	r.clients.Store(name, client)
	return client, cb, nil
}

// GetTransport returns the *http.Transport for the named proxy.
// Used for health checks which need direct transport access.
func (r *Registry) GetTransport(name string) *http.Transport {
	// Direct connection - return default transport
	if name == "" {
		return newDirectTransport(DefaultTransportConfig)
	}

	cfg, found := r.configs[name]
	if !found {
		return nil
	}

	transport, err := newTransport(cfg)
	if err != nil {
		return nil
	}
	return transport
}

// getDirectClient returns a fresh client for direct connections with optional timeout.
func (r *Registry) getDirectClient(timeout time.Duration) *http.Client {
	dialer := &net.Dialer{}
	t := &http.Transport{
		MaxIdleConns:          DefaultTransportConfig.MaxIdleConns,
		MaxIdleConnsPerHost:   DefaultTransportConfig.MaxIdleConnsPerHost,
		IdleConnTimeout:       DefaultTransportConfig.IdleConnTimeout,
		DialContext:           dialer.DialContext,
		ResponseHeaderTimeout: timeout,
	}
	return &http.Client{
		Transport: t,
		Timeout:   timeout,
	}
}

// Close closes all cached clients and their idle connections.
func (r *Registry) Close() {
	r.clients.Range(func(key, value any) bool {
		if client, ok := value.(*http.Client); ok {
			client.CloseIdleConnections()
		}
		return true
	})
	r.clients = sync.Map{}
}

// newClient creates a new http.Client for the given proxy config.
func newClient(cfg Config) (*http.Client, error) {
	transport, err := newTransport(cfg)
	if err != nil {
		return nil, err
	}

	tc := getTransportConfig(cfg)
	applyTransportConfig(transport, tc)

	return &http.Client{
		Transport: transport,
		Timeout:   0, // Timeout is managed by the gateway
	}, nil
}

// newDirectTransport creates a transport for direct connections.
func newDirectTransport(tc TransportConfig) *http.Transport {
	t := http.DefaultTransport.(*http.Transport).Clone()
	t.DialContext = (&net.Dialer{}).DialContext // Respect context deadline
	applyTransportConfig(t, tc)
	return t
}

// newTransport creates a new http.Transport for the given proxy config.
func newTransport(cfg Config) (*http.Transport, error) {
	tc := getTransportConfig(cfg)
	transport := http.DefaultTransport.(*http.Transport).Clone()
	applyTransportConfig(transport, tc)

	switch cfg.Type {
	case "http", "https":
		proxyURL, err := buildProxyURL(cfg)
		if err != nil {
			return nil, fmt.Errorf("building proxy URL: %w", err)
		}
		transport.Proxy = http.ProxyURL(proxyURL)

	case "socks5":
		dialer, err := proxy.SOCKS5("tcp", cfg.Address, &proxy.Auth{
			User:     cfg.Username,
			Password: cfg.Password,
		}, proxy.Direct)
		if err != nil {
			return nil, fmt.Errorf("creating SOCKS5 dialer: %w", err)
		}

		// Wrap with context-aware dialer
		transport.DialContext = makeSOCKS5DialContext(dialer, cfg.Address)

	default:
		return nil, fmt.Errorf("unsupported proxy type: %s", cfg.Type)
	}

	return transport, nil
}

// makeSOCKS5DialContext creates a context-aware DialContext for SOCKS5.
func makeSOCKS5DialContext(dialer proxy.Dialer, proxyAddr string) func(ctx context.Context, network, addr string) (net.Conn, error) {
	return func(ctx context.Context, network, addr string) (net.Conn, error) {
		// Check context deadline before dialing
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		// Create a channel to receive the connection
		type connResult struct {
			conn net.Conn
			err  error
		}
		resultCh := make(chan connResult, 1)

		go func() {
			conn, err := dialer.Dial(network, addr)
			resultCh <- connResult{conn, err}
		}()

		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case result := <-resultCh:
			return result.conn, result.err
		}
	}
}

func buildProxyURL(cfg Config) (*url.URL, error) {
	u, err := url.Parse(cfg.Address)
	if err != nil {
		return nil, err
	}
	if cfg.Username != "" {
		u.User = url.UserPassword(cfg.Username, cfg.Password)
	}
	return u, nil
}

func getTransportConfig(cfg Config) TransportConfig {
	tc := DefaultTransportConfig
	if cfg.MaxIdleConns > 0 {
		tc.MaxIdleConns = cfg.MaxIdleConns
	}
	if cfg.MaxConnsPerHost > 0 {
		tc.MaxIdleConnsPerHost = cfg.MaxConnsPerHost
	}
	if cfg.IdleConnTimeout > 0 {
		tc.IdleConnTimeout = cfg.IdleConnTimeout
	}
	return tc
}

func applyTransportConfig(t *http.Transport, tc TransportConfig) {
	if tc.MaxIdleConns > 0 {
		t.MaxIdleConns = tc.MaxIdleConns
	}
	if tc.MaxIdleConnsPerHost > 0 {
		t.MaxIdleConnsPerHost = tc.MaxIdleConnsPerHost
	}
	if tc.IdleConnTimeout > 0 {
		t.IdleConnTimeout = tc.IdleConnTimeout
	}
}
