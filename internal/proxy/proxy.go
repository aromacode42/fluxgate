package proxy

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"sync"

	"golang.org/x/net/proxy"
)

type Config struct {
	Name     string
	Type     string // "http" or "socks5"
	Address  string
	Username string
	Password string
}

// Registry holds named proxy configs with cached transports.
type Registry struct {
	configs   map[string]Config
	transport sync.Map // name -> *http.Transport
}

// NewRegistry creates a registry from a list of proxy configs.
func NewRegistry(configs []Config) *Registry {
	r := &Registry{
		configs: make(map[string]Config),
	}
	for _, cfg := range configs {
		r.configs[cfg.Name] = cfg
	}
	return r
}

// Get returns the cached transport for the named proxy, or a direct transport if name is empty.
func (r *Registry) Get(name string) (*http.Transport, error) {
	if name == "" {
		// Direct connection (no proxy)
		return http.DefaultTransport.(*http.Transport).Clone(), nil
	}

	// Check cache first
	if cached, ok := r.transport.Load(name); ok {
		return cached.(*http.Transport), nil
	}

	cfg, found := r.configs[name]
	if !found {
		return nil, fmt.Errorf("proxy %q not found", name)
	}

	transport, err := newTransport(cfg)
	if err != nil {
		return nil, err
	}

	// Cache it
	r.transport.Store(name, transport)
	return transport, nil
}

// newTransport creates a new http.Transport for the given proxy config.
func newTransport(cfg Config) (*http.Transport, error) {
	switch cfg.Type {
	case "http", "https":
		proxyURL, err := buildProxyURL(cfg)
		if err != nil {
			return nil, fmt.Errorf("building proxy URL: %w", err)
		}
		transport := http.DefaultTransport.(*http.Transport).Clone()
		transport.Proxy = http.ProxyURL(proxyURL)
		return transport, nil

	case "socks5":
		dialer, err := proxy.SOCKS5("tcp", cfg.Address, &proxy.Auth{
			User:     cfg.Username,
			Password: cfg.Password,
		}, proxy.Direct)
		if err != nil {
			return nil, fmt.Errorf("creating SOCKS5 dialer: %w", err)
		}

		transport := http.DefaultTransport.(*http.Transport).Clone()
		transport.DialContext = func(ctx context.Context, network, addr string) (net.Conn, error) {
			return dialer.Dial(network, addr)
		}
		return transport, nil

	default:
		return nil, fmt.Errorf("unsupported proxy type: %s", cfg.Type)
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