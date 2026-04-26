package provider

import (
	"context"
	"fmt"
	"net/http"
	"sync"
	"sync/atomic"
	"time"
)

type Provider struct {
	Name      string
	Type      string // "openai" or "anthropic"
	BaseURL   string
	APIKeys   []string
	ProxyName string   // references a named proxy (empty = direct)
	Models    []string // models this provider supports (empty = all)
	Weight    int
	Disabled  bool

	keyIndex atomic.Uint64
	healthy  atomic.Bool
	mu       sync.RWMutex
}

func New(name, providerType, baseURL string, apiKeys []string, weight int) *Provider {
	p := &Provider{
		Name:    name,
		Type:    providerType,
		BaseURL: baseURL,
		APIKeys: apiKeys,
		Weight:  weight,
	}
	if weight <= 0 {
		p.Weight = 1
	}
	p.healthy.Store(true)
	return p
}

func (p *Provider) NextKey() string {
	if len(p.APIKeys) == 0 {
		return ""
	}
	idx := p.keyIndex.Add(1) - 1
	return p.APIKeys[idx%uint64(len(p.APIKeys))]
}

func (p *Provider) IsHealthy() bool {
	return p.healthy.Load() && !p.Disabled
}

func (p *Provider) SetHealthy(h bool) {
	p.healthy.Store(h)
}

func (p *Provider) CheckHealth(ctx context.Context, timeout time.Duration) error {
	client := &http.Client{Timeout: timeout}
	url := p.HealthURL()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("creating health check request: %w", err)
	}

	key := p.NextKey()
	if p.Type == "anthropic" {
		req.Header.Set("x-api-key", key)
		req.Header.Set("anthropic-version", "2023-06-01")
	} else {
		req.Header.Set("Authorization", "Bearer "+key)
	}

	resp, err := client.Do(req)
	if err != nil {
		p.SetHealthy(false)
		return fmt.Errorf("health check failed: %w", err)
	}
	defer resp.Body.Close()

	// If unauthorized and anthropic-type, retry with Bearer auth
	if resp.StatusCode == http.StatusUnauthorized && p.Type == "anthropic" {
		req.Header.Set("Authorization", "Bearer "+key)
		req.Header.Del("x-api-key")
		resp, err = client.Do(req)
		if err != nil {
			p.SetHealthy(false)
			return fmt.Errorf("health check failed: %w", err)
		}
		defer resp.Body.Close()
	}

	if resp.StatusCode >= 500 {
		p.SetHealthy(false)
		return fmt.Errorf("health check returned status %d", resp.StatusCode)
	}

	p.SetHealthy(true)
	return nil
}

func (p *Provider) HealthURL() string {
	switch p.Type {
	case "openai":
		return p.BaseURL + "/models"
	case "anthropic":
		return p.BaseURL + "/v1/models"
	default:
		return p.BaseURL
	}
}

// SupportsModel returns true if this provider can serve the given model.
// An empty Models list means the provider accepts any model.
func (p *Provider) SupportsModel(model string) bool {
	if len(p.Models) == 0 {
		return true
	}
	for _, m := range p.Models {
		if m == model {
			return true
		}
	}
	return false
}

type Registry struct {
	providers []*Provider
	mu        sync.RWMutex
}

func NewRegistry() *Registry {
	return &Registry{}
}

func (r *Registry) Add(p *Provider) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.providers = append(r.providers, p)
}

func (r *Registry) All() []*Provider {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]*Provider, len(r.providers))
	copy(out, r.providers)
	return out
}

func (r *Registry) Healthy() []*Provider {
	r.mu.RLock()
	defer r.mu.RUnlock()
	var out []*Provider
	for _, p := range r.providers {
		if p.IsHealthy() {
			out = append(out, p)
		}
	}
	return out
}

func (r *Registry) GetByName(name string) *Provider {
	r.mu.RLock()
	defer r.mu.RUnlock()
	for _, p := range r.providers {
		if p.Name == name {
			return p
		}
	}
	return nil
}
