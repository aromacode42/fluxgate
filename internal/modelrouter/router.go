package modelrouter

import (
	"fmt"
	"strings"
	"sync"

	"github.com/jacek/fluxgate/internal/provider"
	pkgconfig "github.com/jacek/fluxgate/pkg/config"
)

// BackendRef is a resolved backend pointing to a specific provider and real model.
type BackendRef struct {
	ProviderName string
	RealModel    string
	Provider     *provider.Provider
}

// Entry is a virtual model with its ordered list of backends.
type Entry struct {
	VirtualModel string
	Backends     []BackendRef
}

// Router maps virtual model names to entries with backend lists.
type Router struct {
	entries     map[string]*Entry
	providerReg *provider.Registry
	mu          sync.RWMutex
}

// NewRouter builds the routing table from config model entries and the provider registry.
func NewRouter(cfgModels []pkgconfig.ModelConfig, registry *provider.Registry) (*Router, error) {
	r := &Router{
		entries:     make(map[string]*Entry),
		providerReg: registry,
	}

	for _, m := range cfgModels {
		entry := &Entry{
			VirtualModel: m.Name,
		}

		for _, backend := range m.Backends {
			parts := strings.SplitN(backend, "/", 2)
			if len(parts) != 2 {
				return nil, fmt.Errorf("invalid backend %q for model %q: must be provider/model", backend, m.Name)
			}

			p := registry.GetByName(parts[0])
			if p == nil {
				return nil, fmt.Errorf("provider %q not found for model %q", parts[0], m.Name)
			}

			entry.Backends = append(entry.Backends, BackendRef{
				ProviderName: parts[0],
				RealModel:    parts[1],
				Provider:     p,
			})
		}

		r.entries[m.Name] = entry
	}

	return r, nil
}

// Resolve looks up a virtual model name. Returns nil if not found.
func (r *Router) Resolve(virtualModel string) *Entry {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.entries[virtualModel]
}

// HealthyBackends returns backends whose providers are healthy, in config order.
func (r *Router) HealthyBackends(virtualModel string) []BackendRef {
	entry := r.Resolve(virtualModel)
	if entry == nil {
		return nil
	}

	var healthy []BackendRef
	for _, b := range entry.Backends {
		if b.Provider.IsHealthy() {
			healthy = append(healthy, b)
		}
	}
	return healthy
}

// PassthroughCandidates returns providers that can serve the given model
// for the given provider type. Providers with empty Models lists accept all.
func (r *Router) PassthroughCandidates(model string, providerType string) []*provider.Provider {
	var candidates []*provider.Provider
	for _, p := range r.providerReg.Healthy() {
		if providerType != "" && p.Type != providerType {
			continue
		}
		if p.SupportsModel(model) {
			candidates = append(candidates, p)
		}
	}
	return candidates
}

// HasEntries returns true if any model entries are configured.
func (r *Router) HasEntries() bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.entries) > 0
}

// AllEntries returns a snapshot of all virtual model entries.
func (r *Router) AllEntries() []*Entry {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]*Entry, 0, len(r.entries))
	for _, e := range r.entries {
		out = append(out, e)
	}
	return out
}
