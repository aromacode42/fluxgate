package modelrouter

import (
	"testing"

	"github.com/jacek/fluxgate/internal/provider"
	pkgconfig "github.com/jacek/fluxgate/pkg/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func setupRegistry() *provider.Registry {
	reg := provider.NewRegistry()
	reg.Add(provider.New("openai-us", "openai", "https://api.openai.com", []string{"key1"}, 2))
	reg.Add(provider.New("openai-backup", "openai", "https://api.backup.com", []string{"key2"}, 1))
	reg.Add(provider.New("anthropic-jp", "anthropic", "https://api.anthropic.com", []string{"key3"}, 1))
	return reg
}

func TestNewRouter_ValidConfig(t *testing.T) {
	reg := setupRegistry()
	cfgModels := []pkgconfig.ModelConfig{
		{
			Name:     "gpt-5",
			Backends: []string{"openai-us/gpt-4o", "openai-backup/gpt-4o", "anthropic-jp/claude-sonnet-4-20250514"},
		},
		{
			Name:     "fast-chat",
			Backends: []string{"openai-us/gpt-4o-mini", "openai-backup/gpt-4o-mini"},
		},
	}

	r, err := NewRouter(cfgModels, reg)
	require.NoError(t, err)
	assert.True(t, r.HasEntries())
}

func TestNewRouter_UnknownProvider(t *testing.T) {
	reg := setupRegistry()
	cfgModels := []pkgconfig.ModelConfig{
		{Name: "test", Backends: []string{"nonexistent/model"}},
	}

	_, err := NewRouter(cfgModels, reg)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not found")
}

func TestNewRouter_InvalidBackendFormat(t *testing.T) {
	reg := setupRegistry()
	cfgModels := []pkgconfig.ModelConfig{
		{Name: "test", Backends: []string{"invalid-no-slash"}},
	}

	_, err := NewRouter(cfgModels, reg)
	assert.Error(t, err)
}

func TestNewRouter_Empty(t *testing.T) {
	reg := setupRegistry()
	r, err := NewRouter(nil, reg)
	require.NoError(t, err)
	assert.False(t, r.HasEntries())
}

func TestRouter_Resolve(t *testing.T) {
	reg := setupRegistry()
	r, _ := NewRouter([]pkgconfig.ModelConfig{
		{Name: "gpt-5", Backends: []string{"openai-us/gpt-4o", "openai-backup/gpt-4o"}},
	}, reg)

	entry := r.Resolve("gpt-5")
	require.NotNil(t, entry)
	assert.Equal(t, "gpt-5", entry.VirtualModel)
	assert.Len(t, entry.Backends, 2)
	assert.Equal(t, "gpt-4o", entry.Backends[0].RealModel)
	assert.Equal(t, "openai-us", entry.Backends[0].ProviderName)

	assert.Nil(t, r.Resolve("nonexistent"))
}

func TestRouter_HealthyBackends(t *testing.T) {
	reg := setupRegistry()
	r, _ := NewRouter([]pkgconfig.ModelConfig{
		{Name: "gpt-5", Backends: []string{"openai-us/gpt-4o", "openai-backup/gpt-4o"}},
	}, reg)

	// All healthy
	backends := r.HealthyBackends("gpt-5")
	assert.Len(t, backends, 2)

	// Mark one unhealthy
	reg.GetByName("openai-us").SetHealthy(false)
	backends = r.HealthyBackends("gpt-5")
	assert.Len(t, backends, 1)
	assert.Equal(t, "openai-backup", backends[0].ProviderName)

	// Unknown model
	assert.Nil(t, r.HealthyBackends("unknown"))
}

func TestRouter_PassthroughCandidates(t *testing.T) {
	reg := provider.NewRegistry()
	p1 := provider.New("openai-us", "openai", "https://api.openai.com", []string{"key1"}, 1)
	p1.Models = []string{"gpt-4o", "gpt-4o-mini"}
	reg.Add(p1)

	p2 := provider.New("openai-direct", "openai", "https://api.openai.com", []string{"key2"}, 1)
	// p2 has no Models list = accepts everything
	reg.Add(p2)

	p3 := provider.New("anthropic", "anthropic", "https://api.anthropic.com", []string{"key3"}, 1)
	reg.Add(p3)

	r, _ := NewRouter(nil, reg)

	// gpt-4o matches p1 (explicit) and p2 (wildcard)
	candidates := r.PassthroughCandidates("gpt-4o", "openai")
	assert.Len(t, candidates, 2)

	// unknown-model matches only p2 (wildcard)
	candidates = r.PassthroughCandidates("unknown-model", "openai")
	assert.Len(t, candidates, 1)
	assert.Equal(t, "openai-direct", candidates[0].Name)

	// anthropic type
	candidates = r.PassthroughCandidates("anything", "anthropic")
	assert.Len(t, candidates, 1)

	// empty type = match all (but p1 only has specific models, so only p2+p3)
	candidates = r.PassthroughCandidates("anything", "")
	assert.Len(t, candidates, 2)
}

func TestRouter_AllEntries(t *testing.T) {
	reg := setupRegistry()
	r, _ := NewRouter([]pkgconfig.ModelConfig{
		{Name: "gpt-5", Backends: []string{"openai-us/gpt-4o"}},
		{Name: "fast-chat", Backends: []string{"openai-us/gpt-4o-mini"}},
	}, reg)

	entries := r.AllEntries()
	assert.Len(t, entries, 2)

	names := make(map[string]bool)
	for _, e := range entries {
		names[e.VirtualModel] = true
	}
	assert.True(t, names["gpt-5"])
	assert.True(t, names["fast-chat"])
}

func TestRouter_AllEntries_Empty(t *testing.T) {
	reg := setupRegistry()
	r, _ := NewRouter(nil, reg)

	entries := r.AllEntries()
	assert.Empty(t, entries)
}
