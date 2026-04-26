package provider

import (
	"testing"
)

func TestNew(t *testing.T) {
	p := New("test", "openai", "https://api.openai.com", []string{"key1", "key2"}, 2)
	if p.Name != "test" {
		t.Errorf("expected name 'test', got %s", p.Name)
	}
	if p.Type != "openai" {
		t.Errorf("expected type 'openai', got %s", p.Type)
	}
	if p.Weight != 2 {
		t.Errorf("expected weight 2, got %d", p.Weight)
	}
	if !p.IsHealthy() {
		t.Error("new provider should be healthy")
	}
}

func TestNew_DefaultWeight(t *testing.T) {
	p := New("test", "openai", "https://api.openai.com", []string{"key1"}, 0)
	if p.Weight != 1 {
		t.Errorf("expected default weight 1, got %d", p.Weight)
	}
}

func TestProvider_NextKey(t *testing.T) {
	p := New("test", "openai", "https://api.openai.com", []string{"key1", "key2", "key3"}, 1)

	keys := make(map[string]int)
	for i := 0; i < 30; i++ {
		key := p.NextKey()
		keys[key]++
	}

	if len(keys) != 3 {
		t.Errorf("expected 3 unique keys, got %d", len(keys))
	}
	for _, key := range []string{"key1", "key2", "key3"} {
		if keys[key] != 10 {
			t.Errorf("expected key %s to appear 10 times, got %d", key, keys[key])
		}
	}
}

func TestProvider_NextKey_Empty(t *testing.T) {
	p := New("test", "openai", "https://api.openai.com", []string{}, 1)
	key := p.NextKey()
	if key != "" {
		t.Errorf("expected empty key, got %s", key)
	}
}

func TestProvider_Health(t *testing.T) {
	p := New("test", "openai", "https://api.openai.com", []string{"key1"}, 1)

	if !p.IsHealthy() {
		t.Error("provider should start healthy")
	}

	p.SetHealthy(false)
	if p.IsHealthy() {
		t.Error("provider should be unhealthy after SetHealthy(false)")
	}

	p.SetHealthy(true)
	if !p.IsHealthy() {
		t.Error("provider should be healthy after SetHealthy(true)")
	}
}

func TestProvider_Disabled(t *testing.T) {
	p := New("test", "openai", "https://api.openai.com", []string{"key1"}, 1)
	p.Disabled = true

	if p.IsHealthy() {
		t.Error("disabled provider should not be healthy")
	}
}

func TestProvider_HealthURL(t *testing.T) {
	testCases := []struct {
		providerType string
		baseURL      string
		expected     string
	}{
		{"openai", "https://api.openai.com", "https://api.openai.com/models"},
		{"anthropic", "https://api.anthropic.com", "https://api.anthropic.com/v1/models"},
		{"unknown", "https://api.example.com", "https://api.example.com"},
	}

	for _, tc := range testCases {
		p := New("test", tc.providerType, tc.baseURL, []string{"key1"}, 1)
		if got := p.HealthURL(); got != tc.expected {
			t.Errorf("type=%s: expected %s, got %s", tc.providerType, tc.expected, got)
		}
	}
}

func TestRegistry(t *testing.T) {
	reg := NewRegistry()

	p1 := New("p1", "openai", "https://api1.com", []string{"key1"}, 1)
	p2 := New("p2", "anthropic", "https://api2.com", []string{"key2"}, 1)

	reg.Add(p1)
	reg.Add(p2)

	all := reg.All()
	if len(all) != 2 {
		t.Errorf("expected 2 providers, got %d", len(all))
	}

	healthy := reg.Healthy()
	if len(healthy) != 2 {
		t.Errorf("expected 2 healthy providers, got %d", len(healthy))
	}

	p1.SetHealthy(false)
	healthy = reg.Healthy()
	if len(healthy) != 1 {
		t.Errorf("expected 1 healthy provider, got %d", len(healthy))
	}
}

func TestRegistry_GetByName(t *testing.T) {
	reg := NewRegistry()
	p1 := New("p1", "openai", "https://api1.com", []string{"key1"}, 1)
	reg.Add(p1)

	found := reg.GetByName("p1")
	if found == nil || found.Name != "p1" {
		t.Error("expected to find p1")
	}

	notFound := reg.GetByName("nonexistent")
	if notFound != nil {
		t.Error("expected nil for nonexistent provider")
	}
}
