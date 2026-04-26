package balancer

import (
	"testing"

	"github.com/jacek/fluxgate/internal/provider"
)

func TestRoundRobin_Next(t *testing.T) {
	rr := NewRoundRobin()
	p1 := provider.New("p1", "openai", "https://api1.com", []string{"key1"}, 1)
	p2 := provider.New("p2", "openai", "https://api2.com", []string{"key2"}, 1)
	providers := []*provider.Provider{p1, p2}

	// Test round-robin distribution
	results := make(map[string]int)
	for i := 0; i < 100; i++ {
		p, err := rr.Next(providers)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		results[p.Name]++
	}

	// Should be roughly evenly distributed
	if results["p1"] != 50 || results["p2"] != 50 {
		t.Errorf("expected equal distribution (50/50), got p1=%d, p2=%d", results["p1"], results["p2"])
	}
}

func TestRoundRobin_NoProviders(t *testing.T) {
	rr := NewRoundRobin()
	_, err := rr.Next([]*provider.Provider{})
	if err == nil {
		t.Error("expected error with no providers")
	}
}

func TestWeighted_Next(t *testing.T) {
	w := NewWeighted()
	p1 := provider.New("p1", "openai", "https://api1.com", []string{"key1"}, 3)
	p2 := provider.New("p2", "openai", "https://api2.com", []string{"key2"}, 1)
	providers := []*provider.Provider{p1, p2}

	// Test weighted distribution
	results := make(map[string]int)
	for i := 0; i < 100; i++ {
		p, err := w.Next(providers)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		results[p.Name]++
	}

	// p1 should get roughly 75% of requests (weight 3 vs 1)
	if results["p1"] < 60 || results["p1"] > 90 {
		t.Errorf("expected p1 to get ~75%% of requests, got p1=%d, p2=%d", results["p1"], results["p2"])
	}
}

func TestWeighted_NoProviders(t *testing.T) {
	w := NewWeighted()
	_, err := w.Next([]*provider.Provider{})
	if err == nil {
		t.Error("expected error with no providers")
	}
}

func TestNewBalancer(t *testing.T) {
	rr := NewBalancer("round_robin")
	if rr == nil {
		t.Error("expected non-nil round_robin balancer")
	}

	w := NewBalancer("weighted")
	if w == nil {
		t.Error("expected non-nil weighted balancer")
	}

	// Default should be round_robin
	def := NewBalancer("unknown")
	if def == nil {
		t.Error("expected non-nil default balancer")
	}
}