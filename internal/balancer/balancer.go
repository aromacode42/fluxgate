package balancer

import (
	"errors"
	"sync"
	"sync/atomic"

	"github.com/jacek/fluxgate/internal/provider"
)

type Balancer interface {
	Next(providers []*provider.Provider) (*provider.Provider, error)
}

type RoundRobin struct {
	counter atomic.Uint64
}

func NewRoundRobin() *RoundRobin {
	return &RoundRobin{}
}

func (rr *RoundRobin) Next(providers []*provider.Provider) (*provider.Provider, error) {
	if len(providers) == 0 {
		return nil, errors.New("no providers available")
	}
	idx := rr.counter.Add(1) - 1
	return providers[idx%uint64(len(providers))], nil
}

type Weighted struct {
	mu        sync.Mutex
	current   map[string]int
}

func NewWeighted() *Weighted {
	return &Weighted{
		current: make(map[string]int),
	}
}

func (w *Weighted) Next(providers []*provider.Provider) (*provider.Provider, error) {
	if len(providers) == 0 {
		return nil, errors.New("no providers available")
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	var best *provider.Provider
	maxWeight := -1

	for _, p := range providers {
		cur := w.current[p.Name]
		eff := cur + p.Weight
		w.current[p.Name] = eff

		if eff > maxWeight {
			maxWeight = eff
			best = p
		}
	}

	if best != nil {
		w.current[best.Name] -= totalWeight(providers)
	}

	if best == nil {
		return nil, errors.New("no provider selected")
	}
	return best, nil
}

func totalWeight(providers []*provider.Provider) int {
	var total int
	for _, p := range providers {
		total += p.Weight
	}
	return total
}

func NewBalancer(balancerType string) Balancer {
	switch balancerType {
	case "weighted":
		return NewWeighted()
	case "round_robin":
		fallthrough
	default:
		return NewRoundRobin()
	}
}
