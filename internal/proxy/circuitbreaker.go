package proxy

import (
	"sync"
	"sync/atomic"
	"time"
)

// CircuitState represents the state of a circuit breaker.
type CircuitState int32

const (
	CircuitClosed   CircuitState = 0
	CircuitHalfOpen CircuitState = 1
	CircuitOpen     CircuitState = 2
)

func (s CircuitState) String() string {
	switch s {
	case CircuitClosed:
		return "closed"
	case CircuitHalfOpen:
		return "half-open"
	case CircuitOpen:
		return "open"
	default:
		return "unknown"
	}
}

// CircuitBreaker implements the circuit breaker pattern per proxy.
type CircuitBreaker struct {
	name string

	// Configuration
	failureThreshold int
	recoveryTimeout time.Duration
	halfOpenMax     int

	// State
	state            atomic.Int32
	failureCount     atomic.Int64
	successCount     atomic.Int64
	lastFailureTime  atomic.Int64 // Unix timestamp in nanoseconds
	halfOpenAttempts atomic.Int32

	mu sync.Mutex
}

// NewCircuitBreaker creates a new circuit breaker.
func NewCircuitBreaker(name string, failureThreshold int, recoveryTimeout time.Duration, halfOpenMax int) *CircuitBreaker {
	return &CircuitBreaker{
		name:            name,
		failureThreshold: failureThreshold,
		recoveryTimeout: recoveryTimeout,
		halfOpenMax:     halfOpenMax,
	}
}

// Allow checks if a request should be allowed through the circuit.
func (cb *CircuitBreaker) Allow() bool {
	switch CircuitState(cb.state.Load()) {
	case CircuitClosed:
		return true
	case CircuitHalfOpen:
		attempts := cb.halfOpenAttempts.Load()
		if attempts < int32(cb.halfOpenMax) {
			cb.halfOpenAttempts.Add(1)
			return true
		}
		return false
	case CircuitOpen:
		// Check if recovery timeout has passed
		lastFailure := cb.lastFailureTime.Load()
		if lastFailure == 0 {
			return true
		}
		elapsed := time.Since(time.Unix(0, lastFailure))
		if elapsed >= cb.recoveryTimeout {
			cb.transitionToHalfOpen()
			return true
		}
		return false
	default:
		return true
	}
}

// RecordSuccess records a successful request.
func (cb *CircuitBreaker) RecordSuccess() {
	switch CircuitState(cb.state.Load()) {
	case CircuitClosed:
		cb.failureCount.Store(0)
	case CircuitHalfOpen:
		if cb.successCount.Add(1) >= int64(cb.halfOpenMax) {
			cb.transitionToClosed()
		}
	}
}

// RecordFailure records a failed request.
func (cb *CircuitBreaker) RecordFailure() {
	cb.lastFailureTime.Store(time.Now().UnixNano())

	switch CircuitState(cb.state.Load()) {
	case CircuitClosed:
		if cb.failureCount.Add(1) >= int64(cb.failureThreshold) {
			cb.transitionToOpen()
		}
	case CircuitHalfOpen:
		cb.transitionToOpen()
	}
}

// State returns the current circuit state.
func (cb *CircuitBreaker) State() CircuitState {
	return CircuitState(cb.state.Load())
}

func (cb *CircuitBreaker) transitionToOpen() {
	cb.state.Store(int32(CircuitOpen))
	cb.halfOpenAttempts.Store(0)
}

func (cb *CircuitBreaker) transitionToHalfOpen() {
	cb.state.Store(int32(CircuitHalfOpen))
	cb.halfOpenAttempts.Store(0)
	cb.successCount.Store(0)
}

func (cb *CircuitBreaker) transitionToClosed() {
	cb.state.Store(int32(CircuitClosed))
	cb.failureCount.Store(0)
	cb.successCount.Store(0)
	cb.halfOpenAttempts.Store(0)
}

// CircuitBreakerRegistry manages circuit breakers per proxy.
type CircuitBreakerRegistry struct {
	breakers map[string]*CircuitBreaker
	mu       sync.RWMutex
	config   CircuitBreakerConfig
}

type CircuitBreakerConfig struct {
	FailureThreshold int
	RecoveryTimeout time.Duration
	HalfOpenMax     int
}

func NewCircuitBreakerRegistry(cfg CircuitBreakerConfig) *CircuitBreakerRegistry {
	if cfg.FailureThreshold <= 0 {
		cfg.FailureThreshold = 5
	}
	if cfg.RecoveryTimeout <= 0 {
		cfg.RecoveryTimeout = 30 * time.Second
	}
	if cfg.HalfOpenMax <= 0 {
		cfg.HalfOpenMax = 3
	}
	return &CircuitBreakerRegistry{
		breakers: make(map[string]*CircuitBreaker),
		config:   cfg,
	}
}

func (r *CircuitBreakerRegistry) Get(name string) *CircuitBreaker {
	if name == "" {
		return nil
	}
	r.mu.RLock()
	cb, ok := r.breakers[name]
	r.mu.RUnlock()
	if ok {
		return cb
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if cb, ok = r.breakers[name]; ok {
		return cb
	}
	cb = NewCircuitBreaker(name, r.config.FailureThreshold, r.config.RecoveryTimeout, r.config.HalfOpenMax)
	r.breakers[name] = cb
	return cb
}
