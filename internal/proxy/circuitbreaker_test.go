package proxy

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// ============== CircuitState String tests ==============

func TestCircuitState_String(t *testing.T) {
	tests := []struct {
		state    CircuitState
		expected string
	}{
		{CircuitClosed, "closed"},
		{CircuitHalfOpen, "half-open"},
		{CircuitOpen, "open"},
		{CircuitState(99), "unknown"},
	}
	for _, tc := range tests {
		t.Run(tc.expected, func(t *testing.T) {
			assert.Equal(t, tc.expected, tc.state.String())
		})
	}
}

// ============== CircuitBreaker creation tests ==============

func TestNewCircuitBreaker(t *testing.T) {
	cb := NewCircuitBreaker("test-proxy", 5, 30*time.Second, 3)
	assert.Equal(t, "test-proxy", cb.name)
	assert.Equal(t, 5, cb.failureThreshold)
	assert.Equal(t, 30*time.Second, cb.recoveryTimeout)
	assert.Equal(t, 3, cb.halfOpenMax)
	assert.Equal(t, CircuitClosed, cb.State())
}

// ============== CircuitBreaker Allow tests ==============

func TestCircuitBreaker_Allow_Closed(t *testing.T) {
	cb := NewCircuitBreaker("test", 5, 30*time.Second, 3)
	assert.True(t, cb.Allow())
}

func TestCircuitBreaker_Allow_OpenThenRecovery(t *testing.T) {
	cb := NewCircuitBreaker("test", 3, 50*time.Millisecond, 3)

	// Trip the circuit
	cb.RecordFailure()
	cb.RecordFailure()
	cb.RecordFailure()

	assert.Equal(t, CircuitOpen, cb.State())
	assert.False(t, cb.Allow())

	// Wait for recovery timeout
	time.Sleep(60 * time.Millisecond)
	assert.True(t, cb.Allow())
	assert.Equal(t, CircuitHalfOpen, cb.State())
}

func TestCircuitBreaker_Allow_HalfOpen_LimitedAttempts(t *testing.T) {
	// Test that half-open limits the number of probe requests
	// With halfOpenMax=2: transition call + 2 probe calls allowed = 3 successful
	cb := NewCircuitBreaker("test", 3, 50*time.Millisecond, 2)

	// Trip the circuit and wait for recovery
	cb.RecordFailure()
	cb.RecordFailure()
	cb.RecordFailure()
	time.Sleep(60 * time.Millisecond)

	// First Allow() in Open: transition to HalfOpen, returns true
	first := cb.Allow()
	assert.True(t, first)
	assert.Equal(t, CircuitHalfOpen, cb.State())

	// Second Allow() in HalfOpen: probe 1, returns true
	second := cb.Allow()
	assert.True(t, second)

	// Third Allow() in HalfOpen: probe 2, returns true
	third := cb.Allow()
	assert.True(t, third)

	// Fourth Allow() in HalfOpen: probe limit reached, returns false
	fourth := cb.Allow()
	assert.False(t, fourth)
}

func TestCircuitBreaker_Allow_HalfOpen_TransitionsToClosed(t *testing.T) {
	cb := NewCircuitBreaker("test", 3, 50*time.Millisecond, 3)

	// Trip the circuit and wait for recovery
	cb.RecordFailure()
	cb.RecordFailure()
	cb.RecordFailure()
	time.Sleep(60 * time.Millisecond)

	// Transition to half-open
	cb.Allow()
	assert.Equal(t, CircuitHalfOpen, cb.State())

	// Record enough successes to close
	cb.RecordSuccess()
	cb.RecordSuccess()
	cb.RecordSuccess()

	assert.Equal(t, CircuitClosed, cb.State())
}

// ============== CircuitBreaker RecordSuccess tests ==============

func TestCircuitBreaker_RecordSuccess_Closed(t *testing.T) {
	cb := NewCircuitBreaker("test", 5, 30*time.Second, 3)
	cb.RecordSuccess()
	assert.Equal(t, int64(0), cb.failureCount.Load())
}

func TestCircuitBreaker_RecordSuccess_HalfOpen_ClosesOnThreshold(t *testing.T) {
	cb := NewCircuitBreaker("test", 3, 50*time.Millisecond, 3)

	// Trip then recover to half-open
	cb.RecordFailure()
	cb.RecordFailure()
	cb.RecordFailure()
	time.Sleep(60 * time.Millisecond)
	cb.Allow() // Transition to half-open

	// Record 2 successes (not enough)
	cb.RecordSuccess()
	cb.RecordSuccess()
	assert.Equal(t, CircuitHalfOpen, cb.State())

	// Third success closes the circuit
	cb.RecordSuccess()
	assert.Equal(t, CircuitClosed, cb.State())
}

// ============== CircuitBreaker RecordFailure tests ==============

func TestCircuitBreaker_RecordFailure_Closed_TripsOnThreshold(t *testing.T) {
	cb := NewCircuitBreaker("test", 3, 30*time.Second, 3)

	cb.RecordFailure()
	assert.Equal(t, CircuitClosed, cb.State())

	cb.RecordFailure()
	assert.Equal(t, CircuitClosed, cb.State())

	cb.RecordFailure()
	assert.Equal(t, CircuitOpen, cb.State())
}

func TestCircuitBreaker_RecordFailure_HalfOpen_Opens(t *testing.T) {
	cb := NewCircuitBreaker("test", 3, 50*time.Millisecond, 3)

	// Trip then recover to half-open
	cb.RecordFailure()
	cb.RecordFailure()
	cb.RecordFailure()
	time.Sleep(60 * time.Millisecond)
	cb.Allow()

	assert.Equal(t, CircuitHalfOpen, cb.State())

	// Failure in half-open opens immediately
	cb.RecordFailure()
	assert.Equal(t, CircuitOpen, cb.State())
}

// ============== CircuitBreaker transition tests ==============

func TestCircuitBreaker_TransitionToOpen(t *testing.T) {
	cb := NewCircuitBreaker("test", 5, 30*time.Second, 3)
	cb.transitionToOpen()
	assert.Equal(t, CircuitOpen, cb.State())
	assert.Equal(t, int32(0), cb.halfOpenAttempts.Load())
}

func TestCircuitBreaker_TransitionToHalfOpen(t *testing.T) {
	cb := NewCircuitBreaker("test", 5, 30*time.Second, 3)
	cb.transitionToOpen()
	cb.transitionToHalfOpen()
	assert.Equal(t, CircuitHalfOpen, cb.State())
	assert.Equal(t, int32(0), cb.halfOpenAttempts.Load())
	assert.Equal(t, int64(0), cb.successCount.Load())
}

func TestCircuitBreaker_TransitionToClosed(t *testing.T) {
	cb := NewCircuitBreaker("test", 5, 30*time.Second, 3)
	cb.transitionToOpen()
	cb.transitionToClosed()
	assert.Equal(t, CircuitClosed, cb.State())
	assert.Equal(t, int64(0), cb.failureCount.Load())
	assert.Equal(t, int64(0), cb.successCount.Load())
	assert.Equal(t, int32(0), cb.halfOpenAttempts.Load())
}

// ============== CircuitBreakerRegistry tests ==============

func TestNewCircuitBreakerRegistry_Defaults(t *testing.T) {
	reg := NewCircuitBreakerRegistry(CircuitBreakerConfig{})
	assert.Equal(t, 5, reg.config.FailureThreshold)
	assert.Equal(t, 30*time.Second, reg.config.RecoveryTimeout)
	assert.Equal(t, 3, reg.config.HalfOpenMax)
}

func TestCircuitBreakerRegistry_Get_EmptyName(t *testing.T) {
	reg := NewCircuitBreakerRegistry(CircuitBreakerConfig{})
	cb := reg.Get("")
	assert.Nil(t, cb)
}

func TestCircuitBreakerRegistry_Get_CreatesNew(t *testing.T) {
	reg := NewCircuitBreakerRegistry(CircuitBreakerConfig{})
	cb1 := reg.Get("test-proxy")
	cb2 := reg.Get("test-proxy")
	assert.Same(t, cb1, cb2)
}

func TestCircuitBreakerRegistry_Get_DifferentNames(t *testing.T) {
	reg := NewCircuitBreakerRegistry(CircuitBreakerConfig{})
	cb1 := reg.Get("proxy-a")
	cb2 := reg.Get("proxy-b")
	assert.NotSame(t, cb1, cb2)
	assert.Equal(t, "proxy-a", cb1.name)
	assert.Equal(t, "proxy-b", cb2.name)
}

func TestCircuitBreakerRegistry_Get_Concurrent(t *testing.T) {
	reg := NewCircuitBreakerRegistry(CircuitBreakerConfig{})
	done := make(chan bool, 10)

	for i := 0; i < 10; i++ {
		go func(idx int) {
			for j := 0; j < 100; j++ {
				reg.Get("proxy")
			}
			done <- true
		}(i)
	}

	for i := 0; i < 10; i++ {
		<-done
	}
}

// ============== CircuitBreaker edge cases ==============

func TestCircuitBreaker_Allow_Open_NoLastFailureTime(t *testing.T) {
	cb := &CircuitBreaker{
		name:             "test",
		failureThreshold: 3,
		recoveryTimeout:  30 * time.Second,
		halfOpenMax:     3,
	}
	cb.state.Store(int32(CircuitOpen))
	cb.lastFailureTime.Store(0) // No failure recorded

	// If no failure time, should allow (treat as ready for recovery)
	assert.True(t, cb.Allow())
}

func TestCircuitBreaker_AllStates(t *testing.T) {
	cb := NewCircuitBreaker("test", 3, 50*time.Millisecond, 3)

	// Initial state
	assert.Equal(t, CircuitClosed, cb.State())

	// Close -> Open
	cb.RecordFailure()
	cb.RecordFailure()
	cb.RecordFailure()
	assert.Equal(t, CircuitOpen, cb.State())

	// Open -> HalfOpen (after timeout)
	time.Sleep(60 * time.Millisecond)
	cb.Allow()
	assert.Equal(t, CircuitHalfOpen, cb.State())

	// HalfOpen -> Closed (on success threshold)
	cb.RecordSuccess()
	cb.RecordSuccess()
	cb.RecordSuccess()
	assert.Equal(t, CircuitClosed, cb.State())

	// Closed -> Open (failure again)
	cb.RecordFailure()
	cb.RecordFailure()
	cb.RecordFailure()
	assert.Equal(t, CircuitOpen, cb.State())
}