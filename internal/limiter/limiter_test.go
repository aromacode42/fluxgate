package limiter

import (
	"context"
	"testing"
	"time"
)

func TestNewRateLimiter(t *testing.T) {
	rl := NewRateLimiter(10, 5)
	if rl == nil {
		t.Fatal("expected non-nil rate limiter")
	}
}

func TestNewRateLimiter_DefaultBurst(t *testing.T) {
	rl := NewRateLimiter(100, 0)
	if rl == nil {
		t.Fatal("expected non-nil rate limiter with default burst")
	}
}

func TestRateLimiter_Allow(t *testing.T) {
	rl := NewRateLimiter(1, 2)
	if !rl.Allow() {
		t.Error("first request should be allowed")
	}
	if !rl.Allow() {
		t.Error("second request should be allowed (within burst)")
	}
}

func TestRateLimiter_Wait(t *testing.T) {
	rl := NewRateLimiter(1000, 1)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	if err := rl.Wait(ctx); err != nil {
		t.Errorf("Wait should succeed: %v", err)
	}
}

func TestRateLimiter_Wait_ContextCanceled(t *testing.T) {
	rl := NewRateLimiter(1, 1)
	// Exhaust the burst
	rl.Allow()
	rl.Allow()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if err := rl.Wait(ctx); err == nil {
		t.Error("expected error on canceled context")
	}
}
