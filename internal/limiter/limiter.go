package limiter

import (
	"context"

	"golang.org/x/time/rate"
)

type RateLimiter struct {
	limiter *rate.Limiter
}

func NewRateLimiter(rps float64, burst int) *RateLimiter {
	if burst <= 0 {
		burst = int(rps) * 2
		if burst < 1 {
			burst = 1
		}
	}
	return &RateLimiter{
		limiter: rate.NewLimiter(rate.Limit(rps), burst),
	}
}

func (rl *RateLimiter) Wait(ctx context.Context) error {
	return rl.limiter.Wait(ctx)
}

func (rl *RateLimiter) Allow() bool {
	return rl.limiter.Allow()
}

func (rl *RateLimiter) Reserve() {
	rl.limiter.Reserve()
}
