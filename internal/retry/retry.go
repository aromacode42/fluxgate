package retry

import (
	"context"
	"errors"
	"math/rand"
	"net/http"
	"time"
)

type Config struct {
	Max              int
	WaitMin          time.Duration
	WaitMax          time.Duration
	MaxErrorAttempts int // Max consecutive network errors before giving up (default 3)
}

type Retryer struct {
	cfg Config
	rng *rand.Rand
}

func New(cfg Config) *Retryer {
	if cfg.MaxErrorAttempts <= 0 {
		cfg.MaxErrorAttempts = 3
	}
	return &Retryer{
		cfg: cfg,
		rng: rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

func (r *Retryer) Do(ctx context.Context, fn func() (*http.Response, error)) (*http.Response, error) {
	var lastErr error
	var resp *http.Response
	consecutiveErrors := 0

	for attempt := 0; attempt <= r.cfg.Max; attempt++ {
		// Check context before every attempt
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		if attempt > 0 {
			wait := r.Backoff(attempt)
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(wait):
			}
		}

		resp, lastErr = fn()
		if lastErr != nil {
			consecutiveErrors++
			if consecutiveErrors > r.cfg.MaxErrorAttempts {
				return nil, errors.New("max retry attempts exceeded: " + lastErr.Error())
			}
			continue
		}
		consecutiveErrors = 0

		if resp == nil {
			return nil, errors.New("nil response received")
		}

		if r.ShouldRetry(resp.StatusCode) {
			resp.Body.Close()
			continue
		}

		return resp, nil
	}

	if lastErr != nil {
		return nil, lastErr
	}
	return nil, errors.New("max retries exceeded")
}

// Backoff returns a duration for the given attempt number (1-based).
func (r *Retryer) Backoff(attempt int) time.Duration {
	min := float64(r.cfg.WaitMin)
	max := float64(r.cfg.WaitMax)
	if max <= min {
		return r.cfg.WaitMin
	}
	wait := min + r.rng.Float64()*(max-min)
	multiplier := 1 << (attempt - 1)
	if multiplier > 32 {
		multiplier = 32
	}
	exponential := wait * float64(multiplier)
	if exponential > max {
		exponential = max
	}
	return time.Duration(exponential)
}

// ShouldRetry returns true for status codes that warrant a retry.
func (r *Retryer) ShouldRetry(statusCode int) bool {
	switch statusCode {
	case http.StatusTooManyRequests,
		http.StatusInternalServerError,
		http.StatusBadGateway,
		http.StatusServiceUnavailable,
		http.StatusGatewayTimeout:
		return true
	default:
		return false
	}
}