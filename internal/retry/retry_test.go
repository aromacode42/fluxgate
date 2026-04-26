package retry

import (
	"context"
	"net/http"
	"testing"
	"time"
)

func TestRetryer_Do_Success(t *testing.T) {
	cfg := Config{
		Max:     3,
		WaitMin: 10 * time.Millisecond,
		WaitMax: 50 * time.Millisecond,
	}
	r := New(cfg)

	callCount := 0
	resp, err := r.Do(context.Background(), func() (*http.Response, error) {
		callCount++
		return &http.Response{StatusCode: http.StatusOK, Body: http.NoBody}, nil
	})

	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Errorf("expected status 200, got %d", resp.StatusCode)
	}
	if callCount != 1 {
		t.Errorf("expected 1 call, got %d", callCount)
	}
}

func TestRetryer_Do_RetryOn5xx(t *testing.T) {
	cfg := Config{
		Max:     3,
		WaitMin: 10 * time.Millisecond,
		WaitMax: 50 * time.Millisecond,
	}
	r := New(cfg)

	callCount := 0
	_, err := r.Do(context.Background(), func() (*http.Response, error) {
		callCount++
		return &http.Response{StatusCode: http.StatusInternalServerError, Body: http.NoBody}, nil
	})

	if err == nil {
		t.Error("expected error after max retries")
	}
	if callCount != cfg.Max+1 {
		t.Errorf("expected %d calls, got %d", cfg.Max+1, callCount)
	}
}

func TestRetryer_Do_RetryOn429(t *testing.T) {
	cfg := Config{
		Max:     2,
		WaitMin: 10 * time.Millisecond,
		WaitMax: 50 * time.Millisecond,
	}
	r := New(cfg)

	callCount := 0
	_, err := r.Do(context.Background(), func() (*http.Response, error) {
		callCount++
		if callCount < 3 {
			return &http.Response{StatusCode: http.StatusTooManyRequests, Body: http.NoBody}, nil
		}
		return &http.Response{StatusCode: http.StatusOK, Body: http.NoBody}, nil
	})

	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if callCount != 3 {
		t.Errorf("expected 3 calls, got %d", callCount)
	}
}

func TestRetryer_Do_ContextCanceled(t *testing.T) {
	cfg := Config{
		Max:     5,
		WaitMin: 100 * time.Millisecond,
		WaitMax: 200 * time.Millisecond,
	}
	r := New(cfg)

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	_, err := r.Do(ctx, func() (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusOK, Body: http.NoBody}, nil
	})

	if err == nil {
		t.Error("expected error on canceled context")
	}
}

func TestRetryer_ShouldRetry(t *testing.T) {
	r := New(Config{})

	testCases := []struct {
		status   int
		expected bool
	}{
		{http.StatusOK, false},
		{http.StatusBadRequest, false},
		{http.StatusTooManyRequests, true},
		{http.StatusInternalServerError, true},
		{http.StatusBadGateway, true},
		{http.StatusServiceUnavailable, true},
		{http.StatusGatewayTimeout, true},
	}

	for _, tc := range testCases {
		result := r.ShouldRetry(tc.status)
		if result != tc.expected {
			t.Errorf("status %d: expected %v, got %v", tc.status, tc.expected, result)
		}
	}
}

func TestRetryer_Backoff(t *testing.T) {
	cfg := Config{
		Max:     3,
		WaitMin: 100 * time.Millisecond,
		WaitMax: 500 * time.Millisecond,
	}
	r := New(cfg)

	for attempt := 1; attempt <= 5; attempt++ {
		wait := r.Backoff(attempt)
		if wait < cfg.WaitMin {
			t.Errorf("attempt %d: wait %v is less than WaitMin %v", attempt, wait, cfg.WaitMin)
		}
		if wait > cfg.WaitMax*10 { // Allow for exponential growth but cap reasonably
			t.Errorf("attempt %d: wait %v is too large", attempt, wait)
		}
	}
}
