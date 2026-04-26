package server

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestAuthMiddleware_DisabledWhenNoKeys(t *testing.T) {
	called := false
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	})

	handler := authMiddleware(nil, next)
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	assert.True(t, called, "next handler should be called when no keys configured")
	assert.Equal(t, http.StatusOK, w.Code)
}

func TestAuthMiddleware_EmptyKeySlice(t *testing.T) {
	called := false
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	})

	handler := authMiddleware([]string{}, next)
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	assert.True(t, called, "next handler should be called when keys slice is empty")
}

func TestAuthMiddleware_ValidBearerKey(t *testing.T) {
	called := false
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	})

	handler := authMiddleware([]string{"secret-key"}, next)
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req.Header.Set("Authorization", "Bearer secret-key")
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	assert.True(t, called)
	assert.Equal(t, http.StatusOK, w.Code)
}

func TestAuthMiddleware_ValidXAPIKey(t *testing.T) {
	called := false
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	})

	handler := authMiddleware([]string{"secret-key"}, next)
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req.Header.Set("x-api-key", "secret-key")
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	assert.True(t, called)
	assert.Equal(t, http.StatusOK, w.Code)
}

func TestAuthMiddleware_InvalidKey(t *testing.T) {
	called := false
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	})

	handler := authMiddleware([]string{"correct-key"}, next)
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req.Header.Set("Authorization", "Bearer wrong-key")
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	assert.False(t, called)
	assert.Equal(t, http.StatusUnauthorized, w.Code)
}

func TestAuthMiddleware_MissingKey(t *testing.T) {
	called := false
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	})

	handler := authMiddleware([]string{"secret-key"}, next)
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	assert.False(t, called)
	assert.Equal(t, http.StatusUnauthorized, w.Code)
}

func TestAuthMiddleware_MultipleKeys(t *testing.T) {
	called := false
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	})

	handler := authMiddleware([]string{"key-one", "key-two", "key-three"}, next)

	for _, key := range []string{"key-one", "key-two", "key-three"} {
		called = false
		req := httptest.NewRequest(http.MethodGet, "/", nil)
		req.Header.Set("Authorization", "Bearer "+key)
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)
		assert.True(t, called, "key %s should be accepted", key)
	}
}

func TestExtractAPIKey(t *testing.T) {
	tests := []struct {
		name     string
		headers  map[string]string
		expected string
	}{
		{"bearer", map[string]string{"Authorization": "Bearer my-key"}, "my-key"},
		{"x-api-key", map[string]string{"x-api-key": "my-key"}, "my-key"},
		{"bearer takes precedence", map[string]string{"Authorization": "Bearer key1", "x-api-key": "key2"}, "key1"},
		{"no headers", nil, ""},
		{"empty bearer", map[string]string{"Authorization": "Bearer "}, ""},
		{"wrong scheme", map[string]string{"Authorization": "Basic dXNlcjpwYXNz"}, ""},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, "/", nil)
			for k, v := range tc.headers {
				req.Header.Set(k, v)
			}
			assert.Equal(t, tc.expected, extractAPIKey(req))
		})
	}
}
