package server

import (
	"encoding/json"
	"net/http"
	"strings"
)

// authMiddleware wraps an http.Handler with API key authentication.
// If apiKeys is empty, auth is disabled and the next handler is returned directly.
func authMiddleware(apiKeys []string, next http.Handler) http.Handler {
	if len(apiKeys) == 0 {
		return next
	}

	keySet := make(map[string]bool, len(apiKeys))
	for _, k := range apiKeys {
		keySet[k] = true
	}

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		key := extractAPIKey(r)
		if key == "" || !keySet[key] {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnauthorized)
			json.NewEncoder(w).Encode(map[string]interface{}{
				"error": map[string]string{
					"message": "invalid or missing API key",
					"type":    "authentication_error",
				},
			})
			return
		}
		next.ServeHTTP(w, r)
	})
}

// extractAPIKey extracts the API key from request headers.
// Checks Authorization: Bearer <key> first, then x-api-key.
func extractAPIKey(r *http.Request) string {
	auth := r.Header.Get("Authorization")
	if strings.HasPrefix(auth, "Bearer ") {
		return strings.TrimPrefix(auth, "Bearer ")
	}
	if key := r.Header.Get("x-api-key"); key != "" {
		return key
	}
	return ""
}
