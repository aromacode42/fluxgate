package gateway

import (
	"encoding/json"
)

// extractModel reads the "model" field from a JSON request body.
// Returns empty string if parsing fails or field is missing.
func extractModel(body []byte) string {
	var req struct {
		Model string `json:"model"`
	}
	json.Unmarshal(body, &req) //nolint:errcheck // model="" on failure is fine
	return req.Model
}

// rewriteModel replaces the "model" field in a JSON body.
// Returns the original body unchanged if parsing fails.
func rewriteModel(body []byte, newModel string) []byte {
	var m map[string]interface{}
	if err := json.Unmarshal(body, &m); err != nil {
		return body
	}
	m["model"] = newModel
	rewritten, err := json.Marshal(m)
	if err != nil {
		return body
	}
	return rewritten
}
