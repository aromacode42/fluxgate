package adapter

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAnthropicToOpenAI_BasicMessages(t *testing.T) {
	anthropicReq := map[string]interface{}{
		"model":      "claude-3",
		"max_tokens": 1024,
		"messages": []map[string]interface{}{
			{"role": "user", "content": "Hello"},
		},
	}
	body, _ := json.Marshal(anthropicReq)

	result, err := AnthropicToOpenAI(body)
	require.NoError(t, err)

	var openaiReq map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &openaiReq))

	assert.Equal(t, "claude-3", openaiReq["model"])
	assert.Equal(t, float64(1024), openaiReq["max_tokens"])

	msgs := openaiReq["messages"].([]interface{})
	assert.Len(t, msgs, 1)
	msg := msgs[0].(map[string]interface{})
	assert.Equal(t, "user", msg["role"])
	assert.Equal(t, "Hello", msg["content"])
}

func TestAnthropicToOpenAI_WithSystem(t *testing.T) {
	anthropicReq := map[string]interface{}{
		"model":      "claude-3",
		"max_tokens": 1024,
		"system":     "You are helpful",
		"messages": []map[string]interface{}{
			{"role": "user", "content": "Hi"},
		},
	}
	body, _ := json.Marshal(anthropicReq)

	result, err := AnthropicToOpenAI(body)
	require.NoError(t, err)

	var openaiReq map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &openaiReq))

	msgs := openaiReq["messages"].([]interface{})
	// System message should be first
	first := msgs[0].(map[string]interface{})
	assert.Equal(t, "system", first["role"])
	assert.Equal(t, "You are helpful", first["content"])
}

func TestOpenAIToAnthropic_BasicMessages(t *testing.T) {
	openaiReq := map[string]interface{}{
		"model": "gpt-4",
		"messages": []map[string]interface{}{
			{"role": "system", "content": "You are helpful"},
			{"role": "user", "content": "Hello"},
			{"role": "assistant", "content": "Hi there"},
		},
	}
	body, _ := json.Marshal(openaiReq)

	result, err := OpenAIToAnthropic(body)
	require.NoError(t, err)

	var anthropicReq map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &anthropicReq))

	assert.Equal(t, "gpt-4", anthropicReq["model"])
	assert.Equal(t, "You are helpful", anthropicReq["system"])

	msgs := anthropicReq["messages"].([]interface{})
	assert.Len(t, msgs, 2) // system extracted to top-level
}

func TestOpenAIToAnthropic_ToolCalls(t *testing.T) {
	openaiReq := map[string]interface{}{
		"model": "gpt-4",
		"messages": []map[string]interface{}{
			{"role": "user", "content": "What's the weather?"},
			{"role": "assistant", "content": nil, "tool_calls": []map[string]interface{}{
				{
					"id":   "call_123",
					"type": "function",
					"function": map[string]interface{}{
						"name":      "get_weather",
						"arguments": `{"city":"NYC"}`,
					},
				},
			}},
			{"role": "tool", "tool_call_id": "call_123", "content": "72F sunny"},
		},
	}
	body, _ := json.Marshal(openaiReq)

	result, err := OpenAIToAnthropic(body)
	require.NoError(t, err)

	var anthropicReq map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &anthropicReq))

	msgs := anthropicReq["messages"].([]interface{})
	// Should have user, assistant with tool_use, user with tool_result
	assert.Len(t, msgs, 3)

	// Assistant message should have tool_use content block
	assistantMsg := msgs[1].(map[string]interface{})
	content := assistantMsg["content"].([]interface{})
	toolUseBlock := content[0].(map[string]interface{})
	assert.Equal(t, "tool_use", toolUseBlock["type"])
	assert.Equal(t, "call_123", toolUseBlock["id"])
}

func TestDetectRequestFormat(t *testing.T) {
	tests := []struct {
		name     string
		body     map[string]interface{}
		expected string
	}{
		{
			"anthropic with max_tokens and array content",
			map[string]interface{}{
				"max_tokens": 1024,
				"messages": []map[string]interface{}{
					{"role": "user", "content": []map[string]interface{}{{"type": "text", "text": "hi"}}},
				},
			},
			"anthropic",
		},
		{
			"openai format",
			map[string]interface{}{
				"messages": []map[string]interface{}{
					{"role": "user", "content": "hi"},
				},
			},
			"openai",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			body, _ := json.Marshal(tc.body)
			got := DetectRequestFormat(body)
			assert.Equal(t, tc.expected, got)
		})
	}
}

func TestAnthropicToOpenAI_InvalidJSON(t *testing.T) {
	_, err := AnthropicToOpenAI([]byte("not json"))
	assert.Error(t, err)
}

func TestOpenAIToAnthropic_InvalidJSON(t *testing.T) {
	_, err := OpenAIToAnthropic([]byte("not json"))
	assert.Error(t, err)
}
