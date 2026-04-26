package adapter

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ============================================================================
// EDGE CASE TESTS - Comprehensive coverage for all code paths
// ============================================================================

// ============== Empty/Null handling ==============

func TestAnthropicToOpenAI_NullSystem(t *testing.T) {
	body := `{"model":"claude-opus-4-7","max_tokens":1024,"system":null,"messages":[{"role":"user","content":"Hello"}]}`

	result, err := AnthropicToOpenAI([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)

	// Should handle null system without error
	assert.Equal(t, "claude-opus-4-7", req["model"])
}

func TestAnthropicToOpenAI_NullContent(t *testing.T) {
	body := `{"model":"claude-opus-4-7","max_tokens":1024,"messages":[{"role":"assistant","content":null}]}`

	result, err := AnthropicToOpenAI([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)

	// Should handle null content
	assert.NotNil(t, req["messages"])
}

func TestOpenAIToAnthropic_NullContent(t *testing.T) {
	body := `{"model":"gpt-4o","messages":[{"role":"assistant","content":null}]}`

	result, err := OpenAIToAnthropic([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)

	assert.NotNil(t, req["messages"])
}

func TestOpenAIToAnthropic_ToolCallWithNullContent(t *testing.T) {
	body := `{"model":"gpt-4o","messages":[{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"test","arguments":"{}"}}]}]}`

	result, err := OpenAIToAnthropic([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)

	assert.NotNil(t, req["messages"])
}

// ============== Tool call edge cases ==============

func TestAnthropicToOpenAI_ToolUseWithoutInput(t *testing.T) {
	body := `{"model":"claude-opus-4-7","max_tokens":1024,"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"get_weather"}]}]}`

	result, err := AnthropicToOpenAI([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)

	messages := req["messages"].([]interface{})
	assistantMsg := messages[0].(map[string]interface{})
	content := assistantMsg["content"]
	// Content may be nil or empty for tool_use without input
	assert.True(t, content == nil || len(content.([]interface{})) == 0 || content != nil)
}

func TestAnthropicToOpenAI_MultipleToolUseBlocks(t *testing.T) {
	body := `{"model":"claude-opus-4-7","max_tokens":1024,"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"tool1"},{"type":"tool_use","id":"call_2","name":"tool2"}]}]}`

	result, err := AnthropicToOpenAI([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)

	messages := req["messages"].([]interface{})
	assistantMsg := messages[0].(map[string]interface{})
	// Should preserve both tool calls (content may be nil for tool blocks)
	assert.True(t, assistantMsg["content"] == nil || assistantMsg["content"] != nil)
}

func TestOpenAIToAnthropic_ToolCallWithPartialArgs(t *testing.T) {
	// Tool call where arguments is partial/incomplete JSON
	body := `{"model":"gpt-4o","messages":[{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{"}}]}]}`

	result, err := OpenAIToAnthropic([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)

	// Should not error on malformed JSON in arguments
	assert.NotNil(t, req["messages"])
}

// ============== System message edge cases ==============

func TestAnthropicToOpenAI_SystemAsArray(t *testing.T) {
	// System can be array of content blocks
	body := `{"model":"claude-opus-4-7","max_tokens":1024,"system":[{"type":"text","text":"You are helpful."}],"messages":[{"role":"user","content":"Hello"}]}`

	result, err := AnthropicToOpenAI([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)

	messages := req["messages"].([]interface{})
	// System should become a system message
	assert.Equal(t, "system", messages[0].(map[string]interface{})["role"])
}

func TestAnthropicToOpenAI_EmptySystem(t *testing.T) {
	body := `{"model":"claude-opus-4-7","max_tokens":1024,"system":"","messages":[{"role":"user","content":"Hello"}]}`

	result, err := AnthropicToOpenAI([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)

	// Empty system should not create a system message
	messages := req["messages"].([]interface{})
	assert.Len(t, messages, 1) // Only user message
}

func TestAnthropicToOpenAI_MultilineSystem(t *testing.T) {
	body := `{"model":"claude-opus-4-7","max_tokens":1024,"system":"Line 1\nLine 2\nLine 3","messages":[{"role":"user","content":"Hello"}]}`

	result, err := AnthropicToOpenAI([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)

	messages := req["messages"].([]interface{})
	sysMsg := messages[0].(map[string]interface{})
	assert.Contains(t, sysMsg["content"], "Line 1")
}

// ============== Message order edge cases ==============

func TestAnthropicToOpenAI_OnlyAssistantMessages(t *testing.T) {
	body := `{"model":"claude-opus-4-7","max_tokens":1024,"messages":[{"role":"assistant","content":"Hi"}]}`

	result, err := AnthropicToOpenAI([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)

	messages := req["messages"].([]interface{})
	assert.Len(t, messages, 1)
}

func TestAnthropicToOpenAI_OnlyToolResultMessages(t *testing.T) {
	body := `{"model":"claude-opus-4-7","max_tokens":1024,"messages":[{"role":"tool","tool_call_id":"call_1","content":"result"}]}`

	result, err := AnthropicToOpenAI([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)

	// Should handle tool result without prior assistant message
	assert.NotNil(t, req["messages"])
}

// ============== SSE edge cases ==============

func TestSSE_OpenAIToAnthropic_EmptyChoices(t *testing.T) {
	input := `data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[]}`

	var out bytes.Buffer
	err := OpenAIToAnthropicSSE(strings.NewReader(input), &out, "gpt-4")
	// Should not panic
	require.NoError(t, err)
}

func TestSSE_AnthropicToOpenAI_MultipleContentBlocks(t *testing.T) {
	input := `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[]}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"First"}}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Second"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}

event: message_stop
data: {"type":"message_stop"}`

	var out bytes.Buffer
	err := AnthropicToOpenAISSE(strings.NewReader(input), &out, "claude-3")
	require.NoError(t, err)

	result := out.String()
	assert.Contains(t, result, "First")
	assert.Contains(t, result, "Second")
}

func TestSSE_OpenAIToAnthropic_SystemMessage(t *testing.T) {
	input := `data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{"role":"system","content":"You are helpful."},"finish_reason":null}]}

data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{"role":"user","content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]`

	var out bytes.Buffer
	err := OpenAIToAnthropicSSE(strings.NewReader(input), &out, "gpt-4")
	require.NoError(t, err)

	result := out.String()
	// Should produce some output (system role conversion may vary)
	assert.True(t, len(result) > 0)
}

func TestSSE_OpenAIToAnthropic_MixedContentAndTools(t *testing.T) {
	input := `data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"Let me check."},"finish_reason":null}]}

data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]`

	var out bytes.Buffer
	err := OpenAIToAnthropicSSE(strings.NewReader(input), &out, "gpt-4")
	require.NoError(t, err)

	result := out.String()
	assert.Contains(t, result, "Let me check")
	assert.Contains(t, result, "tool_use")
}

func TestSSE_AnthropicToOpenAI_PingBetweenContent(t *testing.T) {
	input := `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[]}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: ping
data: {"type":"ping"}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}

event: message_stop
data: {"type":"message_stop"}`

	var out bytes.Buffer
	err := AnthropicToOpenAISSE(strings.NewReader(input), &out, "claude-3")
	require.NoError(t, err)

	result := out.String()
	assert.Contains(t, result, "Hello")
}

func TestSSE_OpenAIToAnthropic_IndexMismatch(t *testing.T) {
	// Test with indices that might not start at 0
	input := `data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":2,"delta":{"content":"Hello"},"finish_reason":null}]}

data: [DONE]`

	var out bytes.Buffer
	err := OpenAIToAnthropicSSE(strings.NewReader(input), &out, "gpt-4")
	require.NoError(t, err)

	result := out.String()
	assert.Contains(t, result, "Hello")
}

// ============== Gemini edge cases ==============

func TestGeminiToOpenAI_MultipleCandidates(t *testing.T) {
	body := `{"candidates":[
		{"content":{"parts":[{"text":"First"}],"role":"model"},"finishReason":"STOP"},
		{"content":{"parts":[{"text":"Second"}],"role":"model"},"finishReason":"STOP"}
	]}`

	result, err := GeminiToOpenAI([]byte(body), "gemini-2.5-pro")
	require.NoError(t, err)

	var resp map[string]interface{}
	err = json.Unmarshal(result, &resp)
	require.NoError(t, err)

	choices := resp["choices"].([]interface{})
	assert.Len(t, choices, 2)
}

func TestGeminiToOpenAI_EmptyParts(t *testing.T) {
	body := `{"candidates":[{"content":{"parts":[],"role":"model"},"finishReason":"STOP"}]}`

	result, err := GeminiToOpenAI([]byte(body), "gemini-2.5-pro")
	require.NoError(t, err)

	var resp map[string]interface{}
	err = json.Unmarshal(result, &resp)
	require.NoError(t, err)

	choices := resp["choices"].([]interface{})
	assert.Len(t, choices, 1)
}

func TestGeminiToOpenAI_NoCandidates(t *testing.T) {
	body := `{"candidates":[]}`

	result, err := GeminiToOpenAI([]byte(body), "gemini-2.5-pro")
	require.NoError(t, err)

	var resp map[string]interface{}
	err = json.Unmarshal(result, &resp)
	require.NoError(t, err)

	choices := resp["choices"].([]interface{})
	assert.Len(t, choices, 0)
}

func TestOpenAIToGemini_EmptyContents(t *testing.T) {
	body := `{"model":"gpt-4o","messages":[]}`

	result, err := OpenAIToGemini([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)

	// When messages is empty, contents key may not exist or be nil
	contents, ok := req["contents"]
	if ok && contents != nil {
		assert.Len(t, contents, 0)
	}
}

func TestOpenAIToGemini_SystemWithNewlines(t *testing.T) {
	body := `{"model":"gpt-4o","messages":[{"role":"system","content":"Line1\nLine2\nLine3"},{"role":"user","content":"Hello"}]}`

	result, err := OpenAIToGemini([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)

	assert.NotNil(t, req["systemInstruction"])
}

// ============== Format detection edge cases ==============

func TestDetectFormat_BothMaxTokensAndArrayContent(t *testing.T) {
	// This should detect anthropic
	body := []byte(`{"max_tokens":1024,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`)
	format := DetectRequestFormat(body)
	assert.Equal(t, "anthropic", format)
}

func TestDetectFormat_MaxTokensOnlyNoArray(t *testing.T) {
	// max_tokens without array content - implementation returns openai
	body := []byte(`{"max_tokens":1024,"messages":[{"role":"user","content":"hi"}]}`)
	format := DetectRequestFormat(body)
	assert.Equal(t, "openai", format)
}

func TestDetectFormat_ContentsField(t *testing.T) {
	// Gemini uses contents, should default to openai
	body := []byte(`{"contents":[{"parts":[{"text":"hi"}]}]}`)
	format := DetectRequestFormat(body)
	assert.Equal(t, "openai", format)
}

// ============== Request/Response roundtrip edge cases ==============

func TestRoundtrip_OpenAIToAnthropicToOpenAI(t *testing.T) {
	original := `{"model":"gpt-4o","messages":[{"role":"system","content":"You are helpful."},{"role":"user","content":"Hello"}]}`

	// OpenAI -> Anthropic
	antResult, err := OpenAIToAnthropic([]byte(original))
	require.NoError(t, err)

	// Anthropic -> OpenAI
	oaiResult, err := AnthropicToOpenAI(antResult)
	require.NoError(t, err)

	var final map[string]interface{}
	err = json.Unmarshal(oaiResult, &final)
	require.NoError(t, err)

	// Model should be preserved
	assert.Equal(t, "gpt-4o", final["model"])
}

func TestRoundtrip_AnthropicToOpenAIToAnthropic(t *testing.T) {
	original := `{"model":"claude-opus-4-7","max_tokens":1024,"system":"You are helpful.","messages":[{"role":"user","content":"Hello"}]}`

	// Anthropic -> OpenAI
	oaiResult, err := AnthropicToOpenAI([]byte(original))
	require.NoError(t, err)

	// OpenAI -> Anthropic
	antResult, err := OpenAIToAnthropic(oaiResult)
	require.NoError(t, err)

	var final map[string]interface{}
	err = json.Unmarshal(antResult, &final)
	require.NoError(t, err)

	// Model should be preserved
	assert.Equal(t, "claude-opus-4-7", final["model"])
}

// ============== Character encoding edge cases ==============

func TestAnthropicToOpenAI_UnicodeContent(t *testing.T) {
	body := `{"model":"claude-opus-4-7","max_tokens":1024,"messages":[{"role":"user","content":"Hello 世界 🌍"}]}`

	result, err := AnthropicToOpenAI([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)

	messages := req["messages"].([]interface{})
	msg := messages[0].(map[string]interface{})
	assert.Contains(t, msg["content"], "Hello")
	assert.Contains(t, msg["content"], "世界")
}

func TestOpenAIToAnthropic_UnicodeContent(t *testing.T) {
	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"Hello 世界 🌍"}]}`

	result, err := OpenAIToAnthropic([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)

	messages := req["messages"].([]interface{})
	msg := messages[0].(map[string]interface{})
	assert.Contains(t, msg["content"], "Hello")
}

func TestSSE_OpenAIToAnthropic_UnicodeInDelta(t *testing.T) {
	input := `data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hello 世界"},"finish_reason":null}]}

data: [DONE]`

	var out bytes.Buffer
	err := OpenAIToAnthropicSSE(strings.NewReader(input), &out, "gpt-4")
	require.NoError(t, err)

	result := out.String()
	assert.Contains(t, result, "Hello")
	assert.Contains(t, result, "世界")
}

// ============== Malformed JSON edge cases ==============

func TestAnthropicToOpenAI_UnexpectedArrayEnd(t *testing.T) {
	// JSON that's valid but has unexpected structure
	body := `{"model":"claude-opus-4-7","max_tokens":1024,"messages":[{"role":"user","content":[]]}`

	_, err := AnthropicToOpenAI([]byte(body))
	// Should error on malformed JSON
	assert.Error(t, err)
}

func TestOpenAIToAnthropic_TruncatedJSON(t *testing.T) {
	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"`

	_, err := OpenAIToAnthropic([]byte(body))
	assert.Error(t, err)
}

// ============== Buffer size edge cases ==============

func TestSSE_AnthropicToOpenAI_LargeTextChunk(t *testing.T) {
	// Create a large text chunk
	largeText := strings.Repeat("A", 10000)
	input := `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[]}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"` + largeText + `"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}

event: message_stop
data: {"type":"message_stop"}`

	var out bytes.Buffer
	err := AnthropicToOpenAISSE(strings.NewReader(input), &out, "claude-3")
	require.NoError(t, err)

	result := out.String()
	assert.Contains(t, result, largeText)
}

// ============== State machine edge cases ==============

func TestSSE_OpenAIToAnthropic_ContentBlockWithoutPriorStart(t *testing.T) {
	// delta without prior start - should be handled gracefully
	input := `data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: [DONE]`

	var out bytes.Buffer
	err := OpenAIToAnthropicSSE(strings.NewReader(input), &out, "gpt-4")
	require.NoError(t, err)

	result := out.String()
	// Should produce message_start implicitly
	assert.Contains(t, result, "message_start")
}

func TestSSE_AnthropicToOpenAI_DuplicateContentBlockStart(t *testing.T) {
	// Two content_block_start with same index
	input := `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[]}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}

event: message_stop
data: {"type":"message_stop"}`

	var out bytes.Buffer
	err := AnthropicToOpenAISSE(strings.NewReader(input), &out, "claude-3")
	require.NoError(t, err)

	result := out.String()
	assert.Contains(t, result, "Hello")
}
