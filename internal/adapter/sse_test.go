package adapter

import (
	"bytes"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Helper: simulate OpenAI SSE stream
func openaiSSEChunks(chunks ...string) string {
	var b strings.Builder
	for _, c := range chunks {
		b.WriteString("data: " + c + "\n\n")
	}
	b.WriteString("data: [DONE]\n\n")
	return b.String()
}

func TestOpenAIToAnthropicSSE_TextContent(t *testing.T) {
	input := openaiSSEChunks(
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
	)

	var out bytes.Buffer
	err := OpenAIToAnthropicSSE(strings.NewReader(input), &out, "gpt-4")
	require.NoError(t, err)

	result := out.String()
	assert.Contains(t, result, "event: message_start")
	assert.Contains(t, result, "event: content_block_start")
	assert.Contains(t, result, `"type":"text_delta"`)
	assert.Contains(t, result, `"text":"Hello"`)
	assert.Contains(t, result, `"text":" world"`)
	assert.Contains(t, result, "event: content_block_stop")
	assert.Contains(t, result, "event: message_delta")
	assert.Contains(t, result, `"stop_reason":"end_turn"`)
	assert.Contains(t, result, "event: message_stop")
}

func TestOpenAIToAnthropicSSE_ReasoningContent(t *testing.T) {
	input := openaiSSEChunks(
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{"reasoning_content":"Let me think"},"finish_reason":null}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{"content":"The answer is 42"},"finish_reason":null}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
	)

	var out bytes.Buffer
	err := OpenAIToAnthropicSSE(strings.NewReader(input), &out, "gpt-4")
	require.NoError(t, err)

	result := out.String()
	assert.Contains(t, result, `"type":"thinking"`)
	assert.Contains(t, result, `"thinking":"Let me think"`)
	assert.Contains(t, result, `"type":"signature_delta"`)
	assert.Contains(t, result, `"type":"text_delta"`)
	assert.Contains(t, result, `"text":"The answer is 42"`)
}

func TestOpenAIToAnthropicSSE_ToolUse(t *testing.T) {
	input := openaiSSEChunks(
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{"content":"Let me check"},"finish_reason":null}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]},"finish_reason":null}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"NYC\"}"}}]},"finish_reason":null}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
	)

	var out bytes.Buffer
	err := OpenAIToAnthropicSSE(strings.NewReader(input), &out, "gpt-4")
	require.NoError(t, err)

	result := out.String()
	assert.Contains(t, result, `"type":"tool_use"`)
	assert.Contains(t, result, `"id":"call_abc"`)
	assert.Contains(t, result, `"name":"get_weather"`)
	assert.Contains(t, result, `"type":"input_json_delta"`)
	assert.Contains(t, result, `"stop_reason":"tool_use"`)
}

func TestOpenAIToAnthropicSSE_EmptyStream(t *testing.T) {
	input := "data: [DONE]\n\n"
	var out bytes.Buffer
	err := OpenAIToAnthropicSSE(strings.NewReader(input), &out, "gpt-4")
	require.NoError(t, err)
	// Should not crash, may emit minimal events
}

func TestOpenAIToAnthropicSSE_UnparseableChunk(t *testing.T) {
	input := "data: {not-json}\n\ndata: [DONE]\n\n"
	var out bytes.Buffer
	err := OpenAIToAnthropicSSE(strings.NewReader(input), &out, "gpt-4")
	require.NoError(t, err)
	// Should not crash, forward unparseable chunk
}

// Anthropic → OpenAI SSE translation tests

func anthropicSSE(events ...string) string {
	var b strings.Builder
	for _, e := range events {
		b.WriteString(e + "\n\n")
	}
	b.WriteString("data: [DONE]\n\n")
	return b.String()
}

func TestAnthropicToOpenAISSE_TextContent(t *testing.T) {
	input := anthropicSSE(
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-3\",\"content\":[]}}",
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}",
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}",
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}",
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}}",
		"event: message_stop\ndata: {\"type\":\"message_stop\"}",
	)

	var out bytes.Buffer
	err := AnthropicToOpenAISSE(strings.NewReader(input), &out, "claude-3")
	require.NoError(t, err)

	result := out.String()
	assert.Contains(t, result, `"content":"Hello"`)
	assert.Contains(t, result, `"object":"chat.completion.chunk"`)
	assert.Contains(t, result, "data: [DONE]")
}

func TestAnthropicToOpenAISSE_ToolUse(t *testing.T) {
	input := anthropicSSE(
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-3\",\"content\":[]}}",
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"tool_1\",\"name\":\"get_weather\",\"input\":{}}}",
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\\\"NYC\\\"}\"}}",
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}",
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":10}}",
		"event: message_stop\ndata: {\"type\":\"message_stop\"}",
	)

	var out bytes.Buffer
	err := AnthropicToOpenAISSE(strings.NewReader(input), &out, "claude-3")
	require.NoError(t, err)

	result := out.String()
	assert.Contains(t, result, `"tool_calls"`)
	assert.Contains(t, result, `"function"`)
	assert.Contains(t, result, `"finish_reason":"tool_calls"`)
}

func TestAnthropicToOpenAISSE_ThinkingDelta(t *testing.T) {
	input := anthropicSSE(
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-3\",\"content\":[]}}",
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"hmm\"}}",
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}}",
		"event: message_stop\ndata: {\"type\":\"message_stop\"}",
	)

	var out bytes.Buffer
	err := AnthropicToOpenAISSE(strings.NewReader(input), &out, "claude-3")
	require.NoError(t, err)

	result := out.String()
	assert.Contains(t, result, `"reasoning_content":"hmm"`)
}

func TestAnthropicToOpenAISSE_EmptyStream(t *testing.T) {
	input := "data: [DONE]\n\n"
	var out bytes.Buffer
	err := AnthropicToOpenAISSE(strings.NewReader(input), &out, "claude-3")
	require.NoError(t, err)
}
