package adapter

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ============== Helper function tests ==============

func TestExtractText_StringContent(t *testing.T) {
	input := json.RawMessage(`"hello world"`)
	result := extractText(input)
	assert.Equal(t, "hello world", result)
}

func TestExtractText_EmptyContent(t *testing.T) {
	input := json.RawMessage(`""`)
	result := extractText(input)
	assert.Equal(t, "", result)
}

func TestExtractText_ArrayOfBlocks(t *testing.T) {
	input := json.RawMessage(`[{"type":"text","text":"hello"},{"type":"text","text":" world"}]`)
	result := extractText(input)
	assert.Equal(t, "hello\n world", result)
}

func TestExtractText_ArrayWithNonTextBlocks(t *testing.T) {
	input := json.RawMessage(`[{"type":"text","text":"hello"},{"type":"tool_use","id":"123"}]`)
	result := extractText(input)
	assert.Equal(t, "hello", result)
}

func TestExtractText_RawString(t *testing.T) {
	input := json.RawMessage(`no quotes`)
	result := extractText(input)
	assert.Equal(t, "no quotes", result)
}

func TestExtractTextFromRaw_Nil(t *testing.T) {
	result := extractTextFromRaw(nil)
	assert.Equal(t, "", result)
}

func TestExtractTextFromRaw_Interface(t *testing.T) {
	result := extractTextFromRaw("direct string")
	assert.Equal(t, "direct string", result)
}

func TestExtractTextFromRaw_Map(t *testing.T) {
	// When passing a map, it gets JSON-marshaled to {"text":"hello"} and extractText returns the raw string
	result := extractTextFromRaw(map[string]interface{}{"text": "hello"})
	// The function doesn't recursively extract from maps - it returns the JSON representation
	assert.Equal(t, `{"text":"hello"}`, result)
}

func TestStringOrNil_Nil(t *testing.T) {
	result := stringOrNil(nil)
	assert.Equal(t, "", result)
}

func TestStringOrNil_String(t *testing.T) {
	result := stringOrNil("hello")
	assert.Equal(t, "hello", result)
}

func TestStringOrNil_OtherType(t *testing.T) {
	result := stringOrNil(123)
	assert.Equal(t, "123", result)
}

func TestJsonEscape_Basic(t *testing.T) {
	result := jsonEscape(`hello"world`)
	assert.Equal(t, `hello\"world`, result)
}

func TestJsonEscape_Newlines(t *testing.T) {
	result := jsonEscape("hello\nworld")
	assert.Contains(t, result, `\n`)
}

func TestJsonEscape_Empty(t *testing.T) {
	result := jsonEscape("")
	assert.Equal(t, "", result)
}

func TestRandomHex_Length(t *testing.T) {
	result := randomHex(16)
	assert.Len(t, result, 16)
}

func TestRandomHex_Uniqueness(t *testing.T) {
	results := make(map[string]bool)
	for i := 0; i < 100; i++ {
		r := randomHex(16)
		assert.False(t, results[r], "randomHex should generate unique values")
		results[r] = true
	}
}

// ============== DetectRequestFormat tests ==============

func TestDetectRequestFormat_OpenAI_WithoutMaxTokens(t *testing.T) {
	body := map[string]interface{}{
		"model": "gpt-4",
		"messages": []map[string]interface{}{
			{"role": "user", "content": "hi"},
		},
	}
	data, _ := json.Marshal(body)
	result := DetectRequestFormat(data)
	assert.Equal(t, "openai", result)
}

func TestDetectRequestFormat_Anthropic_WithMaxTokens(t *testing.T) {
	body := map[string]interface{}{
		"model":      "claude-3",
		"max_tokens": 1024,
		"messages": []map[string]interface{}{
			{"role": "user", "content": []map[string]interface{}{{"type": "text", "text": "hi"}}},
		},
	}
	data, _ := json.Marshal(body)
	result := DetectRequestFormat(data)
	assert.Equal(t, "anthropic", result)
}

func TestDetectRequestFormat_EmptyBody(t *testing.T) {
	result := DetectRequestFormat([]byte("{}"))
	assert.Equal(t, "openai", result)
}

func TestDetectRequestFormat_NilContent(t *testing.T) {
	body := map[string]interface{}{
		"model": "gpt-4",
		"messages": []map[string]interface{}{
			{"role": "user", "content": nil},
		},
	}
	data, _ := json.Marshal(body)
	result := DetectRequestFormat(data)
	assert.Equal(t, "openai", result)
}

// ============== AnthropicToOpenAI edge cases ==============

func TestAnthropicToOpenAI_EmptyMessages(t *testing.T) {
	anthropicReq := map[string]interface{}{
		"model":      "claude-3",
		"max_tokens": 1024,
	}
	body, _ := json.Marshal(anthropicReq)

	result, err := AnthropicToOpenAI(body)
	require.NoError(t, err)

	var openaiReq map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &openaiReq))

	assert.Equal(t, "claude-3", openaiReq["model"])
}

func TestAnthropicToOpenAI_SystemAsMap(t *testing.T) {
	anthropicReq := map[string]interface{}{
		"model":      "claude-3",
		"max_tokens": 1024,
		"system":     map[string]interface{}{"text": "You are helpful"},
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
	first := msgs[0].(map[string]interface{})
	assert.Equal(t, "system", first["role"])
}

func TestAnthropicToOpenAI_ToolUseBlocks(t *testing.T) {
	anthropicReq := map[string]interface{}{
		"model":      "claude-3",
		"max_tokens": 1024,
		"messages": []map[string]interface{}{
			{
				"role":    "assistant",
				"content": []map[string]interface{}{
					{"type": "text", "text": "Let me check the weather"},
					{"type": "tool_use", "id": "tool_1", "name": "get_weather", "input": map[string]interface{}{"city": "NYC"}},
				},
			},
		},
	}
	body, _ := json.Marshal(anthropicReq)

	result, err := AnthropicToOpenAI(body)
	require.NoError(t, err)

	var openaiReq map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &openaiReq))

	msgs := openaiReq["messages"].([]interface{})
	msg := msgs[0].(map[string]interface{})
	assert.Equal(t, "assistant", msg["role"])
	// Tool calls should be extracted
	tc, ok := msg["tool_calls"].([]interface{})
	require.True(t, ok)
	assert.Len(t, tc, 1)
}

func TestAnthropicToOpenAI_Tools(t *testing.T) {
	anthropicReq := map[string]interface{}{
		"model":      "claude-3",
		"max_tokens": 1024,
		"messages": []map[string]interface{}{
			{"role": "user", "content": "What's the weather?"},
		},
		"tools": []map[string]interface{}{
			{
				"name":        "get_weather",
				"description": "Get weather for a city",
				"input_schema": map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"city": map[string]interface{}{"type": "string"},
					},
				},
			},
		},
	}
	body, _ := json.Marshal(anthropicReq)

	result, err := AnthropicToOpenAI(body)
	require.NoError(t, err)

	var openaiReq map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &openaiReq))

	tools, ok := openaiReq["tools"].([]interface{})
	require.True(t, ok)
	assert.Len(t, tools, 1)
}

func TestAnthropicToOpenAI_ToolChoice(t *testing.T) {
	anthropicReq := map[string]interface{}{
		"model":      "claude-3",
		"max_tokens": 1024,
		"messages": []map[string]interface{}{
			{"role": "user", "content": "Use a tool"},
		},
		"tool_choice": map[string]interface{}{
			"type": "tool",
			"name": "get_weather",
		},
	}
	body, _ := json.Marshal(anthropicReq)

	result, err := AnthropicToOpenAI(body)
	require.NoError(t, err)

	var openaiReq map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &openaiReq))

	assert.NotNil(t, openaiReq["tool_choice"])
}

func TestAnthropicToOpenAI_Metadata(t *testing.T) {
	anthropicReq := map[string]interface{}{
		"model":      "claude-3",
		"max_tokens": 1024,
		"messages": []map[string]interface{}{
			{"role": "user", "content": "Hi"},
		},
		"metadata": map[string]interface{}{
			"user_id": "123",
		},
	}
	body, _ := json.Marshal(anthropicReq)

	result, err := AnthropicToOpenAI(body)
	require.NoError(t, err)

	var openaiReq map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &openaiReq))

	assert.Equal(t, "claude-3", openaiReq["model"])
}

func TestAnthropicToOpenAI_InvalidJSONEdge(t *testing.T) {
	_, err := AnthropicToOpenAI([]byte("not json"))
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "parsing anthropic request")
}

func TestAnthropicToOpenAI_Stream(t *testing.T) {
	anthropicReq := map[string]interface{}{
		"model":      "claude-3",
		"max_tokens": 1024,
		"stream":     true,
		"messages": []map[string]interface{}{
			{"role": "user", "content": "Hi"},
		},
	}
	body, _ := json.Marshal(anthropicReq)

	result, err := AnthropicToOpenAI(body)
	require.NoError(t, err)

	var openaiReq map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &openaiReq))

	assert.Equal(t, true, openaiReq["stream"])
}

// ============== OpenAIToAnthropic edge cases ==============

func TestOpenAIToAnthropic_EmptyMessages(t *testing.T) {
	openaiReq := map[string]interface{}{
		"model":    "gpt-4",
		"messages": []map[string]interface{}{},
	}
	body, _ := json.Marshal(openaiReq)

	result, err := OpenAIToAnthropic(body)
	require.NoError(t, err)

	var anthropicReq map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &anthropicReq))

	assert.Equal(t, "gpt-4", anthropicReq["model"])
	// MaxTokens defaults to 0 which unmarshals as float64
	assert.Equal(t, float64(0), anthropicReq["max_tokens"])
}

func TestOpenAIToAnthropic_AssistantWithToolCalls(t *testing.T) {
	openaiReq := map[string]interface{}{
		"model": "gpt-4",
		"messages": []map[string]interface{}{
			{"role": "user", "content": "What's the weather?"},
			{
				"role":         "assistant",
				"content":      nil,
				"tool_calls": []map[string]interface{}{
					{
						"id":   "call_abc",
						"type": "function",
						"function": map[string]interface{}{
							"name":      "get_weather",
							"arguments": `{"city":"NYC"}`,
						},
					},
				},
			},
		},
	}
	body, _ := json.Marshal(openaiReq)

	result, err := OpenAIToAnthropic(body)
	require.NoError(t, err)

	var anthropicReq map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &anthropicReq))

	msgs := anthropicReq["messages"].([]interface{})
	// Should have user message + assistant message with tool_use blocks
	assert.Len(t, msgs, 2)

	assistantMsg := msgs[1].(map[string]interface{})
	content := assistantMsg["content"].([]interface{})
	assert.Len(t, content, 1)
	toolUse := content[0].(map[string]interface{})
	assert.Equal(t, "tool_use", toolUse["type"])
}

func TestOpenAIToAnthropic_ToolResultAsUser(t *testing.T) {
	openaiReq := map[string]interface{}{
		"model": "gpt-4",
		"messages": []map[string]interface{}{
			{
				"role":       "tool",
				"tool_call_id": "call_abc",
				"content":    "72F sunny",
			},
		},
	}
	body, _ := json.Marshal(openaiReq)

	result, err := OpenAIToAnthropic(body)
	require.NoError(t, err)

	var anthropicReq map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &anthropicReq))

	msgs := anthropicReq["messages"].([]interface{})
	msg := msgs[0].(map[string]interface{})
	assert.Equal(t, "user", msg["role"])
	content := msg["content"].([]interface{})
	toolResult := content[0].(map[string]interface{})
	assert.Equal(t, "tool_result", toolResult["type"])
}

func TestOpenAIToAnthropic_AssistantWithTextAndToolCalls(t *testing.T) {
	openaiReq := map[string]interface{}{
		"model": "gpt-4",
		"messages": []map[string]interface{}{
			{
				"role":    "assistant",
				"content": "Let me check that for you",
				"tool_calls": []map[string]interface{}{
					{
						"id":   "call_123",
						"type": "function",
						"function": map[string]interface{}{
							"name":      "get_weather",
							"arguments": `{"city":"LA"}`,
						},
					},
				},
			},
		},
	}
	body, _ := json.Marshal(openaiReq)

	result, err := OpenAIToAnthropic(body)
	require.NoError(t, err)

	var anthropicReq map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &anthropicReq))

	msgs := anthropicReq["messages"].([]interface{})
	msg := msgs[0].(map[string]interface{})
	content := msg["content"].([]interface{})
	// Should have both text and tool_use blocks
	assert.Len(t, content, 2)
}

func TestOpenAIToAnthropic_SystemMessage(t *testing.T) {
	openaiReq := map[string]interface{}{
		"model": "gpt-4",
		"messages": []map[string]interface{}{
			{"role": "system", "content": "You are helpful"},
			{"role": "user", "content": "Hi"},
		},
	}
	body, _ := json.Marshal(openaiReq)

	result, err := OpenAIToAnthropic(body)
	require.NoError(t, err)

	var anthropicReq map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &anthropicReq))

	assert.Equal(t, "You are helpful", anthropicReq["system"])
}

func TestOpenAIToAnthropic_Tools(t *testing.T) {
	openaiReq := map[string]interface{}{
		"model": "gpt-4",
		"messages": []map[string]interface{}{
			{"role": "user", "content": "Hi"},
		},
		"tools": []map[string]interface{}{
			{
				"type": "function",
				"function": map[string]interface{}{
					"name":        "get_weather",
					"description": "Get weather",
					"parameters":  map[string]interface{}{"type": "object"},
				},
			},
		},
	}
	body, _ := json.Marshal(openaiReq)

	result, err := OpenAIToAnthropic(body)
	require.NoError(t, err)

	var anthropicReq map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &anthropicReq))

	tools := anthropicReq["tools"].([]interface{})
	assert.Len(t, tools, 1)
	tool := tools[0].(map[string]interface{})
	assert.Equal(t, "get_weather", tool["name"])
}

func TestOpenAIToAnthropic_Temperature(t *testing.T) {
	temp := 0.7
	openaiReq := map[string]interface{}{
		"model":       "gpt-4",
		"temperature": temp,
		"messages": []map[string]interface{}{
			{"role": "user", "content": "Hi"},
		},
	}
	body, _ := json.Marshal(openaiReq)

	result, err := OpenAIToAnthropic(body)
	require.NoError(t, err)

	var anthropicReq map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &anthropicReq))

	// Temperature is not directly mapped in OpenAI to Anthropic
	assert.Equal(t, "gpt-4", anthropicReq["model"])
}

func TestOpenAIToAnthropic_InvalidJSONEdge(t *testing.T) {
	_, err := OpenAIToAnthropic([]byte("not json"))
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "parsing openai request")
}

// ============== OpenAIToGemini tests ==============

func TestOpenAIToGemini_Basic(t *testing.T) {
	openaiReq := map[string]interface{}{
		"model": "gpt-4",
		"messages": []map[string]interface{}{
			{"role": "user", "content": "Hello"},
		},
	}
	body, _ := json.Marshal(openaiReq)

	result, err := OpenAIToGemini(body)
	require.NoError(t, err)

	var geminiReq map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &geminiReq))

	contents := geminiReq["contents"].([]interface{})
	assert.Len(t, contents, 1)
}

func TestOpenAIToGemini_SystemMessage(t *testing.T) {
	openaiReq := map[string]interface{}{
		"model": "gpt-4",
		"messages": []map[string]interface{}{
			{"role": "system", "content": "You are helpful"},
			{"role": "user", "content": "Hi"},
		},
	}
	body, _ := json.Marshal(openaiReq)

	result, err := OpenAIToGemini(body)
	require.NoError(t, err)

	var geminiReq map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &geminiReq))

	assert.NotNil(t, geminiReq["systemInstruction"])
}

func TestOpenAIToGemini_ToolCalls(t *testing.T) {
	openaiReq := map[string]interface{}{
		"model": "gpt-4",
		"messages": []map[string]interface{}{
			{"role": "user", "content": "Use get_weather"},
			{
				"role":    "assistant",
				"content": nil,
				"tool_calls": []map[string]interface{}{
					{
						"id":   "call_123",
						"type": "function",
						"function": map[string]interface{}{
							"name":      "get_weather",
							"arguments": `{"city":"NYC"}`,
						},
					},
				},
			},
		},
	}
	body, _ := json.Marshal(openaiReq)

	result, err := OpenAIToGemini(body)
	require.NoError(t, err)

	var geminiReq map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &geminiReq))

	contents := geminiReq["contents"].([]interface{})
	assert.Len(t, contents, 2) // user + assistant with function call
}

func TestOpenAIToGemini_ToolResult(t *testing.T) {
	openaiReq := map[string]interface{}{
		"model": "gpt-4",
		"messages": []map[string]interface{}{
			{
				"role":       "tool",
				"tool_call_id": "call_123",
				"content":    "72F sunny",
			},
		},
	}
	body, _ := json.Marshal(openaiReq)

	result, err := OpenAIToGemini(body)
	require.NoError(t, err)

	var geminiReq map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &geminiReq))

	contents := geminiReq["contents"].([]interface{})
	msg := contents[0].(map[string]interface{})
	assert.Equal(t, "user", msg["role"])
}

func TestOpenAIToGemini_GenerationConfig(t *testing.T) {
	openaiReq := map[string]interface{}{
		"model":     "gpt-4",
		"max_tokens": 100,
		"messages": []map[string]interface{}{
			{"role": "user", "content": "Hi"},
		},
	}
	body, _ := json.Marshal(openaiReq)

	result, err := OpenAIToGemini(body)
	require.NoError(t, err)

	var geminiReq map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &geminiReq))

	genConfig := geminiReq["generationConfig"].(map[string]interface{})
	assert.Equal(t, 100, int(genConfig["maxOutputTokens"].(float64)))
}

func TestOpenAIToGemini_InvalidJSON(t *testing.T) {
	_, err := OpenAIToGemini([]byte("not json"))
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "parsing OpenAI request")
}

// ============== GeminiToOpenAI tests ==============

func TestGeminiToOpenAI_Basic(t *testing.T) {
	geminiResp := map[string]interface{}{
		"candidates": []map[string]interface{}{
			{
				"content": map[string]interface{}{
					"parts": []map[string]interface{}{
						{"text": "Hello!"},
					},
				},
				"finishReason": "STOP",
			},
		},
		"usageMetadata": map[string]interface{}{
			"promptTokenCount":     10,
			"candidatesTokenCount": 5,
			"totalTokenCount":      15,
		},
	}
	body, _ := json.Marshal(geminiResp)

	result, err := GeminiToOpenAI(body, "gemini-pro")
	require.NoError(t, err)

	var openaiResp map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &openaiResp))

	assert.Equal(t, "chat.completion", openaiResp["object"])
	assert.Equal(t, "gemini-pro", openaiResp["model"])
	choices := openaiResp["choices"].([]interface{})
	assert.Len(t, choices, 1)
	usage := openaiResp["usage"].(map[string]interface{})
	assert.Equal(t, 10, int(usage["prompt_tokens"].(float64)))
}

func TestGeminiToOpenAI_FunctionCall(t *testing.T) {
	geminiResp := map[string]interface{}{
		"candidates": []map[string]interface{}{
			{
				"content": map[string]interface{}{
					"parts": []map[string]interface{}{
						{
							"functionCall": map[string]interface{}{
								"name": "get_weather",
								"args": map[string]interface{}{"city": "NYC"},
							},
						},
					},
				},
				"finishReason": "STOP",
			},
		},
	}
	body, _ := json.Marshal(geminiResp)

	result, err := GeminiToOpenAI(body, "gemini-pro")
	require.NoError(t, err)

	var openaiResp map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &openaiResp))

	choices := openaiResp["choices"].([]interface{})
	choice := choices[0].(map[string]interface{})
	msg := choice["message"].(map[string]interface{})
	assert.Equal(t, "assistant", msg["role"])
	assert.NotNil(t, msg["tool_calls"])
}

func TestGeminiToOpenAI_DifferentFinishReasons(t *testing.T) {
	tests := []struct {
		geminiReason string
		expected    string
	}{
		{"STOP", "stop"},
		{"MAX_TOKENS", "length"},
		{"SAFETY", "content_filter"},
		{"OTHER", "content_filter"},
		{"RECITATION", "content_filter"},
		{"UNKNOWN", "stop"},
	}

	for _, tc := range tests {
		t.Run(tc.geminiReason, func(t *testing.T) {
			geminiResp := map[string]interface{}{
				"candidates": []map[string]interface{}{
					{
						"content": map[string]interface{}{
							"parts": []map[string]interface{}{
								{"text": "test"},
							},
						},
						"finishReason": tc.geminiReason,
					},
				},
			}
			body, _ := json.Marshal(geminiResp)
			result, err := GeminiToOpenAI(body, "gemini")
			require.NoError(t, err)
			var openaiResp map[string]interface{}
			require.NoError(t, json.Unmarshal(result, &openaiResp))
			choices := openaiResp["choices"].([]interface{})
			choice := choices[0].(map[string]interface{})
			assert.Equal(t, tc.expected, choice["finish_reason"])
		})
	}
}

func TestGeminiToOpenAI_InvalidJSON(t *testing.T) {
	_, err := GeminiToOpenAI([]byte("not json"), "gemini")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "parsing Gemini response")
}

// ============== Gemini Role conversion ==============

func TestGeminiRole_Conversions(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"user", "user"},
		{"assistant", "model"},
		{"system", "system"},
		{"unknown", "user"},
	}

	for _, tc := range tests {
		t.Run(tc.input, func(t *testing.T) {
			result := geminiRole(tc.input)
			assert.Equal(t, tc.expected, result)
		})
	}
}

// ============== SSE Writer tests ==============

func TestSSEWriter_WriteLine(t *testing.T) {
	var buf bytes.Buffer
	writer := &sseWriter{w: &buf}
	writer.writeLine("hello")
	assert.Equal(t, "hello\n", buf.String())
}

func TestSSEWriter_WriteEvent(t *testing.T) {
	var buf bytes.Buffer
	writer := &sseWriter{w: &buf}
	writer.writeEvent("test", map[string]interface{}{"key": "value"})
	assert.Contains(t, buf.String(), "event: test\n")
	assert.Contains(t, buf.String(), "data: ")
}

func TestTryFlusher_NilWriter(t *testing.T) {
	var buf bytes.Buffer
	flusher := tryFlusher(&buf)
	assert.Nil(t, flusher)
}

type mockFlusher struct {
	flushed bool
}

func (m *mockFlusher) Flush() {
	m.flushed = true
}

type mockWriterWithFlusher struct {
	buf bytes.Buffer
	mockFlusher
}

func (m *mockWriterWithFlusher) Write(p []byte) (int, error) {
	return m.buf.Write(p)
}

func TestTryFlusher_WithFlusher(t *testing.T) {
	w := &mockWriterWithFlusher{}
	flusher := tryFlusher(w)
	assert.NotNil(t, flusher)
	flusher.Flush()
	assert.True(t, w.flushed)
}

// ============== SSE Gemini tests ==============

func TestGeminiToOpenAISSE_Basic(t *testing.T) {
	input := `data: {"candidates":[{"content":{"parts":[{"text":"Hello"}]}}]}
data: [DONE]
`
	var out bytes.Buffer
	err := GeminiToOpenAISSE(strings.NewReader(input), &out, "gemini-pro")
	require.NoError(t, err)
	assert.Contains(t, out.String(), "chat.completion.chunk")
	assert.Contains(t, out.String(), `"content":"Hello"`)
}

func TestGeminiToOpenAISSE_FunctionCall(t *testing.T) {
	// Gemini streaming chunk format
	input := `data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather","args":{"city":"NYC"}}}}]},"finishReason":"STOP"}]}
data: [DONE]
`
	var out bytes.Buffer
	err := GeminiToOpenAISSE(strings.NewReader(input), &out, "gemini")
	require.NoError(t, err)
	// The output should indicate a function call was processed
	result := out.String()
	// Function calls produce tool_calls in the output
	assert.Contains(t, result, "function")
}

func TestGeminiToOpenAISSE_EmptyStream(t *testing.T) {
	input := `data: [DONE]
`
	var out bytes.Buffer
	err := GeminiToOpenAISSE(strings.NewReader(input), &out, "gemini")
	require.NoError(t, err)
}

func TestGeminiToOpenAISSE_InvalidJSON(t *testing.T) {
	input := `data: not-json
data: [DONE]
`
	var out bytes.Buffer
	err := GeminiToOpenAISSE(strings.NewReader(input), &out, "gemini")
	require.NoError(t, err)
	// Should forward unparseable lines
}

func TestOpenAIToGeminiSSE_Basic(t *testing.T) {
	input := `data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}
data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}
data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
data: [DONE]
`
	var out bytes.Buffer
	err := OpenAIToGeminiSSE(strings.NewReader(input), &out, "gemini")
	require.NoError(t, err)
	assert.Contains(t, out.String(), "message_start")
}

func TestOpenAIToGeminiSSE_EmptyStream(t *testing.T) {
	input := `data: [DONE]
`
	var out bytes.Buffer
	err := OpenAIToGeminiSSE(strings.NewReader(input), &out, "gemini")
	require.NoError(t, err)
}
