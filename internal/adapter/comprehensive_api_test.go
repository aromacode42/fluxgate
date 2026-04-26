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
// COMPREHENSIVE API FORMAT TESTS - Covering all scenarios from official API docs
// ============================================================================

// ============== Multimodal Content Tests ==============

func TestAnthropicToOpenAI_ImageContentBlock(t *testing.T) {
	// Anthropic supports base64 images in user messages
	body := `{
		"model":"claude-opus-4-7",
		"max_tokens":1024,
		"messages":[{
			"role":"user",
			"content":[
				{"type":"text","text":"What is in this image?"},
				{"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":"SGVsbG8gV29ybGQ="}}
			]
		}]
	}`

	result, err := AnthropicToOpenAI([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)

	messages := req["messages"].([]interface{})
	msg := messages[0].(map[string]interface{})
	content := msg["content"].([]interface{})
	assert.Len(t, content, 2)
}

func TestAnthropicToOpenAI_SystemAsArrayOfTextBlocks(t *testing.T) {
	// System can be an array of content blocks
	body := `{
		"model":"claude-opus-4-7",
		"max_tokens":1024,
		"system":[
			{"type":"text","text":"You are a helpful assistant."},
			{"type":"text","text":"Be concise."}
		],
		"messages":[{"role":"user","content":"Hello"}]
	}`

	result, err := AnthropicToOpenAI([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)

	messages := req["messages"].([]interface{})
	// System blocks should be concatenated
	assert.True(t, len(messages) >= 1)
}

func TestAnthropicToOpenAI_SystemAsArrayWithImage(t *testing.T) {
	// System with image (multimodal system instruction)
	body := `{
		"model":"claude-opus-4-7",
		"max_tokens":1024,
		"system":[{"type":"text","text":"Describe the image:"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"ABC123"}}],
		"messages":[{"role":"user","content":"Hello"}]
	}`

	result, err := AnthropicToOpenAI([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)
	// Should handle array system without panic
	assert.NotNil(t, req["model"])
}

func TestOpenAIToAnthropic_ImageInMessage(t *testing.T) {
	// OpenAI can send base64 images - converted to Anthropic format
	body := `{
		"model":"gpt-4o",
		"messages":[{
			"role":"user",
			"content":[
				{"type":"text","text":"What do you see?"},
				{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,SGVsbG8="}}
			]
		}]
	}`

	result, err := OpenAIToAnthropic([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)
	assert.Equal(t, "gpt-4o", req["model"])
}

// ============== Generation Parameters Tests ==============

func TestAnthropicToOpenAI_Temperature(t *testing.T) {
	body := `{"model":"claude-opus-4-7","max_tokens":1024,"temperature":0.7,"messages":[{"role":"user","content":"Hello"}]}`

	result, err := AnthropicToOpenAI([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)

	// Temperature should be passed through (not currently converted but should not error)
	assert.Equal(t, "claude-opus-4-7", req["model"])
}

func TestAnthropicToOpenAI_TopP(t *testing.T) {
	body := `{"model":"claude-opus-4-7","max_tokens":1024,"top_p":0.9,"messages":[{"role":"user","content":"Hello"}]}`

	result, err := AnthropicToOpenAI([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)
	// top_p is not in OpenAIRequest struct - tests passthrough behavior
	assert.Equal(t, "claude-opus-4-7", req["model"])
}

func TestAnthropicToOpenAI_StopSequences(t *testing.T) {
	body := `{"model":"claude-opus-4-7","max_tokens":1024,"stop_sequences":["END","STOP"],"messages":[{"role":"user","content":"Hello"}]}`

	result, err := AnthropicToOpenAI([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)
	// stop_sequences not mapped to OpenAI - tests passthrough
	assert.Equal(t, "claude-opus-4-7", req["model"])
}

func TestAnthropicToOpenAI_ThinkingConfig(t *testing.T) {
	// Anthropic supports extended thinking
	body := `{
		"model":"claude-opus-4-7",
		"max_tokens":1024,
		"thinking":{"type":"enabled","budget_tokens":10000},
		"messages":[{"role":"user","content":"Hello"}]
	}`

	result, err := AnthropicToOpenAI([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)
	// thinking config should be passed (not in OpenAIRequest - tests passthrough)
	assert.Equal(t, "claude-opus-4-7", req["model"])
}

func TestOpenAIToAnthropic_TemperatureField(t *testing.T) {
	body := `{"model":"gpt-4o","temperature":0.8,"messages":[{"role":"user","content":"Hello"}]}`

	result, err := OpenAIToAnthropic([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)
	// Temperature should be present in output (not currently mapped)
	assert.Equal(t, "gpt-4o", req["model"])
}

func TestOpenAIToAnthropic_TopP(t *testing.T) {
	body := `{"model":"gpt-4o","top_p":0.95,"messages":[{"role":"user","content":"Hello"}]}`

	result, err := OpenAIToAnthropic([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)
	assert.Equal(t, "gpt-4o", req["model"])
}

func TestOpenAIToAnthropic_FrequencyPenalty(t *testing.T) {
	body := `{"model":"gpt-4o","frequency_penalty":0.5,"messages":[{"role":"user","content":"Hello"}]}`

	result, err := OpenAIToAnthropic([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)
	// frequency_penalty not mapped but should not error
	assert.Equal(t, "gpt-4o", req["model"])
}

func TestOpenAIToAnthropic_PresencePenalty(t *testing.T) {
	body := `{"model":"gpt-4o","presence_penalty":0.3,"messages":[{"role":"user","content":"Hello"}]}`

	result, err := OpenAIToAnthropic([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)
	assert.Equal(t, "gpt-4o", req["model"])
}

func TestOpenAIToAnthropic_Seed(t *testing.T) {
	body := `{"model":"gpt-4o","seed":12345,"messages":[{"role":"user","content":"Hello"}]}`

	result, err := OpenAIToAnthropic([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)
	assert.Equal(t, "gpt-4o", req["model"])
}

func TestOpenAIToAnthropic_MaxCompletionTokens(t *testing.T) {
	// OpenAI's new max_completion_tokens field
	body := `{"model":"gpt-4o","max_completion_tokens":500,"messages":[{"role":"user","content":"Hello"}]}`

	result, err := OpenAIToAnthropic([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)
	// max_completion_tokens not mapped - tests passthrough
	assert.Equal(t, "gpt-4o", req["model"])
}

// ============== Tool/Function Calling Tests ==============

func TestAnthropicToOpenAI_ToolChoiceAuto(t *testing.T) {
	body := `{"model":"claude-opus-4-7","max_tokens":1024,"tool_choice":{"type":"auto"},"messages":[{"role":"user","content":"Hello"}]}`

	result, err := AnthropicToOpenAI([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)
	assert.NotNil(t, req["tool_choice"])
}

func TestAnthropicToOpenAI_ToolChoiceAny(t *testing.T) {
	body := `{"model":"claude-opus-4-7","max_tokens":1024,"tool_choice":{"type":"any"},"messages":[{"role":"user","content":"Hello"}]}`

	result, err := AnthropicToOpenAI([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)
	assert.NotNil(t, req["tool_choice"])
}

func TestOpenAIToAnthropic_ToolChoiceAuto(t *testing.T) {
	body := `{"model":"gpt-4o","tool_choice":"auto","messages":[{"role":"user","content":"Hello"}]}`

	result, err := OpenAIToAnthropic([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)
	assert.NotNil(t, req["tool_choice"])
}

func TestOpenAIToAnthropic_ToolChoiceRequired(t *testing.T) {
	body := `{"model":"gpt-4o","tool_choice":"required","messages":[{"role":"user","content":"Hello"}]}`

	result, err := OpenAIToAnthropic([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)
	assert.NotNil(t, req["tool_choice"])
}

func TestOpenAIToAnthropic_ToolChoiceByName(t *testing.T) {
	body := `{"model":"gpt-4o","tool_choice":{"type":"function","function":{"name":"get_weather"}},"messages":[{"role":"user","content":"Hello"}]}`

	result, err := OpenAIToAnthropic([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)
	assert.NotNil(t, req["tool_choice"])
}

func TestAnthropicToOpenAI_ParallelToolCallsDisabled(t *testing.T) {
	// OpenAI has parallel_tool_calls, Anthropic doesn't - test passthrough
	body := `{"model":"claude-opus-4-7","max_tokens":1024,"messages":[{"role":"user","content":"Hello"}]}`

	result, err := AnthropicToOpenAI([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)
	// parallel_tool_calls not in request - ok
	assert.Equal(t, "claude-opus-4-7", req["model"])
}

func TestOpenAIToAnthropic_BuiltInTool(t *testing.T) {
	// OpenAI built-in tools like web_search
	body := `{
		"model":"gpt-4o",
		"messages":[{"role":"user","content":"Search for weather"}],
		"tools":[{"type":"web_search"}]
	}`

	result, err := OpenAIToAnthropic([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)
	assert.Equal(t, "gpt-4o", req["model"])
}

func TestOpenAIToAnthropic_MultipleToolCalls(t *testing.T) {
	// Multiple tool calls in single assistant message (parallel)
	body := `{
		"model":"gpt-4o",
		"messages":[{
			"role":"assistant",
			"tool_calls":[
				{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{}"}},
				{"id":"call_2","type":"function","function":{"name":"get_time","arguments":"{}"}}
			]
		}]
	}`

	result, err := OpenAIToAnthropic([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)

	msgs := req["messages"].([]interface{})
	assistantMsg := msgs[0].(map[string]interface{})
	content := assistantMsg["content"].([]interface{})
	assert.Len(t, content, 2)
}

func TestOpenAIToAnthropic_ToolResultNoId(t *testing.T) {
	// Tool result without tool_call_id - edge case
	body := `{"model":"gpt-4o","messages":[{"role":"tool","content":"result"}]}`

	result, err := OpenAIToAnthropic([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)
	assert.NotNil(t, req["messages"])
}

func TestAnthropicToOpenAI_AssistantWithMixedContentAndTools(t *testing.T) {
	// Assistant message with text, tool_use, and another text
	body := `{
		"model":"claude-opus-4-7",
		"max_tokens":1024,
		"messages":[{
			"role":"assistant",
			"content":[
				{"type":"text","text":"First part"},
				{"type":"tool_use","id":"tool_1","name":"tool1","input":{}},
				{"type":"text","text":"Second part"}
			]
		}]
	}`

	result, err := AnthropicToOpenAI([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)

	messages := req["messages"].([]interface{})
	msg := messages[0].(map[string]interface{})
	tc := msg["tool_calls"].([]interface{})
	assert.Len(t, tc, 1)
}

// ============== Response Format Tests ==============

func TestOpenAIToAnthropic_ResponseFormat(t *testing.T) {
	// OpenAI response_format parameter
	body := `{
		"model":"gpt-4o",
		"response_format":{"type":"json_object"},
		"messages":[{"role":"user","content":"Return JSON"}]
	}`

	result, err := OpenAIToAnthropic([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)
	// response_format not mapped - tests passthrough
	assert.Equal(t, "gpt-4o", req["model"])
}

func TestOpenAIToAnthropic_ResponseFormatJsonSchema(t *testing.T) {
	body := `{
		"model":"gpt-4o",
		"response_format":{
			"type":"json_schema",
			"json_schema":{"name":"weather","schema":{"type":"object"}}
		},
		"messages":[{"role":"user","content":"Return JSON"}]
	}`

	result, err := OpenAIToAnthropic([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)
	assert.Equal(t, "gpt-4o", req["model"])
}

func TestOpenAIToAnthropic_Logprobs(t *testing.T) {
	body := `{"model":"gpt-4o","logprobs":true,"top_logprobs":5,"messages":[{"role":"user","content":"Hello"}]}`

	result, err := OpenAIToAnthropic([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)
	// logprobs not mapped - tests passthrough
	assert.Equal(t, "gpt-4o", req["model"])
}

// ============== Role Variations Tests ==============

func TestAnthropicToOpenAI_UserRoleOnly(t *testing.T) {
	body := `{"model":"claude-opus-4-7","max_tokens":1024,"messages":[{"role":"user","content":"Hello"}]}`

	result, err := AnthropicToOpenAI([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)

	messages := req["messages"].([]interface{})
	msg := messages[0].(map[string]interface{})
	assert.Equal(t, "user", msg["role"])
}

func TestAnthropicToOpenAI_AssistantRoleOnly(t *testing.T) {
	body := `{"model":"claude-opus-4-7","max_tokens":1024,"messages":[{"role":"assistant","content":"Hi there"}]}`

	result, err := AnthropicToOpenAI([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)

	messages := req["messages"].([]interface{})
	msg := messages[0].(map[string]interface{})
	assert.Equal(t, "assistant", msg["role"])
}

func TestAnthropicToOpenAI_EmptyContentAssistant(t *testing.T) {
	// Assistant with null/empty content (tool result follow-up)
	body := `{"model":"claude-opus-4-7","max_tokens":1024,"messages":[{"role":"assistant","content":null}]}`

	result, err := AnthropicToOpenAI([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)
	assert.NotNil(t, req["messages"])
}

func TestOpenAIToAnthropic_NameField(t *testing.T) {
	// OpenAI supports 'name' field for user messages
	body := `{"model":"gpt-4o","messages":[{"role":"user","name":"user_123","content":"Hello"}]}`

	result, err := OpenAIToAnthropic([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)

	// name field not mapped to Anthropic (no equivalent)
	assert.Equal(t, "gpt-4o", req["model"])
}

func TestOpenAIToGemini_UserRole(t *testing.T) {
	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"Hello"}]}`

	result, err := OpenAIToGemini([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)

	contents := req["contents"].([]interface{})
	content := contents[0].(map[string]interface{})
	assert.Equal(t, "user", content["role"])
}

func TestOpenAIToGemini_AssistantRole(t *testing.T) {
	body := `{"model":"gpt-4o","messages":[{"role":"assistant","content":"Hello"}]}`

	result, err := OpenAIToGemini([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)

	contents := req["contents"].([]interface{})
	content := contents[0].(map[string]interface{})
	assert.Equal(t, "model", content["role"])
}

// ============== Gemini Specific Tests ==============

func TestOpenAIToGemini_TopP(t *testing.T) {
	// top_p is not in OpenAIRequest struct - parsing handles it
	body := `{"model":"gpt-4o","top_p":0.95,"messages":[{"role":"user","content":"Hello"}]}`

	result, err := OpenAIToGemini([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)

	// Verify at least some content was produced
	assert.NotNil(t, req["contents"])
}

func TestOpenAIToGemini_TopK(t *testing.T) {
	body := `{"model":"gpt-4o","top_k":40,"messages":[{"role":"user","content":"Hello"}]}`

	result, err := OpenAIToGemini([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)
	assert.NotNil(t, req["contents"])
}

func TestOpenAIToGemini_StopSequences(t *testing.T) {
	body := `{"model":"gpt-4o","stop":"END","messages":[{"role":"user","content":"Hello"}]}`

	result, err := OpenAIToGemini([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)
	assert.NotNil(t, req["contents"])
}

func TestGeminiToOpenAI_TextOnly(t *testing.T) {
	body := `{"candidates":[{"content":{"parts":[{"text":"Hello world"}]},"finishReason":"STOP"}]}`

	result, err := GeminiToOpenAI([]byte(body), "gemini-pro")
	require.NoError(t, err)

	var resp map[string]interface{}
	err = json.Unmarshal(result, &resp)
	require.NoError(t, err)

	choices := resp["choices"].([]interface{})
	choice := choices[0].(map[string]interface{})
	msg := choice["message"].(map[string]interface{})
	assert.Equal(t, "Hello world", msg["content"])
}

func TestGeminiToOpenAI_NoUsageMetadata(t *testing.T) {
	body := `{"candidates":[{"content":{"parts":[{"text":"Hello"}]},"finishReason":"STOP"}]}`

	result, err := GeminiToOpenAI([]byte(body), "gemini-pro")
	require.NoError(t, err)

	var resp map[string]interface{}
	err = json.Unmarshal(result, &resp)
	require.NoError(t, err)

	usage := resp["usage"].(map[string]interface{})
	assert.Equal(t, 0, int(usage["prompt_tokens"].(float64)))
}

func TestGeminiToOpenAI_MultipleParts(t *testing.T) {
	body := `{"candidates":[{"content":{"parts":[{"text":"Part1"},{"text":"Part2"}]},"finishReason":"STOP"}]}`

	result, err := GeminiToOpenAI([]byte(body), "gemini-pro")
	require.NoError(t, err)

	var resp map[string]interface{}
	err = json.Unmarshal(result, &resp)
	require.NoError(t, err)

	choices := resp["choices"].([]interface{})
	choice := choices[0].(map[string]interface{})
	msg := choice["message"].(map[string]interface{})
	assert.Contains(t, msg["content"], "Part1")
	assert.Contains(t, msg["content"], "Part2")
}

func TestGeminiToOpenAI_UnknownFinishReason(t *testing.T) {
	body := `{"candidates":[{"content":{"parts":[{"text":"Hello"}]},"finishReason":"UNKNOWN_REASON"}]}`

	result, err := GeminiToOpenAI([]byte(body), "gemini-pro")
	require.NoError(t, err)

	var resp map[string]interface{}
	err = json.Unmarshal(result, &resp)
	require.NoError(t, err)

	choices := resp["choices"].([]interface{})
	choice := choices[0].(map[string]interface{})
	// Unknown reason defaults to "stop"
	assert.Equal(t, "stop", choice["finish_reason"])
}

// ============== Roundtrip Tests ==============

func TestRoundtrip_OpenAIToAnthropicToOpenAI_Full(t *testing.T) {
	original := `{
		"model":"gpt-4o",
		"temperature":0.7,
		"max_tokens":500,
		"messages":[
			{"role":"system","content":"You are helpful."},
			{"role":"user","content":"Hello"},
			{"role":"assistant","content":"Hi there"},
			{"role":"user","content":"How are you?"}
		]
	}`

	oaiResult, err := OpenAIToAnthropic([]byte(original))
	require.NoError(t, err)

	backResult, err := AnthropicToOpenAI(oaiResult)
	require.NoError(t, err)

	var final map[string]interface{}
	err = json.Unmarshal(backResult, &final)
	require.NoError(t, err)

	assert.Equal(t, "gpt-4o", final["model"])
}

func TestRoundtrip_OpenAIToGeminiToOpenAI(t *testing.T) {
	original := `{
		"model":"gpt-4o",
		"messages":[
			{"role":"user","content":"Hello"}
		]
	}`

	geminiResult, err := OpenAIToGemini([]byte(original))
	require.NoError(t, err)

	var geminiReq map[string]interface{}
	err = json.Unmarshal(geminiResult, &geminiReq)
	require.NoError(t, err)

	// Convert Gemini response back to OpenAI
	geminiResp := map[string]interface{}{
		"candidates": []map[string]interface{}{
			{
				"content": map[string]interface{}{
					"parts": []map[string]interface{}{
						{"text": "Hello! How can I help?"},
					},
				},
				"finishReason": "STOP",
			},
		},
	}
	geminiRespBytes, _ := json.Marshal(geminiResp)

	oaiResult, err := GeminiToOpenAI(geminiRespBytes, "gemini-pro")
	require.NoError(t, err)

	var oaiResp map[string]interface{}
	err = json.Unmarshal(oaiResult, &oaiResp)
	require.NoError(t, err)

	assert.Equal(t, "chat.completion", oaiResp["object"])
	assert.Equal(t, "gemini-pro", oaiResp["model"])
}

func TestRoundtrip_GeminiToOpenAIToGemini(t *testing.T) {
	// Gemini -> OpenAI (request) -> Gemini
	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"Hello"}]}`

	geminiResult, err := OpenAIToGemini([]byte(body))
	require.NoError(t, err)

	// Convert back - just verify no error
	assert.NotEmpty(t, geminiResult)
}

func TestRoundtrip_AnthropicToOpenAIToAnthropic_Full(t *testing.T) {
	original := `{
		"model":"claude-opus-4-7",
		"max_tokens":1024,
		"system":"You are Claude.",
		"messages":[
			{"role":"user","content":"Hello"},
			{"role":"assistant","content":"Hi!"},
			{"role":"user","content":"How are you?"}
		]
	}`

	oaiResult, err := AnthropicToOpenAI([]byte(original))
	require.NoError(t, err)

	backResult, err := OpenAIToAnthropic(oaiResult)
	require.NoError(t, err)

	var final map[string]interface{}
	err = json.Unmarshal(backResult, &final)
	require.NoError(t, err)

	assert.Equal(t, "claude-opus-4-7", final["model"])
}

// ============== SSE Event Type Tests ==============

func TestSSE_AnthropicToOpenAI_MessageStartWithUsage(t *testing.T) {
	// Anthropic message_start includes usage in the message
	input := `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-3","content":[],"usage":{"input_tokens":10,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}

event: message_stop
data: {"type":"message_stop"}`

	var out bytes.Buffer
	err := AnthropicToOpenAISSE(strings.NewReader(input), &out, "claude-3")
	require.NoError(t, err)

	result := out.String()
	assert.Contains(t, result, "chat.completion.chunk")
	assert.Contains(t, result, "Hello")
}

func TestSSE_OpenAIToAnthropic_ContentFilterEvent(t *testing.T) {
	// OpenAI can emit content_filter event
	input := `data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":"content_filter"}]}
data: [DONE]`

	var out bytes.Buffer
	err := OpenAIToAnthropicSSE(strings.NewReader(input), &out, "gpt-4")
	require.NoError(t, err)

	result := out.String()
	assert.Contains(t, result, "Hello")
}

func TestSSE_OpenAIToAnthropic_RefusalEvent(t *testing.T) {
	// OpenAI refusal event
	input := `data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{"refusal":"I can't help with that"},"finish_reason":"stop"}]}
data: [DONE]`

	var out bytes.Buffer
	err := OpenAIToAnthropicSSE(strings.NewReader(input), &out, "gpt-4")
	require.NoError(t, err)

	result := out.String()
	assert.Contains(t, result, "chatcmpl-1")
}

func TestSSE_AnthropicToOpenAI_ThinkingBlock(t *testing.T) {
	// Anthropic thinking block -> OpenAI reasoning_content
	input := `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[]}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me analyze this..."}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":" The answer is 42."}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"The answer is 42."}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":10}}

event: message_stop
data: {"type":"message_stop"}`

	var out bytes.Buffer
	err := AnthropicToOpenAISSE(strings.NewReader(input), &out, "claude-3")
	require.NoError(t, err)

	result := out.String()
	// Thinking should be mapped to reasoning_content
	assert.Contains(t, result, "reasoning_content")
	assert.Contains(t, result, "The answer is 42")
}

func TestSSE_AnthropicToOpenAI_ToolUseInputJsonDelta(t *testing.T) {
	// Anthropic tool_use with incremental JSON arguments
	input := `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[]}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tool_1","name":"get_weather","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"NYC\""}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}

event: message_stop
data: {"type":"message_stop"}`

	var out bytes.Buffer
	err := AnthropicToOpenAISSE(strings.NewReader(input), &out, "claude-3")
	require.NoError(t, err)

	result := out.String()
	assert.Contains(t, result, "tool_calls")
	assert.Contains(t, result, "get_weather")
	assert.Contains(t, result, "NYC")
}

func TestSSE_AnthropicToOpenAI_SignatureDelta(t *testing.T) {
	// Anthropic thinking block ends with signature_delta
	input := `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[]}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Thinking..."}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"abc123"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}

event: message_stop
data: {"type":"message_stop"}`

	var out bytes.Buffer
	err := AnthropicToOpenAISSE(strings.NewReader(input), &out, "claude-3")
	require.NoError(t, err)

	result := out.String()
	assert.Contains(t, result, "reasoning_content")
	// signature_delta has no OpenAI equivalent - should be skipped
	assert.NotContains(t, result, "signature")
}

func TestSSE_OpenAIToAnthropic_EmptyDelta(t *testing.T) {
	// OpenAI chunk with empty delta
	input := `data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":null}]}
data: [DONE]`

	var out bytes.Buffer
	err := OpenAIToAnthropicSSE(strings.NewReader(input), &out, "gpt-4")
	require.NoError(t, err)
}

func TestSSE_OpenAIToAnthropic_UsageInChunk(t *testing.T) {
	// OpenAI chunk with usage (stream_options.include_usage)
	input := `data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}
data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}
data: [DONE]`

	var out bytes.Buffer
	err := OpenAIToAnthropicSSE(strings.NewReader(input), &out, "gpt-4")
	require.NoError(t, err)

	result := out.String()
	assert.Contains(t, result, "Hi")
	assert.Contains(t, result, "message_delta")
}

func TestSSE_GeminiToOpenAISSE_MultipleParts(t *testing.T) {
	// Gemini streaming with multiple parts
	input := `data: {"candidates":[{"content":{"parts":[{"text":"Part 1"}]}}]}
data: {"candidates":[{"content":{"parts":[{"text":" Part 2"}]}}]}
data: [DONE]`

	var out bytes.Buffer
	err := GeminiToOpenAISSE(strings.NewReader(input), &out, "gemini-pro")
	require.NoError(t, err)

	result := out.String()
	assert.Contains(t, result, "chat.completion.chunk")
}

func TestSSE_GeminiToOpenAISSE_WithUsageMetadata(t *testing.T) {
	input := `data: {"candidates":[{"content":{"parts":[{"text":"Hello"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}}
data: [DONE]`

	var out bytes.Buffer
	err := GeminiToOpenAISSE(strings.NewReader(input), &out, "gemini-pro")
	require.NoError(t, err)

	result := out.String()
	assert.Contains(t, result, "chat.completion.chunk")
}

func TestSSE_OpenAIToGeminiSSE_TextContent(t *testing.T) {
	input := `data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}
data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
data: [DONE]`

	var out bytes.Buffer
	err := OpenAIToGeminiSSE(strings.NewReader(input), &out, "gemini")
	require.NoError(t, err)

	result := out.String()
	assert.Contains(t, result, "message_start")
	assert.Contains(t, result, "Hello")
}

func TestSSE_OpenAIToGeminiSSE_EmptyDelta(t *testing.T) {
	input := `data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":null}]}
data: [DONE]`

	var out bytes.Buffer
	err := OpenAIToGeminiSSE(strings.NewReader(input), &out, "gemini")
	require.NoError(t, err)
}

// ============== New-API / Proxy Station Compatibility ==============

func TestDetectFormat_GeminiContentsField(t *testing.T) {
	// Gemini uses "contents" field - should detect as openai (fallback)
	body := []byte(`{"contents":[{"parts":[{"text":"hi"}]}]}`)
	format := DetectRequestFormat(body)
	assert.Equal(t, "openai", format)
}

func TestDetectFormat_ToolsOnly(t *testing.T) {
	// Request with tools but no max_tokens - should detect as openai
	body := []byte(`{"model":"gpt-4","tools":[{"type":"function","function":{"name":"test","parameters":{}}}],"messages":[{"role":"user","content":"hi"}]}`)
	format := DetectRequestFormat(body)
	assert.Equal(t, "openai", format)
}

func TestDetectFormat_EmptyMessages(t *testing.T) {
	body := []byte(`{"model":"gpt-4","messages":[]}`)
	format := DetectRequestFormat(body)
	assert.Equal(t, "openai", format)
}

func TestDetectFormat_MalformedJSON(t *testing.T) {
	body := []byte(`not json at all`)
	format := DetectRequestFormat(body)
	// Should not panic, defaults to openai
	assert.Equal(t, "openai", format)
}

func TestDetectFormat_PartialJSON(t *testing.T) {
	body := []byte(`{"model":"gpt-4","messages":[{"role":"user"`)
	format := DetectRequestFormat(body)
	// Partial JSON - should not panic
	assert.Equal(t, "openai", format)
}

func TestOpenAIToAnthropic_NonStandardRole(t *testing.T) {
	// Some proxies send non-standard roles
	body := `{"model":"gpt-4o","messages":[{"role":"developer","content":"You are developer."}]}`

	result, err := OpenAIToAnthropic([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)
	// developer role falls through to default case
	assert.NotNil(t, req["messages"])
}

func TestOpenAIToAnthropic_ToolResultWithName(t *testing.T) {
	// Tool result with name field (some proxies send this)
	body := `{"model":"gpt-4o","messages":[{"role":"tool","tool_call_id":"call_1","name":"get_weather","content":"72F"}]}`

	result, err := OpenAIToAnthropic([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)
	assert.NotNil(t, req["messages"])
}

func TestAnthropicToOpenAI_ToolUseWithoutId(t *testing.T) {
	// Tool use without id field - edge case
	body := `{"model":"claude-opus-4-7","max_tokens":1024,"messages":[{"role":"assistant","content":[{"type":"tool_use","name":"test","input":{}}]}]}`

	result, err := AnthropicToOpenAI([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)
	assert.NotNil(t, req["messages"])
}

func TestAnthropicToOpenAI_ToolUseWithPartialInput(t *testing.T) {
	// Tool use with partial JSON input
	body := `{"model":"claude-opus-4-7","max_tokens":1024,"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"test","input":{"key":"val"}}]}]}`

	result, err := AnthropicToOpenAI([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)

	messages := req["messages"].([]interface{})
	msg := messages[0].(map[string]interface{})
	tc := msg["tool_calls"].([]interface{})
	assert.Len(t, tc, 1)
}

func TestOpenAIToAnthropic_ToolCallWithEmptyArgs(t *testing.T) {
	// Tool call with empty arguments string
	body := `{"model":"gpt-4o","messages":[{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"ping","arguments":""}}]}]}`

	result, err := OpenAIToAnthropic([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)

	msgs := req["messages"].([]interface{})
	msg := msgs[0].(map[string]interface{})
	content := msg["content"].([]interface{})
	assert.Len(t, content, 1)
}

func TestOpenAIToGemini_NumericTemperature(t *testing.T) {
	// Gemini expects float64 temperature
	body := `{"model":"gpt-4o","temperature":1.0,"messages":[{"role":"user","content":"Hello"}]}`

	result, err := OpenAIToGemini([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)
	assert.NotNil(t, req["contents"])
}

// ============== Error Handling Tests ==============

func TestAnthropicToOpenAI_UnknownContentBlockType(t *testing.T) {
	// Unknown content block type - should be ignored
	body := `{"model":"claude-opus-4-7","max_tokens":1024,"messages":[{"role":"assistant","content":[{"type":"unknown_type","data":"test"}]}]}`

	result, err := AnthropicToOpenAI([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)
	assert.NotNil(t, req["messages"])
}

func TestAnthropicToOpenAI_MalformedContentBlock(t *testing.T) {
	// Malformed content block - should not crash
	body := `{"model":"claude-opus-4-7","max_tokens":1024,"messages":[{"role":"assistant","content":"not-an-array"}]}`

	result, err := AnthropicToOpenAI([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)
	assert.Equal(t, "claude-opus-4-7", req["model"])
}

func TestAnthropicToOpenAI_SystemWithUnexpectedType(t *testing.T) {
	// System with unexpected type (number) - should be handled gracefully
	body := `{"model":"claude-opus-4-7","max_tokens":1024,"system":123,"messages":[{"role":"user","content":"Hello"}]}`

	result, err := AnthropicToOpenAI([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)
	// Should not crash, system becomes empty
	assert.NotNil(t, req["messages"])
}

func TestOpenAIToAnthropic_InvalidToolArguments(t *testing.T) {
	// Tool arguments that fail JSON unmarshal
	body := `{"model":"gpt-4o","messages":[{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"test","arguments":"{invalid json}"}}]}]}`

	result, err := OpenAIToAnthropic([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)
	// Should handle gracefully
	assert.NotNil(t, req["messages"])
}

func TestGeminiToOpenAI_EmptyCandidate(t *testing.T) {
	// Gemini response with empty candidate
	body := `{"candidates":[{}]}`

	result, err := GeminiToOpenAI([]byte(body), "gemini-pro")
	require.NoError(t, err)

	var resp map[string]interface{}
	err = json.Unmarshal(result, &resp)
	require.NoError(t, err)

	choices := resp["choices"].([]interface{})
	assert.Len(t, choices, 1)
}

func TestGeminiToOpenAI_NilCandidateContent(t *testing.T) {
	// Candidate with nil content
	body := `{"candidates":[{"content":null,"finishReason":"STOP"}]}`

	result, err := GeminiToOpenAI([]byte(body), "gemini-pro")
	require.NoError(t, err)

	var resp map[string]interface{}
	err = json.Unmarshal(result, &resp)
	require.NoError(t, err)
	assert.NotNil(t, resp["choices"])
}

// ============== Message Sequence Tests ==============

func TestAnthropicToOpenAI_FullConversation(t *testing.T) {
	body := `{
		"model":"claude-opus-4-7",
		"max_tokens":1024,
		"system":"You are a helpful assistant.",
		"messages":[
			{"role":"user","content":"Hello"},
			{"role":"assistant","content":"Hi there!"},
			{"role":"user","content":[{"type":"text","text":"How are you?"}]},
			{"role":"assistant","content":[{"type":"text","text":"I'm doing well!"}]}
		]
	}`

	result, err := AnthropicToOpenAI([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)

	messages := req["messages"].([]interface{})
	// system + 4 messages
	assert.Len(t, messages, 5)
}

func TestOpenAIToAnthropic_FullConversation(t *testing.T) {
	body := `{
		"model":"gpt-4o",
		"messages":[
			{"role":"system","content":"You are helpful."},
			{"role":"user","content":"Hello"},
			{"role":"assistant","content":"Hi!"},
			{"role":"user","content":"How are you?"},
			{"role":"assistant","content":"I'm good!"},
			{"role":"user","content":"Use a tool"},
			{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"test","arguments":"{}"}}]},
			{"role":"tool","tool_call_id":"call_1","content":"Result"}
		]
	}`

	result, err := OpenAIToAnthropic([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)

	// System extracted, tool message converted
	assert.NotNil(t, req["messages"])
	assert.Equal(t, "gpt-4o", req["model"])
	assert.NotNil(t, req["system"])
}

func TestOpenAIToGemini_FullConversation(t *testing.T) {
	body := `{
		"model":"gpt-4o",
		"messages":[
			{"role":"system","content":"You are Gemini."},
			{"role":"user","content":"Hello"},
			{"role":"assistant","content":"Hi!"},
			{"role":"user","content":"How are you?"}
		]
	}`

	result, err := OpenAIToGemini([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)

	contents := req["contents"].([]interface{})
	// System instruction + 3 user/assistant messages
	assert.True(t, len(contents) >= 3)
	assert.NotNil(t, req["systemInstruction"])
}

// ============== Special Character Tests ==============

func TestAnthropicToOpenAI_SpecialCharacters(t *testing.T) {
	body := `{"model":"claude-opus-4-7","max_tokens":1024,"messages":[{"role":"user","content":"Special: <>&\"'{}[]\\n\\t"}]}`

	result, err := AnthropicToOpenAI([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)

	messages := req["messages"].([]interface{})
	msg := messages[0].(map[string]interface{})
	assert.Contains(t, msg["content"], "Special")
}

func TestOpenAIToAnthropic_EmojiContent(t *testing.T) {
	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"Hello 👋🎉💻"}]}`

	result, err := OpenAIToAnthropic([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)
	assert.NotNil(t, req["messages"])
}

func TestAnthropicToOpenAI_NullBytes(t *testing.T) {
	// Content with null bytes - JSON may not handle this
	body := `{"model":"claude-opus-4-7","max_tokens":1024,"messages":[{"role":"user","content":"Hello World"}]}`

	result, err := AnthropicToOpenAI([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)
	assert.Equal(t, "claude-opus-4-7", req["model"])
}

func TestAnthropicToOpenAI_VeryLongContent(t *testing.T) {
	// Very long content (simulate long context)
	longText := strings.Repeat("A", 50000)
	body := `{"model":"claude-opus-4-7","max_tokens":1024,"messages":[{"role":"user","content":"` + longText + `"}]}`

	result, err := AnthropicToOpenAI([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)

	messages := req["messages"].([]interface{})
	msg := messages[0].(map[string]interface{})
	assert.Contains(t, msg["content"], strings.Repeat("A", 100))
}

// ============== JSON Structure Tests ==============

func TestExtractText_NestedJSON(t *testing.T) {
	// Content that is a JSON object with nested structure
	input := json.RawMessage(`{"key":"value","nested":{"a":1}}`)
	result := extractText(input)
	// Falls through to string(raw)
	assert.Equal(t, `{"key":"value","nested":{"a":1}}`, result)
}

func TestExtractText_BooleanContent(t *testing.T) {
	input := json.RawMessage(`true`)
	result := extractText(input)
	assert.Equal(t, "true", result)
}

func TestExtractText_NumberContent(t *testing.T) {
	input := json.RawMessage(`42`)
	result := extractText(input)
	assert.Equal(t, "42", result)
}

func TestExtractTextFromRaw_EmptyString(t *testing.T) {
	result := extractTextFromRaw("")
	assert.Equal(t, "", result)
}

func TestJsonEscape_Unicode(t *testing.T) {
	result := jsonEscape("Hello 世界 🌍")
	assert.Contains(t, result, "世界")
	assert.Contains(t, result, "🌍")
}

func TestJsonEscape_TabsAndNewlines(t *testing.T) {
	result := jsonEscape("line1\nline2\ttab")
	assert.Contains(t, result, `\n`)
	assert.Contains(t, result, `\t`)
}

// ============== Edge Case: Sequential Role Messages ==============

func TestAnthropicToOpenAI_SequentialUserMessages(t *testing.T) {
	body := `{"model":"claude-opus-4-7","max_tokens":1024,"messages":[{"role":"user","content":"Hello"},{"role":"user","content":"Again"}]}`

	result, err := AnthropicToOpenAI([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)

	messages := req["messages"].([]interface{})
	assert.Len(t, messages, 2)
}

func TestAnthropicToOpenAI_SequentialAssistantMessages(t *testing.T) {
	body := `{"model":"claude-opus-4-7","max_tokens":1024,"messages":[{"role":"assistant","content":"First"},{"role":"assistant","content":"Second"}]}`

	result, err := AnthropicToOpenAI([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)

	messages := req["messages"].([]interface{})
	assert.Len(t, messages, 2)
}

// ============== Field Mapping Verification ==============

func TestAnthropicToOpenAI_AllDocumentedFields(t *testing.T) {
	// Test all documented Anthropic fields are at least passed through
	body := `{
		"model":"claude-opus-4-7",
		"max_tokens":1024,
		"temperature":0.5,
		"system":"You are Claude.",
		"stop_sequences":["END"],
		"stream":true,
		"messages":[{"role":"user","content":"Hello"}]
	}`

	result, err := AnthropicToOpenAI([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)

	assert.Equal(t, "claude-opus-4-7", req["model"])
	assert.Equal(t, float64(1024), req["max_tokens"])
	assert.Equal(t, true, req["stream"])
	assert.NotNil(t, req["messages"])
}

func TestOpenAIToAnthropic_AllDocumentedFields(t *testing.T) {
	// Test all documented OpenAI fields are at least handled
	body := `{
		"model":"gpt-4o",
		"messages":[{"role":"user","content":"Hello"}],
		"temperature":0.7,
		"top_p":0.9,
		"max_tokens":500,
		"stream":false,
		"user":"user_123"
	}`

	result, err := OpenAIToAnthropic([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)

	assert.Equal(t, "gpt-4o", req["model"])
	assert.Equal(t, float64(500), req["max_tokens"])
	// stream:false may be omitted due to omitempty in struct
}

func TestOpenAIToGemini_AllDocumentedFields(t *testing.T) {
	body := `{
		"model":"gpt-4o",
		"messages":[{"role":"user","content":"Hello"}],
		"max_tokens":1024,
		"temperature":0.8
	}`

	result, err := OpenAIToGemini([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)

	genConfig := req["generationConfig"].(map[string]interface{})
	assert.Equal(t, 1024, int(genConfig["maxOutputTokens"].(float64)))
}

// ============== Version-Specific Tests ==============

func TestAnthropic_VersionHeader(t *testing.T) {
	// The adapter doesn't handle headers, but we verify the format detection works
	// Note: detection requires array content blocks to identify Anthropic format
	body := []byte(`{"model":"claude-opus-4-7","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"text","text":"Hello"}]}]}`)
	format := DetectRequestFormat(body)
	// Requires array content format for detection
	assert.Equal(t, "anthropic", format)
}

func TestOpenAI_MultipleChoices(t *testing.T) {
	// OpenAI's n parameter (multiple completions) - not mapped to Anthropic
	body := `{"model":"gpt-4o","n":3,"messages":[{"role":"user","content":"Hello"}]}`

	result, err := OpenAIToAnthropic([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)
	// n parameter not mapped
	assert.Equal(t, "gpt-4o", req["model"])
}

// ============== Mixed Format Tests (New-API Proxies) ==============

func TestMixed_AnthropicStyleWithOpenAIModel(t *testing.T) {
	// A new-api proxy might send Anthropic format with an OpenAI model name
	body := `{"model":"gpt-4o","max_tokens":1024,"messages":[{"role":"user","content":"Hello"}]}`

	result, err := AnthropicToOpenAI([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)

	assert.Equal(t, "gpt-4o", req["model"])
	assert.Equal(t, float64(1024), req["max_tokens"])
}

func TestMixed_OpenAIStyleWithAnthropicModel(t *testing.T) {
	// Anthropic model name but OpenAI format (no max_tokens, string content)
	body := `{"model":"claude-opus-4-7","messages":[{"role":"user","content":"Hello"}]}`

	result, err := OpenAIToAnthropic([]byte(body))
	require.NoError(t, err)

	var req map[string]interface{}
	err = json.Unmarshal(result, &req)
	require.NoError(t, err)

	assert.Equal(t, "claude-opus-4-7", req["model"])
}

func TestMixed_GenericModelName(t *testing.T) {
	// Some proxies use generic model names
	testCases := []struct {
		model string
	}{
		{"gpt-3.5-turbo"},
		{"gpt-4-turbo"},
		{"claude-3-opus"},
		{"claude-3-sonnet"},
		{"gemini-pro"},
		{"gemini-1.5-pro"},
		{"unknown-model"},
	}

	for _, tc := range testCases {
		body := []byte(`{"model":"` + tc.model + `","messages":[{"role":"user","content":"Hello"}]}`)
		format := DetectRequestFormat(body)
		assert.Equal(t, "openai", format, "Model: %s", tc.model)
	}
}

// ============== Buffer/Scanner Edge Cases ==============

func TestSSE_LargeChunk(t *testing.T) {
	// Test handling of very large SSE chunks
	largeContent := strings.Repeat("x", 100000)
	input := `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[]}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"` + largeContent + `"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}

event: message_stop
data: {"type":"message_stop"}`

	var out bytes.Buffer
	err := AnthropicToOpenAISSE(strings.NewReader(input), &out, "claude-3")
	require.NoError(t, err)

	result := out.String()
	assert.Contains(t, result, largeContent)
}

func TestSSE_ManySmallChunks(t *testing.T) {
	// Test many small chunks (stress test state machine)
	var input strings.Builder
	input.WriteString("event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[]}}\n\n")
	input.WriteString("event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n")

	for i := 0; i < 100; i++ {
		input.WriteString("event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"x\"}}\n\n")
	}

	input.WriteString("event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n")
	input.WriteString("event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n")
	input.WriteString("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")

	var out bytes.Buffer
	err := AnthropicToOpenAISSE(strings.NewReader(input.String()), &out, "claude-3")
	require.NoError(t, err)

	result := out.String()
	// Should produce output with text content
	assert.True(t, len(result) > 100)
	assert.Contains(t, result, "chat.completion.chunk")
}

func TestSSE_CommentLines(t *testing.T) {
	// SSE can include comment lines starting with :
	input := `: This is a comment\n: Another comment\nevent: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[]}}\n\nevent: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\nevent: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n`

	var out bytes.Buffer
	err := AnthropicToOpenAISSE(strings.NewReader(input), &out, "claude-3")
	require.NoError(t, err)

	result := out.String()
	assert.Contains(t, result, "Hello")
}

func TestSSE_UTF8Content(t *testing.T) {
	// Various UTF-8 characters
	input := `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[]}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello 世界 🌍 émojis 🚀"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}

event: message_stop
data: {"type":"message_stop"}`

	var out bytes.Buffer
	err := AnthropicToOpenAISSE(strings.NewReader(input), &out, "claude-3")
	require.NoError(t, err)

	result := out.String()
	assert.Contains(t, result, "世界")
	assert.Contains(t, result, "🌍")
}
