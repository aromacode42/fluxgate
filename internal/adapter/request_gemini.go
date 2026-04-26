package adapter

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// Gemini API request/response types
type GeminiRequest struct {
	Contents         []GeminiContent `json:"contents"`
	SystemInstruction interface{}     `json:"systemInstruction,omitempty"`
	Tools            []GeminiTool    `json:"tools,omitempty"`
	SafetySettings   []interface{}   `json:"safetySettings,omitempty"`
	GenerationConfig *GeminiGenConfig `json:"generationConfig,omitempty"`
}

type GeminiContent struct {
	Role    string                `json:"role,omitempty"`
	Parts   []GeminiPart          `json:"parts"`
}

type GeminiPart struct {
	Text     string              `json:"text,omitempty"`
	Function *GeminiFunctionCall `json:"functionCall,omitempty"`
}

type GeminiFunctionCall struct {
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"args,omitempty"`
}

type GeminiTool struct {
	FunctionDeclarations []GeminiFunctionDecl `json:"functionDeclarations,omitempty"`
}

type GeminiFunctionDecl struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"`
}

type GeminiGenConfig struct {
	Temperature     *float64 `json:"temperature,omitempty"`
	TopP            *float64 `json:"topP,omitempty"`
	TopK            *int     `json:"topK,omitempty"`
	MaxOutputTokens *int     `json:"maxOutputTokens,omitempty"`
	StopSequences   []string `json:"stopSequences,omitempty"`
}

type GeminiResponse struct {
	Candidates []GeminiCandidate `json:"candidates"`
	UsageMetadata *struct {
		PromptTokenCount     int `json:"promptTokenCount"`
		CandidatesTokenCount int `json:"candidatesTokenCount"`
		TotalTokenCount      int `json:"totalTokenCount"`
	} `json:"usageMetadata,omitempty"`
}

type GeminiCandidate struct {
	Content       GeminiContent `json:"content"`
	FinishReason  string        `json:"finishReason,omitempty"`
	AvgLatencyMs  *float64      `json:"avgLatencyMs,omitempty"`
}

type GeminiStreamingChunk struct {
	Candidates []struct {
		Content       GeminiContent `json:"content"`
		FinishReason string        `json:"finishReason,omitempty"`
	} `json:"candidates"`
	usageMetadata *struct {
		PromptTokenCount     int `json:"promptTokenCount"`
		CandidatesTokenCount int `json:"candidatesTokenCount"`
		TotalTokenCount      int `json:"totalTokenCount"`
	} `json:"usageMetadata,omitempty"`
}

// OpenAI to Gemini conversion
func OpenAIToGemini(body []byte) ([]byte, error) {
	var openAIReq OpenAIRequest
	if err := json.Unmarshal(body, &openAIReq); err != nil {
		return nil, fmt.Errorf("parsing OpenAI request: %w", err)
	}

	geminiReq := GeminiRequest{
		GenerationConfig: &GeminiGenConfig{},
	}

	// Convert system message
	for _, msg := range openAIReq.Messages {
		if msg.Role == "system" && msg.Content != nil {
			text := extractText(msg.Content)
			if text != "" {
				geminiReq.SystemInstruction = map[string]interface{}{
					"parts": []GeminiPart{{Text: text}},
				}
			}
		}
	}

	// Convert messages
	for _, msg := range openAIReq.Messages {
		if msg.Role == "system" {
			continue // already handled
		}

		geminiContent := GeminiContent{
			Role:  geminiRole(msg.Role),
			Parts: make([]GeminiPart, 0),
		}

		if msg.Content != nil {
			text := extractText(msg.Content)
			if text != "" {
				geminiContent.Parts = append(geminiContent.Parts, GeminiPart{Text: text})
			}
		}

		// Handle tool calls (OpenAI) → function calls (Gemini)
		if len(msg.ToolCalls) > 0 {
			for _, tc := range msg.ToolCalls {
				args := make(map[string]interface{})
				if tc.Function.Arguments != "" {
					json.Unmarshal([]byte(tc.Function.Arguments), &args)
				}
				geminiContent.Parts = append(geminiContent.Parts, GeminiPart{
					Function: &GeminiFunctionCall{
						Name:      tc.Function.Name,
						Arguments: args,
					},
				})
			}
		}

		// Handle tool result → role: "user" with function response
		if msg.Role == "tool" {
			geminiContent.Role = "user"
			result := extractText(msg.Content)
			// Build function response part
			parts := make([]GeminiPart, 0)
			// Gemini doesn't have a direct tool_result concept, so we use a structured text
			parts = append(parts, GeminiPart{
				Text: fmt.Sprintf("tool_result: %s", result),
			})
			geminiContent.Parts = parts
		}

		geminiReq.Contents = append(geminiReq.Contents, geminiContent)
	}

	// Convert tools
	for _, tool := range openAIReq.Tools {
		decls := make([]GeminiFunctionDecl, 0, len(tool.Function.Parameters))
		decls = append(decls, GeminiFunctionDecl{
			Name:        tool.Function.Name,
			Description: tool.Function.Description,
			Parameters:  tool.Function.Parameters,
		})
		geminiReq.Tools = append(geminiReq.Tools, GeminiTool{FunctionDeclarations: decls})
	}

	// Convert generation config
	if openAIReq.MaxTokens > 0 {
		geminiReq.GenerationConfig.MaxOutputTokens = &openAIReq.MaxTokens
	}
	if openAIReq.Temperature != nil {
		geminiReq.GenerationConfig.Temperature = openAIReq.Temperature
	}

	return json.Marshal(geminiReq)
}

// Gemini to OpenAI conversion
func GeminiToOpenAI(body []byte, model string) ([]byte, error) {
	var geminiResp GeminiResponse
	if err := json.Unmarshal(body, &geminiResp); err != nil {
		return nil, fmt.Errorf("parsing Gemini response: %w", err)
	}

	choices := make([]map[string]interface{}, 0, len(geminiResp.Candidates))
	for i, cand := range geminiResp.Candidates {
		content := ""
		finishReason := ""

		if len(cand.Content.Parts) > 0 {
			for _, part := range cand.Content.Parts {
				if part.Text != "" {
					content += part.Text
				}
				if part.Function != nil {
					// Gemini function call - return as tool call format
					argsBytes, _ := json.Marshal(part.Function.Arguments)
					choices = append(choices, map[string]interface{}{
						"index": i,
						"message": map[string]interface{}{
							"role": "assistant",
							"content": nil,
							"tool_calls": []map[string]interface{}{
								{
									"id":   fmt.Sprintf("call_%d_%s", i, randomHex(16)),
									"type": "function",
									"function": map[string]interface{}{
										"name":      part.Function.Name,
										"arguments": string(argsBytes),
									},
								},
							},
						},
						"finish_reason": "tool_calls",
					})
					continue
				}
			}
		}

		fr := geminiFinishReasonToOpenAI(cand.FinishReason)
		choices = append(choices, map[string]interface{}{
			"index": i,
			"message": map[string]interface{}{
				"role":    "assistant",
				"content": content,
			},
			"finish_reason": fr,
		})
	}

	usage := map[string]interface{}{"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
	if geminiResp.UsageMetadata != nil {
		usage = map[string]interface{}{
			"prompt_tokens":     geminiResp.UsageMetadata.PromptTokenCount,
			"completion_tokens": geminiResp.UsageMetadata.CandidatesTokenCount,
			"total_tokens":      geminiResp.UsageMetadata.TotalTokenCount,
		}
	}

	response := map[string]interface{}{
		"id":      "gemini-" + randomHex(24),
		"object":  "chat.completion",
		"created": 0,
		"model":   model,
		"choices": choices,
		"usage":   usage,
	}

	return json.Marshal(response)
}

func geminiRole(role string) string {
	switch role {
	case "assistant":
		return "model"
	case "user":
		return "user"
	case "system":
		return "system"
	default:
		return "user"
	}
}

func geminiFinishReasonToOpenAI(reason string) string {
	switch reason {
	case "STOP":
		return "stop"
	case "MAX_TOKENS":
		return "length"
	case "SAFETY", "OTHER", "RECITATION":
		return "content_filter"
	default:
		return "stop"
	}
}