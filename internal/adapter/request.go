package adapter

import (
	"encoding/json"
	"fmt"
)

// ---------- Anthropic request types ----------

type AnthropicRequest struct {
	Model     string                 `json:"model"`
	Messages  []AnthropicMessage     `json:"messages"`
	MaxTokens int                    `json:"max_tokens"`
	System    interface{}            `json:"system,omitempty"`
	Stream    bool                   `json:"stream,omitempty"`
	Tools     []AnthropicTool        `json:"tools,omitempty"`
	ToolChoice interface{}           `json:"tool_choice,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

type AnthropicMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

type AnthropicTool struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	InputSchema map[string]interface{} `json:"input_schema"`
}

// ---------- OpenAI request types ----------

type OpenAIRequest struct {
	Model       string            `json:"model"`
	Messages    []OpenAIMessage   `json:"messages"`
	Stream      bool              `json:"stream,omitempty"`
	Tools       []OpenAITool      `json:"tools,omitempty"`
	ToolChoice  interface{}       `json:"tool_choice,omitempty"`
	MaxTokens   int               `json:"max_tokens,omitempty"`
	Temperature *float64          `json:"temperature,omitempty"`
}

type OpenAIMessage struct {
	Role         string          `json:"role"`
	Content      json.RawMessage `json:"content"`
	Name         string          `json:"name,omitempty"`
	ToolCalls    []OpenAIToolCall `json:"tool_calls,omitempty"`
	ToolCallID   string          `json:"tool_call_id,omitempty"`
}

type OpenAIToolCall struct {
	ID       string             `json:"id"`
	Type     string             `json:"type"`
	Function OpenAIFunctionCall `json:"function"`
}

type OpenAIFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type OpenAITool struct {
	Type     string              `json:"type"`
	Function OpenAIToolFunction  `json:"function"`
}

type OpenAIToolFunction struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"`
}

// ---------- AnthropicToOpenAI converts an Anthropic request to OpenAI format ----------

func AnthropicToOpenAI(body []byte) ([]byte, error) {
	var req AnthropicRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, fmt.Errorf("parsing anthropic request: %w", err)
	}

	oai := OpenAIRequest{
		Model:    req.Model,
		Stream:   req.Stream,
		MaxTokens: req.MaxTokens,
	}

	// System message
	if req.System != nil {
		sysContent := extractTextFromRaw(req.System)
		if sysContent != "" {
			sysJSON, _ := json.Marshal(sysContent)
			oai.Messages = append(oai.Messages, OpenAIMessage{
				Role:    "system",
				Content: sysJSON,
			})
		}
	}

	// Convert messages
	for _, msg := range req.Messages {
		oaiMsg, err := anthropicMsgToOpenAI(msg)
		if err != nil {
			return nil, err
		}
		oai.Messages = append(oai.Messages, oaiMsg)
	}

	// Convert tools
	for _, tool := range req.Tools {
		oai.Tools = append(oai.Tools, OpenAITool{
			Type: "function",
			Function: OpenAIToolFunction{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  tool.InputSchema,
			},
		})
	}

	if req.ToolChoice != nil {
		oai.ToolChoice = req.ToolChoice
	}

	return json.Marshal(oai)
}

func anthropicMsgToOpenAI(msg AnthropicMessage) (OpenAIMessage, error) {
	oai := OpenAIMessage{Role: msg.Role}

	switch msg.Role {
	case "user":
		oai.Content = msg.Content
	case "assistant":
		// Check if content has tool_use blocks
		var blocks []map[string]interface{}
		if err := json.Unmarshal(msg.Content, &blocks); err == nil {
			hasToolUse := false
			for _, b := range blocks {
				if b["type"] == "tool_use" {
					hasToolUse = true
					break
				}
			}
			if hasToolUse {
				var textParts []string
				var toolCalls []OpenAIToolCall
				for _, b := range blocks {
					switch b["type"] {
					case "text":
						if t, ok := b["text"].(string); ok {
							textParts = append(textParts, t)
						}
					case "tool_use":
						args, _ := json.Marshal(b["input"])
						toolCalls = append(toolCalls, OpenAIToolCall{
							ID:   stringOrNil(b["id"]),
							Type: "function",
							Function: OpenAIFunctionCall{
								Name:      stringOrNil(b["name"]),
								Arguments: string(args),
							},
						})
					}
				}
				oai.ToolCalls = toolCalls
				if len(textParts) > 0 {
					combined := ""
					for i, t := range textParts {
						if i > 0 {
							combined += "\n"
						}
						combined += t
					}
					oai.Content = json.RawMessage(`"` + jsonEscape(combined) + `"`)
				} else {
					oai.Content = json.RawMessage(`null`)
				}
				return oai, nil
			}
		}
		oai.Content = msg.Content

	default:
		oai.Content = msg.Content
	}

	return oai, nil
}

// ---------- OpenAIToAnthropic converts an OpenAI request to Anthropic format ----------

func OpenAIToAnthropic(body []byte) ([]byte, error) {
	var req OpenAIRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, fmt.Errorf("parsing openai request: %w", err)
	}

	ant := AnthropicRequest{
		Model:     req.Model,
		Stream:    req.Stream,
		MaxTokens: req.MaxTokens,
	}

	for _, msg := range req.Messages {
		switch msg.Role {
		case "system":
			ant.System = extractText(msg.Content)
		case "tool":
			// Convert tool result to user message with tool_result content block
			result := extractText(msg.Content)
			content, _ := json.Marshal([]map[string]interface{}{
				{
					"type":       "tool_result",
					"tool_use_id": msg.ToolCallID,
					"content":    result,
				},
			})
			ant.Messages = append(ant.Messages, AnthropicMessage{
				Role:    "user",
				Content: content,
			})
		case "assistant":
			if len(msg.ToolCalls) > 0 {
				var blocks []map[string]interface{}
				text := extractText(msg.Content)
				if text != "" && string(msg.Content) != "null" {
					blocks = append(blocks, map[string]interface{}{
						"type": "text",
						"text": text,
					})
				}
				for _, tc := range msg.ToolCalls {
					var input interface{}
					json.Unmarshal([]byte(tc.Function.Arguments), &input)
					blocks = append(blocks, map[string]interface{}{
						"type":  "tool_use",
						"id":    tc.ID,
						"name":  tc.Function.Name,
						"input": input,
					})
				}
				content, _ := json.Marshal(blocks)
				ant.Messages = append(ant.Messages, AnthropicMessage{
					Role:    "assistant",
					Content: content,
				})
			} else {
				ant.Messages = append(ant.Messages, AnthropicMessage{
					Role:    "assistant",
					Content: msg.Content,
				})
			}
		default:
			ant.Messages = append(ant.Messages, AnthropicMessage{
				Role:    msg.Role,
				Content: msg.Content,
			})
		}
	}

	for _, tool := range req.Tools {
		ant.Tools = append(ant.Tools, AnthropicTool{
			Name:        tool.Function.Name,
			Description: tool.Function.Description,
			InputSchema: tool.Function.Parameters,
		})
	}

	if req.ToolChoice != nil {
		ant.ToolChoice = req.ToolChoice
	}

	return json.Marshal(ant)
}

// ---------- Helpers ----------

func extractTextFromRaw(v interface{}) string {
	if v == nil {
		return ""
	}
	raw, err := json.Marshal(v)
	if err != nil {
		return ""
	}
	return extractText(raw)
}

func extractText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	// Try string
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	// Try array of content blocks
	var blocks []map[string]interface{}
	if json.Unmarshal(raw, &blocks) == nil {
		var parts []string
		for _, b := range blocks {
			if b["type"] == "text" {
				if t, ok := b["text"].(string); ok {
					parts = append(parts, t)
				}
			}
		}
		result := ""
		for i, p := range parts {
			if i > 0 {
				result += "\n"
			}
			result += p
		}
		return result
	}
	return string(raw)
}

func stringOrNil(v interface{}) string {
	if v == nil {
		return ""
	}
	s, ok := v.(string)
	if !ok {
		return fmt.Sprintf("%v", v)
	}
	return s
}

func jsonEscape(s string) string {
	b, _ := json.Marshal(s)
	if len(b) >= 2 {
		return string(b[1 : len(b)-1])
	}
	return s
}

// DetectRequestFormat detects the format of a request body.
// Returns "openai" or "anthropic".
func DetectRequestFormat(body []byte) string {
	var probe struct {
		Messages  json.RawMessage `json:"messages"`
		MaxTokens int             `json:"max_tokens"`
	}
	json.Unmarshal(body, &probe)

	// Anthropic always has max_tokens, OpenAI usually doesn't
	if probe.MaxTokens > 0 {
		// Check message format
		var msgs []json.RawMessage
		if json.Unmarshal(probe.Messages, &msgs) == nil && len(msgs) > 0 {
			var first struct {
				Role    string `json:"role"`
				Content json.RawMessage `json:"content"`
			}
			json.Unmarshal(msgs[0], &first)
			// Anthropic often has array content blocks
			var arr []interface{}
			if json.Unmarshal(first.Content, &arr) == nil {
				return "anthropic"
			}
		}
	}
	return "openai"
}
