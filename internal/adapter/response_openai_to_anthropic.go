package adapter

import (
	"bufio"
	"encoding/json"
	"io"
	"strings"
)

// OpenAIToAnthropicSSE reads OpenAI SSE chunks and writes Anthropic SSE events.
// This fixes all the new-api bugs:
// - Always emits content_block_start before content_block_delta
// - Correct monotonically increasing index per block
// - Proper thinking block handling with signature_delta
// - Proper tool_use with input_json_delta chunking
func OpenAIToAnthropicSSE(src io.Reader, dst io.Writer, model string) error {
	writer := &sseWriter{w: dst, flusher: tryFlusher(dst)}
	scanner := bufio.NewScanner(src)
	// Increase scanner buffer for large chunks
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	state := &oai2antState{
		model:     model,
		index:     -1,
		started:   false,
		writer:    writer,
		blockType: "",
	}

	for scanner.Scan() {
		line := scanner.Text()

		if !strings.HasPrefix(line, "data: ") {
			// Forward non-data lines (comments, empty lines)
			if line != "" {
				writer.writeLine(line)
			}
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			state.finish()
			writer.writeLine("data: [DONE]")
			return nil
		}

		var chunk openAIChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			// Forward unparseable chunks as-is
			writer.writeLine(line)
			continue
		}

		if err := state.processChunk(chunk); err != nil {
			return err
		}
	}

	if !state.finished {
		state.finish()
	}
	return scanner.Err()
}

type oai2antState struct {
	model     string
	msgID     string
	index     int
	started   bool
	finished  bool
	blockType string // current block type: "thinking", "text", "tool_use"
	writer    *sseWriter
	usage     struct {
		inputTokens  int
		outputTokens int
	}
}

type openAIChunk struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Model   string `json:"model"`
	Choices []struct {
		Index        int             `json:"index"`
		Delta        json.RawMessage `json:"delta"`
		FinishReason *string         `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

type openAIDelta struct {
	Role         string          `json:"role,omitempty"`
	Content      *string         `json:"content,omitempty"`
	Reasoning    *string         `json:"reasoning_content,omitempty"`
	ToolCalls    []openAIToolCallDelta `json:"tool_calls,omitempty"`
}

type openAIToolCallDelta struct {
	Index    int    `json:"index"`
	ID       string `json:"id,omitempty"`
	Type     string `json:"type,omitempty"`
	Function struct {
		Name      string `json:"name,omitempty"`
		Arguments string `json:"arguments,omitempty"`
	} `json:"function"`
}

func (s *oai2antState) processChunk(chunk openAIChunk) error {
	if s.msgID == "" {
		s.msgID = chunk.ID
		if s.msgID == "" {
			s.msgID = "msg_" + randomHex(24)
		}
	}
	if chunk.Model != "" {
		s.model = chunk.Model
	}

	if !s.started {
		s.started = true
		s.writer.writeEvent("message_start", map[string]interface{}{
			"type": "message_start",
			"message": map[string]interface{}{
				"id":       s.msgID,
				"type":     "message",
				"role":     "assistant",
				"content":  []interface{}{},
				"model":    s.model,
				"usage":    map[string]interface{}{"input_tokens": 0, "output_tokens": 0},
			},
		})
	}

	for _, choice := range chunk.Choices {
		var delta openAIDelta
		if err := json.Unmarshal(choice.Delta, &delta); err != nil {
			continue
		}

		// Handle role (first chunk)
		if delta.Role == "assistant" && len(delta.ToolCalls) == 0 && delta.Content == nil && delta.Reasoning == nil {
			continue
		}

		// Reasoning content → thinking block
		if delta.Reasoning != nil {
			text := *delta.Reasoning
			if text != "" {
				if s.blockType != "thinking" {
					s.closeCurrentBlock()
					s.index++
					s.blockType = "thinking"
					s.writer.writeEvent("content_block_start", map[string]interface{}{
						"type":          "content_block_start",
						"index":         s.index,
						"content_block": map[string]interface{}{"type": "thinking", "thinking": ""},
					})
				}
				s.writer.writeEvent("content_block_delta", map[string]interface{}{
					"type":  "content_block_delta",
					"index": s.index,
					"delta": map[string]interface{}{
						"type":         "thinking_delta",
						"thinking":     text,
					},
				})
			}
		}

		// Tool calls
		if len(delta.ToolCalls) > 0 {
			for _, tc := range delta.ToolCalls {
				if tc.ID != "" {
					// New tool call
					s.closeCurrentBlock()
					s.index++
					s.blockType = "tool_use"
					s.writer.writeEvent("content_block_start", map[string]interface{}{
						"type":          "content_block_start",
						"index":         s.index,
						"content_block": map[string]interface{}{
							"type":  "tool_use",
							"id":    tc.ID,
							"name":  tc.Function.Name,
							"input": map[string]interface{}{},
						},
					})
				}
				if tc.Function.Arguments != "" {
					s.writer.writeEvent("content_block_delta", map[string]interface{}{
						"type":  "content_block_delta",
						"index": s.index,
						"delta": map[string]interface{}{
							"type":          "input_json_delta",
							"partial_json":  tc.Function.Arguments,
						},
					})
				}
			}
			continue
		}

		// Text content
		if delta.Content != nil {
			text := *delta.Content
			if text != "" {
				if s.blockType != "text" {
					s.closeCurrentBlock()
					s.index++
					s.blockType = "text"
					s.writer.writeEvent("content_block_start", map[string]interface{}{
						"type":          "content_block_start",
						"index":         s.index,
						"content_block": map[string]interface{}{"type": "text", "text": ""},
					})
				}
				s.writer.writeEvent("content_block_delta", map[string]interface{}{
					"type":  "content_block_delta",
					"index": s.index,
					"delta": map[string]interface{}{
						"type":      "text_delta",
						"text":      text,
					},
				})
			}
		}

		// Finish
		if choice.FinishReason != nil {
			s.closeCurrentBlock()

			stopReason := *choice.FinishReason
			antStop := "end_turn"
			switch stopReason {
			case "tool_calls":
				antStop = "tool_use"
			case "stop":
				antStop = "end_turn"
			case "length":
				antStop = "max_tokens"
			}

			s.writer.writeEvent("message_delta", map[string]interface{}{
				"type": "message_delta",
				"delta": map[string]interface{}{
					"stop_reason": antStop,
				},
				"usage": map[string]interface{}{
					"output_tokens": 1,
				},
			})
			s.writer.writeEvent("message_stop", map[string]interface{}{
				"type": "message_stop",
			})
			s.finished = true
		}
	}
	return nil
}

func (s *oai2antState) closeCurrentBlock() {
	if s.blockType != "" && s.index >= 0 {
		// For thinking blocks, add signature placeholder
		if s.blockType == "thinking" {
			s.writer.writeEvent("content_block_delta", map[string]interface{}{
				"type":  "content_block_delta",
				"index": s.index,
				"delta": map[string]interface{}{
					"type":      "signature_delta",
					"signature": "ErUB3tq2",
				},
			})
		}
		s.writer.writeEvent("content_block_stop", map[string]interface{}{
			"type":  "content_block_stop",
			"index": s.index,
		})
		s.blockType = ""
	}
}

func (s *oai2antState) finish() {
	if s.finished {
		return
	}
	s.closeCurrentBlock()
	if s.started {
		s.writer.writeEvent("message_delta", map[string]interface{}{
			"type": "message_delta",
			"delta": map[string]interface{}{
				"stop_reason": "end_turn",
			},
			"usage": map[string]interface{}{
				"output_tokens": 1,
			},
		})
		s.writer.writeEvent("message_stop", map[string]interface{}{
			"type": "message_stop",
		})
	}
	s.finished = true
}

