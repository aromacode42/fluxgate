package adapter

import (
	"bufio"
	"encoding/json"
	"io"
	"strings"
)

// Gemini to OpenAI SSE translation
// Gemini SSE format: "data: {...json...}"
// OpenAI SSE format: "data: {...chat.completion.chunk...}"
func GeminiToOpenAISSE(src io.Reader, dst io.Writer, model string) error {
	writer := &sseWriter{w: dst, flusher: tryFlusher(dst)}
	scanner := bufio.NewScanner(src)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	state := &gemini2oaiState{
		model:   model,
		index:   -1,
		started: false,
		writer:  writer,
	}

	for scanner.Scan() {
		line := scanner.Text()

		if !strings.HasPrefix(line, "data: ") {
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

		var chunk GeminiStreamingChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			writer.writeLine(line)
			continue
		}

		state.processChunk(chunk)
	}

	if !state.finished {
		state.finish()
	}
	return scanner.Err()
}

type gemini2oaiState struct {
	model    string
	index    int
	started  bool
	finished bool
	writer   *sseWriter
}

func (s *gemini2oaiState) processChunk(chunk GeminiStreamingChunk) {
	if !s.started && len(chunk.Candidates) > 0 {
		s.started = true
		s.index = 0

		// Extract content from first candidate
		content := ""
		hasContent := false
		if len(chunk.Candidates) > 0 && len(chunk.Candidates[0].Content.Parts) > 0 {
			for _, part := range chunk.Candidates[0].Content.Parts {
				if part.Text != "" {
					content += part.Text
					hasContent = true
				}
			}
		}

		// message_start equivalent
		s.writer.writeEvent("chunk", map[string]interface{}{
			"id": "gemini-" + randomHex(16),
			"object": "chat.completion.chunk",
			"model": s.model,
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"delta":  map[string]interface{}{},
				},
			},
		})

		if hasContent {
			s.writer.writeEvent("chunk", map[string]interface{}{
				"id": "gemini-" + randomHex(16),
				"object": "chat.completion.chunk",
				"model": s.model,
				"choices": []map[string]interface{}{
					{
						"index": 0,
						"delta": map[string]interface{}{
							"content": content,
						},
					},
				},
			})
		}
		return
	}

	// Process deltas (incremental content)
	for _, cand := range chunk.Candidates {
		for _, part := range cand.Content.Parts {
			if part.Text != "" {
				s.index++
				s.writer.writeEvent("chunk", map[string]interface{}{
					"id": "gemini-" + randomHex(16),
					"object": "chat.completion.chunk",
					"model": s.model,
					"choices": []map[string]interface{}{
						{
							"index": s.index,
							"delta": map[string]interface{}{
								"content": part.Text,
							},
						},
					},
				})
			}

			// Handle function calls
			if part.Function != nil {
				s.index++
				args, _ := json.Marshal(part.Function.Arguments)
				s.writer.writeEvent("chunk", map[string]interface{}{
					"id": "gemini-" + randomHex(16),
					"object": "chat.completion.chunk",
					"model": s.model,
					"choices": []map[string]interface{}{
						{
							"index": s.index,
							"delta": map[string]interface{}{
								"tool_calls": []map[string]interface{}{
									{
										"index": s.index,
										"id":    "call_" + randomHex(16),
										"type":  "function",
										"function": map[string]interface{}{
											"name":      part.Function.Name,
											"arguments": string(args),
										},
									},
								},
							},
						},
					},
				})
			}
		}

		// Handle finish
		if cand.FinishReason != "" {
			fr := geminiFinishReasonToOpenAI(cand.FinishReason)
			s.writer.writeEvent("chunk", map[string]interface{}{
				"id": "gemini-" + randomHex(16),
				"object": "chat.completion.chunk",
				"model": s.model,
				"choices": []map[string]interface{}{
					{
						"index": 0,
						"delta": map[string]interface{}{},
						"finish_reason": fr,
					},
				},
			})
			s.finished = true
		}
	}

	// Handle usage in streaming
	if chunk.usageMetadata != nil {
		s.writer.writeEvent("chunk", map[string]interface{}{
			"id": "gemini-" + randomHex(16),
			"object": "chat.completion.chunk",
			"model": s.model,
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"delta": map[string]interface{}{},
				},
			},
			"usage": map[string]interface{}{
				"prompt_tokens":     chunk.usageMetadata.PromptTokenCount,
				"completion_tokens": chunk.usageMetadata.CandidatesTokenCount,
				"total_tokens":     chunk.usageMetadata.TotalTokenCount,
			},
		})
	}
}

func (s *gemini2oaiState) finish() {
	if s.finished {
		return
	}
	s.writer.writeEvent("chunk", map[string]interface{}{
		"id": "gemini-" + randomHex(16),
		"object": "chat.completion.chunk",
		"model": s.model,
		"choices": []map[string]interface{}{
			{
				"index": 0,
				"delta": map[string]interface{}{},
				"finish_reason": "stop",
			},
		},
	})
	s.finished = true
}

// OpenAI to Gemini SSE translation
// OpenAI SSE format: "data: {...chat.completion.chunk...}"
// Gemini SSE format: "data: {...}"
func OpenAIToGeminiSSE(src io.Reader, dst io.Writer, model string) error {
	writer := &sseWriter{w: dst, flusher: tryFlusher(dst)}
	scanner := bufio.NewScanner(src)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	state := &oai2gemState{
		model:   model,
		index:   0,
		started: false,
		writer:  writer,
	}

	for scanner.Scan() {
		line := scanner.Text()

		if !strings.HasPrefix(line, "data: ") {
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
			writer.writeLine(line)
			continue
		}

		state.processChunk(chunk)
	}

	if !state.finished {
		state.finish()
	}
	return scanner.Err()
}

type oai2gemState struct {
	model     string
	index     int
	started   bool
	finished  bool
	writer    *sseWriter
	content   strings.Builder
	hasContent bool
}

func (s *oai2gemState) processChunk(chunk openAIChunk) {
	if len(chunk.Choices) == 0 {
		return
	}

	if !s.started {
		s.started = true
		s.index = 0
		// Emit initial Gemini event
		s.writer.writeEvent("gemini", map[string]interface{}{
			"type": "message_start",
		})
	}

	for _, choice := range chunk.Choices {
		var delta openAIDelta
		if err := json.Unmarshal(choice.Delta, &delta); err != nil {
			continue
		}

		// Text content
		if delta.Content != nil && *delta.Content != "" {
			s.hasContent = true
			s.content.WriteString(*delta.Content)
		}

		// Finish
		if choice.FinishReason != nil {
			s.finish()
		}
	}
}

func (s *oai2gemState) finish() {
	if s.finished {
		return
	}

	// Emit content
	if s.hasContent {
		s.writer.writeEvent("gemini", map[string]interface{}{
			"type": "content_block_start",
			"index": 0,
			"block": map[string]interface{}{
				"type": "text",
			},
		})
		s.writer.writeEvent("gemini", map[string]interface{}{
			"type": "content_block_delta",
			"index": 0,
			"delta": map[string]interface{}{
				"text": s.content.String(),
			},
		})
		s.writer.writeEvent("gemini", map[string]interface{}{
			"type": "content_block_stop",
			"index": 0,
		})
	}

	s.writer.writeEvent("gemini", map[string]interface{}{
		"type": "message_end",
	})

	s.finished = true
}
