package adapter

import (
	"bufio"
	"encoding/json"
	"io"
	"strings"
)

// AnthropicToOpenAISSE reads Anthropic SSE events and writes OpenAI SSE chunks.
func AnthropicToOpenAISSE(src io.Reader, dst io.Writer, model string) error {
	writer := &sseWriter{w: dst}
	scanner := bufio.NewScanner(src)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	state := &ant2oaiState{
		model:      model,
		writer:     writer,
		toolCallID: make(map[int]string),
		toolCallFn: make(map[int]string),
	}

	for scanner.Scan() {
		line := scanner.Text()

		if strings.HasPrefix(line, "event: ") {
			state.currentEvent = strings.TrimPrefix(line, "event: ")
			continue
		}

		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				writer.writeLine("data: [DONE]")
				return nil
			}
			if err := state.processEvent(state.currentEvent, data); err != nil {
				// Forward unparseable events
				writer.writeLine(line)
			}
			state.currentEvent = ""
			continue
		}

		// Forward empty lines and comments
		if line != "" {
			writer.writeLine(line)
		}
	}

	if state.started && !state.finished {
		state.emitDone()
	}
	return scanner.Err()
}

type ant2oaiState struct {
	model        string
	msgID        string
	started      bool
	finished     bool
	currentEvent string
	writer       *sseWriter
	toolCallID   map[int]string
	toolCallFn   map[int]string
	usage        struct {
		inputTokens  int
		outputTokens int
	}
}

func (s *ant2oaiState) processEvent(event, data string) error {
	if event == "" || data == "" {
		return nil
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal([]byte(data), &parsed); err != nil {
		return err
	}

	switch event {
	case "message_start":
		s.started = true
		if msg, ok := parsed["message"].(map[string]interface{}); ok {
			if id, ok := msg["id"].(string); ok {
				s.msgID = id
			}
			if m, ok := msg["model"].(string); ok {
				s.model = m
			}
			if usage, ok := msg["usage"].(map[string]interface{}); ok {
				if it, ok := usage["input_tokens"].(float64); ok {
					s.usage.inputTokens = int(it)
				}
			}
		}
		s.emitChunk("", "", nil, nil)

	case "content_block_start":
		block, _ := parsed["content_block"].(map[string]interface{})
		idx, _ := parsed["index"].(float64)
		if block != nil {
			blockType, _ := block["type"].(string)
			switch blockType {
			case "tool_use":
				id, _ := block["id"].(string)
				name, _ := block["name"].(string)
				i := int(idx)
				s.toolCallID[i] = id
				s.toolCallFn[i] = name
			}
		}

	case "content_block_delta":
		idx, _ := parsed["index"].(float64)
		delta, _ := parsed["delta"].(map[string]interface{})
		if delta == nil {
			return nil
		}
		deltaType, _ := delta["type"].(string)

		switch deltaType {
		case "text_delta":
			text, _ := delta["text"].(string)
			s.emitChunk(text, "", nil, nil)

		case "thinking_delta":
			// Skip thinking in OpenAI format (or map to reasoning_content)
			// Some OpenAI-compatible APIs support reasoning_content
			thinking, _ := delta["thinking"].(string)
			s.emitChunk("", thinking, nil, nil)

		case "input_json_delta":
			partial, _ := delta["partial_json"].(string)
			i := int(idx)
			s.emitChunk("", "", nil, &toolCallDelta{
				index:     i,
				id:        s.toolCallID[i],
				name:      s.toolCallFn[i],
				arguments: partial,
			})

		case "signature_delta":
			// Skip, no OpenAI equivalent
		}

	case "content_block_stop":
		// Nothing to emit

	case "message_delta":
		delta, _ := parsed["delta"].(map[string]interface{})
		stopReason := "stop"
		if delta != nil {
			if sr, ok := delta["stop_reason"].(string); ok && sr != "" {
				switch sr {
				case "tool_use":
					stopReason = "tool_calls"
				case "max_tokens":
					stopReason = "length"
				default:
					stopReason = "stop"
				}
			}
		}
		usage, _ := parsed["usage"].(map[string]interface{})
		if usage != nil {
			if ot, ok := usage["output_tokens"].(float64); ok {
				s.usage.outputTokens = int(ot)
			}
		}
		s.emitDoneWithReason(stopReason)

	case "message_stop":
		s.finished = true

	case "ping":
		// Ignore
	}

	return nil
}

type toolCallDelta struct {
	index     int
	id        string
	name      string
	arguments string
}

func (s *ant2oaiState) emitChunk(content, reasoning string, finish *string, toolDelta *toolCallDelta) {
	if !s.started {
		return
	}

	id := s.msgID
	if id == "" {
		id = "chatcmpl-fluxgate"
	}

	choice := map[string]interface{}{
			"index": 0,
			"delta": s.buildDelta(content, reasoning, finish, toolDelta),
		}
		if finish != nil {
			choice["finish_reason"] = *finish
		}

		chunk := map[string]interface{}{
			"id":      id,
			"object":  "chat.completion.chunk",
			"created": 0,
			"model":   s.model,
			"choices": []map[string]interface{}{choice},
		}

	jsonData, _ := json.Marshal(chunk)
	s.writer.writeLine("data: " + string(jsonData))
}

func (s *ant2oaiState) buildDelta(content, reasoning string, finish *string, toolDelta *toolCallDelta) map[string]interface{} {
	delta := map[string]interface{}{}

	if content != "" {
		delta["content"] = content
	}
	if reasoning != "" {
		delta["reasoning_content"] = reasoning
	}
	if toolDelta != nil {
		tc := map[string]interface{}{
			"index": toolDelta.index,
		}
		if toolDelta.id != "" {
			tc["id"] = toolDelta.id
			tc["type"] = "function"
			tc["function"] = map[string]interface{}{
				"name":      toolDelta.name,
				"arguments": toolDelta.arguments,
			}
		} else {
			tc["function"] = map[string]interface{}{
				"arguments": toolDelta.arguments,
			}
		}
		delta["tool_calls"] = []map[string]interface{}{tc}
	}
	if finish != nil {
		delta["content"] = ""
	}

	return delta
}

func (s *ant2oaiState) emitDone() {
	s.emitDoneWithReason("stop")
}

func (s *ant2oaiState) emitDoneWithReason(reason string) {
	s.emitChunk("", "", &reason, nil)
	s.writer.writeLine("data: [DONE]")
	s.finished = true
}
