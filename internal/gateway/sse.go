package gateway

import (
	"bufio"
	"io"
	"strings"

	"github.com/jacek/fluxgate/internal/adapter"
)

// sseTranslate converts SSE format between OpenAI and Anthropic.
func sseTranslate(src io.Reader, dst io.Writer, entryFormat, backendFormat, model string) {
	switch {
	case entryFormat == "anthropic" && backendFormat == "openai":
		adapter.OpenAIToAnthropicSSE(src, dst, model)
	case entryFormat == "openai" && backendFormat == "anthropic":
		adapter.AnthropicToOpenAISSE(src, dst, model)
	case entryFormat == "gemini" && backendFormat == "openai":
		adapter.GeminiToOpenAISSE(src, dst, model)
	case entryFormat == "openai" && backendFormat == "gemini":
		adapter.OpenAIToGeminiSSE(src, dst, model)
	default:
		// Fallback to passthrough if formats match or are unknown
		ssePassthrough(src, dst)
	}
}

// isSSE checks if the content type indicates SSE streaming.
func isSSE(contentType string) bool {
	return strings.Contains(strings.ToLower(contentType), "text/event-stream")
}

// ssePassthrough copies SSE data line-by-line from dst to src with immediate flush.
// Preserves SSE event boundaries (double newlines between events).
func ssePassthrough(src io.Reader, dst io.Writer) error {
	flusher, canFlush := dst.(interface{ Flush() })
	scanner := bufio.NewScanner(src)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	for scanner.Scan() {
		line := scanner.Text()
		if _, err := dst.Write([]byte(line + "\n")); err != nil {
			return err
		}

		// Flush after each SSE event boundary (empty line)
		if line == "" && canFlush {
			flusher.Flush()
		}
	}

	// Final flush for any buffered data
	if canFlush {
		flusher.Flush()
	}

	return scanner.Err()
}

// rawPassthrough copies raw bytes for non-SSE responses.
func rawPassthrough(src io.Reader, dst io.Writer) error {
	_, err := io.Copy(dst, src)
	return err
}
