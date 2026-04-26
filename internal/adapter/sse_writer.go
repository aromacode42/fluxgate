package adapter

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
)

// httpFlusher is implemented by http.ResponseWriter and similar writers.
type httpFlusher interface {
	Flush()
}

// tryFlusher attempts to cast an io.Writer to httpFlusher.
// Returns nil if the writer doesn't implement Flush().
func tryFlusher(w io.Writer) httpFlusher {
	if f, ok := w.(httpFlusher); ok {
		return f
	}
	return nil
}

// sseWriter wraps an io.Writer for SSE event output with optional flushing.
type sseWriter struct {
	w       io.Writer
	flusher httpFlusher
}

func (w *sseWriter) writeLine(line string) {
	fmt.Fprintf(w.w, "%s\n", line)
	w.flush()
}

func (w *sseWriter) writeEvent(event string, data interface{}) {
	jsonData, err := json.Marshal(data)
	if err != nil {
		return
	}
	fmt.Fprintf(w.w, "event: %s\ndata: %s\n\n", event, string(jsonData))
	w.flush()
}

func (w *sseWriter) flush() {
	if w.flusher != nil {
		w.flusher.Flush()
	}
}

// randomHex generates a cryptographically random hex string.
func randomHex(n int) string {
	b := make([]byte, (n+1)/2)
	rand.Read(b)
	return hex.EncodeToString(b)[:n]
}
