# FluxGate

Production-grade local API gateway for AI providers with model-level routing, automatic failover, named proxies, and load balancing.

## Features

- **Model routing**: Define virtual models (e.g. "gpt-5") that map to real backends across providers
- **Automatic failover**: If one backend fails, seamlessly switches to the next — API never breaks
- **Named proxies**: HTTP/SOCKS5 proxies assigned per-provider
- **Multi-provider**: OpenAI and Anthropic API formats
- **Load balancing**: Round-robin and weighted algorithms
- **Auto rate limiting**: Computed from API key count
- **Health checks**: Periodic upstream monitoring

## Quick Start

```bash
go build -o fluxgate ./cmd/server
./fluxgate -config config.yaml
```

## Configuration

```yaml
providers:
  - name: "openai-us"
    type: "openai"
    base_url: "https://api.openai.com"
    api_keys: ["sk-key1", "sk-key2"]
    models: ["gpt-4o", "gpt-4o-mini"]
    proxy: "us-proxy"

  - name: "anthropic-jp"
    type: "anthropic"
    base_url: "https://api.anthropic.com"
    api_keys: ["sk-ant-key"]
    models: ["claude-sonnet-4-20250514"]
    proxy: "jp-socks"

models:
  - name: "gpt-5"
    backends:
      - "openai-us/gpt-4o"
      - "anthropic-jp/claude-sonnet-4-20250514"

  - name: "fast-chat"
    backends:
      - "openai-us/gpt-4o-mini"
```

### How it works

1. Client sends `{"model": "gpt-5", ...}` to `/v1/chat/completions`
2. FluxGate looks up "gpt-5" in the models table → backends: `[openai-us/gpt-4o, anthropic-jp/claude-sonnet-4-20250514]`
3. Tries first healthy backend, rewrites model field in body
4. On failure (429/5xx/timeout), automatically tries next backend
5. Only errors when ALL backends are exhausted

### Key concepts

- **providers.models**: Which real models this provider supports. Empty = accepts anything.
- **models**: Virtual model entries with ordered backends. Backends tried in order.
- **models.backends**: Format `"provider-name/real-model"`. Model field is rewritten automatically.
- **Failover**: Cross-provider, cross-type. If OpenAI fails, can fall back to Anthropic.

## API Usage

### OpenAI format

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-5","messages":[{"role":"user","content":"Hello"}]}'
```

### Anthropic format

```bash
curl http://localhost:8080/anthropic/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4-20250514","max_tokens":1024,"messages":[{"role":"user","content":"Hello"}]}'
```

### Health

```bash
curl http://localhost:8080/health
```

## Testing

```bash
# Unit tests (57 tests)
go test -count=1 ./internal/balancer/ ./internal/gateway/ ./internal/limiter/ ./internal/provider/ ./internal/proxy/ ./internal/retry/ ./pkg/config/ ./internal/modelrouter/

# Integration tests
go test -v ./tests/integration_test.go

# E2E tests
go test -v ./tests/e2e/e2e_test.go
```

## Project Structure

```
cmd/server/            # Entry point
internal/
  gateway/             # HTTP routing, model routing, failover
  modelrouter/         # Model routing table, backend resolution
  balancer/            # Load balancing (round-robin, weighted)
  limiter/             # Rate limiting (auto-computed)
  provider/            # Provider registry, health, model support
  proxy/               # Named proxy registry with transport cache
  retry/               # Exponential backoff retry
  server/              # Server bootstrap
pkg/
  config/              # YAML config loading and validation
tests/
  integration_test.go
  e2e/e2e_test.go
```
