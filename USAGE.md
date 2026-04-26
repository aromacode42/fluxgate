# FluxGate 使用指南

FluxGate 是一个生产级的本地 API 网关，用于管理多个 AI 服务商（OpenAI、Anthropic 等）的 API 端点。支持负载均衡、故障转移、自动格式转换、SSE 流式传输、代理路由等功能。

## 目录

- [快速开始](#快速开始)
- [安装与构建](#安装与构建)
- [配置详解](#配置详解)
  - [服务器配置](#服务器配置)
  - [代理配置](#代理配置)
  - [服务商配置](#服务商配置)
  - [模型路由配置](#模型路由配置)
  - [网关配置](#网关配置)
- [API 入口格式](#api-入口格式)
  - [OpenAI 格式](#openai-格式)
  - [Anthropic 格式](#anthropic-格式)
- [查询可用模型](#查询可用模型)
- [入站鉴权](#入站鉴权)
- [请求流程](#请求流程)
- [格式自动转换](#格式自动转换)
- [模型路由与故障转移](#模型路由与故障转移)
- [负载均衡](#负载均衡)
- [速率限制](#速率限制)
- [健康检查](#健康检查)
- [使用示例](#使用示例)
  - [curl 示例](#curl-示例)
  - [Claude Code 对接](#claude-code-对接)
  - [IDE / Cursor / Continue 对接](#ide--cursor--continue-对接)
  - [OpenAI SDK 对接](#openai-sdk-对接)
- [环境变量](#环境变量)
- [配置验证](#配置验证)

---

## 快速开始

```bash
# 1. 克隆项目
git clone <repo-url> && cd FluxGate

# 2. 构建
go build -o fluxgate ./cmd/server

# 3. 复制并编辑配置
cp config.yaml my-config.yaml
# 编辑 my-config.yaml，填入你的 API keys

# 4. 启动
./fluxgate -config my-config.yaml
```

服务默认监听 `http://0.0.0.0:8080`。

---

## 安装与构建

### 前提条件

- Go 1.22+（推荐最新稳定版）
- 如果在中国大陆，设置 Go 代理：

```bash
export GOPROXY=https://goproxy.cn,direct
export GOSUMDB=sum.golang.google.cn
```

### 构建

```bash
go build -o fluxgate ./cmd/server
```

### 运行测试

```bash
# 全部测试
go test ./...

# 详细输出
go test -v ./...

# 运行 e2e 测试
go test -v ./tests/e2e/
```

---

## 配置详解

FluxGate 通过 YAML 配置文件驱动。完整示例见 `config.yaml`。

### 服务器配置

```yaml
server:
  host: "0.0.0.0"       # 监听地址
  port: 8080             # 监听端口
  read_timeout: 30s      # 读超时
  write_timeout: 120s    # 写超时（SSE 流式需要较长超时）
  shutdown_timeout: 10s  # 优雅关闭超时
```

### 代理配置

定义命名代理，供服务商引用。支持 HTTP 和 SOCKS5 两种类型。

```yaml
proxies:
  - name: "us-proxy"           # 代理名称（唯一标识）
    type: "http"               # 类型：http 或 socks5
    address: "http://127.0.0.1:7890"  # HTTP 代理地址

  - name: "jp-socks"
    type: "socks5"
    address: "127.0.0.1:1080"
    username: ""               # 可选认证
    password: ""

  - name: "hk-proxy"
    type: "http"
    address: "http://proxy.hk:8080"
```

**规则：**
- `name` 必须唯一
- `type` 只能是 `http` 或 `socks5`
- `address` 不能为空
- `username` / `password` 为可选认证字段

### 服务商配置

定义上游 AI 服务商的连接信息。

```yaml
providers:
  - name: "openai-us"              # 服务商名称（唯一标识）
    type: "openai"                  # 类型：openai 或 anthropic
    base_url: "https://api.openai.com"  # 上游 API 地址
    api_keys:                       # API 密钥列表（支持轮换）
      - "sk-xxxxxxxxxxxxxxxx"
      - "sk-yyyyyyyyyyyyyyyy"
    proxy: "us-proxy"               # 引用的代理名称（可选）
    models:                         # 该服务商支持的模型列表
      - "gpt-4o"                    # 留空 = 接受任何模型（兜底直连）
      - "gpt-4o-mini"
    weight: 2                       # 负载均衡权重（默认 1）
    disabled: false                 # 是否禁用（默认 false）

  - name: "anthropic-jp"
    type: "anthropic"
    base_url: "https://api.anthropic.com"
    api_keys:
      - "sk-ant-aaaaaaaaaaaaa"
    proxy: "jp-socks"
    models:
      - "claude-sonnet-4-20250514"
    weight: 1
```

**关键字段说明：**

| 字段 | 说明 |
|------|------|
| `name` | 服务商唯一标识，在模型路由中引用 |
| `type` | API 格式类型：`openai` 或 `anthropic` |
| `base_url` | 上游 API 基础地址 |
| `api_keys` | 支持多个密钥，自动轮换，均摊请求量 |
| `proxy` | 引用 `proxies` 中定义的代理名称，留空则直连 |
| `models` | 该服务商支持的模型列表。**留空表示接受任何模型** |
| `weight` | 负载均衡权重。权重 2 的服务商获得的请求数是权重 1 的两倍 |
| `disabled` | 设为 `true` 可临时禁用该服务商 |

### 模型路由配置

定义虚拟模型名到实际后端的映射，支持跨服务商故障转移。

```yaml
models:
  - name: "gpt-5"                    # 对外暴露的虚拟模型名
    backends:
      - "openai-us/gpt-4o"           # 优先级从高到低
      - "openai-backup/gpt-4o"       # 格式：provider-name/real-model
      - "anthropic-jp/claude-sonnet-4-20250514"  # 跨服务商故障转移

  - name: "fast-chat"
    backends:
      - "openai-us/gpt-4o-mini"
      - "openai-backup/gpt-4o-mini"

  - name: "heavy-reasoning"
    backends:
      - "openai-us/o1-pro"
```

**路由规则：**
1. 请求到达时，提取请求中的 `model` 字段
2. 如果匹配 `models` 中的 `name`，按 `backends` 顺序尝试
3. 第一个后端失败，自动切换下一个（指数退避）
4. 所有后端失败才返回错误
5. 如果模型不在路由表中，走直通路径（passthrough），按服务商类型匹配

### 网关配置

```yaml
gateway:
  api_keys:                      # 入站鉴权密钥（留空关闭鉴权）
    - "my-secret-key-1"
    - "my-secret-key-2"
  balancer_type: "round_robin"  # 负载均衡策略：round_robin 或 weighted
  retry_max: 5                  # 后端级别重试次数（每个后端）
  retry_wait_min: 500ms         # 重试最小等待时间
  retry_wait_max: 5s            # 重试最大等待时间
  gateway_retry_max: 0          # 网关级别重试次数（0 = 无限，客户端连接断开才停止）
  health_check: true            # 是否启用健康检查
  health_interval: 30s          # 健康检查间隔
  health_timeout: 5s            # 健康检查超时
  request_timeout: 300s         # 单次请求超时（AI 模型可能响应较慢）
```

**重试机制说明：**
- `retry_max`：后端级别重试，同一后端失败后重试次数
- `gateway_retry_max`：网关级别重试，整个请求失败后重试次数
  - `0` 表示无限重试，只要客户端连接保持就会一直尝试
  - 使用指数退避：2s → 4s → 8s → 16s → 30s（上限）
  - 适合 Claude Code 等需要持续运行的工具，不会因临时故障中断

**速率限制说明：**
速率限制根据 API 密钥总数自动计算，无需手动配置：
- 每秒请求数 (RPS) = `密钥总数 × 5`（最低 100）
- 突发容量 (Burst) = `密钥总数 × 10`（最低 200）

---

## API 入口格式

FluxGate 同时支持 OpenAI 和 Anthropic 两种 API 格式作为入口。

### OpenAI 格式

标准 OpenAI Chat Completions API：

```
POST /v1/chat/completions
```

请求示例：
```json
{
  "model": "gpt-5",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "stream": true
}
```

### Anthropic 格式

标准 Anthropic Messages API：

```
POST /v1/messages
```

或兼容路径：
```
POST /anthropic/v1/messages
```

请求示例：
```json
{
  "model": "gpt-5",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 4096,
  "stream": true
}
```

---

## 查询可用模型

FluxGate 提供兼容 OpenAI 的 `/v1/models` 端点，用于查询当前可用的模型列表：

```bash
curl http://localhost:8080/v1/models \
  -H "Authorization: Bearer your-key"
```

响应格式：

```json
{
  "object": "list",
  "data": [
    {"id": "gpt-5", "object": "model", "created": 0, "owned_by": "fluxgate"},
    {"id": "fast-chat", "object": "model", "created": 0, "owned_by": "fluxgate"},
    {"id": "gpt-4o", "object": "model", "created": 0, "owned_by": "openai-us"},
    {"id": "claude-sonnet-4-20250514", "object": "model", "created": 0, "owned_by": "anthropic-jp"}
  ]
}
```

**说明：**
- `owned_by: "fluxgate"` 表示这是虚拟模型（来自模型路由配置）
- `owned_by: "provider-name"` 表示这是服务商提供的真实模型
- 虚拟模型优先显示，重复的模型名不会重复列出
- 仅支持 GET 请求
- 如果配置了入站鉴权，此端点需要提供有效的 API Key

## 入站鉴权

FluxGate 支持为入站请求配置 API Key 鉴权，防止未授权访问。

### 启用鉴权

在 `config.yaml` 的 `gateway` 部分添加 `api_keys`：

```yaml
gateway:
  api_keys:
    - "my-secret-key-1"
    - "my-secret-key-2"
  balancer_type: "round_robin"
  # ...
```

### 鉴权方式

支持两种标准 API Key 传递方式：

**OpenAI 风格（Bearer Token）：**
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer my-secret-key-1" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-5","messages":[{"role":"user","content":"Hi"}]}'
```

**Anthropic 风格（x-api-key）：**
```bash
curl http://localhost:8080/v1/messages \
  -H "x-api-key: my-secret-key-1" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-5","messages":[{"role":"user","content":"Hi"}]}'
```

### 规则

- `api_keys` 为空或未设置时，鉴权**关闭**（向后兼容）
- `Authorization: Bearer` 优先于 `x-api-key`
- 无效或缺失 API Key 返回 HTTP 401
- `/health` 端点始终无需鉴权
- 支持多个 Key，客户端使用任意一个即可

---

## 请求流程

```
客户端请求
    │
    ▼
┌──────────────────┐
│  检测入口格式     │  根据 URL 路径判断 OpenAI/Anthropic
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  速率限制检查     │  Token Bucket 算法
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  提取模型名称     │  从请求体 JSON 中读取 model 字段
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  模型路由匹配     │  查找虚拟模型 → 后端列表
└──────────────────┘
    │
    ├── 匹配到路由 ──▶ 按优先级遍历后端，故障自动切换
    │
    └── 未匹配 ──────▶ 直通路径：按服务商类型选择
    │
    ▼
┌──────────────────┐
│  格式转换（如需）  │  入口格式 ≠ 后端格式时自动翻译请求体
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  发送请求到后端   │  通过指定代理或直连
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  响应处理         │  SSE 流式翻译（如需）或直接透传
└──────────────────┘
    │
    ▼
  返回客户端
```

---

## 格式自动转换

当客户端使用的 API 格式与后端服务商的格式不同时，FluxGate 自动进行双向转换。

### 转换场景

| 客户端格式 | 后端格式 | 请求转换 | 响应转换 |
|-----------|---------|---------|---------|
| OpenAI | OpenAI | 直接透传 | 直接透传 |
| Anthropic | Anthropic | 直接透传 | 直接透传 |
| **OpenAI** | **Anthropic** | OpenAI → Anthropic | Anthropic SSE → OpenAI SSE |
| **Anthropic** | **OpenAI** | Anthropic → OpenAI | OpenAI SSE → Anthropic SSE |

### 请求体转换

**Anthropic → OpenAI：**
- `system` 顶层字段 → `system` 角色 message
- `messages` 中的 `content` blocks → 字符串拼接
- `tool_use` blocks → `tool_calls` 数组
- `tool_result` 角色 → `tool` 角色
- `max_tokens` → `max_tokens`
- `tools` 格式转换

**OpenAI → Anthropic：**
- `system` 角色 message → 顶层 `system` 字段
- `tool_calls` → `tool_use` content blocks
- `tool` 角色 → `tool_result` content blocks
- `tools` 格式转换

### SSE 响应转换

**OpenAI SSE → Anthropic SSE（Claude Code 关键路径）：**

正确处理以下所有场景：
- `content_block_start` 在 `content_block_delta` 之前发出
- 每个 content block 使用唯一递增的 index
- `reasoning_content` → `thinking` block + `signature_delta`
- `tool_calls` → `tool_use` block + `input_json_delta`
- 正确的 `message_start` / `message_delta` / `message_stop` 事件序列

**Anthropic SSE → OpenAI SSE：**
- `text_delta` → `content` delta
- `thinking_delta` → `reasoning_content` delta
- `input_json_delta` → `tool_calls` function arguments delta
- `stop_reason` 映射：`end_turn` → `stop`，`tool_use` → `tool_calls`

---

## 模型路由与故障转移

### 工作原理

1. 在 `models` 中定义虚拟模型名（如 `gpt-5`）
2. 每个虚拟模型对应一组后端，按优先级排列
3. 第一个可用后端处理请求
4. 如果失败（5xx、429、网络错误），自动切换到下一个后端
5. 每次重试使用指数退避 + 随机抖动

### 跨服务商故障转移

```yaml
models:
  - name: "gpt-5"
    backends:
      - "openai-us/gpt-4o"                        # OpenAI 格式
      - "anthropic-jp/claude-sonnet-4-20250514"    # Anthropic 格式
```

当 `openai-us/gpt-4o` 失败时，自动切换到 `anthropic-jp/claude-sonnet-4-20250514`。FluxGate 会：
1. 将请求体从 OpenAI 格式翻译为 Anthropic 格式
2. 将 SSE 响应从 Anthropic 格式翻译回 OpenAI 格式（因为客户端期望 OpenAI 格式）

### 直通模式

如果请求的模型不在 `models` 路由表中，FluxGate 会走直通路径：
1. 根据入口格式筛选匹配类型的服务商
2. 在匹配的服务商中按负载均衡策略选择一个
3. 如果没有类型匹配的服务商，使用所有可用服务商
4. 无需格式转换，直接透传

---

## 负载均衡

支持两种策略：

### Round Robin（轮询）

```yaml
gateway:
  balancer_type: "round_robin"
```

依次轮流选择服务商，均匀分配请求。

### Weighted（加权）

```yaml
gateway:
  balancer_type: "weighted"
```

根据 `weight` 字段分配请求比例。使用平滑加权轮询算法（Smooth Weighted Round-Robin）。

示例：
- `openai-us` weight=2, `openai-backup` weight=1
- 请求分配比例约为 2:1

---

## 速率限制

基于 Token Bucket 算法，自动根据配置的 API 密钥总数计算限额：

```
RPS   = max(密钥总数 × 5, 100)
Burst = max(密钥总数 × 10, 200)
```

超出限制时返回 HTTP 429：

```json
{
  "error": {
    "message": "rate limit exceeded",
    "type": "gateway_error"
  }
}
```

---

## 健康检查

启用后，FluxGate 定期对每个服务商发送健康检查请求：

```yaml
gateway:
  health_check: true
  health_interval: 30s
  health_timeout: 5s
```

不健康的服务商会被自动跳过，直到恢复。健康检查在服务器关闭时自动停止。

---

## 使用示例

### curl 示例

**OpenAI 格式请求：**

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-key" \
  -d '{
    "model": "gpt-5",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

**Anthropic 格式请求：**

```bash
curl http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk-ant-your-key" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "gpt-5",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 4096,
    "stream": true
  }'
```

**健康检查端点：**

```bash
curl http://localhost:8080/health
# {"status":"ok"}
```

### Claude Code 对接

Claude Code 使用 Anthropic API 格式。配置 FluxGate 后，在 Claude Code 中设置：

```
ANTHROPIC_BASE_URL=http://localhost:8080
```

这样 Claude Code 的所有请求（`/v1/messages`）都会通过 FluxGate 路由。

如果后端是 OpenAI 格式的服务商（如 new-api），FluxGate 会自动：
1. 将 Anthropic 请求体翻译为 OpenAI 格式
2. 将 OpenAI SSE 响应翻译为 Anthropic SSE 格式
3. 正确处理 thinking blocks、tool_use、content_block_start/delta 序列

### IDE / Cursor / Continue 对接

这些工具通常使用 OpenAI 格式。设置 base URL 指向 FluxGate：

```
OPENAI_BASE_URL=http://localhost:8080/v1
```

或配置为：
```
Base URL: http://localhost:8080
API Key: sk-your-key
```

### OpenAI SDK 对接

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-your-key"  # 任意值，FluxGate 会替换为配置的密钥
)

response = client.chat.completions.create(
    model="gpt-5",  # 虚拟模型名
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

---

## 环境变量

如果没有配置文件，FluxGate 可以从环境变量读取基本配置：

| 环境变量 | 说明 |
|---------|------|
| `OPENAI_API_KEY` | OpenAI API 密钥 |
| `OPENAI_BASE_URL` | OpenAI API 地址（默认 `https://api.openai.com`） |
| `ANTHROPIC_API_KEY` | Anthropic API 密钥 |
| `ANTHROPIC_BASE_URL` | Anthropic API 地址（默认 `https://api.anthropic.com`） |

---

## 配置验证

FluxGate 在启动时自动验证配置，包括：

- 端口号范围（1-65535）
- 至少配置一个服务商
- 代理名称唯一性
- 代理类型必须为 `http` 或 `socks5`
- 服务商名称唯一性
- 服务商 `base_url` 非空
- 服务商至少一个 `api_key`
- 服务商引用的代理必须存在
- 模型名称唯一性
- 模型后端格式必须为 `provider/model`
- 模型后端引用的 provider 必须存在

配置无效时，启动会报错并退出。
