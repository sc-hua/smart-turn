# WebSocket API 文档

## 概述

Smart Turn WebSocket 服务器提供实时语音端点检测服务，支持 VAD（语音活动检测）和端点预测。

**服务地址**：`ws://localhost:8765`（默认）

## 连接流程

```
Client                          Server
  │                               │
  ├─── 建立 WebSocket 连接 ────────▶│
  │                               │
  │◀─── ready 消息 ───────────────┤
  │     (包含默认配置)             │
  │                               │
  ├─── config 消息 ────────────────▶│
  │     (客户端配置)               │
  │                               │
  │◀─── config 确认 ──────────────┤
  │                               │
  ├─── 二进制音频帧 ───────────────▶│
  │     (16kHz mono int16)        │
  │                               │
  │◀─── vad 事件 ─────────────────┤
  │◀─── prediction 结果 ──────────┤
  │                               │
```

## 消息类型

### 1. ready（服务端 → 客户端）

连接建立后服务端发送的初始消息。

```json
{
  "type": "ready",
  "message": "send config JSON as first text frame...",
  "defaults": {
    "vad_threshold": 0.5,
    "prediction_threshold": 0.5,
    "min_duration_seconds": 0.0,
    "vad_type": "silero",
    "denoise_type": "none",
    "denoise_mix": 1.0
  },
  "supported_denoise_types": ["dtln", "none", "noisereduce"],
  "session_id": "uuid-string"
}
```

### 2. config（客户端 → 服务端）

客户端配置请求，必须作为第一个文本帧发送。

```json
{
  "vad_type": "silero",
  "vad_threshold": 0.5,
  "prediction_threshold": 0.5,
  "min_duration_seconds": 1.0,
  "denoise_type": "dtln",
  "denoise_mix": 0.8
}
```

**配置字段说明**：

| 字段 | 类型 | 范围 | 默认值 | 说明 |
|------|------|------|--------|------|
| `vad_type` | string | silero/ten/fsmn | silero | VAD 类型 |
| `vad_threshold` | float | 0.0-1.0 | 0.5 | VAD 语音概率阈值 |
| `prediction_threshold` | float | 0.0-1.0 | 0.5 | 端点预测阈值 |
| `min_duration_seconds` | float | 0.0-8.0 | 0.0 | 最短语音段时长 |
| `denoise_type` | string | none/dtln/noisereduce | none | 降噪类型 |
| `denoise_mix` | float | 0.0-1.0 | 1.0 | 降噪混合比例 |

### 3. config（服务端 → 客户端）

配置确认消息。

```json
{
  "type": "config",
  "message": "config applied",
  "config": {
    "vad_threshold": 0.5,
    "prediction_threshold": 0.5,
    "min_duration_seconds": 1.0,
    "vad_type": "silero",
    "denoise_type": "dtln",
    "denoise_mix": 0.8
  },
  "session_id": "uuid-string"
}
```

### 4. audio（客户端 → 服务端）

二进制音频帧。

- **格式**：little-endian int16 PCM
- **采样率**：16000 Hz
- **声道**：单声道（mono）
- **帧大小**：建议 512 samples（32ms）

### 5. vad（服务端 → 客户端）

VAD 状态变化事件。

```json
{
  "type": "vad",
  "speech": true,
  "probability": 0.82,
  "vad_threshold": 0.5,
  "timestamp_ms": 1712345678901,
  "segment_id": 1,
  "session_id": "uuid-string"
}
```

**字段说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `speech` | bool | 当前是否检测到语音 |
| `probability` | float | VAD 语音概率 |
| `segment_id` | int | 当前语音段 ID |

### 6. prediction（服务端 → 客户端）

端点预测结果。

```json
{
  "type": "prediction",
  "segment_id": 1,
  "prediction": 1,
  "probability": 0.9673,
  "duration_seconds": 1.824,
  "inference_ms": 55.88,
  "timestamp_ms": 1712345678901,
  "vad_probability": 0.0168,
  "vad_speech": false,
  "vad_threshold": 0.5,
  "prediction_threshold": 0.5,
  "denoise_type": "dtln",
  "denoise_mix": 0.8,
  "denoise_applied": true,
  "denoise_status": "ok",
  "denoise_ms": 12.34,
  "session_id": "uuid-string"
}
```

**预测字段说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `prediction` | int | 1=完整（端点到达），0=不完整 |
| `probability` | float | Smart Turn 模型置信度 |
| `duration_seconds` | float | 语音段时长 |
| `inference_ms` | float | 总推理耗时（含降噪） |

**降噪字段说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `denoise_type` | string | 使用的降噪类型 |
| `denoise_mix` | float | 降噪混合比例 |
| `denoise_applied` | bool | 是否实际应用了降噪 |
| `denoise_status` | string | ok/skipped/failed |
| `denoise_ms` | float | 降噪处理耗时（可选） |
| `denoise_error` | string | 错误信息（可选） |
| `denoise_skip_reason` | string | 跳过原因（可选） |

### 7. skip（服务端 → 客户端）

语音段过短，跳过预测。

```json
{
  "type": "skip",
  "segment_id": 2,
  "reason": "min_duration_not_met",
  "duration_seconds": 0.21,
  "min_duration_seconds": 1.0,
  "timestamp_ms": 1712345678901,
  "session_id": "uuid-string"
}
```

### 8. reset（双向）

客户端发送 `"reset"` 文本清空 VAD/降噪状态。

```json
{
  "type": "reset"
}
```

### 9. error（服务端 → 客户端）

错误消息。

```json
{
  "type": "error",
  "message": "invalid config: denoise_type must be one of: dtln, none, noisereduce",
  "session_id": "uuid-string"
}
```

## 降噪状态枚举

| 状态 | 说明 |
|------|------|
| `ok` | 降噪成功应用 |
| `skipped` | 降噪被跳过（disabled/mix_zero/empty_audio） |
| `failed` | 降噪处理失败，回退到原始音频 |

## 跳过原因枚举

| 原因 | 说明 |
|------|------|
| `disabled` | denoise_type 为 none |
| `mix_zero` | denoise_mix 为 0 |
| `empty_audio` | 音频为空 |
| `no_denoiser` | 降噪器未初始化 |

## 客户端示例

### Python

```python
import asyncio
import json
import websockets

async def main():
    async with websockets.connect("ws://localhost:8765") as ws:
        # 等待 ready
        ready = await ws.recv()
        print(f"Ready: {ready}")

        # 发送配置
        config = {
            "vad_type": "silero",
            "vad_threshold": 0.5,
            "denoise_type": "dtln",
            "denoise_mix": 0.8,
        }
        await ws.send(json.dumps(config))

        # 等待配置确认
        ack = await ws.recv()
        print(f"Config ACK: {ack}")

        # 发送音频（示例）
        # audio_bytes = read_audio_from_mic()
        # await ws.send(audio_bytes)

        # 接收结果
        async for msg in ws:
            data = json.loads(msg)
            if data["type"] == "prediction":
                print(f"Prediction: {data['prediction']}, prob: {data['probability']}")
                print(f"Denoise: {data['denoise_type']}, applied: {data['denoise_applied']}")

asyncio.run(main())
```

### 命令行客户端

```bash
# 基本使用
python ws_mic_client.py --url ws://localhost:8765

# 使用 DTLN 降噪
python ws_mic_client.py --denoise-type dtln --denoise-mix 0.8

# 使用 noisereduce 降噪
python ws_mic_client.py --denoise-type noisereduce

# 完整参数
python ws_mic_client.py \
  --url ws://localhost:8765 \
  --vad-type silero \
  --vad-threshold 0.5 \
  --prediction-threshold 0.5 \
  --min-duration-seconds 1.0 \
  --denoise-type dtln \
  --denoise-mix 0.8 \
  --debug
```

## 错误处理

### 配置错误

```json
{
  "type": "error",
  "message": "invalid config: vad_threshold must be between 0 and 1"
}
```

### 音频格式错误

```json
{
  "type": "error",
  "message": "audio payload must be int16 PCM"
}
```

### 降噪错误

降噪错误不会导致连接关闭，而是在 prediction 中通过 `denoise_status` 和 `denoise_error` 报告：

```json
{
  "type": "prediction",
  "denoise_applied": false,
  "denoise_status": "failed",
  "denoise_error": "DTLN model_1 not found: onnx_model/dtln/model_1.onnx"
}
```
