# Smart Turn WebSocket 服务接口文档

本文档基于 `server.py`，说明如何向 Smart Turn WebSocket 服务器推送音频流并解析其返回的端点预测结果，供接入方查阅。

## 概述
- **协议**：WebSocket（客户端发送二进制音频帧，服务器返回 JSON 文本帧）
- **编码**：小端 `int16` PCM，采样率 16 kHz，单声道
- **内部处理**：服务器以 512 个采样（32 ms）为窗口喂给 Silero VAD
- **触发条件**：Silero VAD 判定语音段结束（至少 1 s 静音或达到 8 s 上限）即执行 `predict_endpoint`

## 服务端地址
- 默认监听：`ws://0.0.0.0:8765`
- 环境变量覆盖：
  - `SMART_TURN_WS_HOST`：监听网卡（如 `127.0.0.1` 或公网 IP）
  - `SMART_TURN_WS_PORT`：监听端口（默认 `8765`）

```
SMART_TURN_WS_HOST=0.0.0.0 SMART_TURN_WS_PORT=8765 python server.py
```

## 连接流程
1. 客户端建立 WebSocket 连接。
2. 服务器立即回一条 JSON 文本：
   ```json
  {
    "type": "ready",
    "message": "send config JSON as first text frame (e.g. {\"vad_threshold\": 0.5, \"prediction_threshold\": 0.5, \"min_duration_seconds\": 0}), then 16kHz mono int16 PCM as binary frames; text \"reset\" to clear state",
    "defaults": {
      "vad_threshold": 0.5,
      "prediction_threshold": 0.5,
      "min_duration_seconds": 0
    },
    "session_id": "c3b3..."
  }
   ```
3. 客户端发送配置 JSON 文本帧（如 `{"vad_threshold": 0.5, "prediction_threshold": 0.6, "min_duration_seconds": 0.3}`），服务器返回 `config` 确认。
4. 收到确认后，客户端开始发送二进制音频帧（格式要求见下节）。如需动态调整阈值，可再次发送配置 JSON。
5. 服务器在 VAD 语音/静音切换时发送 `vad` 消息，在语音段结束时发送 `prediction` 消息。
6. 客户端可按需发送文本消息 `reset`，服务器会清空该连接的 VAD 状态并回执。

## 音频帧要求
- 仅发送二进制帧（不要 Base64、不要放进 JSON）。
- 帧长度必须是 2 的倍数，以便解析为 `int16`。
- 采样率固定 16 kHz，单声道。
- 帧时长不限。过小会增加网络开销，过大增加延迟，建议 10–100 ms。

### Python 客户端示例
```python
import asyncio
import json
import numpy as np
import websockets


async def stream(audio_path: str):
    async with websockets.connect("ws://localhost:8765") as ws:
        print(await ws.recv())  # ready 消息
        await ws.send(
            json.dumps(
                {
                    "vad_threshold": 0.6,
                    "prediction_threshold": 0.55,
                    "min_duration_seconds": 0.3,
                }
            )
        )  # 先下发配置
        print(await ws.recv())  # config 回执
        audio = np.fromfile(audio_path, dtype=np.int16)  # 16 kHz 单声道 PCM
        chunk = 1600  # 100 ms
        for i in range(0, len(audio), chunk):
            await ws.send(audio[i : i + chunk].tobytes())
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.01)
                print("server ->", msg)
            except asyncio.TimeoutError:
                pass
        await ws.send("reset")


asyncio.run(stream("/path/to/audio.raw"))

```

## 消息类型
| `type`        | 方向 | 说明 |
|---------------|------|------|
| `ready`       | 服务器 → 客户端 | 握手成功后发送，描述期望 payload 和默认配置。
| `config`      | 服务器 → 客户端 | 客户端发送配置 JSON 后的确认回执，包含生效配置。
| `prediction`  | 服务器 → 客户端 | 语音段结束后的端点判定（详见下文 schema）。
| `vad`         | 服务器 → 客户端 | VAD 语音/静音切换时返回概率及阈值。
| `skip`        | 服务器 → 客户端 | 语音段时长未达到最短长度时返回说明，未执行预测。
| `reset`       | 双向 | 客户端发送文本 `reset` 触发清空，服务器回执确认。
| `error`       | 服务器 → 客户端 | 出现协议或处理错误时返回，详情见 `message`。
> 服务器返回的所有消息均包含 `session_id`，便于日志关联。

### 配置 JSON
- `vad_threshold`：数值 `0-1`，默认 `0.5`，Silero VAD 判定语音的概率阈值。
- `prediction_threshold`：数值 `0-1`，默认 `0.5`，端点概率阈值，大于该值输出 `prediction=1`。
- `min_duration_seconds`：数值 `>=0`，默认 `0`。单段语音长度不足该值时不执行端点预测（会返回 `skip` 提示）。

客户端发送示例（首条文本帧必须为配置）：
```json
{
    "vad_threshold": 0.6,
    "prediction_threshold": 0.55,
    "min_duration_seconds": 0.3,
    "session_id": "c3b3..."
}
```

服务端回执示例：
```json
{
    "type": "config",
    "message": "config applied",
    "config": {
        "vad_threshold": 0.6,
        "prediction_threshold": 0.55,
        "min_duration_seconds": 0.3
    }
}
```

### `prediction` payload
```json
{
  "type": "prediction",
  "prediction": 1,            // 1=已到端点, 0=继续监听
  "probability": 0.73,        // 模型置信度 0-1
  "duration_seconds": 3.12,   // 刚结束的语音段长度
  "inference_ms": 18.4,       // 端点模型推理耗时
  "timestamp_ms": 1712345678901, // 服务器发出该消息的 UNIX 毫秒时间戳
  "vad_probability": 0.12,     // 切换时最新的 VAD 概率
  "vad_speech": false,         // 该概率下是否被判定为语音
  "vad_threshold": 0.6,        // 生效的 VAD 阈值
  "prediction_threshold": 0.55, // 生效的端点概率阈值
  "session_id": "c3b3..."
}
```

### `vad` payload
当 VAD 判定由静音变为语音，或由语音变为静音时返回：
```json
{
  "type": "vad",
  "speech": true,            // 当前判定是否为语音
  "probability": 0.82,       // 对应的 VAD 概率
  "vad_threshold": 0.6,      // 生效的 VAD 阈值
  "timestamp_ms": 1712345678901,
  "session_id": "c3b3..."
}
```

### `skip` payload
当单段语音长度不足 `min_duration_seconds` 时返回：
```json
{
  "type": "skip",
  "reason": "min_duration_not_met",
  "duration_seconds": 0.21,
  "min_duration_seconds": 0.3,
  "vad_probability": 0.08,
  "vad_speech": false,
  "vad_threshold": 0.5,
  "prediction_threshold": 0.55,
  "timestamp_ms": 1712345678901,
  "session_id": "c3b3..."
}
```

### 常见错误
- `"audio payload must be int16 PCM"`：二进制帧长度不是 2 的倍数。
- `"unsupported message type"`：发送了非文本也非二进制的帧。
- `"send audio as binary frames"`：发送了除 `reset` 以外的文本。
- `"server exception"`：服务器内部出现异常，请查看日志。

## 语音段行为
- 服务器会保留最多 200 ms 的前置缓冲，使预测段包含少量起始前滚。
- 静音阈值：1,000 ms（对应 `STOP_MS`）。
- 最长语音段：8 s。超时会强制输出一次预测并立即开始下一段。
- 每段结束后状态自动重置，仅在切换说话人或希望手动控制时发送 `reset`。

## 测试建议
- 使用 `python ws_mic_client.py` 将麦克风音频实时推送到服务器。
- 通过公网时注意 TCP 缓冲区配置，避免任何转码，始终传输原始 PCM。
- 可使用本地 16 kHz 单声道 `.wav` 做回放测试，并记录服务器返回的 `prediction`。
