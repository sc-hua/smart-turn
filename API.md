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
     "message": "send 16kHz mono int16 PCM as binary frames; text 'reset' to clear state"
   }
   ```
3. 客户端开始发送二进制音频帧（格式要求见下节）。
4. 每当语音段结束，服务器发送一条 `prediction` JSON 文本。
5. 客户端可按需发送文本消息 `reset`，服务器会清空该连接的 VAD 状态并回执。

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
        audio = np.fromfile(audio_path, dtype=np.int16)  # 16 kHz 单声道 PCM
        chunk = 1600  # 100 ms
        for i in range(0, len(audio), chunk):
            await ws.send(audio[i:i+chunk].tobytes())
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
| `ready`       | 服务器 → 客户端 | 握手成功后发送，描述期望 payload。
| `prediction`  | 服务器 → 客户端 | 语音段结束后的端点判定（详见下文 schema）。
| `reset`       | 双向 | 客户端发送文本 `reset` 触发清空，服务器回执确认。
| `error`       | 服务器 → 客户端 | 出现协议或处理错误时返回，详情见 `message`。

### `prediction` payload
```json
{
  "type": "prediction",
  "prediction": 1,            // 1=已到端点, 0=继续监听
  "probability": 0.73,        // 模型置信度 0-1
  "duration_seconds": 3.12,   // 刚结束的语音段长度
  "inference_ms": 18.4,       // 端点模型推理耗时
  "timestamp_ms": 1712345678901 // 服务器发出该消息的 UNIX 毫秒时间戳
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
