"""
WebSocket server that accepts streaming 16 kHz mono audio and returns Smart Turn
endpoint predictions whenever a VAD-detected segment completes.

Protocol
--------
- Clients first send a config JSON text frame (e.g. {"vad_threshold": 0.5});
  server replies with a config echo.
- Clients then send binary frames containing raw little-endian int16 PCM audio
  sampled at 16 kHz, mono. Text "reset" clears VAD/segment state.
- Audio can arrive in any chunk size; the server buffers and processes it in
  VAD 专属窗口（Silero 默认 512 样本 ≈ 32 ms，FSMN 默认 200 ms）。
- Whenever VAD speech/silence flips, the server sends a JSON text frame:

    {"type": "vad", "speech": true, "probability": 0.82, "vad_threshold": 0.5, "timestamp_ms": ...}

- When the server decides a speech segment has ended (silence or max duration),
  it runs `predict_endpoint` on the collected audio (up to 8 seconds) and sends
  a JSON text frame like:

    {
      "type": "prediction",
      "prediction": 1,              # 1 = complete, 0 = incomplete
      "probability": 0.73,
      "duration_seconds": 3.12,
      "timestamp_ms": 1712345678901,
      "vad_probability": 0.12,
      "vad_speech": false,
      "vad_threshold": 0.5
    }

"""

import asyncio
import json
import os
import time
import uuid
from typing import List
from datetime import datetime, timedelta, timezone

import numpy as np
import websockets
from websockets.asyncio.server import serve

from inference import predict_endpoint  # expects 16 kHz mono float32 input
from vad import (
    DEFAULT_VAD_TYPE,
    SUPPORTED_VAD_TYPES,
    FSMNVADPipeline,
    SileroVADPipeline,
    TenVADPipeline,
    VAD_THRESHOLD,
    MAX_DURATION_SECONDS,
    RATE,
    fmt4,
)

PREDICTION_THRESHOLD = 0.5
MIN_DURATION_SECONDS = 0.0
SHANGHAI_TZ = timezone(timedelta(hours=8))


def format_config(cfg: dict) -> dict:
    """返回浮点数保留 4 位的小数配置副本。"""
    return {
        k: fmt4(v) if isinstance(v, (int, float)) else v
        for k, v in cfg.items()
    }


def log(msg: str):
    """打印带上海时区的简洁时间戳。"""
    ts = datetime.now(SHANGHAI_TZ).strftime("%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


class StreamingEndpointSession:
    """Connection-scoped state for streaming VAD + endpoint prediction。"""

    def __init__(self):
        self.leftover = np.array([], dtype=np.int16)
        self.config = {
            "vad_threshold": VAD_THRESHOLD,
            "prediction_threshold": PREDICTION_THRESHOLD,
            "min_duration_seconds": MIN_DURATION_SECONDS,
            "vad_type": DEFAULT_VAD_TYPE,
        }
        self.vad_pipeline = self._build_vad_pipeline(self.config["vad_type"])
        self.active_segment_id = None
        self.next_segment_id = 1

    def _build_vad_pipeline(self, vad_type: str):
        if vad_type == "fsmn":
            return FSMNVADPipeline(self.config)
        elif vad_type == "ten":
            return TenVADPipeline(self.config)
        return SileroVADPipeline(self.config)

    def _set_vad_pipeline(self, vad_type: str):
        self.leftover = np.array([], dtype=np.int16)
        self.vad_pipeline = self._build_vad_pipeline(vad_type)

    def apply_config(self, config: dict) -> dict:
        """Apply client配置，当前支持 vad_threshold/prediction_threshold/min_duration_seconds/vad_type。"""
        if not isinstance(config, dict):
            raise ValueError("config must be a JSON object")

        unknown_keys = set(config.keys()) - {
            "vad_threshold",
            "prediction_threshold",
            "min_duration_seconds",
            "vad_type",
        }
        if unknown_keys:
            raise ValueError(f"unsupported config fields: {', '.join(sorted(unknown_keys))}")

        new_config = dict(self.config)
        if "vad_threshold" in config:
            try:
                threshold = float(config["vad_threshold"])
            except (TypeError, ValueError):
                raise ValueError("vad_threshold must be a number between 0 and 1")
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("vad_threshold must be between 0 and 1")
            new_config["vad_threshold"] = threshold
        if "prediction_threshold" in config:
            try:
                threshold = float(config["prediction_threshold"])
            except (TypeError, ValueError):
                raise ValueError("prediction_threshold must be a number between 0 and 1")
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("prediction_threshold must be between 0 and 1")
            new_config["prediction_threshold"] = threshold
        if "min_duration_seconds" in config:
            try:
                min_dur = float(config["min_duration_seconds"])
            except (TypeError, ValueError):
                raise ValueError("min_duration_seconds must be a number >= 0")
            if min_dur < 0.0:
                raise ValueError("min_duration_seconds must be >= 0")
            if min_dur > MAX_DURATION_SECONDS:
                raise ValueError(f"min_duration_seconds must be <= {MAX_DURATION_SECONDS}")
            new_config["min_duration_seconds"] = min_dur
        if "vad_type" in config:
            vad_type = str(config["vad_type"]).strip().lower()
            if vad_type not in SUPPORTED_VAD_TYPES:
                raise ValueError(f"vad_type must be one of: {', '.join(sorted(SUPPORTED_VAD_TYPES))}")
            new_config["vad_type"] = vad_type

        switched_type = new_config.get("vad_type") != self.config.get("vad_type")
        self.config = new_config

        if switched_type:
            self._set_vad_pipeline(self.config["vad_type"])
        else:
            self.vad_pipeline.update_config(self.config)

        return dict(self.config)

    def reset(self):
        """Clear VAD buffers and segment state."""
        self.leftover = np.array([], dtype=np.int16)
        self.vad_pipeline.reset()

    def process_audio(self, int16_audio: np.ndarray) -> List[dict]:
        """
        Ingest new int16 samples, process in VAD 专属窗口, and return any
        payloads (vad变化或预测结果) that should be sent to the client.
        """
        messages: List[dict] = []

        if int16_audio.size == 0:
            return messages

        if self.leftover.size:
            int16_audio = np.concatenate([self.leftover, int16_audio])

        chunk_size = self.vad_pipeline.chunk_size
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")

        full_chunks = int16_audio.size // chunk_size
        self.leftover = int16_audio[full_chunks * chunk_size :]

        for i in range(full_chunks):
            start = i * chunk_size
            end = start + chunk_size
            chunk_int16 = int16_audio[start:end]
            events, segments = self.vad_pipeline.process_chunk(chunk_int16)
            # 先处理 VAD 事件，标记 segment_id
            clear_after_segment = False
            for event in events:
                if event.get("type") == "vad":
                    if bool(event.get("speech")):
                        if self.active_segment_id is None:
                            self.active_segment_id = self.next_segment_id
                            self.next_segment_id += 1
                    else:
                        clear_after_segment = True
                    if self.active_segment_id is not None:
                        event["segment_id"] = self.active_segment_id
                messages.append(event)
            # 再处理语音段（预测/skip），携带 segment_id
            for seg in segments:
                messages.extend(self._process_segment(seg, segment_id=self.active_segment_id))
                clear_after_segment = True
            if clear_after_segment:
                self.active_segment_id = None

        return messages

    def _process_segment(self, segment: dict, segment_id: int | None = None) -> List[dict]:
        """对单段语音执行最短时长校验与端点预测。"""
        events: List[dict] = []
        audio = segment.get("audio", np.array([], dtype=np.float32))
        if audio.size == 0:
            return events

        dur_sec = audio.size / RATE
        dur_sec_fmt = fmt4(dur_sec)
        min_dur = float(self.config.get("min_duration_seconds", MIN_DURATION_SECONDS))
        min_dur_fmt = fmt4(min_dur)

        vad_prob = fmt4(segment.get("vad_probability", 0.0))
        vad_speech = bool(segment.get("vad_speech", False))
        vad_threshold_fmt = fmt4(self.config.get("vad_threshold", VAD_THRESHOLD))

        if dur_sec < min_dur:
            events.append(
                {
                    "type": "skip",
                    "segment_id": segment_id,
                    "reason": "min_duration_not_met",
                    "duration_seconds": dur_sec_fmt,
                    "min_duration_seconds": min_dur_fmt,
                    "timestamp_ms": int(time.time() * 1000),
                    "vad_probability": vad_prob,
                    "vad_speech": vad_speech,
                    "vad_threshold": vad_threshold_fmt,
                    "prediction_threshold": fmt4(
                        self.config.get("prediction_threshold", PREDICTION_THRESHOLD)
                    ),
                }
            )
            return events

        t0 = time.perf_counter()
        result = predict_endpoint(audio.astype(np.float32))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        prob = float(result.get("probability", 0.0))
        prob_fmt = fmt4(prob)
        pred_threshold = float(self.config.get("prediction_threshold", PREDICTION_THRESHOLD))
        pred_threshold_fmt = fmt4(pred_threshold)
        pred = 1 if prob > pred_threshold else 0

        events.append(
            {
                "type": "prediction",
                "segment_id": segment_id,
                "prediction": pred,
                "probability": prob_fmt,
                "duration_seconds": dur_sec_fmt,
                "inference_ms": fmt4(latency_ms),
                "timestamp_ms": int(time.time() * 1000),
                "vad_probability": vad_prob,
                "vad_speech": vad_speech,
                "vad_threshold": vad_threshold_fmt,
                "prediction_threshold": pred_threshold_fmt,
            }
        )
        return events


async def handle_connection(websocket):
    peer = websocket.remote_address
    session_id = str(uuid.uuid4())
    log(f"[server] connection opened from {peer} session_id={session_id}")
    session = StreamingEndpointSession()
    config_received = False
    closed_logged = False
    ready_defaults = format_config(
        {
            "vad_threshold": VAD_THRESHOLD,
            "prediction_threshold": PREDICTION_THRESHOLD,
            "min_duration_seconds": MIN_DURATION_SECONDS,
            "vad_type": DEFAULT_VAD_TYPE,
        }
    )
    await websocket.send(
        json.dumps(
            {
                "type": "ready",
                "message": 'send config JSON as first text frame (e.g. {"vad_type": "silero", "vad_threshold": 0.5, "prediction_threshold": 0.5}), then 16kHz mono int16 PCM as binary frames; text "reset" to clear state',
                "defaults": ready_defaults,
                "session_id": session_id,
            }
        )
    )

    try:
        async for message in websocket:
            if not config_received:
                if not isinstance(message, str):
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "error",
                                "message": "send config JSON as first text frame before audio",
                                "session_id": session_id,
                            }
                        )
                    )
                    await websocket.close(code=1002, reason="config required before audio")
                    break
                try:
                    config = json.loads(message)
                    applied_config = session.apply_config(config if config is not None else {})
                    formatted_config = format_config(applied_config)
                except json.JSONDecodeError:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "error",
                                "message": "config must be a JSON object (text frame)",
                                "session_id": session_id,
                            }
                        )
                    )
                    await websocket.close(code=1002, reason="invalid config")
                    break
                except ValueError as exc:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "error",
                                "message": f"invalid config: {exc}",
                                "session_id": session_id,
                            }
                        )
                    )
                    await websocket.close(code=1002, reason="invalid config")
                    break

                config_received = True
                log(f"[server] config applied for {peer}: {applied_config}")
                await websocket.send(
                    json.dumps(
                        {
                            "type": "config",
                            "message": "config applied",
                            "config": formatted_config,
                            "session_id": session_id,
                        }
                    )
                )
                continue

            if isinstance(message, str):
                stripped = message.strip().lower()
                if stripped == "reset":
                    session.reset()
                    log(f"[server] state reset for {peer}")
                    await websocket.send(json.dumps({"type": "reset"}))
                    continue

                try:
                    config = json.loads(message)
                    if not isinstance(config, dict):
                        raise ValueError("config must be a JSON object")
                    applied_config = session.apply_config(config)
                    formatted_config = format_config(applied_config)
                    log(f"[server] config updated for {peer}: {applied_config}")
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "config",
                                "message": "config applied",
                                "config": formatted_config,
                                "session_id": session_id,
                            }
                        )
                    )
                except json.JSONDecodeError:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "error",
                                "message": "text frames must be config JSON or 'reset'",
                                "session_id": session_id,
                            }
                        )
                    )
                except ValueError as exc:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "error",
                                "message": f"invalid config: {exc}",
                                "session_id": session_id,
                            }
                        )
                    )
                continue

            # Binary audio frame
            if not isinstance(message, (bytes, bytearray)):
                await websocket.send(
                    json.dumps(
                        {
                            "type": "error",
                            "message": "unsupported message type",
                            "session_id": session_id,
                        }
                    )
                )
                continue

            if len(message) % 2 != 0:
                await websocket.send(
                    json.dumps(
                        {
                            "type": "error",
                            "message": "audio payload must be int16 PCM",
                            "session_id": session_id,
                        }
                    )
                )
                continue

            samples = np.frombuffer(message, dtype=np.int16)
            for payload in session.process_audio(samples):
                payload["session_id"] = session_id
                if payload.get("type") == "prediction":
                    log(
                        "[server] prediction"
                        f" pred={payload.get('prediction')}"
                        f" prob={payload.get('probability'):.4f}"
                        f" dur={payload.get('duration_seconds'):.2f}s"
                        f" infer_ms={payload.get('inference_ms'):.1f}"
                        f" from {peer}"
                    )
                await websocket.send(json.dumps(payload))

    except websockets.ConnectionClosed as exc:
        log(f"[server] connection closed from {peer} code={exc.code} reason={exc.reason}")
        closed_logged = True
    except Exception as exc:  # pragma: no cover - defensive logging for server mode
        err = {"type": "error", "message": f"server exception: {exc!r}", "session_id": session_id}
        try:
            await websocket.send(json.dumps(err))
        finally:
            raise
    finally:
        if not closed_logged:
            code = getattr(websocket, "close_code", None)
            reason = getattr(websocket, "close_reason", None)
            log(f"[server] connection closed from {peer} session_id={session_id} code={code} reason={reason}")


async def main():
    host = os.getenv("SMART_TURN_WS_HOST", "0.0.0.0")
    port = int(os.getenv("SMART_TURN_WS_PORT", "8765"))
    log(f"Starting Smart Turn WebSocket server on ws://{host}:{port}")
    async with serve(handle_connection, host, port, max_size=None):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
