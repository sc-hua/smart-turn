"""
Real-time microphone client for the Smart Turn WebSocket server.

Usage examples:
  # List input devices (indexes for --device-index)
  python ws_mic_client.py --list-devices

  # Stream default mic to a remote server
  python ws_mic_client.py --url ws://localhost:8765

  # Stream a specific input device
  python ws_mic_client.py --device-index 3 --url ws://localhost:8765
  
  # example with debug output
  python ws_mic_client.py --device-index 3 --url ws://localhost:8765 --debug
  [server] {"type": "ready", "message": "send 16kHz mono int16 PCM as binary frames; text 'reset' to clear state"}
  [client] streaming microphone -> ws://localhost:8765 (Ctrl+C to stop)
  [client] sent 50 frames (~1.6s)
  [server] {"type": "prediction", "prediction": 1, "probability": 0.9763804078102112, "duration_seconds": 1.76, "inference_ms": 48.0109520140104, "timestamp_ms": 1764208874456}
  [client] sent 100 frames (~3.2s)
  [client] sent 150 frames (~4.8s)
  [server] {"type": "prediction", "prediction": 1, "probability": 0.9785922765731812, "duration_seconds": 1.504, "inference_ms": 50.1353699946776, "timestamp_ms": 1764208877467}
  [client] sent 200 frames (~6.4s)
  [client] sent 250 frames (~8.0s)
  [client] sent 300 frames (~9.6s)
  [server] {"type": "prediction", "prediction": 0, "probability": 0.05131623148918152, "duration_seconds": 1.376, "inference_ms": 28.71188599965535, "timestamp_ms": 1764208881474}
  [client] sent 350 frames (~11.2s)
  [client] sent 400 frames (~12.8s)
  [client] sent 450 frames (~14.4s)
  [client] sent 500 frames (~16.0s)
  [server] {"type": "prediction", "prediction": 1, "probability": 0.9739460945129395, "duration_seconds": 2.336, "inference_ms": 43.89235298731364, "timestamp_ms": 1764208887382}
  [client] sent 550 frames (~17.6s)
"""

import argparse
import asyncio
import json
from contextlib import suppress

import pyaudio
import websockets
from websockets.client import connect
from websockets.exceptions import ConnectionClosed, ConnectionClosedError, ConnectionClosedOK

RATE = 16000
CHUNK = 512  # Must match server VAD window


def list_input_devices(pa: pyaudio.PyAudio):
    print("Available input devices:")
    for i in range(pa.get_device_count()):
        try:
            info = pa.get_device_info_by_index(i)
        except Exception:
            continue
        if info.get("maxInputChannels", 0) > 0:
            name = info.get("name", "Unknown")
            print(f"  [{i}] {name} (channels={info.get('maxInputChannels')})")


async def recv_printer(ws, stop_event: asyncio.Event):
    """Background task: print any server messages."""
    try:
        async for msg in ws:
            try:
                parsed = json.loads(msg)
                print(f"[server] {json.dumps(parsed)}", flush=True)
            except Exception:
                print(f"[server] {msg}", flush=True)
    except ConnectionClosedOK as exc:
        print(f"[client] server closed connection (normal) code={exc.code} reason={exc.reason}", flush=True)
    except ConnectionClosedError as exc:
        print(f"[client] server closed connection (error) code={exc.code} reason={exc.reason}", flush=True)
    finally:
        stop_event.set()


async def send_microphone(ws, stream, debug: bool, stop_event: asyncio.Event):
    """Read mic frames and send to server."""
    frame_count = 0
    loop = asyncio.get_running_loop()
    try:
        while not stop_event.is_set():
            data = await loop.run_in_executor(
                None, lambda: stream.read(CHUNK, exception_on_overflow=False)
            )
            if stop_event.is_set():
                break
            frame_count += 1
            await ws.send(data)
            if debug and frame_count % 50 == 0:
                sent_seconds = frame_count * CHUNK / RATE
                print(f"[client] sent {frame_count} frames (~{sent_seconds:.1f}s)", flush=True)
    except ConnectionClosed as exc:
        print(f"[client] send loop closed code={exc.code} reason={exc.reason}", flush=True)
    except asyncio.CancelledError:
        pass
    finally:
        stop_event.set()


async def stream_microphone(
    url: str,
    device_index: int | None,
    debug: bool,
    vad_threshold: float,
    prediction_threshold: float,
    min_duration_seconds: float,
):
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        input_device_index=device_index,
    )

    async with connect(url, max_size=None) as ws:
        stop_event = asyncio.Event()

        # Print initial ready message if present
        try:
            ready = await asyncio.wait_for(ws.recv(), timeout=2.0)
            print(f"[server] {ready}", flush=True)
        except asyncio.TimeoutError:
            pass

        # 配置 VAD 阈值，服务端要求建立连接后先下发配置
        config_msg = json.dumps(
            {
                "vad_threshold": vad_threshold,
                "prediction_threshold": prediction_threshold,
                "min_duration_seconds": min_duration_seconds,
            }
        )
        await ws.send(config_msg)
        try:
            raw_ack = await asyncio.wait_for(ws.recv(), timeout=2.0)
            print(f"[server] {raw_ack}", flush=True)
            try:
                config_ack = json.loads(raw_ack)
            except Exception:
                print("[client] invalid config ack, exiting", flush=True)
                return
            if config_ack.get("type") != "config":
                print("[client] config not accepted, exiting", flush=True)
                return
        except asyncio.TimeoutError:
            print("[client] did not receive config ack, exiting", flush=True)
            return

        recv_task = asyncio.create_task(recv_printer(ws, stop_event))
        send_task = asyncio.create_task(send_microphone(ws, stream, debug, stop_event))
        stop_waiter = asyncio.create_task(stop_event.wait())
        print(f"[client] streaming microphone -> {url} (Ctrl+C to stop)", flush=True)

        async def cancel_task(task, timeout: float = 1.0):
            task.cancel()
            with suppress(asyncio.TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=timeout)

        async def graceful_close():
            stop_event.set()
            # 停止采集，释放音频资源
            with suppress(Exception):
                stream.stop_stream()
            with suppress(Exception):
                stream.close()
            with suppress(Exception):
                pa.terminate()
            # 主动关闭 websocket，唤醒 recv
            with suppress(Exception):
                await ws.close(code=1000, reason="client shutdown")
                await ws.wait_closed()
            # 结束后台任务
            await cancel_task(send_task)
            await cancel_task(recv_task)
            await cancel_task(stop_waiter)

        try:
            await asyncio.wait(
                {recv_task, send_task, stop_waiter}, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            await graceful_close()


def parse_args():
    parser = argparse.ArgumentParser(description="Stream microphone audio to Smart Turn WebSocket server.")
    parser.add_argument("--url", default="ws://localhost:8765", help="WebSocket server URL.")
    parser.add_argument("-d", "--device-index", type=int, default=None, help="PyAudio input device index.")
    parser.add_argument("-l", "--list-devices", action="store_true", help="List available input devices and exit.")
    parser.add_argument("--debug", action="store_true", help="Print send-side frame counters.")
    parser.add_argument("--vad-threshold", type=float, default=0.5, help="Silero VAD threshold, 0-1.")
    parser.add_argument(
        "--prediction-threshold",
        type=float,
        default=0.5,
        help="Endpoint probability threshold, 0-1.",
    )
    parser.add_argument(
        "--min-duration-seconds",
        type=float,
        default=1.0,
        help="Minimum segment length before running endpoint prediction.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.list_devices:
        pa = pyaudio.PyAudio()
        list_input_devices(pa)
        pa.terminate()
    else:
        asyncio.run(
            stream_microphone(
                args.url,
                args.device_index,
                args.debug,
                args.vad_threshold,
                args.prediction_threshold,
                args.min_duration_seconds,
            )
        )
