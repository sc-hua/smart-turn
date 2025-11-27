"""
Real-time microphone client for the Smart Turn WebSocket server.

Usage examples:
  # List input devices (indexes for --device-index)
  python ws_mic_client.py --list-devices

  # Stream default mic to a remote server
  python ws_mic_client.py --url ws://localhost:8765

  # Stream a specific input device
  python ws_mic_client.py --device-index 3 --url ws://localhost:8765
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


async def recv_printer(ws):
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


async def send_microphone(ws, stream, debug: bool):
    """Read mic frames and send to server."""
    frame_count = 0
    loop = asyncio.get_running_loop()
    try:
        while True:
            data = await loop.run_in_executor(
                None, lambda: stream.read(CHUNK, exception_on_overflow=False)
            )
            frame_count += 1
            await ws.send(data)
            if debug and frame_count % 50 == 0:
                sent_seconds = frame_count * CHUNK / RATE
                print(f"[client] sent {frame_count} frames (~{sent_seconds:.1f}s)", flush=True)
    except ConnectionClosed as exc:
        print(f"[client] send loop closed code={exc.code} reason={exc.reason}", flush=True)


async def stream_microphone(url: str, device_index: int | None, debug: bool):
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
        # Print initial ready message if present
        try:
            ready = await asyncio.wait_for(ws.recv(), timeout=2.0)
            print(f"[server] {ready}", flush=True)
        except asyncio.TimeoutError:
            pass

        recv_task = asyncio.create_task(recv_printer(ws))
        send_task = asyncio.create_task(send_microphone(ws, stream, debug))
        print(f"[client] streaming microphone -> {url} (Ctrl+C to stop)", flush=True)

        try:
            done, pending = await asyncio.wait(
                {recv_task, send_task}, return_when=asyncio.FIRST_EXCEPTION
            )
            for task in done:
                task.result()
        except KeyboardInterrupt:
            pass
        finally:
            send_task.cancel()
            recv_task.cancel()
            with suppress(asyncio.CancelledError):
                await send_task
                await recv_task
            stream.stop_stream()
            stream.close()
            pa.terminate()


def parse_args():
    parser = argparse.ArgumentParser(description="Stream microphone audio to Smart Turn WebSocket server.")
    parser.add_argument("--url", default="ws://localhost:8765", help="WebSocket server URL.")
    parser.add_argument("--device-index", type=int, default=None, help="PyAudio input device index.")
    parser.add_argument("--list-devices", action="store_true", help="List available input devices and exit.")
    parser.add_argument("--debug", action="store_true", help="Print send-side frame counters.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.list_devices:
        pa = pyaudio.PyAudio()
        list_input_devices(pa)
        pa.terminate()
    else:
        asyncio.run(stream_microphone(args.url, args.device_index, args.debug))
