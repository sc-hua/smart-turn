import os
import time
import math
import urllib.request
from collections import deque

import numpy as np
import pyaudio
from scipy.io import wavfile
import onnxruntime as ort

from inference import predict_endpoint  # assumes 16 kHz mono float32 input

# --- Configuration (fixed 16 kHz mono, 512-sample chunks) ---
RATE = 16000
CHUNK = 512                     # Silero VAD expects 512 samples at 16 kHz
FORMAT = pyaudio.paInt16
CHANNELS = 1

VAD_THRESHOLD = 0.5             # speech probability threshold
PRE_SPEECH_MS = 200             # keep this many ms before trigger
STOP_MS = 1000                  # end after this much trailing silence
MAX_DURATION_SECONDS = 8        # hard cap per segment

DEBUG_SAVE_WAV = False
TEMP_OUTPUT_WAV = "temp_output.wav"

# Silero ONNX model
ONNX_MODEL_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
)
ONNX_MODEL_PATH = "silero_vad.onnx"

# Reset VAD internal state every N seconds
MODEL_RESET_STATES_TIME = 5.0


class SileroVAD:
    """Minimal Silero VAD ONNX wrapper for 16 kHz, mono, chunk=512."""

    def __init__(self, model_path: str):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1

        available_providers = ort.get_available_providers()
        providers = []
        if 'CUDAExecutionProvider' in available_providers:
            providers.append('CUDAExecutionProvider')
        if 'CoreMLExecutionProvider' in available_providers:
            providers.append('CoreMLExecutionProvider')
        if 'MPSExecutionProvider' in available_providers:
            providers.append('MPSExecutionProvider')
        providers.append('CPUExecutionProvider')
        print(f"SileroVAD using providers: {providers}")

        self.session = ort.InferenceSession(
            model_path, providers=providers, sess_options=opts
        )
        self.context_size = 64            # Silero uses 64-sample context at 16 kHz
        self._state = None
        self._context = None
        self._last_reset_time = time.time()
        self._init_states()

    def _init_states(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)     # (2, B, 128)
        self._context = np.zeros((1, self.context_size), dtype=np.float32)

    def maybe_reset(self):
        if (time.time() - self._last_reset_time) >= MODEL_RESET_STATES_TIME:
            self._init_states()
            self._last_reset_time = time.time()

    def prob(self, chunk_f32: np.ndarray) -> float:
        """
        Compute speech probability for one chunk of length 512 (float32, mono).
        Returns a scalar float.
        """
        # Ensure shape (1, 512) and concat context
        x = np.reshape(chunk_f32, (1, -1))
        if x.shape[1] != CHUNK:
            raise ValueError(f"Expected {CHUNK} samples, got {x.shape[1]}")
        x = np.concatenate((self._context, x), axis=1)

        # Run ONNX
        ort_inputs = {"input": x.astype(np.float32), "state": self._state, "sr": np.array(16000, dtype=np.int64)}
        out, self._state = self.session.run(None, ort_inputs)

        # Update context (keep last 64 samples)
        self._context = x[:, -self.context_size:]
        self.maybe_reset()

        # out shape is (1, 1) -> return scalar
        return float(out[0][0])


def ensure_model(path: str = ONNX_MODEL_PATH, url: str = ONNX_MODEL_URL) -> str:
    if not os.path.exists(path):
        print("Downloading Silero VAD ONNX model...")
        urllib.request.urlretrieve(url, path)
        print("ONNX model downloaded.")
    return path


def select_input_device(pa: pyaudio.PyAudio) -> int:
    """List input devices and ask user to select one."""
    count = pa.get_device_count()
    input_devices = []
    # print("Scanning for input devices...")
    for i in range(count):
        try:
            dev_info = pa.get_device_info_by_index(i)
            if dev_info.get('maxInputChannels') > 0:
                input_devices.append((i, dev_info.get('name')))
        except Exception:
            pass

    if not input_devices:
        print("No input devices found. Using default.")
        return None

    print("\nAvailable input devices:")
    for idx, (dev_index, name) in enumerate(input_devices):
        print(f"  [{idx}] {name} (ID: {dev_index})")

    default_idx = 0
    print(f"\nSelect device number [0-{len(input_devices)-1}] (default {default_idx}): ", end="")
    
    try:
        selection = input().strip()
        if selection == "":
            selected_dev_index = input_devices[default_idx][0]
        else:
            list_idx = int(selection)
            if 0 <= list_idx < len(input_devices):
                selected_dev_index = input_devices[list_idx][0]
            else:
                print(f"Invalid selection. Using default: {input_devices[default_idx][1]}")
                selected_dev_index = input_devices[default_idx][0]
    except Exception:
        print(f"Invalid input. Using default: {input_devices[default_idx][1]}")
        selected_dev_index = input_devices[default_idx][0]
        
    print(f"Selected device ID: {selected_dev_index}\n")
    return selected_dev_index


def record_and_predict():
    # Derived chunk counts (avoid timestamp tracking)
    chunk_ms = (CHUNK / RATE) * 1000.0
    pre_chunks = math.ceil(PRE_SPEECH_MS / chunk_ms)
    stop_chunks = math.ceil(STOP_MS / chunk_ms)
    max_chunks = math.ceil(MAX_DURATION_SECONDS / (CHUNK / RATE))

    # Pre-speech ring buffer
    pre_buffer = deque(maxlen=pre_chunks)

    # Segment assembly state
    segment = []             # list of float32 chunks (includes pre, speech, trailing silence)
    speech_active = False
    trailing_silence = 0
    since_trigger_chunks = 0

    # Init audio + VAD
    vad = SileroVAD(ensure_model())
    pa = pyaudio.PyAudio()

    input_device_index = select_input_device(pa)

    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=input_device_index,
        frames_per_buffer=CHUNK,
    )

    print("Listening for speech... (Ctrl+C to stop)")
    try:
        while True:
            # Read one chunk and convert ONCE
            data = stream.read(CHUNK, exception_on_overflow=False)
            int16 = np.frombuffer(data, dtype=np.int16)
            f32 = (int16.astype(np.float32)) / 32768.0

            # VAD
            is_speech = vad.prob(f32) > VAD_THRESHOLD

            if not speech_active:
                # Warmup pre-speech buffer until trigger
                pre_buffer.append(f32)
                if is_speech:
                    # Trigger: start a new segment with pre-speech
                    segment = list(pre_buffer)
                    segment.append(f32)
                    speech_active = True
                    trailing_silence = 0
                    since_trigger_chunks = 1
            else:
                # Already in a segment
                segment.append(f32)
                since_trigger_chunks += 1
                if is_speech:
                    trailing_silence = 0
                else:
                    trailing_silence += 1

                # End conditions: long enough silence or duration cap
                if trailing_silence >= stop_chunks or since_trigger_chunks >= max_chunks:
                    # Pause capture while we process
                    stream.stop_stream()
                    _process_segment(np.concatenate(segment, dtype=np.float32))
                    # Reset for next segment
                    segment.clear()
                    speech_active = False
                    trailing_silence = 0
                    since_trigger_chunks = 0
                    pre_buffer.clear()
                    stream.start_stream()
                    print("Listening for speech...")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


def _process_segment(segment_audio_f32: np.ndarray):
    if segment_audio_f32.size == 0:
        print("Captured empty audio segment, skipping prediction.")
        return

    if DEBUG_SAVE_WAV:
        wavfile.write(TEMP_OUTPUT_WAV, RATE, (segment_audio_f32 * 32767.0).astype(np.int16))

    dur_sec = segment_audio_f32.size / RATE
    print(f"Processing segment ({dur_sec:.2f}s)...")

    t0 = time.perf_counter()
    result = predict_endpoint(segment_audio_f32)  # expects 16 kHz float32 mono
    dt_ms = (time.perf_counter() - t0) * 1000.0

    pred = result.get("prediction", 0)
    prob = result.get("probability", float("nan"))

    print("--------")
    print(f"Prediction: {'Complete' if pred == 1 else 'Incomplete'}")
    print(f"Probability of complete: {prob:.4f}")
    print(f"Inference time: {dt_ms:.2f} ms")


if __name__ == "__main__":
    record_and_predict()
