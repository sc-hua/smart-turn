"""DTLN (Dual-signal Transformation LSTM Network) ONNX denoiser.

DTLN is a lightweight deep learning model for real-time speech enhancement.
It uses two ONNX models (model_1 and model_2) in a dual-path architecture
with overlap-add processing.

Key characteristics:
- Sample rate: 16kHz only
- Block length: 512 samples (32ms)
- Block shift: 128 samples (8ms)
- Algorithmic latency: 32ms
- Typical inference: <8ms per block on CPU

Model files required in onnx_model/dtln/:
- model_1.onnx: Frequency domain processing (magnitude masking)
- model_2.onnx: Time domain processing (waveform enhancement)

Reference: https://github.com/breizhn/DTLN
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .base import DenoiserBase
from .registry import register_denoiser


def _normalize_shape(shape: list[Any]) -> list[int]:
    """Convert ONNX shape with dynamic dims to concrete shape.

    Args:
        shape: ONNX shape that may contain None, 0, or string dimensions.

    Returns:
        Shape with all dimensions as positive integers.
    """
    return [1 if (isinstance(d, str) or d is None or d == 0) else int(d) for d in shape]


def _infer_input_rank(shape: list[Any]) -> int:
    """Infer the expected input rank from ONNX shape."""
    return len(shape)


@register_denoiser
class DTLNDenoiser(DenoiserBase):
    """DTLN ONNX denoiser with overlap-add processing.

    Configuration options (via config dict):
        dtln_model_dir: Path to model directory (default: onnx_model/dtln)
        dtln_block_len: FFT block length in samples (default: 512)
        dtln_block_shift: Block shift/hop in samples (default: 128)
        dtln_providers: Comma-separated ONNX providers (default: CPUExecutionProvider)
        dtln_pad_samples: Samples to pad at start to reduce boundary effects (default: 256)
    """

    type_name = "dtln"

    def __init__(self, sample_rate: int = 16000, config: dict[str, Any] | None = None):
        """Initialize DTLN denoiser.

        Args:
            sample_rate: Must be 16000 Hz.
            config: Configuration dictionary.

        Raises:
            ValueError: If sample_rate is not 16000.
            FileNotFoundError: If model files are missing.
            RuntimeError: If onnxruntime is not available.
        """
        super().__init__(sample_rate, config)

        if self.sample_rate != 16000:
            raise ValueError(f"DTLN requires 16kHz sample rate, got {self.sample_rate}")

        # Get model directory
        model_dir = str(self.config.get("dtln_model_dir") or "").strip()
        if not model_dir:
            model_dir = os.getenv("DTLN_MODEL_DIR", "onnx_model/dtln")

        self.model_1_path = os.path.join(model_dir, "model_1.onnx")
        self.model_2_path = os.path.join(model_dir, "model_2.onnx")

        # Validate model files exist
        if not os.path.exists(self.model_1_path):
            raise FileNotFoundError(f"DTLN model_1 not found: {self.model_1_path}")
        if not os.path.exists(self.model_2_path):
            raise FileNotFoundError(f"DTLN model_2 not found: {self.model_2_path}")

        # Processing parameters
        self.block_len = int(self.config.get("dtln_block_len", 512))
        self.block_shift = int(self.config.get("dtln_block_shift", 128))
        self.pad_samples = int(self.config.get("dtln_pad_samples", 256))

        if self.block_len <= 0 or self.block_shift <= 0:
            raise ValueError("dtln_block_len and dtln_block_shift must be > 0")
        if self.block_shift > self.block_len:
            raise ValueError("dtln_block_shift must be <= dtln_block_len")

        # Load ONNX sessions
        self._load_models()
        self.reset()

    def _load_models(self) -> None:
        """Load ONNX model sessions."""
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required for DTLN denoiser. "
                "Install with: pip install onnxruntime"
            ) from exc

        # Session options for optimal performance
        sess_opts = ort.SessionOptions()
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_opts.inter_op_num_threads = 1
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Get execution providers
        requested = self.config.get("dtln_providers") or os.getenv("DTLN_PROVIDERS", "")
        if requested:
            provider_list = [p.strip() for p in str(requested).split(",") if p.strip()]
        else:
            provider_list = ["CPUExecutionProvider"]

        available = ort.get_available_providers()
        providers = [p for p in provider_list if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]

        # Load models
        self.model_1 = ort.InferenceSession(
            self.model_1_path, sess_options=sess_opts, providers=providers
        )
        self.model_2 = ort.InferenceSession(
            self.model_2_path, sess_options=sess_opts, providers=providers
        )

        # Get input/output info for model 1
        m1_inputs = self.model_1.get_inputs()
        m1_outputs = self.model_1.get_outputs()

        if len(m1_inputs) < 2:
            raise ValueError("DTLN model_1 must have at least 2 inputs (signal + state)")
        if len(m1_outputs) < 2:
            raise ValueError("DTLN model_1 must have at least 2 outputs (mask + state)")

        # Model 1: frequency domain (magnitude -> mask)
        self.m1_input_name = m1_inputs[0].name
        self.m1_state_name = m1_inputs[1].name
        self.m1_input_shape = list(m1_inputs[0].shape)
        self.m1_input_rank = _infer_input_rank(self.m1_input_shape)
        self.m1_state_shape = _normalize_shape(list(m1_inputs[1].shape))

        # Find output names by checking shapes (mask is usually variable, state is fixed)
        self.m1_mask_output_name = m1_outputs[0].name
        self.m1_state_output_name = m1_outputs[1].name

        # Get input/output info for model 2
        m2_inputs = self.model_2.get_inputs()
        m2_outputs = self.model_2.get_outputs()

        if len(m2_inputs) < 2:
            raise ValueError("DTLN model_2 must have at least 2 inputs (signal + state)")
        if len(m2_outputs) < 2:
            raise ValueError("DTLN model_2 must have at least 2 outputs (block + state)")

        # Model 2: time domain (block -> enhanced block)
        self.m2_input_name = m2_inputs[0].name
        self.m2_state_name = m2_inputs[1].name
        self.m2_input_shape = list(m2_inputs[0].shape)
        self.m2_input_rank = _infer_input_rank(self.m2_input_shape)
        self.m2_state_shape = _normalize_shape(list(m2_inputs[1].shape))

        self.m2_block_output_name = m2_outputs[0].name
        self.m2_state_output_name = m2_outputs[1].name

    def _reshape_for_model(self, data: np.ndarray, target_rank: int) -> np.ndarray:
        """Reshape data to match model's expected input rank.

        Args:
            data: 1D array of features.
            target_rank: Expected number of dimensions.

        Returns:
            Reshaped array matching the model's expected input shape.
        """
        if target_rank == 2:
            # Shape: (batch, features) or (1, N)
            return data.reshape(1, -1)
        elif target_rank == 3:
            # Shape: (batch, seq, features) or (1, 1, N)
            return data.reshape(1, 1, -1)
        elif target_rank >= 4:
            # Shape: (batch, seq, features, 1) etc
            return data.reshape(1, 1, -1, 1)
        else:
            # Fallback: just add batch dimension
            return data.reshape(1, -1)

    def reset(self) -> None:
        """Reset LSTM states to zeros."""
        self.state_1 = np.zeros(self.m1_state_shape, dtype=np.float32)
        self.state_2 = np.zeros(self.m2_state_shape, dtype=np.float32)

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Process audio segment with DTLN denoising.

        Uses overlap-add processing with optional padding to reduce
        boundary effects at segment start.

        Args:
            audio: Input audio as float32 array, values in [-1, 1].

        Returns:
            Denoised audio as float32 array with same length as input.
        """
        audio = np.asarray(audio, dtype=np.float32).flatten()
        if audio.size == 0:
            return audio

        original_len = audio.size

        # Pad start to reduce boundary effects
        if self.pad_samples > 0:
            audio = np.concatenate([np.zeros(self.pad_samples, dtype=np.float32), audio])

        # Initialize buffers
        in_buf = np.zeros(self.block_len, dtype=np.float32)
        out_buf = np.zeros(self.block_len, dtype=np.float32)
        output = np.zeros(audio.size, dtype=np.float32)

        state_1 = self.state_1
        state_2 = self.state_2

        # Process in overlapping blocks
        for idx in range(0, audio.size, self.block_shift):
            # Shift input buffer and add new samples
            take_len = min(self.block_shift, audio.size - idx)
            in_buf[:-self.block_shift] = in_buf[self.block_shift:]
            in_buf[-self.block_shift:] = 0.0
            in_buf[-self.block_shift:-self.block_shift + take_len] = audio[idx:idx + take_len]

            # FFT to get magnitude and phase
            in_spec = np.fft.rfft(in_buf)
            in_mag = np.abs(in_spec).astype(np.float32)
            in_phase = np.angle(in_spec)

            # Model 1: estimate magnitude mask
            mag_input = self._reshape_for_model(in_mag, self.m1_input_rank)
            m1_out = self.model_1.run(
                None,  # Get all outputs
                {self.m1_input_name: mag_input, self.m1_state_name: state_1},
            )
            mask = np.squeeze(m1_out[0]).astype(np.float32)
            state_1 = m1_out[1]

            # Apply mask and reconstruct
            est_mag = in_mag * mask
            est_spec = est_mag * np.exp(1j * in_phase)
            est_block = np.fft.irfft(est_spec).astype(np.float32)

            # Model 2: time domain enhancement
            block_input = self._reshape_for_model(est_block, self.m2_input_rank)
            m2_out = self.model_2.run(
                None,  # Get all outputs
                {self.m2_input_name: block_input, self.m2_state_name: state_2},
            )
            out_block = np.squeeze(m2_out[0]).astype(np.float32)
            state_2 = m2_out[1]

            # Overlap-add
            out_buf[:-self.block_shift] = out_buf[self.block_shift:]
            out_buf[-self.block_shift:] = 0.0
            out_buf += out_block[:self.block_len]

            # Write to output
            output[idx:idx + take_len] = out_buf[:take_len]

        # Update states for potential stateful mode
        self.state_1 = state_1
        self.state_2 = state_2

        # Remove padding and return
        if self.pad_samples > 0:
            output = output[self.pad_samples:]

        return output[:original_len]
