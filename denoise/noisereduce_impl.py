"""noisereduce library denoiser.

Uses the noisereduce library for spectral gating based noise reduction.
This is more suitable for offline/segment processing than real-time streaming,
but works well for the VAD-segmented audio in this project.

The noisereduce library uses spectral gating with optional noise profiling.
When processing VAD segments, we can use the pre-speech buffer (if available)
as a noise reference for better results.

Reference: https://github.com/timsainb/noisereduce
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import DenoiserBase
from .registry import register_denoiser


@register_denoiser
class NoiseReduceDenoiser(DenoiserBase):
    """noisereduce library based denoiser.

    Configuration options (via config dict):
        noisereduce_stationary: Use stationary noise reduction (default: True)
        noisereduce_prop_decrease: Proportion to reduce noise by (default: 0.8)
        noisereduce_n_fft: FFT size (default: 512)
        noisereduce_hop_length: Hop length (default: 128)
        noisereduce_n_std_thresh: Number of std deviations for threshold (default: 1.5)
        noisereduce_use_tqdm: Show progress bar (default: False)
    """

    type_name = "noisereduce"

    def __init__(self, sample_rate: int = 16000, config: dict[str, Any] | None = None):
        """Initialize noisereduce denoiser.

        Args:
            sample_rate: Audio sample rate in Hz.
            config: Configuration dictionary.

        Raises:
            RuntimeError: If noisereduce library is not available.
        """
        super().__init__(sample_rate, config)

        # Verify library is available
        try:
            import noisereduce as nr
            self._nr = nr
        except ImportError as exc:
            raise RuntimeError(
                "noisereduce library is required. "
                "Install with: pip install noisereduce"
            ) from exc

        # Extract configuration
        self.stationary = bool(self.config.get("noisereduce_stationary", True))
        self.prop_decrease = float(self.config.get("noisereduce_prop_decrease", 0.8))
        self.n_fft = int(self.config.get("noisereduce_n_fft", 512))
        self.hop_length = int(self.config.get("noisereduce_hop_length", 128))
        self.n_std_thresh = float(self.config.get("noisereduce_n_std_thresh", 1.5))
        self.use_tqdm = bool(self.config.get("noisereduce_use_tqdm", False))

    def update_config(self, config: dict[str, Any]) -> None:
        """Update configuration.

        Args:
            config: New configuration dictionary.
        """
        super().update_config(config)
        self.stationary = bool(self.config.get("noisereduce_stationary", True))
        self.prop_decrease = float(self.config.get("noisereduce_prop_decrease", 0.8))
        self.n_fft = int(self.config.get("noisereduce_n_fft", 512))
        self.hop_length = int(self.config.get("noisereduce_hop_length", 128))
        self.n_std_thresh = float(self.config.get("noisereduce_n_std_thresh", 1.5))
        self.use_tqdm = bool(self.config.get("noisereduce_use_tqdm", False))

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Process audio with noisereduce.

        Args:
            audio: Input audio as float32 array, values in [-1, 1].

        Returns:
            Denoised audio as float32 array with same length as input.

        Raises:
            ValueError: If audio is too short for STFT processing.
        """
        audio = np.asarray(audio, dtype=np.float32).flatten()
        if audio.size == 0:
            return audio

        # Minimum length check for STFT
        min_samples = self.n_fft * 2
        if audio.size < min_samples:
            # Too short for processing - raise error so manager marks as failed
            raise ValueError(
                f"Audio too short for noisereduce: {audio.size} samples < {min_samples} required"
            )

        if self.stationary:
            # Stationary noise reduction (faster, assumes constant noise)
            reduced = self._nr.reduce_noise(
                y=audio,
                sr=self.sample_rate,
                stationary=True,
                prop_decrease=self.prop_decrease,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_std_thresh_stationary=self.n_std_thresh,
                use_tqdm=self.use_tqdm,
            )
        else:
            # Non-stationary noise reduction (slower but handles varying noise)
            reduced = self._nr.reduce_noise(
                y=audio,
                sr=self.sample_rate,
                stationary=False,
                prop_decrease=self.prop_decrease,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                use_tqdm=self.use_tqdm,
            )

        return np.asarray(reduced, dtype=np.float32)
