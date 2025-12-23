"""No-op denoiser (passthrough).

This is the default denoiser that simply returns the input audio unchanged.
Useful for baseline comparison and as a fallback when denoising is disabled.
"""

from __future__ import annotations

import numpy as np

from .base import DenoiserBase
from .registry import register_denoiser


@register_denoiser
class NoneDenoiser(DenoiserBase):
    """No-op denoiser that returns input unchanged."""

    type_name = "none"

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Return audio unchanged.

        Args:
            audio: Input audio as float32 array.

        Returns:
            Same audio array unchanged.
        """
        return audio
