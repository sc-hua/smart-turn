"""Base class for audio denoisers.

All denoiser implementations must inherit from DenoiserBase and implement
the process() method. Denoisers process float32 audio arrays at 16kHz.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class DenoiserBase(ABC):
    """Abstract base class for all denoisers.

    Attributes:
        type_name: Unique identifier for this denoiser type (must be set by subclass).
        sample_rate: Audio sample rate in Hz (fixed at 16000 for this project).
        config: Runtime configuration dictionary.
    """

    type_name: str = "base"

    def __init__(self, sample_rate: int = 16000, config: dict[str, Any] | None = None):
        """Initialize denoiser.

        Args:
            sample_rate: Audio sample rate in Hz. Must be 16000.
            config: Optional configuration dictionary.
        """
        self.sample_rate = int(sample_rate)
        self.config = dict(config or {})

    def reset(self) -> None:
        """Reset internal state (if any).

        Called before processing a new audio segment to ensure clean state.
        Subclasses with stateful processing should override this method.
        """
        pass

    def update_config(self, config: dict[str, Any]) -> None:
        """Update runtime configuration.

        Args:
            config: New configuration dictionary to merge.
        """
        self.config.update(config or {})

    @abstractmethod
    def process(self, audio: np.ndarray) -> np.ndarray:
        """Process audio and return denoised result.

        Args:
            audio: Input audio as float32 array, values in [-1, 1].

        Returns:
            Denoised audio as float32 array with same shape as input.

        Raises:
            ValueError: If audio format is invalid.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} type={self.type_name}>"
