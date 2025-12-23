"""Denoiser registry and session-scoped manager.

Provides decorator-based registration for denoiser implementations and
a DenoiseManager class for managing denoiser lifecycle within sessions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Type

import numpy as np

from .base import DenoiserBase


# Global registry mapping type_name -> denoiser class
_REGISTRY: dict[str, Type[DenoiserBase]] = {}


class DenoiseStatus(str, Enum):
    """Status of denoising operation."""

    OK = "ok"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class DenoiseResult:
    """Result of a denoising operation."""

    audio: np.ndarray
    applied: bool
    status: DenoiseStatus
    denoise_type: str
    denoise_mix: float
    latency_ms: float | None = None
    error: str | None = None
    skip_reason: str | None = None


def register_denoiser(cls: Type[DenoiserBase]) -> Type[DenoiserBase]:
    """Decorator to register a denoiser class.

    Args:
        cls: Denoiser class to register.

    Returns:
        The same class (allows use as decorator).

    Raises:
        TypeError: If cls is not a subclass of DenoiserBase.
        ValueError: If type_name is missing or already registered.
    """
    if not issubclass(cls, DenoiserBase):
        raise TypeError(f"{cls.__name__} must inherit from DenoiserBase")

    type_name = getattr(cls, "type_name", None)
    if not type_name or not isinstance(type_name, str):
        raise ValueError(f"{cls.__name__} must define a non-empty type_name")

    if type_name in _REGISTRY:
        existing = _REGISTRY[type_name]
        if existing is not cls:
            raise ValueError(
                f"Denoiser type '{type_name}' already registered by {existing.__name__}"
            )

    _REGISTRY[type_name] = cls
    return cls


def list_denoisers() -> list[str]:
    """Get list of all registered denoiser type names."""
    return sorted(_REGISTRY.keys())


def get_denoiser_class(type_name: str) -> Type[DenoiserBase] | None:
    """Get denoiser class by type name.

    Args:
        type_name: The denoiser type identifier.

    Returns:
        The denoiser class, or None if not found.
    """
    return _REGISTRY.get(type_name)


class DenoiseManager:
    """Session-scoped denoiser controller with wet/dry mixing support.

    Manages denoiser lifecycle for a single WebSocket session, handling
    configuration updates, instance creation, and audio processing with
    optional wet/dry mixing.

    Attributes:
        sample_rate: Audio sample rate in Hz.
        config: Current configuration dictionary.
    """

    def __init__(self, sample_rate: int = 16000, config: dict[str, Any] | None = None):
        """Initialize manager.

        Args:
            sample_rate: Audio sample rate in Hz.
            config: Initial configuration dictionary.
        """
        self.sample_rate = int(sample_rate)
        self.config = dict(config or {})
        self._denoiser: DenoiserBase | None = None
        self._denoise_type: str = "none"
        self._init_denoiser()

    def _init_denoiser(self) -> None:
        """Initialize denoiser based on current config."""
        denoise_type = str(self.config.get("denoise_type", "none")).strip().lower()
        self._create_denoiser(denoise_type)

    def _create_denoiser(self, denoise_type: str) -> None:
        """Create denoiser instance for given type.

        Args:
            denoise_type: The denoiser type to create.

        Raises:
            ValueError: If type is unknown.
        """
        cls = get_denoiser_class(denoise_type)
        if cls is None:
            raise ValueError(
                f"Unknown denoise_type: '{denoise_type}'. "
                f"Available: {', '.join(list_denoisers())}"
            )
        self._denoiser = cls(sample_rate=self.sample_rate, config=self.config)
        self._denoise_type = denoise_type

    @property
    def denoise_type(self) -> str:
        """Current denoiser type name."""
        return self._denoise_type

    def reset(self) -> None:
        """Reset denoiser state."""
        if self._denoiser is not None:
            self._denoiser.reset()

    def update_config(self, config: dict[str, Any]) -> None:
        """Update configuration, recreating denoiser if type changed.

        Args:
            config: New configuration dictionary.

        Raises:
            ValueError: If denoise_type is invalid.
        """
        self.config.update(config or {})
        new_type = str(self.config.get("denoise_type", "none")).strip().lower()

        if new_type != self._denoise_type:
            self._create_denoiser(new_type)
        elif self._denoiser is not None:
            self._denoiser.update_config(self.config)

    def process(self, audio: np.ndarray, reset_state: bool = True) -> DenoiseResult:
        """Process audio segment with denoising.

        Args:
            audio: Input audio as float32 array.
            reset_state: Whether to reset denoiser state before processing.
                         Default True for segment-level processing.

        Returns:
            DenoiseResult containing processed audio and metadata.
        """
        denoise_type = self._denoise_type
        denoise_mix = float(self.config.get("denoise_mix", 1.0))

        # Handle empty audio
        if audio.size == 0:
            return DenoiseResult(
                audio=audio,
                applied=False,
                status=DenoiseStatus.SKIPPED,
                denoise_type=denoise_type,
                denoise_mix=denoise_mix,
                skip_reason="empty_audio",
            )

        # Skip if mix is 0 or type is none
        if denoise_mix <= 0.0 or denoise_type == "none":
            return DenoiseResult(
                audio=audio,
                applied=False,
                status=DenoiseStatus.SKIPPED,
                denoise_type=denoise_type,
                denoise_mix=denoise_mix,
                skip_reason="disabled" if denoise_type == "none" else "mix_zero",
            )

        # Skip if no denoiser available
        if self._denoiser is None:
            return DenoiseResult(
                audio=audio,
                applied=False,
                status=DenoiseStatus.SKIPPED,
                denoise_type=denoise_type,
                denoise_mix=denoise_mix,
                skip_reason="no_denoiser",
            )

        # Reset state if requested (default for segment processing)
        if reset_state:
            self._denoiser.reset()

        # Process with timing
        try:
            t0 = time.perf_counter()
            denoised = self._denoiser.process(audio)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            # Validate output
            denoised = np.asarray(denoised, dtype=np.float32)
            if denoised.size != audio.size:
                raise ValueError(
                    f"Denoiser output length mismatch: {denoised.size} != {audio.size}"
                )
            if np.any(np.isnan(denoised)) or np.any(np.isinf(denoised)):
                raise ValueError("Denoiser output contains NaN or Inf values")

            # Apply wet/dry mix
            if denoise_mix >= 1.0:
                output = denoised
            else:
                output = (1.0 - denoise_mix) * audio + denoise_mix * denoised

            return DenoiseResult(
                audio=output.astype(np.float32),
                applied=True,
                status=DenoiseStatus.OK,
                denoise_type=denoise_type,
                denoise_mix=denoise_mix,
                latency_ms=latency_ms,
            )

        except Exception as exc:
            # On failure, return original audio
            return DenoiseResult(
                audio=audio,
                applied=False,
                status=DenoiseStatus.FAILED,
                denoise_type=denoise_type,
                denoise_mix=denoise_mix,
                error=str(exc),
            )
