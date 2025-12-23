"""Denoise module for audio enhancement before endpoint prediction.

This module provides a modular denoising pipeline that can be inserted between
VAD segment detection and Smart Turn endpoint prediction to improve accuracy.

Supported denoisers:
- none: No-op passthrough (default)
- dtln: DTLN ONNX model (16kHz, ~32ms latency, requires model files)
- noisereduce: Spectral gating based (pip install noisereduce)

Usage:
    from denoise import DenoiseManager, list_denoisers

    # Check available denoisers
    print(list_denoisers())  # ['dtln', 'none', 'noisereduce']

    # Create manager with config
    manager = DenoiseManager(sample_rate=16000, config={
        "denoise_type": "dtln",
        "denoise_mix": 0.8,
    })

    # Process audio segment
    result = manager.process(audio_float32)
    if result.applied:
        denoised_audio = result.audio
"""

from .base import DenoiserBase
from .registry import (
    DenoiseManager,
    DenoiseResult,
    DenoiseStatus,
    get_denoiser_class,
    list_denoisers,
    register_denoiser,
)

# Import implementations to trigger registration
from . import none as _none  # noqa: F401
from . import dtln as _dtln  # noqa: F401
from . import noisereduce_impl as _noisereduce  # noqa: F401

__all__ = [
    "DenoiserBase",
    "DenoiseManager",
    "DenoiseResult",
    "DenoiseStatus",
    "get_denoiser_class",
    "list_denoisers",
    "register_denoiser",
]
