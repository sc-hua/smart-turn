"""TEN VAD 封装，基于 ONNX 模型的高性能 VAD。

特点：
- Chunk 大小：256 samples (16ms @ 16kHz)
- 基于 ONNX 模型，通过编译的 Python 扩展模块调用
"""

import os
import sys
import platform
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

TEN_VAD_CHUNK = 256  # 256 samples @ 16 kHz = 16 ms
TEN_VAD_DEFAULT_THRESHOLD = 0.5

# 项目根目录
_ROOT = Path(__file__).resolve().parent.parent


def _find_file(name: str) -> Optional[Path]:
    """在 onnx_model/ 和 assets/ 目录下查找文件。"""
    for subdir in ("onnx_model", "assets"):
        path = _ROOT / subdir / name
        if path.exists():
            return path
    return None


def _get_extension_name() -> str:
    """获取当前平台的 Python 扩展模块文件名。"""
    system, machine = platform.system(), platform.machine().lower()
    ver = f"{sys.version_info.major}{sys.version_info.minor}"
    
    if system == "Linux":
        arch = "aarch64" if machine in ("aarch64", "arm64") else "x86_64"
        return f"ten_vad_python.cpython-{ver}-{arch}-linux-gnu.so"
    elif system == "Darwin":
        return f"ten_vad_python.cpython-{ver}-darwin.so"
    elif system == "Windows":
        return f"ten_vad_python.cp{ver}-win_amd64.pyd"
    raise RuntimeError(f"不支持的平台: {system} {machine}")


class TenVAD:
    """TEN VAD 封装类。"""
    
    def __init__(
        self,
        hop_size: int = TEN_VAD_CHUNK,
        threshold: float = TEN_VAD_DEFAULT_THRESHOLD,
    ):
        self.hop_size = hop_size
        self.threshold = threshold
        self._vad = None
        
        # 查找扩展模块和模型
        ext_path = _find_file(_get_extension_name())
        model_path = _find_file("ten-vad.onnx")
        
        if not ext_path or not model_path:
            raise RuntimeError(
                "无法初始化 TEN VAD。请确保 onnx_model/ 目录中存在 "
                f"{_get_extension_name()} 和 ten-vad.onnx 文件。"
            )
        
        # 添加扩展模块目录到 Python 路径
        ext_dir = str(ext_path.parent)
        if ext_dir not in sys.path:
            sys.path.insert(0, ext_dir)
        
        # C++ 模块使用相对路径 "onnx_model/ten-vad.onnx"
        cwd_model = Path.cwd() / "onnx_model" / "ten-vad.onnx"
        if not cwd_model.exists():
            raise RuntimeError(
                f"TEN VAD 模型文件不存在: {cwd_model}，"
                "请确保在工作目录下存在 onnx_model/ten-vad.onnx"
            )
        
        import ten_vad_python
        self._vad = ten_vad_python.VAD(hop_size=hop_size, threshold=threshold)
    
    def reset(self):
        """重置 VAD 状态。"""
        import ten_vad_python
        self._vad = ten_vad_python.VAD(
            hop_size=self.hop_size, threshold=self.threshold
        )
    
    def process(self, audio_int16: np.ndarray) -> Tuple[float, int]:
        """处理一帧音频数据。
        
        Args:
            audio_int16: int16 格式音频，长度必须等于 hop_size
            
        Returns:
            (probability, flag): 语音概率和二值标志
        """
        audio_int16 = np.squeeze(audio_int16).astype(np.int16)
        if audio_int16.shape[0] != self.hop_size:
            raise ValueError(
                f"音频长度必须为 {self.hop_size}，实际为 {audio_int16.shape[0]}"
            )
        prob, is_voice = self._vad.process(audio_int16)
        return float(prob), 1 if is_voice else 0
    
    def prob(self, chunk_int16: np.ndarray) -> float:
        """计算语音概率（与 SileroVAD 接口兼容）。"""
        probability, _ = self.process(chunk_int16)
        return probability
    
    def prob_f32(self, chunk_f32: np.ndarray) -> float:
        """计算语音概率（float32 输入）。"""
        chunk_int16 = (chunk_f32 * 32768.0).astype(np.int16)
        return self.prob(chunk_int16)


def ensure_library(path: Optional[str] = None) -> str:
    """确保 TEN VAD 可用。"""
    ext_path = _find_file(_get_extension_name())
    model_path = _find_file("ten-vad.onnx")
    if ext_path and model_path:
        return "onnx"
    raise FileNotFoundError("TEN VAD 不可用。请确保必要文件存在。")

