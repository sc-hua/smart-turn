"""TEN VAD 封装，基于 ONNX 模型的高性能 VAD。

TEN VAD 是一个实时语音活动检测系统，相比 Silero VAD 具有更低的延迟和更高的精度。
特点：
- Chunk 大小：256 samples (16ms @ 16kHz)
- 输出：(probability, flag)
- 基于 ONNX 模型，通过编译的 Python 扩展模块调用

支持两种模式：
1. ONNX 模式（推荐）：使用编译好的 ten_vad_python 扩展模块
2. C 库模式（备选）：使用预编译的动态库（仅限特定平台）
"""

import os
import sys
import platform
from typing import Optional, Tuple

import numpy as np

# TEN VAD 默认 chunk 大小
TEN_VAD_CHUNK = 256  # 256 samples @ 16 kHz = 16 ms

# 默认阈值
TEN_VAD_DEFAULT_THRESHOLD = 0.5

# 获取项目根目录
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_MODULE_DIR)
_ASSETS_DIR = os.path.join(_PROJECT_ROOT, "assets")

# ONNX 模型路径
TEN_VAD_ONNX_MODEL_PATH = os.path.join(_ASSETS_DIR, "ten-vad.onnx")

# 环境变量
TEN_VAD_LIB_PATH_ENV = "TEN_VAD_LIB_PATH"
TEN_VAD_MODE_ENV = "TEN_VAD_MODE"  # "onnx" 或 "ctypes"


def _get_python_extension_name() -> str:
    """获取当前平台的 Python 扩展模块文件名。"""
    system = platform.system()
    machine = platform.machine().lower()
    
    # Python 版本
    py_version = f"{sys.version_info.major}{sys.version_info.minor}"
    
    if system == "Linux":
        if machine in ("aarch64", "arm64"):
            return f"ten_vad_python.cpython-{py_version}-aarch64-linux-gnu.so"
        elif machine in ("x86_64", "amd64"):
            return f"ten_vad_python.cpython-{py_version}-x86_64-linux-gnu.so"
    elif system == "Darwin":
        if machine in ("arm64", "aarch64"):
            return f"ten_vad_python.cpython-{py_version}-darwin.so"
        else:
            return f"ten_vad_python.cpython-{py_version}-darwin.so"
    elif system == "Windows":
        return f"ten_vad_python.cp{py_version}-win_amd64.pyd"
    
    raise RuntimeError(f"不支持的平台: {system} {machine}")


def _find_onnx_extension() -> Optional[str]:
    """查找 ONNX Python 扩展模块。
    
    Returns:
        扩展模块所在目录，如果找不到返回 None
    """
    ext_name = _get_python_extension_name()
    
    # 检查 assets 目录
    ext_path = os.path.join(_ASSETS_DIR, ext_name)
    if os.path.exists(ext_path):
        return _ASSETS_DIR
    
    # 检查 vad 目录
    ext_path = os.path.join(_MODULE_DIR, ext_name)
    if os.path.exists(ext_path):
        return _MODULE_DIR
    
    return None


def _check_onnx_model() -> bool:
    """检查 ONNX 模型是否存在。"""
    return os.path.exists(TEN_VAD_ONNX_MODEL_PATH)


class TenVAD:
    """TEN VAD 封装类，提供与 SileroVAD 类似的接口。
    
    优先使用 ONNX 模式（编译的 Python 扩展），如果不可用则回退到 ctypes 模式。
    
    Args:
        hop_size: 每次处理的样本数，默认 256 (16ms @ 16kHz)
        threshold: VAD 阈值，用于内部 flag 计算，默认 0.5
    """
    
    def __init__(
        self,
        hop_size: int = TEN_VAD_CHUNK,
        threshold: float = TEN_VAD_DEFAULT_THRESHOLD,
    ):
        self.hop_size = hop_size
        self.threshold = threshold
        self._mode = None
        self._vad = None
        
        # 尝试初始化
        mode = os.getenv(TEN_VAD_MODE_ENV, "auto").lower()
        
        if mode == "auto":
            # 优先尝试 ONNX 模式
            if self._try_init_onnx():
                return
            # 回退到 ctypes 模式
            if self._try_init_ctypes():
                return
            raise RuntimeError(
                "无法初始化 TEN VAD。"
                "请确保 assets/ 目录中存在 ten_vad_python.*.so 和 ten-vad.onnx 文件，"
                "或者设置 TEN_VAD_LIB_PATH 环境变量指定动态库路径。"
            )
        elif mode == "onnx":
            if not self._try_init_onnx():
                raise RuntimeError("ONNX 模式初始化失败")
        elif mode == "ctypes":
            if not self._try_init_ctypes():
                raise RuntimeError("ctypes 模式初始化失败")
        else:
            raise ValueError(f"无效的 TEN_VAD_MODE: {mode}")
    
    def _try_init_onnx(self) -> bool:
        """尝试使用 ONNX 模式初始化。"""
        ext_dir = _find_onnx_extension()
        if ext_dir is None:
            return False
        
        if not _check_onnx_model():
            return False
        
        try:
            # 将扩展模块目录添加到 Python 路径
            if ext_dir not in sys.path:
                sys.path.insert(0, ext_dir)
            
            # C++ 模块使用相对路径 "onnx_model/ten-vad.onnx" 查找模型
            # 需要在当前工作目录下创建符号链接
            cwd_onnx_model_dir = os.path.join(os.getcwd(), "onnx_model")
            if not os.path.exists(cwd_onnx_model_dir):
                os.makedirs(cwd_onnx_model_dir, exist_ok=True)
            
            cwd_onnx_link = os.path.join(cwd_onnx_model_dir, "ten-vad.onnx")
            if not os.path.exists(cwd_onnx_link):
                try:
                    os.symlink(TEN_VAD_ONNX_MODEL_PATH, cwd_onnx_link)
                except OSError:
                    # 如果符号链接失败，尝试复制文件
                    import shutil
                    shutil.copy2(TEN_VAD_ONNX_MODEL_PATH, cwd_onnx_link)
            
            import ten_vad_python
            self._vad = ten_vad_python.VAD(
                hop_size=self.hop_size,
                threshold=self.threshold
            )
            self._mode = "onnx"
            print(f"TenVAD initialized in ONNX mode from: {ext_dir}")
            return True
        except Exception as e:
            print(f"ONNX 模式初始化失败: {e}")
            return False
    
    def _try_init_ctypes(self) -> bool:
        """尝试使用 ctypes 模式初始化。"""
        try:
            from ctypes import CDLL, POINTER, c_float, c_int, c_int32, c_size_t, c_void_p
            
            lib_path = self._find_library()
            if lib_path is None:
                return False
            
            self._vad_library = CDLL(lib_path)
            self._vad_handler = c_void_p(0)
            self._out_probability = c_float()
            self._out_flags = c_int32()
            
            # 配置函数签名
            self._vad_library.ten_vad_create.argtypes = [
                POINTER(c_void_p), c_size_t, c_float
            ]
            self._vad_library.ten_vad_create.restype = c_int
            
            self._vad_library.ten_vad_destroy.argtypes = [POINTER(c_void_p)]
            self._vad_library.ten_vad_destroy.restype = c_int
            
            self._vad_library.ten_vad_process.argtypes = [
                c_void_p, c_void_p, c_size_t, POINTER(c_float), POINTER(c_int32)
            ]
            self._vad_library.ten_vad_process.restype = c_int
            
            # 创建 handler
            ret = self._vad_library.ten_vad_create(
                POINTER(c_void_p)(self._vad_handler),
                c_size_t(self.hop_size),
                c_float(self.threshold),
            )
            if ret != 0:
                return False
            
            self._mode = "ctypes"
            print(f"TenVAD initialized in ctypes mode from: {lib_path}")
            return True
        except Exception as e:
            print(f"ctypes 模式初始化失败: {e}")
            return False
    
    def _find_library(self) -> Optional[str]:
        """查找动态库路径。"""
        # 检查环境变量
        env_path = os.getenv(TEN_VAD_LIB_PATH_ENV)
        if env_path and os.path.exists(env_path):
            return env_path
        
        # 检查 assets 目录
        system = platform.system()
        machine = platform.machine().lower()
        
        lib_map = {
            ("Linux", "x86_64"): "libten_vad.so",
            ("Linux", "amd64"): "libten_vad.so",
            ("Linux", "aarch64"): "libten_vad_aarch64.so",
            ("Darwin", "x86_64"): "libten_vad.dylib",
            ("Darwin", "arm64"): "libten_vad.dylib",
            ("Windows", "amd64"): "ten_vad.dll",
        }
        
        key = (system, machine)
        if key in lib_map:
            lib_path = os.path.join(_ASSETS_DIR, lib_map[key])
            if os.path.exists(lib_path):
                return lib_path
        
        return None
    
    def __del__(self):
        """析构时清理资源。"""
        if self._mode == "ctypes" and hasattr(self, "_vad_library"):
            try:
                from ctypes import POINTER, c_void_p
                self._vad_library.ten_vad_destroy(
                    POINTER(c_void_p)(self._vad_handler)
                )
            except Exception:
                pass
    
    def reset(self):
        """重置 VAD 状态。"""
        if self._mode == "onnx":
            # ONNX 模式：重新创建 VAD 实例
            import ten_vad_python
            self._vad = ten_vad_python.VAD(
                hop_size=self.hop_size,
                threshold=self.threshold
            )
        elif self._mode == "ctypes":
            # ctypes 模式：重建 handler
            from ctypes import POINTER, c_void_p, c_size_t, c_float
            self._vad_library.ten_vad_destroy(
                POINTER(c_void_p)(self._vad_handler)
            )
            self._vad_handler = c_void_p(0)
            self._vad_library.ten_vad_create(
                POINTER(c_void_p)(self._vad_handler),
                c_size_t(self.hop_size),
                c_float(self.threshold),
            )
    
    def process(self, audio_int16: np.ndarray) -> Tuple[float, int]:
        """处理一帧音频数据。
        
        Args:
            audio_int16: int16 格式的音频数据，长度必须等于 hop_size
            
        Returns:
            (probability, flag): 语音概率和二值标志
        """
        audio_int16 = np.squeeze(audio_int16).astype(np.int16)
        
        if audio_int16.shape[0] != self.hop_size:
            raise ValueError(
                f"音频长度必须为 {self.hop_size}，实际为 {audio_int16.shape[0]}"
            )
        
        if self._mode == "onnx":
            prob, is_voice = self._vad.process(audio_int16)
            return float(prob), 1 if is_voice else 0
        elif self._mode == "ctypes":
            from ctypes import POINTER, c_float, c_int32, c_size_t, c_void_p
            
            data_pointer = c_void_p(audio_int16.__array_interface__["data"][0])
            self._vad_library.ten_vad_process(
                self._vad_handler,
                data_pointer,
                c_size_t(self.hop_size),
                POINTER(c_float)(self._out_probability),
                POINTER(c_int32)(self._out_flags),
            )
            return self._out_probability.value, self._out_flags.value
        else:
            raise RuntimeError("VAD 未正确初始化")
    
    def prob(self, chunk_int16: np.ndarray) -> float:
        """计算语音概率（与 SileroVAD 接口兼容）。
        
        Args:
            chunk_int16: int16 格式的音频数据
            
        Returns:
            语音概率 (0.0 ~ 1.0)
        """
        probability, _ = self.process(chunk_int16)
        return probability
    
    def prob_f32(self, chunk_f32: np.ndarray) -> float:
        """计算语音概率（float32 输入，与 SileroVAD 接口兼容）。
        
        Args:
            chunk_f32: float32 格式的音频数据 (-1.0 ~ 1.0)
            
        Returns:
            语音概率 (0.0 ~ 1.0)
        """
        chunk_int16 = (chunk_f32 * 32768.0).astype(np.int16)
        return self.prob(chunk_int16)


def ensure_library(path: Optional[str] = None) -> str:
    """确保 TEN VAD 可用（兼容旧接口）。
    
    Returns:
        模式字符串 ("onnx" 或 "ctypes")
    """
    ext_dir = _find_onnx_extension()
    if ext_dir and _check_onnx_model():
        return "onnx"
    
    # 检查 ctypes 库
    system = platform.system()
    machine = platform.machine().lower()
    
    lib_map = {
        ("Linux", "x86_64"): "libten_vad.so",
        ("Darwin", "arm64"): "libten_vad.dylib",
    }
    
    key = (system, machine)
    if key in lib_map:
        lib_path = os.path.join(_ASSETS_DIR, lib_map[key])
        if os.path.exists(lib_path):
            return "ctypes"
    
    raise FileNotFoundError(
        "TEN VAD 不可用。请确保 assets/ 目录中存在必要的文件。"
    )
