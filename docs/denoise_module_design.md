# 降噪模块设计文档

## 概述

本文档描述了在 VAD -> Smart Turn 端点预测管道之间添加的轻量级降噪模块的设计与实现。

### 背景

当前系统流程：
1. VAD（Silero/TEN/FSMN）检测语音片段
2. 语音段结束后，将原始音频发送给 Smart Turn 模型进行端点预测
3. **问题**：原声直接进入 Smart Turn，噪声可能导致误判

### 解决方案

在 VAD 检测到语音段结束后、发送给 Smart Turn 之前，对音频进行轻量级降噪处理。

## 架构设计

### 模块结构

```
denoise/
├── __init__.py          # 模块入口，导出公共接口
├── base.py              # DenoiserBase 抽象基类
├── registry.py          # 注册机制 + DenoiseManager + DenoiseResult
├── none.py              # 无操作降噪器（默认）
├── dtln.py              # DTLN ONNX 降噪器
└── noisereduce_impl.py  # noisereduce 库降噪器
```

### 核心组件

#### 1. DenoiserBase（基类）

```python
class DenoiserBase(ABC):
    type_name: str = "base"  # 唯一标识符

    def __init__(self, sample_rate: int = 16000, config: dict = None): ...
    def reset(self) -> None: ...  # 重置内部状态
    def update_config(self, config: dict) -> None: ...  # 更新配置
    @abstractmethod
    def process(self, audio: np.ndarray) -> np.ndarray: ...  # 处理音频
```

#### 2. DenoiseManager（会话管理器）

- 每个 WebSocket 会话维护独立的降噪器实例
- 支持运行时切换降噪类型
- 支持 wet/dry 混合比例
- 默认 segment-level reset（每段处理前重置状态）

#### 3. DenoiseResult（处理结果）

```python
@dataclass
class DenoiseResult:
    audio: np.ndarray       # 处理后的音频
    applied: bool           # 是否实际应用了降噪
    status: DenoiseStatus   # ok / skipped / failed
    denoise_type: str       # 使用的降噪类型
    denoise_mix: float      # wet/dry 混合比例
    latency_ms: float       # 处理耗时（毫秒）
    error: str              # 错误信息（如果失败）
    skip_reason: str        # 跳过原因（如果跳过）
```

### 数据流

```
VAD Pipeline
     │
     ▼
┌─────────────────────────────────────────────────┐
│ _process_segment()                              │
│                                                 │
│   audio (float32, 16kHz)                       │
│          │                                      │
│          ▼                                      │
│   ┌──────────────────┐                         │
│   │ DenoiseManager   │                         │
│   │   .process()     │                         │
│   └────────┬─────────┘                         │
│            │                                    │
│   ┌────────▼─────────┐                         │
│   │ DenoiseResult    │                         │
│   │  .audio          │                         │
│   │  .applied        │                         │
│   │  .status         │                         │
│   └────────┬─────────┘                         │
│            │                                    │
│            ▼                                    │
│   predict_endpoint(denoised_audio)             │
│                                                 │
└─────────────────────────────────────────────────┘
     │
     ▼
Prediction Result (with denoise metadata)
```

## 已实现功能

### 1. 降噪器类型

| 类型 | 状态 | 说明 |
|------|------|------|
| `none` | ✅ 已实现 | 无操作，直接返回原始音频（默认） |
| `dtln` | ✅ 已实现 | DTLN ONNX 模型，需要模型文件 |
| `noisereduce` | ✅ 已实现 | 基于 noisereduce 库的谱门控降噪 |

### 2. 配置参数

#### WebSocket 配置字段

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `denoise_type` | string | `"none"` | 降噪类型 |
| `denoise_mix` | float | `1.0` | wet/dry 混合比例 (0-1) |

#### DTLN 特定配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dtln_model_dir` | string | `"onnx_model/dtln"` | 模型目录路径 |
| `dtln_block_len` | int | `512` | FFT 块长度（样本数） |
| `dtln_block_shift` | int | `128` | 块移位（样本数） |
| `dtln_pad_samples` | int | `256` | 起始填充样本数 |
| `dtln_providers` | string | `"CPUExecutionProvider"` | ONNX 执行提供者 |

#### noisereduce 特定配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `noisereduce_stationary` | bool | `True` | 使用稳态噪声模式 |
| `noisereduce_prop_decrease` | float | `0.8` | 噪声抑制比例 |
| `noisereduce_n_fft` | int | `512` | FFT 大小 |
| `noisereduce_hop_length` | int | `128` | 帧移 |
| `noisereduce_n_std_thresh` | float | `1.5` | 阈值标准差倍数 |

### 3. WebSocket 协议扩展

#### Ready 消息（新增字段）

```json
{
  "type": "ready",
  "defaults": {
    "denoise_type": "none",
    "denoise_mix": 1.0
  },
  "supported_denoise_types": ["dtln", "none", "noisereduce"]
}
```

#### Config 消息（新增字段）

```json
{
  "denoise_type": "dtln",
  "denoise_mix": 0.8
}
```

#### Prediction 消息（新增字段）

```json
{
  "type": "prediction",
  "denoise_type": "dtln",
  "denoise_mix": 0.8,
  "denoise_applied": true,
  "denoise_status": "ok",
  "denoise_ms": 12.34,
  "denoise_error": null,
  "denoise_skip_reason": null
}
```

### 4. 客户端参数

```bash
python ws_mic_client.py \
  --denoise-type dtln \
  --denoise-mix 0.8
```

## 未实现功能

### 1. 其他降噪方案

| 类型 | 状态 | 说明 |
|------|------|------|
| `rnnoise` | ❌ 未实现 | RNNoise 需要 48kHz，需重采样 |
| `speex` | ❌ 未实现 | SpeexDSP，需要编译绑定 |
| `webrtc_ns` | ❌ 未实现 | WebRTC NS，库维护状态不佳 |

### 2. 高级功能

| 功能 | 状态 | 说明 |
|------|------|------|
| 启动时模型验证 | ❌ 未实现 | 可选的 `--validate-denoise` 启动参数 |
| 噪声参考提取 | ❌ 未实现 | 使用 pre-speech buffer 作为噪声样本 |
| 跨段状态保持 | ❌ 未实现 | `denoise_stateful` 参数支持连续对话 |
| GPU 加速 | ⚠️ 部分支持 | DTLN 支持 CUDA，但未充分测试 |

### 3. 测试与监控

| 功能 | 状态 | 说明 |
|------|------|------|
| 单元测试 | ❌ 未实现 | 需要添加降噪模块的测试用例 |
| 性能基准测试 | ❌ 未实现 | 不同降噪方案的延迟/CPU 对比 |
| A/B 测试框架 | ❌ 未实现 | 比较降噪前后的端点预测准确率 |

## 设计决策

### 1. Segment-level Reset（默认）

**决策**：每次处理新的语音段时，默认重置降噪器的内部状态。

**原因**：
- VAD 分割后的语音段是独立的，段与段之间不连续
- 保留跨段状态可能将前一段的噪声/语音上下文带入后一段
- 对于端点检测场景，独立处理每段更合理

### 2. 错误处理策略

**决策**：降噪失败时静默回退到原始音频，但在返回结果中明确标记状态。

**原因**：
- 不应因降噪失败而阻断整个预测流程
- 通过 `denoise_status` 和 `denoise_error` 字段提供完整的状态信息
- 客户端可根据状态决定是否重试或调整配置

### 3. 输出验证

**决策**：DenoiseManager 验证降噪器输出的长度和数值有效性。

**验证项**：
- 输出长度必须与输入相同
- 输出不能包含 NaN 或 Inf 值

### 4. 模块化注册机制

**决策**：使用装饰器 `@register_denoiser` 自动注册降噪器类。

**优点**：
- 添加新降噪器只需创建新文件并使用装饰器
- 无需修改核心代码
- 便于动态发现可用降噪器

## 使用示例

### 服务端

```python
from denoise import DenoiseManager, list_denoisers

# 查看可用降噪器
print(list_denoisers())  # ['dtln', 'none', 'noisereduce']

# 创建管理器
manager = DenoiseManager(sample_rate=16000, config={
    "denoise_type": "dtln",
    "denoise_mix": 0.8,
})

# 处理音频段
result = manager.process(audio_float32)
if result.applied:
    print(f"降噪成功，耗时 {result.latency_ms:.2f}ms")
    denoised_audio = result.audio
else:
    print(f"降噪跳过/失败: {result.skip_reason or result.error}")
```

### 客户端

```bash
# 使用 DTLN 降噪
python ws_mic_client.py --denoise-type dtln --denoise-mix 0.8

# 使用 noisereduce 降噪
python ws_mic_client.py --denoise-type noisereduce --denoise-mix 1.0

# 禁用降噪（默认）
python ws_mic_client.py --denoise-type none
```

## DTLN 模型准备

DTLN 降噪器需要预训练的 ONNX 模型文件。

### 下载模型

从 [DTLN GitHub](https://github.com/breizhn/DTLN) 获取或转换模型：

```bash
mkdir -p onnx_model/dtln
# 将 model_1.onnx 和 model_2.onnx 放入该目录
```

### 模型要求

- `model_1.onnx`：频域处理（幅度掩码）
- `model_2.onnx`：时域处理（波形增强）
- 采样率：16kHz
- 块长度：512 样本（32ms）
- 块移位：128 样本（8ms）

## 性能特性

### DTLN

- 算法延迟：32ms
- 推理延迟：<8ms/块（CPU）
- 模型大小：约 2MB（两个模型总计）

### noisereduce

- 非实时（批处理）
- 适合离线处理或较长语音段
- 依赖：`pip install noisereduce`

## 后续改进建议

1. **添加 RNNoise 支持**：需要实现 16kHz <-> 48kHz 重采样
2. **噪声参考功能**：利用 VAD 的 pre-speech buffer 提取噪声特征
3. **性能优化**：考虑使用 ONNX Runtime 的并行执行模式
4. **可观测性**：添加降噪效果的指标监控（SNR 改善等）
