# 降噪方案调研报告

## 调研背景

为 realtime-vad 项目选择适合的轻量级降噪方案，用于在 VAD 分段后、Smart Turn 端点预测前对音频进行预处理。

### 需求约束

- **采样率**：16kHz 单声道
- **延迟要求**：< 50ms
- **CPU 友好**：适合实时处理
- **易集成**：pip 安装或 ONNX 模型

## 调研方案汇总

### 1. DTLN (Dual-signal Transformation LSTM Network)

**项目地址**：https://github.com/breizhn/DTLN

| 维度 | 评估 |
|------|------|
| Python 包 | 无官方 pip，需手动集成 |
| 采样率 | ✅ 原生 16kHz |
| 实时性能 | ✅ 1.13ms/块（老 CPU），算法延迟 32ms |
| ONNX 支持 | ✅ 提供 ONNX 模型 |
| GitHub 星数 | ~676 stars |
| 维护状态 | 研究型项目，稳定 |

**优点**：
- 专为 16kHz 设计，无需重采样
- 提供 ONNX 模型，跨平台部署
- 推理速度快，满足实时需求
- 有官方的实时处理脚本参考

**缺点**：
- 需要手动下载/转换模型
- 双模型架构（model_1 + model_2）增加复杂度

**结论**：✅ **推荐，已实现**

---

### 2. noisereduce

**项目地址**：https://github.com/timsainb/noisereduce

| 维度 | 评估 |
|------|------|
| Python 包 | ✅ `pip install noisereduce` |
| 采样率 | ✅ 任意采样率 |
| 实时性能 | ⚠️ 批处理导向，非流式 |
| ONNX 支持 | ❌ 无 |
| GitHub 星数 | ~1.8k stars |
| 维护状态 | 活跃，最新版本 2024-10 |

**优点**：
- pip 一键安装，零配置
- 支持稳态/非稳态噪声
- 对 VAD 分段后的音频块处理友好

**缺点**：
- 不适合真正的流式处理
- 需要 STFT，对短段有最小长度要求
- 可能引入"音乐噪声"伪影

**结论**：✅ **推荐作为备选，已实现**

---

### 3. RNNoise

**项目地址**：https://github.com/xiph/rnnoise

| 维度 | 评估 |
|------|------|
| Python 包 | `pip install pyrnnoise` |
| 采样率 | ⚠️ 48kHz（需重采样） |
| 实时性能 | ✅ 帧级实时处理 |
| ONNX 支持 | ❌ 自有二进制格式 |
| GitHub 星数 | ~5.2k stars |
| 维护状态 | 活跃，上游在 Xiph GitLab |

**优点**：
- 成熟的 RNN 降噪方案
- pyrnnoise 有 Python 绑定
- 专为实时设计

**缺点**：
- 需要 16kHz <-> 48kHz 重采样
- 重采样带来额外延迟和 CPU 开销
- 非 ONNX 格式，部署受限

**结论**：⚠️ **可选，因重采样开销暂未实现**

---

### 4. WebRTC Noise Suppression

**项目地址**：https://github.com/xiongyihui/python-webrtc-audio-processing

| 维度 | 评估 |
|------|------|
| Python 包 | `pip install webrtc-audio-processing` |
| 采样率 | ✅ 16kHz |
| 实时性能 | ✅ 10ms 帧处理 |
| ONNX 支持 | ❌ 传统 DSP |
| GitHub 星数 | ~208 stars |
| 维护状态 | ⚠️ PyPI 版本停留在 2018 |

**优点**：
- 工业级验证，专为实时通信设计
- 原生支持 16kHz
- 低延迟

**缺点**：
- Python 绑定版本过旧
- `webrtc-noise-gain` 已归档
- 依赖维护风险高

**结论**：❌ **不推荐，维护状态不佳**

---

### 5. DeepFilterNet

**项目地址**：https://github.com/Rikorose/DeepFilterNet

| 维度 | 评估 |
|------|------|
| Python 包 | `pip install deepfilternet`（需 PyTorch） |
| 采样率 | ⚠️ 48kHz |
| 实时性能 | ✅ 有实时版本 |
| ONNX 支持 | ✅ 有 ONNX 模型 |
| GitHub 星数 | ~3.6k stars |
| 维护状态 | 活跃 |

**优点**：
- 高质量降噪效果
- 有 ONNX 模型
- 支持实时处理

**缺点**：
- 默认 48kHz，需重采样
- 模型较大，不够轻量
- 依赖 PyTorch

**结论**：⚠️ **可选，因采样率和依赖复杂度暂未实现**

---

### 6. SpeexDSP

**项目地址**：https://github.com/TeaPoly/speexdsp-ns-python

| 维度 | 评估 |
|------|------|
| Python 包 | `pip install pyspeex-noise` |
| 采样率 | ✅ 可配置 |
| 实时性能 | ✅ 10ms 帧处理 |
| ONNX 支持 | ❌ 传统 DSP |
| GitHub 星数 | 较少 |
| 维护状态 | 有限 |

**优点**：
- 极轻量的传统 DSP 方案
- CPU 消耗极低
- 支持 16kHz

**缺点**：
- 需要编译依赖（libspeexdsp-dev, swig）
- 输入需要 int16 格式转换
- API 不统一

**结论**：⚠️ **可选，因编译复杂度暂未实现**

---

### 7. NSNet2 (Microsoft)

**项目地址**：https://github.com/microsoft/DNS-Challenge

| 维度 | 评估 |
|------|------|
| Python 包 | 需手动集成 |
| 采样率 | ⚠️ 48kHz |
| 实时性能 | ⚠️ GRU state 管理困难 |
| ONNX 支持 | ✅ 有 ONNX 模型 |
| 维护状态 | 研究项目 |

**优点**：
- Microsoft 官方基线模型
- 有 ONNX 版本

**缺点**：
- 仅支持 48kHz
- ONNX Runtime 难以维持 GRU hidden state
- 主要为离线设计

**结论**：❌ **不推荐**

---

## 最终选择

### 主推方案：DTLN

- 原生 16kHz 支持
- ONNX 模型，跨平台部署
- 满足实时延迟要求
- 有成熟的处理逻辑参考

### 备选方案：noisereduce

- pip 一键安装，零配置
- 适合 VAD 分段后的批处理场景
- 作为 DTLN 模型不可用时的降级方案

### 未来可扩展

- RNNoise（需实现重采样逻辑）
- SpeexDSP（需解决编译依赖）
- DeepFilterNet（需实现重采样 + 轻量化）

## 性能对比预估

| 方案 | 延迟 | CPU 占用 | 降噪效果 | 集成难度 |
|------|------|----------|----------|----------|
| DTLN | 32ms | 低 | 高 | 中 |
| noisereduce | 变化 | 中 | 中 | 低 |
| RNNoise | ~50ms* | 中 | 中高 | 中 |
| SpeexDSP | 10ms | 极低 | 中低 | 高 |
| WebRTC NS | 10ms | 低 | 中 | 高（维护风险） |

*包含重采样开销
