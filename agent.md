# Agent.md

本文件为 AI 编码助手（Agent）在 arktts 项目中工作时的约定与说明。任何 Agent 在修改本项目代码前应先阅读本文件，遵循其中的项目背景与开发约定。

## 项目说明

arktts 是 Audio8 TTS Preview 0.6B 的纯 Go 实现，基于 ONNX Runtime 在 CPU 上提供零样本语音克隆与文本转语音（TTS）。

- 对应上游 Python 版本：[Audio8_TTS/onnx_runtime](https://github.com/Audio8-AI/Audio8_TTS/tree/master/onnx_runtime)
- 模型权重相同（INT4 DualAR + FP16 codec），仅用 Go 重写推理与服务层
- 无 PyTorch / Transformers 依赖，无 cgo / Rust 外部工具链
- 推理请求串行化（ONNX session 并发 `Run` 不保证线程安全），与 Python 版的 `request_lock` 一致

### 核心特性

- ONNX Runtime CPU 执行，INT4 权重 + FP16 激活，约 1 GiB 常驻内存
- 零样本声音克隆：上传 0.5–30 秒参考音频即可注册新 voice
- HTTP API + OpenAI 兼容端点 + 流式 PCM
- 多语言支持：粤、中、荷、英、法、德、意、日、韩、波兰、西

### 关键包结构

| 包 | 职责 |
|---|---|
| `audio` | WAV / PCM / 重采样 |
| `config` | runtime_manifest 解析 |
| `inference` | ONNX 引擎、采样器、tensor、PCG64 |
| `onnxruntime` | onnxruntime_go 绑定封装 |
| `prompt` | PromptBuilder |
| `registration` | voice 注册（codec encoder） |
| `runtime` | 顶层 Runtime 编排 |
| `service` | HTTP handlers / 路由 / 流式 |
| `tokenizer` | tokenizer 封装 |
| `voices` | VoiceStore（npy 读写） |

## 开发约定

### 编码风格

- 坚持纯 Go 风格：短变量名、哨兵错误（sentinel error）、包级预编译（package-level precompilation）。
- 优先采用 Go-idiomatic 的写法（如接口小、错误处理显式、并发用 goroutine + channel）。
- 代码注释使用中文，与项目现有注释保持一致。

### 依赖约束

- 一律使用纯 Go 实现，避免引入 cgo、Rust 工具链或外部原生依赖。
- 不新增对 PyTorch / Transformers 等 Python 生态的运行时依赖。

### 性能与并发

- 优先考虑降低内存分配，提升并发处理能力。
- 保持既有已验证组件的稳定性（如 PCG64），除非有充分理由并经回归验证，否则不做无谓替换。

### 一致性

- 对系统一致性的改动须谨慎，优化前要求进行全面的回归测试。
- 不在需求范围之外擅自改动已验证的模块。

### 构建与运行

```bash
make            # 构建 cli 和 server
make cli        # 仅 cmd/arktts-cli
make server     # 仅 cmd/arktts-server
make clean      # 清理二进制
```

二进制输出到 `cmd/arktts-cli` 与 `cmd/arktts-server`，需在 `cmd/` 目录下运行（默认相对路径 `model` / `voices` / `onnxruntime-...` 均基于此目录）。

### 目录约定

- `cmd/model/`、`cmd/onnxruntime-*/`、`cmd/voices/` 下的文件不入库。
- 模型与运行时为外部下载资源，路径需与 `defaultOnnxLibPath()` 等默认逻辑保持一致。

## 备注

- 注册 voice 时会先释放在线 session 再加载 codec encoder，峰值内存约 1.55 GiB。
- INT4 量化可能改变采样 token 序列，跨语言/voice 部署前建议各自评估。

## 许可

代码与模型权重采用 Apache License 2.0，详见上游 NOTICE。DualAR 架构源自 Fish Audio S2 Pro。