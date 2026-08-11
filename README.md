# arktts

Audio8 TTS Preview 0.6B 的 Go 实现，基于 ONNX Runtime 在 CPU 上做零样本语音克隆与文本转语音。

对应上游 Python 版本 [Audio8_TTS/onnx_runtime](https://github.com/Audio8-AI/Audio8_TTS/tree/master/onnx_runtime)，模型权重相同（INT4 DualAR + FP16 codec），用纯 Go 重写推理与服务层，无 PyTorch / Transformers 依赖。

## 特性

- 纯 Go 实现：prompt 构造、ONNX 推理、WAV 编码、HTTP 服务
- ONNX Runtime CPU 执行，INT4 权重 + FP16 激活，约 1 GiB 常驻内存
- 零样本声音克隆：上传 0.5–30 秒参考音频即可注册新 voice
- HTTP API + OpenAI 兼容端点 + 流式 PCM
- 多语言：粤、中、荷、英、法、德、意、日、韩、波兰、西

## 目录结构

```text
arktts/
├── cmd/
│   ├── arktts/            # CLI 入口
│   ├── server/            # HTTP 服务入口
│   ├── model/             # 模型文件（hf download 下载，不入库）
│   ├── onnxruntime-*/     # ONNX Runtime 动态库（不入库）
│   └── voices/            # 已注册 voice 数据（不入库）
├── audio/                 # WAV / PCM / 重采样
├── config/                # runtime_manifest 解析
├── inference/             # ONNX 引擎、采样器、tensor、PCG64
├── onnxruntime/           # onnxruntime_go 绑定封装
├── prompt/                # PromptBuilder
├── registration/          # voice 注册（codec encoder）
├── runtime/               # 顶层 Runtime 编排
├── service/               # HTTP handlers / 路由 / 流式
├── tokenizer/             # tokenizer 封装
├── voices/                # VoiceStore（npy 读写）
├── Makefile
└── go.mod
```

## 前置依赖

- Go 1.26.5+
- ONNX Runtime 动态库（macOS arm64 已测试，1.28.0）
- 模型文件（约 572 MiB 在线模型，968 MiB 含注册 encoder）

### 下载模型与运行时

```bash
cd cmd
python3 -m pip install -U "huggingface_hub[cli]"
hf download Audio8/Audio8-TTS-Preview-0.6B-ONNX-INT4 --local-dir model
```

从 [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases) 下载对应平台的预编译包，解压到 `cmd/onnxruntime-osx-arm64-1.28.0/`（路径需与 `defaultOnnxLibPath()` 一致）。模型目录布局：

```text
cmd/model/
├── slow_ar_int4.onnx(.data)
├── fast_ar_int4.onnx(.data)
├── codec_decoder_fp16.onnx(.data)
├── runtime_manifest.json
├── tokenizer/tokenizer.json
└── registration/
    ├── codec_encoder_fp16.onnx(.data)
    └── registration_manifest.json
```

## 构建

```bash
make            # 构建 cli 和 server
make cli        # 仅 cmd/arktts-cli
make server     # 仅 cmd/arktts-server
make clean      # 清理二进制
```

二进制输出到 `cmd/arktts-cli` 与 `cmd/arktts-server`，需在 `cmd/` 目录下运行（默认相对路径 `model` / `voices` / `onnxruntime-...` 均基于此目录）。

## 使用

### CLI

```bash
cd cmd
./arktts-cli --list                                 # 列出已注册 voice
./arktts-cli --text "你好" --voice speaker_a        # 构造 prompt 矩阵
./arktts-cli --synth --text "你好" --voice speaker_a --output out.wav
```

主要参数：`--model-dir`、`--voices-dir`、`--lib`、`--text`、`--voice`、`--synth`、`--output`、`--max-new-tokens`、`--temperature`、`--top-p`、`--top-k`、`--seed`、`--threads`。

### Server

```bash
cd cmd
./arktts-server
# 默认监听 127.0.0.1:8024
```

环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `ARKTTS_MODEL_DIR` | `model` | 模型目录 |
| `ARKTTS_VOICES_DIR` | `voices` | voice 目录 |
| `ARKTTS_REGISTRATION_DIR` | `$MODEL_DIR/registration` | 注册 encoder 目录 |
| `ARKTTS_THREADS` | `5` | ONNX Runtime CPU 线程数 |
| `ARKTTS_ONNX_LIB` | 自动检测 | ONNX Runtime 动态库路径 |
| `HOST` | `127.0.0.1` | 监听地址 |
| `PORT` | `8024` | 监听端口 |

### HTTP API

```bash
# 健康检查
curl http://127.0.0.1:8024/api/health

# 列出 voice
curl http://127.0.0.1:8024/api/voices

# 合成
curl http://127.0.0.1:8024/api/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"Welcome to arktts.","voice_name":"speaker_a","max_new_tokens":256}' \
  -o out.wav

# OpenAI 兼容
curl http://127.0.0.1:8024/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"arktts","input":"Welcome to arktts.","voice":"speaker_a","response_format":"wav"}' \
  -o openai.wav

# 流式 PCM
curl -N http://127.0.0.1:8024/api/tts/stream?text=...&voice_name=speaker_a --output stream.pcm

# 注册新 voice
curl http://127.0.0.1:8024/api/voices/register \
  -F 'audio=@reference.wav' \
  -F 'text=参考录音的精确转写' \
  -F 'name=speaker_b'
```

完整路由：`/api/health`、`/api/voices`、`/api/tts`、`/api/tts/stream`、`/api/tts/cancel`、`/api/voices/register`、`/api/registration/status`、`/api/system`、`/api/runtime/reload`、`/v1/audio/speech`，以及 `/` 的主页。

## 备注

- 推理请求串行化（ONNX session 的并发 `Run` 不保证线程安全），与 Python 版 `request_lock` 一致。
- 注册 voice 时会先释放在线 session 再加载 codec encoder，峰值内存约 1.55 GiB。
- INT4 量化可能改变采样 token 序列，跨语言/voice 部署前建议各自评估。

## 许可

代码与模型权重采用 Apache License 2.0，详见上游 [NOTICE](https://github.com/Audio8-AI/Audio8_TTS/blob/master/NOTICE)。DualAR 架构源自 [Fish Audio S2 Pro](https://github.com/fishaudio/fish-speech)。
