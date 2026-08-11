// Package runtime 集成 tokenizer、PromptBuilder、VoiceStore、manifest 和 ONNX 推理引擎，
// 提供 TTS 推理的统一入口。
package runtime

import (
	"context"
	"fmt"
	"path/filepath"
	stdruntime "runtime"

	"github.com/metaRobin/arktts/config"
	"github.com/metaRobin/arktts/inference"
	"github.com/metaRobin/arktts/prompt"
	"github.com/metaRobin/arktts/tokenizer"
	"github.com/metaRobin/arktts/voices"
)

// Runtime 是 TTS 运行时，集成 tokenizer、PromptBuilder、VoiceStore 和推理引擎。
type Runtime struct {
	manifest *config.RuntimeManifest
	tok      *tokenizer.Tokenizer
	builder  *prompt.PromptBuilder
	voices   *voices.Store
	modelDir string
	engine   *inference.Engine

	// engine 初始化参数，供 ReloadEngine 重新创建时使用
	libPath string
	threads int
}

// New 加载 manifest、tokenizer，创建 PromptBuilder 和 VoiceStore。
// 不初始化推理引擎；调用 InitEngine 加载 ONNX 模型。
func New(modelDir, voicesDir string) (*Runtime, error) {
	manifest, err := config.LoadRuntimeManifest(modelDir)
	if err != nil {
		return nil, fmt.Errorf("load manifest: %w", err)
	}

	tok, err := tokenizer.LoadFromFile(filepath.Join(modelDir, "tokenizer", "tokenizer.json"))
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}

	return &Runtime{
		manifest: manifest,
		tok:      tok,
		builder:  prompt.New(tok, manifest.SemanticBeginID, manifest.NumCodebooks),
		voices:   voices.New(voicesDir, manifest.NumCodebooks),
		modelDir: modelDir,
	}, nil
}

// InitEngine 初始化 ONNX 推理引擎，加载 slow AR、fast AR 和 codec decoder 模型。
// libraryPath 指向 ONNX Runtime 动态库路径。
func (r *Runtime) InitEngine(libraryPath string, threads int) error {
	if r.engine != nil {
		return fmt.Errorf("engine already initialized")
	}
	engine, err := inference.NewEngine(inference.EngineConfig{
		ModelDir:       r.modelDir,
		LibraryPath:    libraryPath,
		Precision:      r.manifest.DefaultPrecision,
		CodecPrecision: r.manifest.DefaultCodecPrecision,
		Threads:        threads,
	}, r.manifest)
	if err != nil {
		return fmt.Errorf("init engine: %w", err)
	}
	r.engine = engine
	r.libPath = libraryPath
	r.threads = threads
	return nil
}

// ReloadEngine 卸载当前推理引擎并重新初始化。
// 用于 /api/runtime/reload：释放 ONNX session 内存 → GC → 重新加载模型。
// 调用方需保证此时无并发推理（service 层用 requestLock 串行化）。
func (r *Runtime) ReloadEngine() error {
	// 1. 关闭旧引擎
	if r.engine != nil {
		if err := r.engine.Close(); err != nil {
			return fmt.Errorf("close old engine: %w", err)
		}
		r.engine = nil
	}

	// 2. 触发 GC 释放底层 ONNX allocator 缓存
	//    对应 Python 的 gc.collect() + malloc_zone_pressure_relief
	stdruntime.GC()

	// 3. 重新加载 manifest（模型可能更新）
	manifest, err := config.LoadRuntimeManifest(r.modelDir)
	if err != nil {
		return fmt.Errorf("reload manifest: %w", err)
	}
	r.manifest = manifest

	// 4. 重建 tokenizer（tokenizer.json 可能更新）
	tok, err := tokenizer.LoadFromFile(filepath.Join(r.modelDir, "tokenizer", "tokenizer.json"))
	if err != nil {
		return fmt.Errorf("reload tokenizer: %w", err)
	}
	r.tok = tok
	r.builder = prompt.New(tok, manifest.SemanticBeginID, manifest.NumCodebooks)

	// 5. 重新创建引擎
	engine, err := inference.NewEngine(inference.EngineConfig{
		ModelDir:       r.modelDir,
		LibraryPath:    r.libPath,
		Precision:      r.manifest.DefaultPrecision,
		CodecPrecision: r.manifest.DefaultCodecPrecision,
		Threads:        r.threads,
	}, r.manifest)
	if err != nil {
		return fmt.Errorf("reinit engine: %w", err)
	}
	r.engine = engine

	// 6. 失效 voice 缓存（voice 目录可能变更）
	r.voices.ClearCache()

	return nil
}

// Close 释放推理引擎和所有资源。
func (r *Runtime) Close() error {
	if r.engine != nil {
		if err := r.engine.Close(); err != nil {
			return err
		}
		r.engine = nil
	}
	return nil
}

// Manifest 返回运行时配置。
func (r *Runtime) Manifest() *config.RuntimeManifest { return r.manifest }

// Tokenizer 返回 tokenizer 实例。
func (r *Runtime) Tokenizer() *tokenizer.Tokenizer { return r.tok }

// PromptBuilder 返回 PromptBuilder 实例。
func (r *Runtime) PromptBuilder() *prompt.PromptBuilder { return r.builder }

// Voices 返回 VoiceStore 实例。
func (r *Runtime) Voices() *voices.Store { return r.voices }

// ModelDir 返回模型目录路径。
func (r *Runtime) ModelDir() string { return r.modelDir }

// BuildPrompt 加载指定 voice 并构造 [1, num_codebooks+1, T] 的 prompt 矩阵。
// voice 数据优先从内存缓存读取，首次调用后后续为零开销。
func (r *Runtime) BuildPrompt(text, voiceName string) ([][][]int64, error) {
	codes, meta, err := r.voices.Load(voiceName)
	if err != nil {
		return nil, err
	}
	return r.builder.Build(text, meta.ReferenceText, codes)
}

// PromptLen 返回 prompt 长度（不构造完整矩阵），用于检查是否超过 max_seq_len。
func (r *Runtime) PromptLen(text, voiceName string) (int, error) {
	codes, meta, err := r.voices.Load(voiceName)
	if err != nil {
		return 0, err
	}
	prefix, suffix := r.builder.BuildPrefixSuffix(text, meta.ReferenceText)
	return len(prefix) + len(codes[0]) + len(suffix), nil
}

// ListVoices 返回所有已注册 voice 的元数据。
func (r *Runtime) ListVoices() ([]voices.Meta, error) {
	return r.voices.List()
}

// ReloadVoice 强制从磁盘重新加载 voice，更新缓存。
// 用于 voice 重新注册后刷新内存。
func (r *Runtime) ReloadVoice(voiceName string) error {
	_, _, err := r.voices.Reload(voiceName)
	return err
}

// InvalidateVoice 从缓存中移除指定 voice。
func (r *Runtime) InvalidateVoice(voiceName string) {
	r.voices.Invalidate(voiceName)
}

// Engine 返回推理引擎实例（可能为 nil，如果未调用 InitEngine）。
func (r *Runtime) Engine() *inference.Engine { return r.engine }

// Synthesize 执行完整 TTS 推理：构造 prompt → 生成 codes → codec 解码 → 音频。
// 需要先调用 InitEngine 初始化推理引擎。
func (r *Runtime) Synthesize(text, voiceName string, opts inference.GenerateOptions) ([]float32, [][]int64, error) {
	if r.engine == nil {
		return nil, nil, fmt.Errorf("engine not initialized; call InitEngine first")
	}

	// 确保使用传入的 text 和 voice
	opts.Text = text
	opts.Voice = voiceName

	// 构造 prompt 矩阵
	promptMatrix, err := r.BuildPrompt(text, voiceName)
	if err != nil {
		return nil, nil, fmt.Errorf("build prompt: %w", err)
	}

	// 检查 prompt 长度
	promptLen := len(promptMatrix[0][0])
	if promptLen >= r.manifest.MaxSeqLen {
		return nil, nil, fmt.Errorf("prompt length %d exceeds max sequence length %d",
			promptLen, r.manifest.MaxSeqLen)
	}

	// 执行推理
	return r.engine.Synthesize(promptMatrix, opts)
}

// Stream 以流式方式生成音频，分块返回。
// 需要先调用 InitEngine 初始化推理引擎。
// chunkFrames 控制每批解码的 codec 帧数（<=0 时默认 12）。
// ctx 取消时尽快停止生成并关闭 channel。
func (r *Runtime) Stream(ctx context.Context, text, voiceName string, opts inference.GenerateOptions, chunkFrames int) (<-chan inference.StreamEvent, error) {
	if r.engine == nil {
		return nil, fmt.Errorf("engine not initialized; call InitEngine first")
	}

	opts.Text = text
	opts.Voice = voiceName

	promptMatrix, err := r.BuildPrompt(text, voiceName)
	if err != nil {
		return nil, fmt.Errorf("build prompt: %w", err)
	}

	promptLen := len(promptMatrix[0][0])
	if promptLen >= r.manifest.MaxSeqLen {
		return nil, fmt.Errorf("prompt length %d exceeds max sequence length %d",
			promptLen, r.manifest.MaxSeqLen)
	}

	return r.engine.Stream(ctx, promptMatrix, opts, chunkFrames)
}
