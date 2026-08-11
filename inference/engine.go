// Package inference 提供 ONNX 模型推理引擎，封装 slow AR、fast AR 和 codec decoder
// 三个模型的推理逻辑，包括 KV cache 管理、采样和生成循环。
//
// 与 Python 端 arktts_runtime/runtime.py 的 ArkTtsRuntime 保持功能一致。
package inference

import (
	"context"
	"fmt"
	"log/slog"
	"path/filepath"
	"sync"

	"github.com/metaRobin/arktts/config"
	"github.com/metaRobin/arktts/onnxruntime"
	ort "github.com/yalue/onnxruntime_go"
)

// destroyAll 销毁所有非 nil 的 ort.Value 并清空切片。
func destroyAll(values []ort.Value) {
	for _, v := range values {
		if v != nil {
			v.Destroy()
		}
	}
}

// destroyOutputs 销毁所有非 nil 的输出张量。
func destroyOutputs(outputs []ort.Value) {
	for _, v := range outputs {
		if v != nil {
			v.Destroy()
		}
	}
}

// CacheBuffer 管理 KV cache 的底层缓冲区。
// data 以 []uint16 存储 float16 位模式（与 ONNX float16 张量二进制兼容）。
type CacheBuffer struct {
	Data    []uint16 // float16 位模式
	Shape   []int64  // [1, heads, seqLen, headDim]
	Heads   int
	SeqLen  int
	HeadDim int
}

// NewCacheBuffer 创建一个零填充的 KV cache 缓冲区。
func NewCacheBuffer(heads, seqLen, headDim int) *CacheBuffer {
	total := heads * seqLen * headDim
	return &CacheBuffer{
		Data:    make([]uint16, total),
		Shape:   []int64{1, int64(heads), int64(seqLen), int64(headDim)},
		Heads:   heads,
		SeqLen:  seqLen,
		HeadDim: headDim,
	}
}

// CreateTensor 从缓存数据创建 float16 ONNX 张量。
// 调用方必须在用完后调用 Destroy。
func (c *CacheBuffer) CreateTensor() (*ort.CustomDataTensor, error) {
	return NewFloat16Tensor(ort.Shape(c.Shape), c.Data)
}

// WriteDelta 将 delta 数据写入缓存的指定位置。
// delta 是 float16 的 []uint16 数据，positions 是序列维度上的位置。
func (c *CacheBuffer) WriteDelta(positions []int, delta []uint16) error {
	return WriteFloat16SliceAtPosition(c.Data, c.Heads, c.SeqLen, c.HeadDim, positions, delta)
}

// Reset 将缓存清零。
func (c *CacheBuffer) Reset() {
	for i := range c.Data {
		c.Data[i] = 0
	}
}

// Engine 是 TTS 推理引擎，管理 slow AR、fast AR 和 codec decoder 三个 ONNX 会话。
type Engine struct {
	manifest  *config.RuntimeManifest
	ort       *onnxruntime.RuntimeImpl
	slowSess  *onnxruntime.SessionImpl
	fastSess  *onnxruntime.SessionImpl
	codecSess *onnxruntime.SessionImpl

	// 预计算输入/输出名称
	slowInputNames  []string
	slowOutputNames []string
	fastInputNames  []string
	fastOutputNames []string

	// 缓存的 buffer 池，避免每次推理重新分配
	slowCachePool sync.Pool
	fastCachePool sync.Pool

	// 预计算的语义采样 allowed IDs，避免每次 SampleSemantic 重建
	semanticAllowedIDs []int

	// 模型元数据
	precision      string
	codecPrecision string
}

// EngineConfig 配置推理引擎。
type EngineConfig struct {
	ModelDir       string
	LibraryPath    string
	Precision      string // "int4"
	CodecPrecision string // "fp16"
	Threads        int
	EnableCPUArena bool
}

// NewEngine 初始化 ONNX Runtime 并加载三个模型会话。
func NewEngine(cfg EngineConfig, manifest *config.RuntimeManifest) (*Engine, error) {
	precision := cfg.Precision
	if precision == "" {
		precision = manifest.DefaultPrecision
	}

	codecPrecision := cfg.CodecPrecision
	if codecPrecision == "" {
		codecPrecision = manifest.DefaultCodecPrecision
	}

	// 验证 precision
	precisionOK := false
	for _, p := range manifest.AvailablePrecisions {
		if p == precision {
			precisionOK = true
			break
		}
	}
	if !precisionOK {
		return nil, fmt.Errorf("unsupported precision: %s (available: %v)", precision, manifest.AvailablePrecisions)
	}

	codecOK := false
	for _, p := range manifest.AvailableCodecPrecisions {
		if p == codecPrecision {
			codecOK = true
			break
		}
	}
	if !codecOK {
		return nil, fmt.Errorf("unsupported codec precision: %s (available: %v)", codecPrecision, manifest.AvailableCodecPrecisions)
	}

	// 初始化 ONNX Runtime
	ortRuntime := onnxruntime.NewRuntime()
	if err := ortRuntime.Initialize(onnxruntime.RuntimeConfig{
		LibraryPath:    cfg.LibraryPath,
		IntraOpThreads: cfg.Threads,
		InterOpThreads: max(1, cfg.Threads/2),
		EnableCPUArena: cfg.EnableCPUArena,
	}); err != nil {
		return nil, fmt.Errorf("init onnxruntime: %w", err)
	}

	e := &Engine{
		manifest:       manifest,
		ort:            ortRuntime,
		precision:      precision,
		codecPrecision: codecPrecision,
	}

	// 加载三个模型
	slowPath := filepath.Join(cfg.ModelDir, fmt.Sprintf("slow_ar_%s.onnx", precision))
	fastPath := filepath.Join(cfg.ModelDir, fmt.Sprintf("fast_ar_%s.onnx", precision))
	codecModelName := manifest.CodecModels[codecPrecision]
	codecPath := filepath.Join(cfg.ModelDir, codecModelName)

	slog.Info("Loading ONNX models",
		slog.String("slow", slowPath),
		slog.String("fast", fastPath),
		slog.String("codec", codecPath))

	slowSess, err := ortRuntime.CreateSession(slowPath)
	if err != nil {
		return nil, fmt.Errorf("load slow model: %w", err)
	}
	fastSess, err := ortRuntime.CreateSession(fastPath)
	if err != nil {
		return nil, fmt.Errorf("load fast model: %w", err)
	}
	codecSess, err := ortRuntime.CreateSession(codecPath)
	if err != nil {
		return nil, fmt.Errorf("load codec model: %w", err)
	}

	e.slowSess = slowSess.(*onnxruntime.SessionImpl)
	e.fastSess = fastSess.(*onnxruntime.SessionImpl)
	e.codecSess = codecSess.(*onnxruntime.SessionImpl)

	e.slowInputNames = e.slowSess.GetInputNames()
	e.slowOutputNames = e.slowSess.GetOutputNames()
	e.fastInputNames = e.fastSess.GetInputNames()
	e.fastOutputNames = e.fastSess.GetOutputNames()

	slog.Info("ONNX models loaded",
		slog.Int("slow_inputs", len(e.slowInputNames)),
		slog.Int("slow_outputs", len(e.slowOutputNames)),
		slog.Int("fast_inputs", len(e.fastInputNames)),
		slog.Int("fast_outputs", len(e.fastOutputNames)),
		slog.Int("num_layers", manifest.NumLayers),
		slog.Int("num_fast_layers", manifest.NumFastLayers))

	// 初始化 cache 池
	e.slowCachePool = sync.Pool{
		New: func() any {
			return e.newSlowCaches()
		},
	}
	e.fastCachePool = sync.Pool{
		New: func() any {
			return e.newFastCaches()
		},
	}

	// 预计算语义采样的 allowed IDs: [begin..end] + [stop]
	// 避免每次 SampleSemantic 调用时重建 4097 元素的切片
	allowedRange := manifest.SemanticEndID - manifest.SemanticBeginID + 1
	e.semanticAllowedIDs = make([]int, allowedRange+1)
	for i := 0; i < allowedRange; i++ {
		e.semanticAllowedIDs[i] = manifest.SemanticBeginID + i
	}
	e.semanticAllowedIDs[allowedRange] = manifest.ImEndID

	return e, nil
}

// Close 释放所有 ONNX 会话和运行时资源。
func (e *Engine) Close() error {
	if e.slowSess != nil {
		e.slowSess.Destroy()
	}
	if e.fastSess != nil {
		e.fastSess.Destroy()
	}
	if e.codecSess != nil {
		e.codecSess.Destroy()
	}
	if e.ort != nil {
		e.ort.Cleanup()
	}
	return nil
}

// ORT 返回底层 ONNX Runtime 实例，供 Encoder 等按需创建额外 session。
func (e *Engine) ORT() *onnxruntime.RuntimeImpl { return e.ort }

// Manifest 返回运行时 manifest。
func (e *Engine) Manifest() *config.RuntimeManifest { return e.manifest }

// newSlowCaches 创建 slow AR 的 KV cache 缓冲区。
// 布局：[1, n_local_heads, max_seq_len, head_dim] float16
func (e *Engine) newSlowCaches() []*CacheBuffer {
	m := e.manifest
	numCaches := 2 * m.NumLayers
	caches := make([]*CacheBuffer, numCaches)
	for i := range caches {
		caches[i] = NewCacheBuffer(m.NLocalHeads, m.MaxSeqLen, m.HeadDim)
	}
	return caches
}

// newFastCaches 创建 fast AR 的 KV cache 缓冲区。
// 布局：[1, fast_n_local_heads, num_codebooks, fast_head_dim] float16
func (e *Engine) newFastCaches() []*CacheBuffer {
	m := e.manifest
	numCaches := 2 * m.NumFastLayers
	caches := make([]*CacheBuffer, numCaches)
	for i := range caches {
		caches[i] = NewCacheBuffer(m.FastNLocalHeads, m.NumCodebooks, m.FastHeadDim)
	}
	return caches
}

// getSlowCaches 从池中获取 slow caches（已清零）。
func (e *Engine) getSlowCaches() []*CacheBuffer {
	caches := e.slowCachePool.Get().([]*CacheBuffer)
	for _, c := range caches {
		c.Reset()
	}
	return caches
}

// putSlowCaches 将 slow caches 归还到池中。
func (e *Engine) putSlowCaches(caches []*CacheBuffer) {
	e.slowCachePool.Put(caches)
}

// getFastCaches 从池中获取 fast caches（已清零）。
func (e *Engine) getFastCaches() []*CacheBuffer {
	caches := e.fastCachePool.Get().([]*CacheBuffer)
	for _, c := range caches {
		c.Reset()
	}
	return caches
}

// putFastCaches 将 fast caches 归还到池中。
func (e *Engine) putFastCaches(caches []*CacheBuffer) {
	e.fastCachePool.Put(caches)
}

// SlowStep 执行一步 slow AR 推理。
//
// codes: [1, num_codebooks+1, T] 展平的 int64 数据
// codesShape: [1, num_codebooks+1, T]
// positions: [T] 位置索引
// caches: KV cache 缓冲区（会被原地更新）
//
// 返回：
//   - logits: [slow_logits_size] 最后一个 token 的 logits
//   - hidden: [fast_dim] 最后一个 token 的 hidden state（float32）
func (e *Engine) SlowStep(codes []int64, codesShape []int64, positions []int64, caches []*CacheBuffer) ([]float32, []float32, error) {
	m := e.manifest
	numLayers := m.NumLayers
	t := int64(len(positions))

	// 构建输入张量
	numInputs := 2 + 2*numLayers
	inputs := make([]ort.Value, 0, numInputs)

	codesTensor, err := ort.NewTensor(ort.Shape(codesShape), codes)
	if err != nil {
		return nil, nil, fmt.Errorf("create codes tensor: %w", err)
	}
	inputs = append(inputs, codesTensor)

	posTensor, err := ort.NewTensor(ort.Shape{t}, positions)
	if err != nil {
		codesTensor.Destroy()
		return nil, nil, fmt.Errorf("create input_pos tensor: %w", err)
	}
	inputs = append(inputs, posTensor)

	for i := 0; i < numLayers; i++ {
		keyTensor, err := caches[2*i].CreateTensor()
		if err != nil {
			destroyAll(inputs)
			return nil, nil, fmt.Errorf("create cache_key_%d tensor: %w", i, err)
		}
		valTensor, err := caches[2*i+1].CreateTensor()
		if err != nil {
			keyTensor.Destroy()
			destroyAll(inputs)
			return nil, nil, fmt.Errorf("create cache_value_%d tensor: %w", i, err)
		}
		inputs = append(inputs, keyTensor, valTensor)
	}

	// 预分配输出张量
	// onnxruntime_go v1.32.0 有 bug：float16 输出自动分配时将元素数当作字节数，
	// 导致 "not enough space" 错误。预分配可绕过此 bug。
	numOutputs := 2 + 2*numLayers
	outputs := make([]ort.Value, numOutputs)
	outputBuffers := make([][]uint16, 0, 2*numLayers+1)

	// Output 0: logits [1, T, slow_logits_size] float32 — 安全，可 auto-allocate
	outputs[0] = nil

	// Output 1: slow_hidden [1, 1, fast_dim] float16 — 需要预分配
	// 模型只输出最后一个位置的 hidden state
	hiddenBuf := make([]uint16, m.FastDim)
	hiddenTensor, err := NewFloat16Tensor(ort.Shape{1, 1, int64(m.FastDim)}, hiddenBuf)
	if err != nil {
		destroyAll(inputs)
		return nil, nil, fmt.Errorf("create slow_hidden output tensor: %w", err)
	}
	outputs[1] = hiddenTensor
	outputBuffers = append(outputBuffers, hiddenBuf)

	// Outputs 2+: key_delta_i / value_delta_i [1, 2, T, head_dim] float16
	deltaShape := ort.Shape{1, int64(m.NLocalHeads), t, int64(m.HeadDim)}
	deltaElems := m.NLocalHeads * int(t) * m.HeadDim
	for i := 0; i < 2*numLayers; i++ {
		buf := make([]uint16, deltaElems)
		tensor, err := NewFloat16Tensor(deltaShape, buf)
		if err != nil {
			destroyAll(inputs)
			destroyOutputs(outputs)
			return nil, nil, fmt.Errorf("create delta output %d tensor: %w", i, err)
		}
		outputs[2+i] = tensor
		outputBuffers = append(outputBuffers, buf)
	}

	// 执行推理
	if err := e.slowSess.RunWithValues(inputs, outputs); err != nil {
		destroyAll(inputs)
		destroyOutputs(outputs)
		return nil, nil, fmt.Errorf("slow inference: %w", err)
	}
	destroyAll(inputs)

	// 读取 logits (output 0): [1, T, slow_logits_size]
	logitsRaw, err := ReadFloat32Tensor(outputs[0])
	if err != nil {
		destroyOutputs(outputs)
		return nil, nil, fmt.Errorf("read logits: %w", err)
	}
	logitsSize := m.SlowLogitsSize
	if len(logitsRaw) < logitsSize {
		destroyOutputs(outputs)
		return nil, nil, fmt.Errorf("logits too short: got %d, need %d", len(logitsRaw), logitsSize)
	}
	logits := make([]float32, logitsSize)
	copy(logits, logitsRaw[len(logitsRaw)-logitsSize:])

	// 读取 hidden (output 1): [1, 1, fast_dim] float16
	hiddenRaw, err := ReadFloat16Tensor(outputs[1])
	if err != nil {
		destroyOutputs(outputs)
		return nil, nil, fmt.Errorf("read hidden: %w", err)
	}
	hidden := make([]float32, m.FastDim)
	copy(hidden, hiddenRaw)

	// 更新 KV caches with deltas (outputs[2:])
	posInts := make([]int, len(positions))
	for i, p := range positions {
		posInts[i] = int(p)
	}
	for i := 0; i < 2*numLayers; i++ {
		deltaRaw, err := readCacheDelta(outputs[2+i])
		if err != nil {
			destroyOutputs(outputs)
			return nil, nil, fmt.Errorf("read cache delta %d: %w", i, err)
		}
		if err := caches[i].WriteDelta(posInts, deltaRaw); err != nil {
			destroyOutputs(outputs)
			return nil, nil, fmt.Errorf("update cache %d: %w", i, err)
		}
	}

	destroyOutputs(outputs)
	return logits, hidden, nil
}

// FastStep 执行一步 fast AR 推理。
//
// hidden: [fast_dim] slow 模型的 hidden state（float32）
// tokenID: 当前 codebook token
// useSlowHidden: 是否使用 slow hidden（仅第一步为 true）
// position: fast 位置 (0..num_codebooks-1)
// caches: fast KV cache 缓冲区（会被原地更新）
//
// 返回: [codebook_size] logits
func (e *Engine) FastStep(hidden []float32, tokenID int64, useSlowHidden bool, position int64, caches []*CacheBuffer) ([]float32, error) {
	m := e.manifest
	numFastLayers := m.NumFastLayers

	// 构建输入张量
	// 顺序: slow_hidden, token_id, use_slow_hidden, input_pos,
	//       cache_key_0, cache_value_0, ..., cache_key_{M-1}, cache_value_{M-1}
	numInputs := 4 + 2*numFastLayers
	inputs := make([]ort.Value, 0, numInputs)

	// slow_hidden: float16 tensor, shape [1, 1, fast_dim]
	hiddenTensor, err := NewFloat16TensorFromFloat32(ort.Shape{1, 1, int64(m.FastDim)}, hidden)
	if err != nil {
		return nil, fmt.Errorf("create slow_hidden tensor: %w", err)
	}
	inputs = append(inputs, hiddenTensor)

	// token_id: int64 tensor, shape [1, 1]
	tokenData := []int64{tokenID}
	tokenTensor, err := ort.NewTensor(ort.Shape{1, 1}, tokenData)
	if err != nil {
		hiddenTensor.Destroy()
		return nil, fmt.Errorf("create token_id tensor: %w", err)
	}
	inputs = append(inputs, tokenTensor)

	// use_slow_hidden: bool tensor, shape [1]
	useHiddenData := []bool{useSlowHidden}
	useHiddenTensor, err := ort.NewTensor(ort.Shape{1}, useHiddenData)
	if err != nil {
		hiddenTensor.Destroy()
		tokenTensor.Destroy()
		return nil, fmt.Errorf("create use_slow_hidden tensor: %w", err)
	}
	inputs = append(inputs, useHiddenTensor)

	// input_pos: int64 tensor, shape [1]
	posData := []int64{position}
	posTensor, err := ort.NewTensor(ort.Shape{1}, posData)
	if err != nil {
		hiddenTensor.Destroy()
		tokenTensor.Destroy()
		useHiddenTensor.Destroy()
		return nil, fmt.Errorf("create input_pos tensor: %w", err)
	}
	inputs = append(inputs, posTensor)

	// cache tensors
	for i := 0; i < numFastLayers; i++ {
		keyTensor, err := caches[2*i].CreateTensor()
		if err != nil {
			destroyAll(inputs)
			return nil, fmt.Errorf("create fast cache_key_%d tensor: %w", i, err)
		}
		valTensor, err := caches[2*i+1].CreateTensor()
		if err != nil {
			keyTensor.Destroy()
			destroyAll(inputs)
			return nil, fmt.Errorf("create fast cache_value_%d tensor: %w", i, err)
		}
		inputs = append(inputs, keyTensor, valTensor)
	}

	// 预分配输出张量（绕过 onnxruntime_go float16 bug）
	numOutputs := 1 + 2*numFastLayers
	outputs := make([]ort.Value, numOutputs)

	// Output 0: logits [1, 1, codebook_size] float32 — 安全，可 auto-allocate
	outputs[0] = nil

	// Outputs 1+: key_delta_i / value_delta_i [1, 2, 1, head_dim] float16
	deltaShape := ort.Shape{1, int64(m.FastNLocalHeads), 1, int64(m.FastHeadDim)}
	deltaElems := m.FastNLocalHeads * m.FastHeadDim
	for i := 0; i < 2*numFastLayers; i++ {
		buf := make([]uint16, deltaElems)
		tensor, err := NewFloat16Tensor(deltaShape, buf)
		if err != nil {
			destroyAll(inputs)
			destroyOutputs(outputs)
			return nil, fmt.Errorf("create fast delta output %d tensor: %w", i, err)
		}
		outputs[1+i] = tensor
	}

	if err := e.fastSess.RunWithValues(inputs, outputs); err != nil {
		destroyAll(inputs)
		destroyOutputs(outputs)
		return nil, fmt.Errorf("fast inference: %w", err)
	}
	destroyAll(inputs)

	// 读取 logits (output 0): [1, 1, codebook_size]
	logitsRaw, err := ReadFloat32Tensor(outputs[0])
	if err != nil {
		destroyOutputs(outputs)
		return nil, fmt.Errorf("read fast logits: %w", err)
	}
	codebookSize := m.CodebookSize
	if len(logitsRaw) < codebookSize {
		destroyOutputs(outputs)
		return nil, fmt.Errorf("fast logits too short: got %d, need %d", len(logitsRaw), codebookSize)
	}
	logits := make([]float32, codebookSize)
	copy(logits, logitsRaw[len(logitsRaw)-codebookSize:])

	// 更新 fast KV caches
	posInt := []int{int(position)}
	for i := 0; i < 2*numFastLayers; i++ {
		deltaRaw, err := readCacheDelta(outputs[1+i])
		if err != nil {
			destroyOutputs(outputs)
			return nil, fmt.Errorf("read fast cache delta %d: %w", i, err)
		}
		if err := caches[i].WriteDelta(posInt, deltaRaw); err != nil {
			destroyOutputs(outputs)
			return nil, fmt.Errorf("update fast cache %d: %w", i, err)
		}
	}

	destroyOutputs(outputs)
	return logits, nil
}

// DecodeCodes 执行 codec 解码。
//
// codes: [1, num_codebooks, T] 展平的 int64 数据
// codesShape: [1, num_codebooks, T]
//
// 返回: 展平的音频样本 []float32
func (e *Engine) DecodeCodes(codes []int64, codesShape []int64) ([]float32, error) {
	// 构建输入: codes int64 tensor
	codesTensor, err := ort.NewTensor(ort.Shape(codesShape), codes)
	if err != nil {
		return nil, fmt.Errorf("create codec codes tensor: %w", err)
	}
	inputs := []ort.Value{codesTensor}

	// 输出: audio
	outputs := make([]ort.Value, 1)

	if err := e.codecSess.RunWithValues(inputs, outputs); err != nil {
		codesTensor.Destroy()
		return nil, fmt.Errorf("codec inference: %w", err)
	}

	codesTensor.Destroy()

	// 读取音频输出
	audio, err := ReadFloat32Tensor(outputs[0])
	if err != nil {
		if outputs[0] != nil {
			outputs[0].Destroy()
		}
		return nil, fmt.Errorf("read codec audio: %w", err)
	}

	if outputs[0] != nil {
		outputs[0].Destroy()
	}

	// 展平
	result := make([]float32, len(audio))
	copy(result, audio)
	return result, nil
}

// readCacheDelta 读取 cache delta 输出张量，返回 []uint16 (float16 位模式)。
func readCacheDelta(v ort.Value) ([]uint16, error) {
	if v == nil {
		return nil, fmt.Errorf("nil cache delta tensor")
	}
	switch t := v.(type) {
	case *ort.CustomDataTensor:
		raw := t.GetData()
		return BytesToUint16(raw)
	case *ort.Tensor[float32]:
		// 模型输出 float32，需要转换为 float16 位模式
		data := t.GetData()
		return F32To16Slice(data), nil
	default:
		return nil, fmt.Errorf("unexpected cache delta tensor type %T", v)
	}
}

// SampleSemantic 语义 token 采样，复刻 Python _sample_semantic。
//
// 当 slow_logits_layout == "semantic_then_eos" 时，logits 直接对应
// [begin..end] + [stop] 的连续区间。否则需要用 allowed_ids 索引。
//
// 如果 normal 采样结果在 previous 中出现过（且在语义区间内），
// 则改用 high 采样结果（temperature=1.0, top_p=0.9）。
func (e *Engine) SampleSemantic(logits []float32, previous []int, temperature, topP float64, topK int, rng *PCG64) int {
	m := e.manifest
	begin := m.SemanticBeginID
	end := m.SemanticEndID

	// 使用预计算的 allowedIDs（在 NewEngine 中一次性构造）
	allowedIDs := e.semanticAllowedIDs

	// 根据 layout 确定 allowed_logits
	var allowedLogits []float32
	if m.SlowLogitsLayout == "semantic_then_eos" {
		// logits 直接对应 allowed_ids 的连续区间
		allowedLogits = logits
	} else {
		// 需要 gather
		allowedLogits = make([]float32, len(allowedIDs))
		for i, id := range allowedIDs {
			if id < len(logits) {
				allowedLogits[i] = logits[id]
			}
		}
	}

	if len(allowedLogits) != len(allowedIDs) {
		panic(fmt.Sprintf("unexpected slow logits size: %d, expected %d", len(allowedLogits), len(allowedIDs)))
	}

	// normal 采样
	normalIdx := Sample(allowedLogits, temperature, topP, topK, rng)
	normal := allowedIDs[normalIdx]

	// high 采样（temperature=1.0, top_p=0.9）
	highIdx := Sample(allowedLogits, 1.0, 0.9, topK, rng)
	high := allowedIDs[highIdx]

	// 如果 normal 在语义区间内且在 previous 中出现过，返回 high
	if normal >= begin && normal <= end {
		for _, p := range previous {
			if p == normal {
				return high
			}
		}
	}

	return normal
}

// GenerateOptions 控制 TTS 生成的参数。
type GenerateOptions struct {
	Text         string
	Voice        string
	MaxNewTokens int
	Temperature  float64
	TopP         float64
	TopK         int
	Seed         int64
}

// DefaultGenerateOptions 返回默认生成参数（与 Python CLI 默认值一致）。
func DefaultGenerateOptions() GenerateOptions {
	return GenerateOptions{
		MaxNewTokens: 1024,
		Temperature:  0.7,
		TopP:         0.9,
		TopK:         50,
		Seed:         42,
	}
}

// GeneratedFrame 表示一个生成的 codec frame（num_codebooks 个 token）。
type GeneratedFrame struct {
	Semantic int     // 语义 token ID
	Codes    []int64 // [num_codebooks] codebook token IDs
}

// NewSlowCaches creates slow AR KV cache buffers (public wrapper for newSlowCaches).
func (e *Engine) NewSlowCaches() []*CacheBuffer {
	return e.newSlowCaches()
}

// NewFastCaches creates fast AR KV cache buffers (public wrapper for newFastCaches).
func (e *Engine) NewFastCaches() []*CacheBuffer {
	return e.newFastCaches()
}

// Argmax returns the index of the maximum value in a float32 slice.
func Argmax(logits []float32) int {
	if len(logits) == 0 {
		return 0
	}
	bestIdx := 0
	bestVal := logits[0]
	for i := 1; i < len(logits); i++ {
		if logits[i] > bestVal {
			bestVal = logits[i]
			bestIdx = i
		}
	}
	return bestIdx
}

// iterResult 是 IterCodesStream 内部的流式结果，可能是一帧或一个错误。
// 最后一帧之后若发生错误，Err 非 nil；正常结束时 channel 直接关闭。
type iterResult struct {
	Frame GeneratedFrame
	Err   error
}

// IterCodesStream 是 ctx 可取消的流式生成循环。
//
// setup 阶段（prompt 校验 + 初始 slow step）同步执行，错误作为第二个返回值。
// 生成阶段在 goroutine 中运行，每生成一帧通过 channel 发送；
// 中途出错（FastStep / SlowStep 失败）发送 iterResult{Err: err} 后关闭 channel。
// ctx 取消时 goroutine 尽快退出，channel 关闭。
func (e *Engine) IterCodesStream(ctx context.Context, promptMatrix [][][]int64, opts GenerateOptions) (<-chan iterResult, error) {
	m := e.manifest
	numCodebooks := m.NumCodebooks
	maxSeqLen := m.MaxSeqLen

	promptLen := len(promptMatrix[0][0])
	if promptLen >= maxSeqLen {
		return nil, fmt.Errorf("prompt length %d exceeds max sequence length %d", promptLen, maxSeqLen)
	}

	maxNewTokens := opts.MaxNewTokens
	if maxNewTokens > maxSeqLen-promptLen {
		maxNewTokens = maxSeqLen - promptLen
	}

	promptFlat := flattenPromptMatrix(promptMatrix, numCodebooks+1, promptLen)
	rng := NewPCG64(opts.Seed)
	slowCaches := e.getSlowCaches()

	// 初始 slow step: 用完整 prompt 填充 KV cache（同步）
	positions := make([]int64, promptLen)
	for i := range positions {
		positions[i] = int64(i)
	}
	logits, hidden, err := e.SlowStep(promptFlat, []int64{1, int64(numCodebooks + 1), int64(promptLen)}, positions, slowCaches)
	if err != nil {
		e.putSlowCaches(slowCaches)
		return nil, fmt.Errorf("initial slow step: %w", err)
	}

	ch := make(chan iterResult, 16)
	go func() {
		defer close(ch)
		defer e.putSlowCaches(slowCaches)

		var previous []int
		begin := m.SemanticBeginID
		stop := m.ImEndID
		codebookSize := m.CodebookSize

		for step := 0; step < maxNewTokens; step++ {
			if ctx.Err() != nil {
				return
			}

			semantic := e.SampleSemantic(logits, previous, opts.Temperature, opts.TopP, opts.TopK, rng)
			if semantic == stop {
				return
			}

			previous = append(previous, semantic)
			if len(previous) > 10 {
				previous = previous[len(previous)-10:]
			}

			fastCaches := e.getFastCaches()

			_, err := e.FastStep(hidden, 0, true, 0, fastCaches)
			if err != nil {
				e.putFastCaches(fastCaches)
				ch <- iterResult{Err: fmt.Errorf("fast step 0: %w", err)}
				return
			}

			token := semantic - begin
			if token < 0 {
				token = 0
			} else if token >= codebookSize {
				token = codebookSize - 1
			}

			codebooks := make([]int64, 0, numCodebooks)
			codebooks = append(codebooks, int64(token))

			for fastPos := 1; fastPos < numCodebooks; fastPos++ {
				fastLogits, err := e.FastStep(hidden, int64(token), false, int64(fastPos), fastCaches)
				if err != nil {
					e.putFastCaches(fastCaches)
					ch <- iterResult{Err: fmt.Errorf("fast step %d: %w", fastPos, err)}
					return
				}
				token = Sample(fastLogits, opts.Temperature, opts.TopP, opts.TopK, rng)
				codebooks = append(codebooks, int64(token))
			}

			e.putFastCaches(fastCaches)

			select {
			case <-ctx.Done():
				return
			case ch <- iterResult{Frame: GeneratedFrame{Semantic: semantic, Codes: codebooks}}:
			}

			if step+1 >= maxNewTokens {
				return
			}

			column := make([]int64, numCodebooks+1)
			column[0] = int64(semantic)
			copy(column[1:], codebooks)

			position := int64(promptLen + step)
			logits, hidden, err = e.SlowStep(column, []int64{1, int64(numCodebooks + 1), 1}, []int64{position}, slowCaches)
			if err != nil {
				ch <- iterResult{Err: fmt.Errorf("slow step %d: %w", step+1, err)}
				return
			}
		}
	}()

	return ch, nil
}

// IterCodes 是 TTS codec 生成循环的同步封装，复刻 Python iter_codes。
// 内部调用 IterCodesStream 并排空 channel，保留中途错误传递。
func (e *Engine) IterCodes(promptMatrix [][][]int64, opts GenerateOptions) ([]GeneratedFrame, error) {
	ch, err := e.IterCodesStream(context.Background(), promptMatrix, opts)
	if err != nil {
		return nil, err
	}

	var frames []GeneratedFrame
	for res := range ch {
		if res.Err != nil {
			return frames, res.Err
		}
		frames = append(frames, res.Frame)
	}
	return frames, nil
}

// StreamEvent 是 Stream 输出的事件。
//
// Type 取值：
//   - "audio_chunk": Audio 字段携带本批 PCM float32 样本
//   - "complete":    FrameCount 为总帧数
//   - "error":       Err 为中途错误
type StreamEvent struct {
	Type       string
	Seq        int
	Audio      []float32
	FrameCount int
	Err        error
}

// Stream 以流式方式生成 codec 帧并分批解码为音频块。
//
// 每 chunkFrames 帧调用一次 codec decoder，发出一个 "audio_chunk" 事件；
// 生成结束后发出 "complete" 事件。ctx 取消时直接关闭 channel（无 cancelled 事件，
// 由上层 HTTP handler 决定如何响应）。
func (e *Engine) Stream(ctx context.Context, promptMatrix [][][]int64, opts GenerateOptions, chunkFrames int) (<-chan StreamEvent, error) {
	if chunkFrames <= 0 {
		chunkFrames = 12
	}

	ch, err := e.IterCodesStream(ctx, promptMatrix, opts)
	if err != nil {
		return nil, err
	}

	eventCh := make(chan StreamEvent, 4)
	go func() {
		defer close(eventCh)

		var buffer []GeneratedFrame
		var seq int
		var totalFrames int

		for res := range ch {
			if res.Err != nil {
				eventCh <- StreamEvent{Type: "error", Err: res.Err}
				return
			}
			if ctx.Err() != nil {
				return
			}

			buffer = append(buffer, res.Frame)
			totalFrames++

			if len(buffer) >= chunkFrames {
				audio, err := e.decodeFrames(buffer)
				if err != nil {
					eventCh <- StreamEvent{Type: "error", Err: fmt.Errorf("decode chunk %d: %w", seq, err)}
					return
				}
				eventCh <- StreamEvent{Type: "audio_chunk", Seq: seq, Audio: audio, FrameCount: totalFrames}
				seq++
				buffer = buffer[:0]
			}
		}

		if ctx.Err() != nil {
			return
		}

		// flush 剩余帧
		if len(buffer) > 0 {
			audio, err := e.decodeFrames(buffer)
			if err != nil {
				eventCh <- StreamEvent{Type: "error", Err: fmt.Errorf("decode tail: %w", err)}
				return
			}
			eventCh <- StreamEvent{Type: "audio_chunk", Seq: seq, Audio: audio, FrameCount: totalFrames}
		}

		eventCh <- StreamEvent{Type: "complete", FrameCount: totalFrames}
	}()

	return eventCh, nil
}

// decodeFrames 将一批 GeneratedFrame 解码为音频样本。
// frames: [T]Frame，每个 Frame.Codes 为 [num_codebooks]int64
// 返回展平的 audio []float32
func (e *Engine) decodeFrames(frames []GeneratedFrame) ([]float32, error) {
	m := e.manifest
	numCodebooks := m.NumCodebooks
	t := len(frames)

	codesFlat := make([]int64, numCodebooks*t)
	for cb := 0; cb < numCodebooks; cb++ {
		for i, frame := range frames {
			codesFlat[cb*t+i] = frame.Codes[cb]
		}
	}

	codesShape := []int64{1, int64(numCodebooks), int64(t)}
	return e.DecodeCodes(codesFlat, codesShape)
}

// flattenPromptMatrix 将 [1, rows, cols] 的 int64 矩阵展平为连续的 []int64。
// ONNX 张量使用 C-contiguous 布局（行优先）。
func flattenPromptMatrix(matrix [][][]int64, rows, cols int) []int64 {
	flat := make([]int64, rows*cols)
	for r := 0; r < rows; r++ {
		copy(flat[r*cols:(r+1)*cols], matrix[0][r])
	}
	return flat
}

// Synthesize 执行完整 TTS 推理：prompt → 生成 codes → codec 解码 → 音频。
//
// promptMatrix: [1, num_codebooks+1, T] 的 prompt 矩阵
// opts: 生成参数
//
// 返回: 音频样本 []float32 和生成的 codes [num_codebooks][T_generated]
func (e *Engine) Synthesize(promptMatrix [][][]int64, opts GenerateOptions) ([]float32, [][]int64, error) {
	frames, err := e.IterCodes(promptMatrix, opts)
	if err != nil {
		return nil, nil, fmt.Errorf("generate codes: %w", err)
	}
	if len(frames) == 0 {
		return nil, nil, fmt.Errorf("model produced no codec frames")
	}

	// 将 frames 转换为 codes 矩阵 [num_codebooks][T]
	m := e.manifest
	numCodebooks := m.NumCodebooks
	codes := make([][]int64, numCodebooks)
	for cb := 0; cb < numCodebooks; cb++ {
		codes[cb] = make([]int64, len(frames))
		for t, frame := range frames {
			codes[cb][t] = frame.Codes[cb]
		}
	}

	// 展平 codes 为 [1, num_codebooks, T] 的连续数组
	codesFlat := make([]int64, numCodebooks*len(frames))
	for cb := 0; cb < numCodebooks; cb++ {
		copy(codesFlat[cb*len(frames):(cb+1)*len(frames)], codes[cb])
	}

	codesShape := []int64{1, int64(numCodebooks), int64(len(frames))}
	audio, err := e.DecodeCodes(codesFlat, codesShape)
	if err != nil {
		return nil, nil, fmt.Errorf("decode codes: %w", err)
	}

	return audio, codes, nil
}
