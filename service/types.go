// Package service 提供 TTS HTTP 服务，封装 runtime 与请求串行锁。
//
// 对应 Python arktts_runtime/service.py 的最简端点集（health / voices / tts）。
package service

// TtsRequest 对应 Python TtsRequest。
// 数值字段用指针类型，以便区分「未提供」与「显式零值」。
type TtsRequest struct {
	Text         string   `json:"text"`
	VoiceName    string   `json:"voice_name"`
	MaxNewTokens *int     `json:"max_new_tokens"`
	Temperature  *float64 `json:"temperature"`
	TopP         *float64 `json:"top_p"`
	TopK         *int     `json:"top_k"`
	Seed         *int64   `json:"seed"`
}

// HealthResponse 是 /api/health 的响应。
type HealthResponse struct {
	Status         string `json:"status"`
	EngineReady    bool   `json:"engine_ready"`
	ModelFamily    string `json:"model_family"`
	Precision      string `json:"precision"`
	CodecPrecision string `json:"codec_precision"`
	SampleRate     int    `json:"sample_rate"`
	NumCodebooks   int    `json:"num_codebooks"`
	Threads        int    `json:"threads"`
}

// OpenAiSpeechRequest 对应 OpenAI /v1/audio/speech 接口。
// 与 Python OpenAiSpeechRequest 一致：
//
//	model: "arktts" | "tts-1"（仅做校验，不区分行为）
//	input: 文本（1..1000）
//	voice: voice 名称
//	response_format: "wav"（默认） | "pcm"
type OpenAiSpeechRequest struct {
	Model          string `json:"model"`
	Input          string `json:"input"`
	Voice          string `json:"voice"`
	ResponseFormat string `json:"response_format"`
}

// errorResponse 是统一的 JSON 错误响应。
type errorResponse struct {
	Error  string `json:"error"`
	Detail string `json:"detail,omitempty"`
}

// NDJSON 流式事件类型，对应 Python service.py 的流式协议。
// 每行一个 JSON 对象，event 字段区分类型。

// streamStartEvent 是流式起始事件。
type streamStartEvent struct {
	Event          string `json:"event"`
	SampleRate     int    `json:"sample_rate"`
	Channels       int    `json:"channels"`
	SampleFormat   string `json:"sample_format"`
	Precision      string `json:"precision"`
	CodecPrecision string `json:"codec_precision"`
}

// streamAudioChunkEvent 是一个音频块事件。
// PCMBase64 为 int16 LE PCM 的 base64 编码。
type streamAudioChunkEvent struct {
	Event      string `json:"event"`
	Seq        int    `json:"seq"`
	FrameCount int    `json:"frame_count"`
	PCMBase64  string `json:"pcm_b64"`
}

// streamCompleteEvent 是流式完成事件。
type streamCompleteEvent struct {
	Event      string `json:"event"`
	FrameCount int    `json:"frame_count"`
}

// streamCancelledEvent 是流式取消事件。
type streamCancelledEvent struct {
	Event string `json:"event"`
}

// SystemResponse 是 /api/system 的响应，返回内存与运行时信息。
// 用 runtime.MemStats 替代 Python 的 /usr/bin/footprint 子进程采样。
type SystemResponse struct {
	// Go runtime 内存统计
	AllocBytes      uint64 `json:"alloc_bytes"`       // 当前堆分配
	TotalAllocBytes uint64 `json:"total_alloc_bytes"` // 累计堆分配
	SysBytes        uint64 `json:"sys_bytes"`         // 从 OS 获取的内存
	HeapInUseBytes  uint64 `json:"heap_inuse_bytes"`  // 堆中正在使用
	StackInUseBytes uint64 `json:"stack_inuse_bytes"` // 栈使用
	NumGC           uint32 `json:"num_gc"`            // GC 次数
	LastGCNs        uint64 `json:"last_gc_ns"`        // 上次 GC 时间（纳秒，epoch）

	// 服务信息
	UptimeSeconds float64 `json:"uptime_seconds"`
	EngineReady   bool    `json:"engine_ready"`
	Threads       int     `json:"threads"`
	GoVersion     string  `json:"go_version"`
	GOMAXPROCS    int     `json:"gomaxprocs"`
}
