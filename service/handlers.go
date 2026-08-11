package service

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	"github.com/metaRobin/arktts/audio"
	"github.com/metaRobin/arktts/inference"
	"github.com/metaRobin/arktts/voices"
)

// health 返回服务与模型健康信息。对应 GET /api/health。
func (s *Server) health(w http.ResponseWriter, r *http.Request) {
	m := s.rt.Manifest()
	writeJSON(w, http.StatusOK, HealthResponse{
		Status:         "ok",
		EngineReady:    s.rt.Engine() != nil,
		ModelFamily:    m.ModelFamily,
		Precision:      m.DefaultPrecision,
		CodecPrecision: m.DefaultCodecPrecision,
		SampleRate:     m.SampleRate,
		NumCodebooks:   m.NumCodebooks,
		Threads:        s.threads,
	})
}

// listVoices 返回已注册 voice 列表。对应 GET /api/voices。
func (s *Server) listVoices(w http.ResponseWriter, r *http.Request) {
	metas, err := s.rt.ListVoices()
	if err != nil {
		writeError(w, http.StatusInternalServerError, "list voices failed", err.Error())
		return
	}
	// 空列表返回 [] 而非 null
	if metas == nil {
		metas = []voices.Meta{}
	}
	writeJSON(w, http.StatusOK, metas)
}

// tts 执行一次性合成，返回 WAV 字节。对应 POST /api/tts。
func (s *Server) tts(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed", "use POST")
		return
	}

	var req TtsRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body", err.Error())
		return
	}

	opts, err := buildOptions(&req)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid request", err.Error())
		return
	}

	audioSamples, err := s.synthesize(req.Text, req.VoiceName, opts)
	if err != nil {
		writeHTTPError(w, err)
		return
	}

	s.respondAudio(w, audioSamples, "wav")
}

// openAISpeech 实现 OpenAI 兼容接口。对应 POST /v1/audio/speech。
// 支持 response_format=wav（默认）或 pcm（裸 int16 LE）。
func (s *Server) openAISpeech(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed", "use POST")
		return
	}

	var req OpenAiSpeechRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body", err.Error())
		return
	}

	// 校验 model：接受 "arktts" 或 "tts-1"，与 Python 一致
	if req.Model != "arktts" && req.Model != "tts-1" {
		writeError(w, http.StatusBadRequest, "invalid model", `model must be "arktts" or "tts-1"`)
		return
	}

	// 校验 response_format，默认 wav
	format := req.ResponseFormat
	if format == "" {
		format = "wav"
	}
	if format != "wav" && format != "pcm" {
		writeError(w, http.StatusBadRequest, "invalid response_format", `response_format must be "wav" or "pcm"`)
		return
	}

	// 复用 TtsRequest 的校验逻辑
	ttsReq := TtsRequest{Text: req.Input, VoiceName: req.Voice}
	opts, err := buildOptions(&ttsReq)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid request", err.Error())
		return
	}

	audioSamples, err := s.synthesize(req.Input, req.Voice, opts)
	if err != nil {
		writeHTTPError(w, err)
		return
	}

	s.respondAudio(w, audioSamples, format)
}

// synthesize 串行化推理请求，返回音频样本。
// 错误会被包装为 *httpError 以便上层映射到合适的 HTTP 状态码。
func (s *Server) synthesize(text, voice string, opts inference.GenerateOptions) ([]float32, error) {
	if s.rt.Engine() == nil {
		return nil, &httpError{status: http.StatusServiceUnavailable, msg: "engine not initialized"}
	}

	s.requestLock.Lock()
	defer s.requestLock.Unlock()

	audioSamples, _, err := s.rt.Synthesize(text, voice, opts)
	if err != nil {
		return nil, &httpError{status: http.StatusInternalServerError, msg: "synthesis failed", detail: err.Error()}
	}
	return audioSamples, nil
}

// respondAudio 按 format 编码音频样本并写入响应。
// format 取值："wav"（audio/wav）或 "pcm"（audio/pcm，裸 int16 LE）。
func (s *Server) respondAudio(w http.ResponseWriter, samples []float32, format string) {
	var data []byte
	contentType := "audio/wav"

	if format == "pcm" {
		data = audio.Float32ToS16LE(samples)
		contentType = "audio/pcm"
	} else {
		var buf bytes.Buffer
		if err := audio.WriteWAV(&buf, samples, s.rt.Manifest().SampleRate); err != nil {
			writeError(w, http.StatusInternalServerError, "encode wav failed", err.Error())
			return
		}
		data = buf.Bytes()
	}

	w.Header().Set("Content-Type", contentType)
	w.Header().Set("Content-Length", fmt.Sprintf("%d", len(data)))
	w.WriteHeader(http.StatusOK)
	w.Write(data)
}

// buildOptions 从请求构造 GenerateOptions，未提供的字段使用默认值
// （与 Python service 默认一致：temperature=0.3, 其余同 CLI）。
func buildOptions(req *TtsRequest) (inference.GenerateOptions, error) {
	if len(req.Text) < 1 || len(req.Text) > 1000 {
		return inference.GenerateOptions{}, fmt.Errorf("text length must be 1..1000, got %d", len(req.Text))
	}
	if req.VoiceName == "" {
		return inference.GenerateOptions{}, fmt.Errorf("voice_name is required")
	}

	opts := inference.GenerateOptions{
		MaxNewTokens: 1024,
		Temperature:  0.3,
		TopP:         0.9,
		TopK:         50,
		Seed:         42,
	}

	if req.MaxNewTokens != nil {
		opts.MaxNewTokens = *req.MaxNewTokens
	}
	if opts.MaxNewTokens < 16 || opts.MaxNewTokens > 2048 {
		return inference.GenerateOptions{}, fmt.Errorf("max_new_tokens must be 16..2048, got %d", opts.MaxNewTokens)
	}

	if req.Temperature != nil {
		opts.Temperature = *req.Temperature
	}
	if opts.Temperature < 0 || opts.Temperature > 2 {
		return inference.GenerateOptions{}, fmt.Errorf("temperature must be 0..2, got %f", opts.Temperature)
	}

	if req.TopP != nil {
		opts.TopP = *req.TopP
	}
	if opts.TopP < 0 || opts.TopP > 1 {
		return inference.GenerateOptions{}, fmt.Errorf("top_p must be 0..1, got %f", opts.TopP)
	}

	if req.TopK != nil {
		opts.TopK = *req.TopK
	}
	if opts.TopK < 1 || opts.TopK > 4096 {
		return inference.GenerateOptions{}, fmt.Errorf("top_k must be 1..4096, got %d", opts.TopK)
	}

	if req.Seed != nil {
		opts.Seed = *req.Seed
	}

	return opts, nil
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg, detail string) {
	writeJSON(w, status, errorResponse{Error: msg, Detail: detail})
}

// httpError 携带 HTTP 状态码，便于 synthesize 等辅助函数
// 在出错时把状态码透传给上层 handler。
type httpError struct {
	status int
	msg    string
	detail string
}

func (e *httpError) Error() string {
	if e.detail != "" {
		return e.msg + ": " + e.detail
	}
	return e.msg
}

// writeHTTPError 若 err 是 *httpError 按其状态码输出，否则按 500 处理。
func writeHTTPError(w http.ResponseWriter, err error) {
	var he *httpError
	if errors.As(err, &he) {
		writeError(w, he.status, he.msg, he.detail)
		return
	}
	writeError(w, http.StatusInternalServerError, "internal error", err.Error())
}
