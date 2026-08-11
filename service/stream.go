package service

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"

	"github.com/metaRobin/arktts/audio"
)

// streamTTS 以 NDJSON 流式合成语音。对应 POST /api/tts/stream。
//
// 协议（每行一个 JSON 对象）：
//
//	start:        {event:"start", sample_rate, channels:1, sample_format:"s16le", precision, codec_precision}
//	audio_chunk:  {event:"audio_chunk", seq, frame_count, pcm_b64}
//	complete:     {event:"complete", frame_count}
//	cancelled:    {event:"cancelled"}
//
// 可被 /api/tts/cancel 取消；客户端断开连接也会通过 r.Context() 取消。
func (s *Server) streamTTS(w http.ResponseWriter, r *http.Request) {
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

	if s.rt.Engine() == nil {
		writeError(w, http.StatusServiceUnavailable, "engine not initialized", "")
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming not supported", "")
		return
	}

	// ctx 绑定客户端连接 + 可被 cancelTTS 取消
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()

	s.setActiveStop(cancel)
	defer s.clearActiveStop()

	// 串行化推理（与 /api/tts 一致）
	s.requestLock.Lock()
	defer s.requestLock.Unlock()

	// 获取锁后再次检查 ctx（可能在等锁期间被取消）
	if ctx.Err() != nil {
		return
	}

	events, err := s.rt.Stream(ctx, req.Text, req.VoiceName, opts, 12)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "stream init failed", err.Error())
		return
	}

	// NDJSON 响应头
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no") // nginx 禁用缓冲
	w.WriteHeader(http.StatusOK)

	enc := json.NewEncoder(w)
	m := s.rt.Manifest()

	// start 事件
	if err := enc.Encode(streamStartEvent{
		Event:          "start",
		SampleRate:     m.SampleRate,
		Channels:       1,
		SampleFormat:   "s16le",
		Precision:      m.DefaultPrecision,
		CodecPrecision: m.DefaultCodecPrecision,
	}); err != nil {
		return
	}
	flusher.Flush()

	// 逐事件输出
	for ev := range events {
		// 客户端断开或被取消
		if ctx.Err() != nil {
			enc.Encode(streamCancelledEvent{Event: "cancelled"})
			flusher.Flush()
			return
		}

		switch ev.Type {
		case "audio_chunk":
			pcm := audio.Float32ToS16LE(ev.Audio)
			if err := enc.Encode(streamAudioChunkEvent{
				Event:      "audio_chunk",
				Seq:        ev.Seq,
				FrameCount: ev.FrameCount,
				PCMBase64:  base64.StdEncoding.EncodeToString(pcm),
			}); err != nil {
				return
			}
		case "complete":
			enc.Encode(streamCompleteEvent{
				Event:      "complete",
				FrameCount: ev.FrameCount,
			})
		case "error":
			enc.Encode(errorResponse{Error: "stream error", Detail: ev.Err.Error()})
		}
		flusher.Flush()
	}
}

// cancelTTS 取消当前流式推理。对应 POST /api/tts/cancel。
// 不需要 requestLock（不涉及推理，只调 cancel）。
func (s *Server) cancelTTS(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed", "use POST")
		return
	}

	if s.cancelActive() {
		writeJSON(w, http.StatusOK, map[string]string{"status": "cancelled"})
	} else {
		writeError(w, http.StatusNotFound, "no active stream", "")
	}
}
