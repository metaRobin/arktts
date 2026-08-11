package service

import (
	"context"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"github.com/metaRobin/arktts/registration"
	"github.com/metaRobin/arktts/runtime"
)

// Server 是 TTS HTTP 服务，封装 runtime 与请求串行锁。
//
// requestLock 串行化所有推理请求（与 Python request_lock 一致），
// 因为 ONNX session 的并发 Run 不保证线程安全。
// activeStopMu 保护 activeStop，供 /api/tts/cancel 取消当前流式推理。
type Server struct {
	rt          *runtime.Runtime
	threads     int
	requestLock sync.Mutex

	registration    *registration.Registration
	registrationDir string

	// startedAt 记录服务启动时间，用于 /api/system 的 uptime 计算。
	startedAt time.Time

	activeStopMu sync.Mutex
	activeStop   context.CancelFunc
}

// New 创建 Server。rt 应已调用 InitEngine；否则 /api/health 报 engine_ready=false，
// /api/tts 返回 503。registrationDir 为 encoder 模型所在目录。
func New(rt *runtime.Runtime, threads int, registrationDir string) *Server {
	var reg *registration.Registration
	if rt.Engine() != nil {
		reg = registration.New(rt.Engine().ORT(), rt.Manifest(), registrationDir, rt.Voices().Root())
	}
	return &Server{
		rt:              rt,
		threads:         threads,
		registration:    reg,
		registrationDir: registrationDir,
		startedAt:       time.Now(),
	}
}

// Routes 返回 HTTP 路由树。
func (s *Server) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/health", s.health)
	mux.HandleFunc("/api/voices", s.listVoices)
	mux.HandleFunc("/api/tts", s.tts)
	mux.HandleFunc("/v1/audio/speech", s.openAISpeech)
	mux.HandleFunc("/api/tts/stream", s.streamTTS)
	mux.HandleFunc("/api/tts/cancel", s.cancelTTS)
	mux.HandleFunc("/api/registration/status", s.registrationStatus)
	mux.HandleFunc("/api/voices/register", s.registerVoice)
	mux.HandleFunc("/api/system", s.systemStats)
	mux.HandleFunc("/api/runtime/reload", s.reloadRuntime)
	mux.HandleFunc("/", s.indexPage)
	return s.logging(mux)
}

// setActiveStop 注册当前流式推理的 cancel 函数。
func (s *Server) setActiveStop(cancel context.CancelFunc) {
	s.activeStopMu.Lock()
	s.activeStop = cancel
	s.activeStopMu.Unlock()
}

// clearActiveStop 清除当前流式推理的 cancel 函数。
func (s *Server) clearActiveStop() {
	s.activeStopMu.Lock()
	s.activeStop = nil
	s.activeStopMu.Unlock()
}

// cancelActive 取消当前流式推理，返回是否有活跃推理被取消。
func (s *Server) cancelActive() bool {
	s.activeStopMu.Lock()
	cancel := s.activeStop
	s.activeStop = nil
	s.activeStopMu.Unlock()
	if cancel != nil {
		cancel()
		return true
	}
	return false
}

// logging 是请求日志中间件，记录方法、路径、状态码、耗时。
func (s *Server) logging(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		sw := &statusWriter{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(sw, r)
		slog.Info("http",
			slog.String("method", r.Method),
			slog.String("path", r.URL.Path),
			slog.Int("status", sw.status),
			slog.Int64("ms", time.Since(start).Milliseconds()),
		)
	})
}

// statusWriter 包装 ResponseWriter 以捕获状态码。
type statusWriter struct {
	http.ResponseWriter
	status int
}

func (w *statusWriter) WriteHeader(code int) {
	w.status = code
	w.ResponseWriter.WriteHeader(code)
}
