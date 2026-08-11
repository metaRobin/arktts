package service

import (
	"net/http"
	stdruntime "runtime"
	"time"

	"github.com/metaRobin/arktts/registration"
)

// systemStats 返回系统内存与运行时信息。对应 GET /api/system。
//
// 用 runtime.MemStats 替代 Python 的 /usr/bin/footprint 子进程采样：
// 更准确（直接读 Go runtime 内部计数）、零 fork 开销、跨平台一致。
func (s *Server) systemStats(w http.ResponseWriter, r *http.Request) {
	var ms stdruntime.MemStats
	stdruntime.ReadMemStats(&ms)

	writeJSON(w, http.StatusOK, SystemResponse{
		AllocBytes:      ms.Alloc,
		TotalAllocBytes: ms.TotalAlloc,
		SysBytes:        ms.Sys,
		HeapInUseBytes:  ms.HeapInuse,
		StackInUseBytes: ms.StackInuse,
		NumGC:           ms.NumGC,
		LastGCNs:        ms.LastGC,
		UptimeSeconds:   time.Since(s.startedAt).Seconds(),
		EngineReady:     s.rt.Engine() != nil,
		Threads:         s.threads,
		GoVersion:       stdruntime.Version(),
		GOMAXPROCS:      stdruntime.GOMAXPROCS(0),
	})
}

// reloadRuntime 卸载并重新加载推理引擎。对应 POST /api/runtime/reload。
//
// 流程：
//  1. 持有 requestLock 串行化（与所有推理/注册请求互斥）
//  2. 调用 rt.ReloadEngine：关闭引擎 → GC → 重新加载模型
//  3. 重建 registration 实例（encoder session 引用已变更的 ORT）
func (s *Server) reloadRuntime(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed", "use POST")
		return
	}

	// 串行化：等所有进行中的推理/注册完成
	s.requestLock.Lock()
	defer s.requestLock.Unlock()

	// 取消任何进行中的流式（防御性，正常情况下 requestLock 已保证无活跃流）
	if s.cancelActive() {
		// 给被取消的流一点时间退出
		time.Sleep(10 * time.Millisecond)
	}

	if err := s.rt.ReloadEngine(); err != nil {
		writeError(w, http.StatusInternalServerError, "reload failed", err.Error())
		return
	}

	// 重建 registration（ORT 引用已更新）
	if s.rt.Engine() != nil {
		s.registration = registration.New(
			s.rt.Engine().ORT(),
			s.rt.Manifest(),
			s.registrationDir,
			s.rt.Voices().Root(),
		)
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"status":         "reloaded",
		"engine_ready":   s.rt.Engine() != nil,
		"uptime_seconds": time.Since(s.startedAt).Seconds(),
	})
}
