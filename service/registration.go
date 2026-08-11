package service

import (
	"encoding/json"
	"net/http"
)

// registrationStatus 返回注册器可用性。对应 GET /api/registration/status。
func (s *Server) registrationStatus(w http.ResponseWriter, r *http.Request) {
	if s.registration == nil {
		writeJSON(w, http.StatusOK, map[string]any{
			"available":          false,
			"ffmpeg_available":   false,
			"fingerprint_match":  false,
			"manifest_exists":    false,
		})
		return
	}
	status := s.registration.Status()
	writeJSON(w, http.StatusOK, status)
}

// registerVoice 处理 multipart 上传音频并注册 voice。
// 对应 POST /api/voices/register。
//
// 表单字段：
//
//	name:           voice 名称（必填）
//	reference_text: 参考文本（必填）
//	audio:          音频文件（必填，wav/mp3/flac/m4a）
//	overwrite:      "true" 覆盖已存在的 voice（可选，默认 false）
func (s *Server) registerVoice(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed", "use POST")
		return
	}

	if s.registration == nil {
		writeError(w, http.StatusServiceUnavailable, "registration not initialized", "")
		return
	}

	// 限制上传大小 50 MiB
	if err := r.ParseMultipartForm(50 << 20); err != nil {
		writeError(w, http.StatusBadRequest, "parse multipart form failed", err.Error())
		return
	}

	name := r.FormValue("name")
	referenceText := r.FormValue("reference_text")
	overwrite := r.FormValue("overwrite") == "true"

	if name == "" {
		writeError(w, http.StatusBadRequest, "name is required", "")
		return
	}
	if referenceText == "" {
		writeError(w, http.StatusBadRequest, "reference_text is required", "")
		return
	}

	file, _, err := r.FormFile("audio")
	if err != nil {
		writeError(w, http.StatusBadRequest, "audio file required", err.Error())
		return
	}
	defer file.Close()

	// 读取音频数据
	buf := make([]byte, 0, 1024*1024)
	chunk := make([]byte, 4096)
	for {
		n, err := file.Read(chunk)
		if n > 0 {
			buf = append(buf, chunk[:n]...)
		}
		if err != nil {
			break
		}
	}

	// 串行化注册请求
	s.requestLock.Lock()
	err = s.registration.Register(name, referenceText, buf, overwrite)
	s.requestLock.Unlock()

	if err != nil {
		writeError(w, http.StatusInternalServerError, "registration failed", err.Error())
		return
	}

	// 注册成功后刷新 voice 缓存
	s.rt.ReloadVoice(name)

	writeJSON(w, http.StatusOK, map[string]string{
		"status": "registered",
		"name":   name,
	})
}

// 避免未使用的 import（json 在 writeJSON 中使用）
var _ = json.NewEncoder
