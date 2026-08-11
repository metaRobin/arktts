// Package registration 实现 voice 注册流程：音频解码 → 重采样 → codec 编码 → 原子写入。
//
// 对应 Python arktts_runtime/registration.py 的 VoiceRegistration 类。
package registration

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/metaRobin/arktts/audio"
	"github.com/metaRobin/arktts/config"
	"github.com/metaRobin/arktts/inference"
	"github.com/metaRobin/arktts/onnxruntime"
	"github.com/metaRobin/arktts/voices"
)

// StatusResponse 是 /api/registration/status 的响应。
type StatusResponse struct {
	Available         bool   `json:"available"`
	EncoderModel      string `json:"encoder_model"`
	ManifestExists    bool   `json:"manifest_exists"`
	FingerprintMatch  bool   `json:"fingerprint_match"`
	FFmpegAvailable   bool   `json:"ffmpeg_available"`
	ModelFingerprint  string `json:"model_fingerprint,omitempty"`
	LocalFingerprint  string `json:"local_fingerprint,omitempty"`
}

// Registration 管理 voice 注册流程。
type Registration struct {
	registrationDir string
	voicesRoot      string
	runtimeManifest *config.RuntimeManifest
	ortRuntime      *onnxruntime.RuntimeImpl
}

// New 创建 Registration。ortRuntime 可为 nil（仅 Status 检查文件是否存在）。
func New(ortRuntime *onnxruntime.RuntimeImpl, runtimeManifest *config.RuntimeManifest, registrationDir, voicesRoot string) *Registration {
	return &Registration{
		registrationDir: registrationDir,
		voicesRoot:      voicesRoot,
		runtimeManifest: runtimeManifest,
		ortRuntime:      ortRuntime,
	}
}

// Status 检查注册器可用性。对应 Python registration.status()。
func (r *Registration) Status() StatusResponse {
	encoderPath := inference.EncoderModelPath(r.registrationDir)
	manifestPath := filepath.Join(r.registrationDir, "registration_manifest.json")

	resp := StatusResponse{
		FFmpegAvailable: audio.HasFFmpeg(),
	}

	if _, err := os.Stat(encoderPath); err == nil {
		resp.EncoderModel = encoderPath
	}
	if _, err := os.Stat(manifestPath); err == nil {
		resp.ManifestExists = true
	}

	if resp.ManifestExists {
		if regManifest, err := config.LoadRegistrationManifest(r.registrationDir); err == nil {
			resp.LocalFingerprint = regManifest.ModelFingerprint
			resp.FingerprintMatch = regManifest.ModelFingerprint == r.runtimeManifest.ModelFingerprint
			resp.ModelFingerprint = r.runtimeManifest.ModelFingerprint
		}
	}

	// 可用条件：encoder 模型存在 + manifest 存在 + 指纹匹配 + ORT 已初始化
	resp.Available = resp.EncoderModel != "" && resp.ManifestExists && resp.FingerprintMatch && r.ortRuntime != nil

	return resp
}

// Register 执行完整 voice 注册流程。
// 对应 Python VoiceRegistration.register(name, reference_text, audio_data, overwrite)。
func (r *Registration) Register(name, referenceText string, audioData []byte, overwrite bool) error {
	// 1. 检查 encoder 可用性
	status := r.Status()
	if !status.Available {
		return fmt.Errorf("registration not available: encoder=%v manifest=%v fingerprint=%v ort=%v",
			status.EncoderModel != "", status.ManifestExists, status.FingerprintMatch, r.ortRuntime != nil)
	}

	// 2. 校验 name 和 reference_text
	if err := validateName(name); err != nil {
		return err
	}
	if referenceText == "" {
		return fmt.Errorf("reference_text is required")
	}

	// 3. 校验音频大小
	const maxAudioSize = 50 * 1024 * 1024 // 50 MiB
	if len(audioData) < 1 {
		return fmt.Errorf("audio data is empty")
	}
	if len(audioData) > maxAudioSize {
		return fmt.Errorf("audio data too large: %d bytes (max %d)", len(audioData), maxAudioSize)
	}

	// 4. 解码音频
	samples, srcSampleRate, err := audio.DecodeAudio(audioData)
	if err != nil {
		return fmt.Errorf("decode audio: %w", err)
	}

	// 5. 校验时长
	duration := float64(len(samples)) / float64(srcSampleRate)
	if duration < 0.5 || duration > 30.0 {
		return fmt.Errorf("audio duration %.2fs out of range [0.5, 30.0]", duration)
	}

	// 6. 校验有限值
	for i, s := range samples {
		if math.IsNaN(float64(s)) || math.IsInf(float64(s), 0) {
			return fmt.Errorf("audio contains non-finite value at sample %d", i)
		}
	}

	// 7. 重采样到目标采样率
	targetRate := r.runtimeManifest.SampleRate
	if srcSampleRate != targetRate {
		samples = audio.ResamplePoly(samples, srcSampleRate, targetRate)
	}

	// 8. pad 到 2048 倍数
	codecFrameSize := r.runtimeManifest.CodecFrameSize
	if codecFrameSize <= 0 {
		codecFrameSize = 2048
	}
	samples = audio.PadToMultiple(samples, codecFrameSize)

	// 9. 创建 encoder session 并编码
	encoder, err := inference.NewEncoder(r.ortRuntime, status.EncoderModel)
	if err != nil {
		return fmt.Errorf("create encoder: %w", err)
	}
	defer encoder.Close()

	codes, err := encoder.Encode(samples)
	if err != nil {
		return fmt.Errorf("encode audio: %w", err)
	}

	// 10. 校验 codes 形状
	numCodebooks := r.runtimeManifest.NumCodebooks
	if len(codes) != numCodebooks {
		return fmt.Errorf("encoder output codebooks %d != expected %d", len(codes), numCodebooks)
	}
	if len(codes[0]) == 0 {
		return fmt.Errorf("encoder produced empty codes")
	}

	// 11. 原子写入
	targetDir := filepath.Join(r.voicesRoot, name)
	if !overwrite {
		if _, err := os.Stat(targetDir); err == nil {
			return fmt.Errorf("voice %q already exists (use overwrite=true to replace)", name)
		}
	}

	// 创建临时目录
	tempDir, err := os.MkdirTemp(r.voicesRoot, "."+name+".*")
	if err != nil {
		return fmt.Errorf("create temp dir: %w", err)
	}

	cleanup := func() {
		os.RemoveAll(tempDir)
	}

	// 12. 写 codes.npy (uint16)
	codesUint16 := make([][]uint16, len(codes))
	for cb := range codes {
		codesUint16[cb] = make([]uint16, len(codes[cb]))
		for i, v := range codes[cb] {
			codesUint16[cb][i] = uint16(v)
		}
	}
	npyPath := filepath.Join(tempDir, "codes.npy")
	if err := voices.WriteNpyUint16File(npyPath, codesUint16); err != nil {
		cleanup()
		return fmt.Errorf("write codes.npy: %w", err)
	}

	// 13. 写 meta.json
	sha256Hash := sha256.Sum256(audioData)
	meta := voices.Meta{
		Name:             name,
		ReferenceText:    referenceText,
		Shape:            []int{numCodebooks, len(codes[0])},
		Dtype:            "uint16",
		SampleRate:       targetRate,
		SourceAudio:      "",
		SourceSampleRate: srcSampleRate,
		SourceSHA256:     hex.EncodeToString(sha256Hash[:]),
		ModelFingerprint: r.runtimeManifest.ModelFingerprint,
		CreatedAt:        time.Now().UTC().Format(time.RFC3339),
		SourceKind:       "web_registration",
	}
	metaPath := filepath.Join(tempDir, "meta.json")
	metaData, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		cleanup()
		return fmt.Errorf("marshal meta: %w", err)
	}
	if err := os.WriteFile(metaPath, metaData, 0644); err != nil {
		cleanup()
		return fmt.Errorf("write meta.json: %w", err)
	}

	// 14. 原子替换：删除旧目录 → rename 临时目录
	if overwrite {
		os.RemoveAll(targetDir)
	}
	if err := os.Rename(tempDir, targetDir); err != nil {
		cleanup()
		return fmt.Errorf("atomic rename voice dir: %w", err)
	}

	return nil
}

// validateName 校验 voice 名称。对齐 Python _validate_name。
func validateName(name string) error {
	if name == "" {
		return fmt.Errorf("name is required")
	}
	if len(name) > 64 {
		return fmt.Errorf("name too long: %d (max 64)", len(name))
	}
	if name == "." || name == ".." {
		return fmt.Errorf("name must not be %q", name)
	}
	// 单段路径：filepath.Base(name) == name
	if filepath.Base(name) != name {
		return fmt.Errorf("name must be a single path segment, got %q", name)
	}
	// 禁止路径分隔符
	if strings.ContainsAny(name, `/\`) {
		return fmt.Errorf("name must not contain path separators")
	}
	return nil
}
