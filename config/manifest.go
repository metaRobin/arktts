// Package config 解析模型 manifest 文件。
package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// RuntimeManifest 对应 model/runtime_manifest.json。
type RuntimeManifest struct {
	ModelFamily              string            `json:"model_family"`
	ActivationDtype          string            `json:"activation_dtype"`
	SlowLogitsLayout         string            `json:"slow_logits_layout"`
	SlowLogitsSize           int               `json:"slow_logits_size"`
	KVAttentionLayout        string            `json:"kv_attention_layout"`
	MaxSeqLen                int               `json:"max_seq_len"`
	NumLayers                int               `json:"num_layers"`
	NumFastLayers            int               `json:"num_fast_layers"`
	NumCodebooks             int               `json:"num_codebooks"`
	NLocalHeads              int               `json:"n_local_heads"`
	FastNLocalHeads          int               `json:"fast_n_local_heads"`
	HeadDim                  int               `json:"head_dim"`
	FastHeadDim              int               `json:"fast_head_dim"`
	FastDim                  int               `json:"fast_dim"`
	VocabSize                int               `json:"vocab_size"`
	CodebookSize             int               `json:"codebook_size"`
	SemanticBeginID          int               `json:"semantic_begin_id"`
	SemanticEndID            int               `json:"semantic_end_id"`
	EosTokenID               int               `json:"eos_token_id"`
	PadTokenID               int               `json:"pad_token_id"`
	CodecSampleRate          int               `json:"codec_sample_rate"`
	CodecFrameSize           int               `json:"codec_frame_size"`
	SampleRate               int               `json:"sample_rate"`
	CodecHopLength           int               `json:"codec_hop_length"`
	StreamContextFrames      int               `json:"stream_context_frames"`
	StreamGuardFrames        int               `json:"stream_guard_frames"`
	DecoderProvider          string            `json:"decoder_provider"`
	DefaultCodecPrecision    string            `json:"default_codec_precision"`
	AvailableCodecPrecisions []string          `json:"available_codec_precisions"`
	CodecModels              map[string]string `json:"codec_models"`
	ImEndID                  int               `json:"im_end_id"`
	ModelFingerprint         string            `json:"model_fingerprint"`
	DefaultPrecision         string            `json:"default_precision"`
	AvailablePrecisions      []string          `json:"available_precisions"`
}

// RegistrationManifest 对应 model/registration_manifest.json。
type RegistrationManifest struct {
	SampleRate       int    `json:"sample_rate"`
	NumCodebooks     int    `json:"num_codebooks"`
	FrameLength      int    `json:"frame_length"`
	ModelFingerprint string `json:"model_fingerprint"`
}

// LoadRuntimeManifest 从 modelDir 读取 runtime_manifest.json。
func LoadRuntimeManifest(modelDir string) (*RuntimeManifest, error) {
	return loadJSON[RuntimeManifest](filepath.Join(modelDir, "runtime_manifest.json"))
}

// LoadRegistrationManifest 从 modelDir 读取 registration_manifest.json。
func LoadRegistrationManifest(modelDir string) (*RegistrationManifest, error) {
	return loadJSON[RegistrationManifest](filepath.Join(modelDir, "registration_manifest.json"))
}

func loadJSON[T any](path string) (*T, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", filepath.Base(path), err)
	}
	var v T
	if err := json.Unmarshal(data, &v); err != nil {
		return nil, fmt.Errorf("parse %s: %w", filepath.Base(path), err)
	}
	return &v, nil
}
