package onnxruntime

import (
	"fmt"
	"log/slog"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

// RunWithValues executes inference using pre-built ort.Value tensors.
// Unlike RunInference, this method supports arbitrary tensor types
// (float16, bool, etc.) and gives the caller full control over tensor
// lifecycle. The caller is responsible for destroying both input and
// output tensors.
//
// outputs must be pre-allocated with the correct length (len(outputNames)).
// Entries set to nil will be auto-allocated by ONNX Runtime and filled in-place.
func (s *SessionImpl) RunWithValues(inputs, outputs []ort.Value) error {
	if s.session == nil {
		return NewRuntimeError("SESSION_NULL", "Session is not initialized", nil)
	}
	if len(inputs) != len(s.inputNames) {
		return NewValidationError("INPUT_COUNT_MISMATCH",
			fmt.Sprintf("expected %d inputs, got %d", len(s.inputNames), len(inputs)), nil)
	}
	if len(outputs) != len(s.outputNames) {
		return NewValidationError("OUTPUT_COUNT_MISMATCH",
			fmt.Sprintf("expected %d outputs, got %d", len(s.outputNames), len(outputs)), nil)
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	start := time.Now()
	if err := s.session.Run(inputs, outputs); err != nil {
		slog.Error("RunWithValues failed", slog.String("file", getFileLine()),
			slog.String("model_path", s.modelPath), slog.String("error", err.Error()),
			slog.Duration("cost_ms", time.Duration(time.Since(start).Milliseconds())))
		return NewRuntimeError("INFERENCE_FAILED", "Model inference failed", err)
	}
	return nil
}

// GetModelInputOutputInfo returns the actual input and output metadata
// (names, shapes, data types) for the loaded model. This is needed to
// determine tensor types (e.g., float16 vs float32 for KV caches).
func (s *SessionImpl) GetModelInputOutputInfo() ([]ort.InputOutputInfo, []ort.InputOutputInfo, error) {
	inputs, outputs, err := ort.GetInputOutputInfo(s.modelPath)
	if err != nil {
		return nil, nil, NewModelError("MODEL_INFO_FAILED",
			fmt.Sprintf("Failed to get model info from %s", s.modelPath), err)
	}
	return inputs, outputs, nil
}
