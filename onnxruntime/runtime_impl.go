package onnxruntime

import (
	"fmt"
	"log/slog"
	"sync"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

const maxInt64BufferCap = 2 * 1024 * 1024   // 2MB cap limit for int64 conversion buffers
const maxFloat32BufferCap = 2 * 1024 * 1024 // 2MB cap limit for float32 conversion buffers

var int64BufferPool = sync.Pool{
	New: func() any { return make([]int64, 0, 64*1024) }, // 64KB initial
}

var float32BufferPool = sync.Pool{
	New: func() any { return make([]float32, 0, 64*1024) }, // 64KB initial
}

func getInt64Buffer(minLen int) []int64 {
	v := int64BufferPool.Get()
	if v == nil {
		return make([]int64, minLen)
	}
	buf := v.([]int64)
	if cap(buf) < minLen {
		return make([]int64, minLen)
	}
	return buf[:minLen]
}

func putInt64Buffer(buf []int64) {
	if buf == nil || cap(buf) > maxInt64BufferCap {
		return
	}
	int64BufferPool.Put(buf[:0])
}

func getFloat32Buffer(minLen int) []float32 {
	v := float32BufferPool.Get()
	if v == nil {
		return make([]float32, minLen)
	}
	buf := v.([]float32)
	if cap(buf) < minLen {
		return make([]float32, minLen)
	}
	return buf[:minLen]
}

func putFloat32Buffer(buf []float32) {
	if buf == nil || cap(buf) > maxFloat32BufferCap {
		return
	}
	float32BufferPool.Put(buf[:0])
}

// RuntimeImpl implements the Runtime interface
type RuntimeImpl struct {
	loader      *LibraryLoader
	platform    PlatformInfo
	config      RuntimeConfig
	initialized bool
	// sessions    map[string]Session // Track loaded sessions by model path
	sessionParams map[string]SessionParams // Track loaded sessionParams by model path
	mu            sync.RWMutex
}

// SessionImpl implements the Session interface
type SessionImpl struct {
	session     *ort.DynamicAdvancedSession
	inputNames  []string
	outputNames []string
	modelPath   string
	mu          sync.RWMutex
}

type SessionParams struct {
	InputNames  []string
	OutputNames []string
	Options     *ort.SessionOptions
}

// NewRuntime creates a new ONNX runtime instance
func NewRuntime() *RuntimeImpl {
	return &RuntimeImpl{
		sessionParams: make(map[string]SessionParams),
	}
}

// Initialize initializes the runtime with the given configuration
func (r *RuntimeImpl) Initialize(config RuntimeConfig) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.initialized {
		return NewRuntimeError("RUNTIME_ALREADY_INITIALIZED", "Runtime already initialized", nil)
	}

	r.config = config
	r.loader = NewLibraryLoader(config.LibraryPath)
	r.platform = r.loader.DetectPlatform()

	// Load the appropriate library
	if err := r.loader.LoadLibrary(r.platform); err != nil {
		slog.Error("Failed to load ONNX runtime library", slog.String("file", getFileLine()), slog.String("error", err.Error()))
		return NewRuntimeError("LIBRARY_LOAD_FAILED", "Failed to load ONNX runtime library", err)
	}

	if err := ort.InitializeEnvironment(ort.WithLogLevelWarning()); err != nil {
		slog.Error("Failed to initialize ONNX runtime environment", slog.String("file", getFileLine()), slog.String("error", err.Error()),
			slog.String("library_path", r.platform.LibraryPath))
		return NewRuntimeError("RUNTIME_INIT_FAILED", "Failed to initialize ONNX runtime environment", err)
	}

	r.initialized = true

	slog.Info("ONNX runtime initialized successfully",
		slog.Int("intra_op_threads", r.config.IntraOpThreads), slog.Int("inter_op_threads", r.config.InterOpThreads))
	return nil
}

// CreateSession creates a new session for the given model path
func (r *RuntimeImpl) CreateSession(modelPath string) (Session, error) {
	// Validate model path
	if modelPath == "" {
		return nil, NewValidationError("EMPTY_MODEL_PATH",
			"Model path cannot be empty", nil)
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	// Check if runtime is initialized
	if !r.initialized {
		return nil, NewRuntimeError("RUNTIME_NOT_INITIALIZED",
			"Runtime must be initialized before loading models", nil)
	}

	// Check if session already exists for this model
	if sessionParams, exists := r.sessionParams[modelPath]; exists {
		slog.Info("SessionParams already exists for model", slog.String("model_path", modelPath))
		session, err := r.createDynamicAdvancedSession(modelPath, sessionParams.InputNames, sessionParams.OutputNames)
		if err != nil {
			return nil, NewModelError("SESSION_CREATION_FAILED", fmt.Sprintf("Failed to create session for %s", modelPath), err)
		}
		return &SessionImpl{
			session:     session,
			inputNames:  sessionParams.InputNames,
			outputNames: sessionParams.OutputNames,
			modelPath:   modelPath,
		}, nil
	}

	// Get input/output information first
	start := time.Now()
	inputs, outputs, err := ort.GetInputOutputInfo(modelPath)
	if err != nil {
		slog.Error("Failed to get model info", slog.String("file", getFileLine()),
			slog.String("model_path", modelPath), slog.String("error", err.Error()))
		return nil, NewModelError("MODEL_INFO_FAILED", fmt.Sprintf("Failed to get model info from %s", modelPath), err)
	}

	slog.Info("Model metadata loaded", slog.String("file", getFileLine()),
		slog.String("model_path", modelPath), slog.Duration("cost_ms", time.Duration(time.Since(start).Milliseconds())))

	// Extract input and output names
	inputNames := make([]string, len(inputs))
	outputNames := make([]string, len(outputs))

	for i, input := range inputs {
		inputNames[i] = input.Name
	}

	for i, output := range outputs {
		outputNames[i] = output.Name
	}

	// Create dynamic advanced session
	session, err := r.createDynamicAdvancedSession(modelPath, inputNames, outputNames)
	if err != nil {
		return nil, NewModelError("SESSION_CREATION_FAILED", fmt.Sprintf("Failed to create session for %s", modelPath), err)
	}

	// Store session params for reuse
	r.sessionParams[modelPath] = SessionParams{
		InputNames:  inputNames,
		OutputNames: outputNames,
	}

	sessionImpl := &SessionImpl{
		session:     session,
		inputNames:  inputNames,
		outputNames: outputNames,
		modelPath:   modelPath,
	}

	return sessionImpl, nil
}

func (r *RuntimeImpl) createDynamicAdvancedSession(modelPath string, inputNames []string, outputNames []string) (*ort.DynamicAdvancedSession, error) {
	sessionStart := time.Now()
	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, NewModelError("SESSION_OPTIONS_FAILED",
			"Failed to create session options", err)
	}
	defer options.Destroy()

	// Set thread count if specified
	if r.config.IntraOpThreads > 0 {
		if e := options.SetIntraOpNumThreads(r.config.IntraOpThreads); e != nil {
			return nil, NewConfigError("THREAD_CONFIG_FAILED", "Failed to set thread count", e)
		}
	}

	if r.config.InterOpThreads > 0 {
		if e := options.SetInterOpNumThreads(r.config.InterOpThreads); e != nil {
			return nil, NewConfigError("THREAD_CONFIG_FAILED", "Failed to set inter op num threads", e)
		}
	}

	// Use sequential execution mode to match Python onnxruntime defaults.
	// Parallel mode can cause non-deterministic floating-point results
	// due to different reduction orders in multi-threaded execution.
	if e := options.SetExecutionMode(ort.ExecutionModeSequential); e != nil {
		return nil, NewConfigError("THREAD_CONFIG_FAILED", "Failed to set ExecutionMode", e)
	}

	// Enable CPU memory arena if configured
	if r.config.EnableCPUArena {
		if e := options.SetCpuMemArena(true); e != nil {
			return nil, NewConfigError("CPU_ARENA_CONFIG_FAILED", "Failed to enable CPU memory arena", e)
		}
	}

	if e := options.SetMemPattern(true); e != nil {
		return nil, NewConfigError("MEM_PATTERN_CONFIG_FAILED", "Failed to set mem pattern", e)
	}

	if e := options.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableAll); e != nil {
		return nil, NewConfigError("GRAPH_OPTIMIZATION_FAILED", "Failed to set graph optimization level", e)
	}

	session, err := ort.NewDynamicAdvancedSession(modelPath, inputNames, outputNames, options)
	if err != nil {
		slog.Error("Failed to create session for model", slog.String("model_path", modelPath),
			slog.String("error", err.Error()), slog.String("file", getFileLine()))
		return nil, NewModelError("MODEL_LOAD_FAILED", fmt.Sprintf("Failed to load model from %s", modelPath), err)
	}

	slog.Info("ONNX session created", slog.String("file", getFileLine()), slog.Duration("cost_ms", time.Duration(time.Since(sessionStart).Milliseconds())))
	return session, nil
}

// GetPlatformInfo returns platform information
func (r *RuntimeImpl) GetPlatformInfo() PlatformInfo {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.platform
}

// Cleanup cleans up runtime resources
func (r *RuntimeImpl) Cleanup() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Destroy all sessions first
	for modelPath, sessionParams := range r.sessionParams {
		if sessionParams.Options != nil {
			if err := sessionParams.Options.Destroy(); err != nil {
				slog.Warn("Failed to destroy session options for model", slog.String("model_path", modelPath),
					slog.String("error", err.Error()))
			}
		}
		delete(r.sessionParams, modelPath)
	}

	// Clear sessions map
	r.sessionParams = make(map[string]SessionParams)

	// Destroy ONNX runtime environment
	if r.initialized {
		ort.DestroyEnvironment()
		r.initialized = false
	}

	return nil
}

// GetLoadedModels returns a list of currently loaded model paths
func (r *RuntimeImpl) GetLoadedModels() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	models := make([]string, 0, len(r.sessionParams))
	for modelPath := range r.sessionParams {
		models = append(models, modelPath)
	}
	return models
}

// UnloadModel removes a specific model session from memory
func (r *RuntimeImpl) UnloadModel(modelPath string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	sessionParams, exists := r.sessionParams[modelPath]
	if !exists {
		return NewValidationError("MODEL_NOT_LOADED",
			fmt.Sprintf("Model %s is not currently loaded", modelPath), nil)
	}

	if err := sessionParams.Options.Destroy(); err != nil {
		return NewRuntimeError("SESSION_DESTROY_FAILED",
			fmt.Sprintf("Failed to destroy session for model %s", modelPath), err)
	}

	delete(r.sessionParams, modelPath)
	return nil
}

// IsInitialized returns whether the runtime has been initialized
func (r *RuntimeImpl) IsInitialized() bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.initialized
}

// RunInference executes inference on the session
func (s *SessionImpl) RunInference(inputs map[string]any, inputShapes map[string][]int64) (map[string][]float32, error) {
	if s.session == nil {
		slog.Error("Inference failed: session is not initialized", slog.String("file", getFileLine()), slog.String("model_path", s.modelPath))
		return nil, NewRuntimeError("SESSION_NULL", "Session is not initialized", nil)
	}

	// Validate inputs
	if len(inputs) == 0 {
		slog.Error("Inference failed: no input tensors provided", slog.String("file", getFileLine()), slog.String("model_path", s.modelPath))
		return nil, NewValidationError("EMPTY_INPUTS", "No input tensors provided", nil)
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	stat := struct {
		input  int64
		run    int64
		output int64
	}{}
	start := time.Now()
	// Pre-allocate input tensors slice
	inputTensors := make([]ort.Value, 0, len(s.inputNames))
	for _, inputName := range s.inputNames {
		data, exists := inputs[inputName]
		if !exists {
			return nil, NewValidationError("MISSING_INPUT", fmt.Sprintf("Missing input tensor: %s", inputName), nil)
		}

		shape, exists := inputShapes[inputName]
		if !exists {
			return nil, NewValidationError("MISSING_INPUT_SHAPE", fmt.Sprintf("Missing input shape for tensor: %s", inputName), nil)
		}

		// Create tensor based on data type
		var tensor ort.Value
		var err error

		switch v := data.(type) {
		case []int32:
			// Convert int32 to int64 for ONNX (reuse buffer from pool)
			int64Data := getInt64Buffer(len(v))
			defer putInt64Buffer(int64Data)
			for i, val := range v {
				int64Data[i] = int64(val)
			}
			tensor, err = ort.NewTensor(shape, int64Data)
		case []int64:
			tensor, err = ort.NewTensor(shape, v)
		case []float32:
			tensor, err = ort.NewTensor(shape, v)
		case []float64:
			// Convert float64 to float32 (reuse buffer from pool)
			float32Data := getFloat32Buffer(len(v))
			defer putFloat32Buffer(float32Data)
			for i, val := range v {
				float32Data[i] = float32(val)
			}
			tensor, err = ort.NewTensor(shape, float32Data)
		default:
			return nil, NewValidationError("UNSUPPORTED_DATA_TYPE",
				fmt.Sprintf("Unsupported data type for input %s: %T", inputName, data), nil)
		}

		if err != nil {
			return nil, NewRuntimeError("TENSOR_CREATION_FAILED",
				fmt.Sprintf("Failed to create tensor for input %s", inputName), err)
		}

		inputTensors = append(inputTensors, tensor)
	}

	// Ensure tensors are cleaned up
	defer func() {
		for _, tensor := range inputTensors {
			tensor.Destroy()
		}
	}()

	outputTensors := make([]ort.Value, len(s.outputNames))
	stat.input = time.Since(start).Milliseconds()

	inferenceStart := time.Now()
	err := s.session.Run(inputTensors, outputTensors)
	if err != nil {
		slog.Error("Inference failed for model", slog.String("file", getFileLine()),
			slog.String("model_path", s.modelPath), slog.String("error", err.Error()))
		return nil, NewRuntimeError("INFERENCE_FAILED", "Model inference failed", err)
	}
	stat.run = time.Since(inferenceStart).Milliseconds()
	inferenceStart = time.Now()

	defer func() {
		for _, tensor := range outputTensors {
			if tensor != nil {
				tensor.Destroy()
			}
		}
	}()

	// Pre-allocate output map to avoid rehashing
	outputs := make(map[string][]float32, len(s.outputNames))
	for i, outputName := range s.outputNames {
		if i >= len(outputTensors) || outputTensors[i] == nil {
			return nil, NewRuntimeError("OUTPUT_MISMATCH",
				fmt.Sprintf("Expected output %s not found in results", outputName), nil)
		}

		// Get the output tensor
		outputTensor := outputTensors[i]

		// Extract data from the ONNX tensor
		// Get tensor info
		tensorType := outputTensor.DataType()

		// Handle different tensor types
		switch tensorType {
		case ort.TensorElementDataTypeFloat:
			tensor, ok := outputTensor.(*ort.Tensor[float32])
			if !ok {
				return nil, NewRuntimeError("TENSOR_TYPE_ASSERTION_FAILED",
					fmt.Sprintf("Failed to type assert output tensor %s to *ort.Tensor[float32]", outputName), nil)
			}
			outputs[outputName] = tensor.GetData()
		case ort.TensorElementDataTypeInt64:
			tensor, ok := outputTensor.(*ort.Tensor[int64])
			if !ok {
				return nil, NewRuntimeError("TENSOR_TYPE_ASSERTION_FAILED",
					fmt.Sprintf("Failed to type assert output tensor %s to *ort.Tensor[int64]", outputName), nil)
			}
			int64Data := tensor.GetData()
			float32Data := make([]float32, len(int64Data))
			for i, v := range int64Data {
				float32Data[i] = float32(v)
			}
			outputs[outputName] = float32Data
		default:
			return nil, NewRuntimeError("UNSUPPORTED_TENSOR_TYPE",
				fmt.Sprintf("Unsupported tensor type for output %s: %v", outputName, tensorType), nil)
		}
	}
	stat.output = time.Since(inferenceStart).Milliseconds()

	return outputs, nil
}

// GetInputInfo returns information about model inputs
func (s *SessionImpl) GetInputInfo() []TensorInfo {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if len(s.inputNames) == 0 {
		return nil
	}

	info := make([]TensorInfo, len(s.inputNames))
	for i, name := range s.inputNames {
		// For BERT-like models, typical input shapes
		var shape []int64
		var tensorType string

		switch name {
		case "input_ids", "attention_mask", "token_type_ids":
			shape = []int64{-1, -1} // Dynamic batch size and sequence length
			tensorType = "int64"
		default:
			shape = []int64{-1, -1} // Default dynamic shape
			tensorType = "float32"
		}

		info[i] = TensorInfo{
			Name:  name,
			Shape: shape,
			Type:  tensorType,
		}
	}
	return info
}

// GetOutputInfo returns information about model outputs
func (s *SessionImpl) GetOutputInfo() []TensorInfo {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if len(s.outputNames) == 0 {
		return nil
	}

	info := make([]TensorInfo, len(s.outputNames))
	for i, name := range s.outputNames {
		// For sentence transformer models, typical output is embeddings
		var shape []int64
		var tensorType string

		switch name {
		case "last_hidden_state", "pooler_output", "sentence_embedding":
			shape = []int64{-1, 384} // Dynamic batch size, 384 embedding dimension for all-MiniLM-L6-v2
			tensorType = "float32"
		default:
			shape = []int64{-1, -1} // Default dynamic shape
			tensorType = "float32"
		}

		info[i] = TensorInfo{
			Name:  name,
			Shape: shape,
			Type:  tensorType,
		}
	}
	return info
}

// Destroy releases the session resources
func (s *SessionImpl) Destroy() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.session != nil {
		s.session.Destroy()
		s.session = nil
	}

	return nil
}

// GetModelPath returns the path of the model used by this session
func (s *SessionImpl) GetModelPath() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.modelPath
}

// IsValid returns whether the session is still valid and usable
func (s *SessionImpl) IsValid() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.session != nil
}

// GetInputNames returns the names of all input tensors
func (s *SessionImpl) GetInputNames() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()

	names := make([]string, len(s.inputNames))
	copy(names, s.inputNames)
	return names
}

// GetOutputNames returns the names of all output tensors
func (s *SessionImpl) GetOutputNames() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()

	names := make([]string, len(s.outputNames))
	copy(names, s.outputNames)
	return names
}
