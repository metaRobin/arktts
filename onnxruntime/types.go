package onnxruntime

// PlatformInfo contains information about the current platform
type PlatformInfo struct {
	OS           string // linux, darwin, windows
	Architecture string // amd64, arm64
	LibraryPath  string // path to the corresponding ONNX runtime library
	Supported    bool   // whether the platform is supported
}

// RuntimeConfig contains configuration for ONNX runtime
type RuntimeConfig struct {
	LibraryPath    string // path to ONNX runtime libraries
	IntraOpThreads int    // number of threads, 0 = auto
	InterOpThreads int    // number of threads, 0 = auto
	EnableCPUArena bool   // enable CPU memory arena
}

// TensorInfo contains information about tensor inputs/outputs
type TensorInfo struct {
	Name  string
	Shape []int64
	Type  string
}

// TensorType represents the data type of tensors
type TensorType int

const (
	TensorTypeInt32 TensorType = iota
	TensorTypeFloat32
	TensorTypeInt64
)

// TensorData represents tensor data with shape and type information
type TensorData struct {
	Shape []int64
	Data  any // []int32 for input_ids, []float32 for embeddings
	Type  TensorType
}

// Runtime interface defines the ONNX runtime wrapper
type Runtime interface {
	// Initialize the runtime and load appropriate library
	Initialize(config RuntimeConfig) error

	// Create ONNX model session
	CreateSession(modelPath string) (Session, error)

	// Get platform information
	GetPlatformInfo() PlatformInfo

	// Session management
	GetLoadedModels() []string
	UnloadModel(modelPath string) error
	IsInitialized() bool

	// Cleanup resources
	Cleanup() error
}

// Session interface defines model session operations
type Session interface {
	// Run inference with input tensors and return output tensors
	RunInference(inputs map[string]any, inputShapes map[string][]int64) (map[string][]float32, error)

	// Get input/output information
	GetInputInfo() []TensorInfo
	GetOutputInfo() []TensorInfo

	// Get input/output names
	GetInputNames() []string
	GetOutputNames() []string

	// Get model path
	GetModelPath() string

	// Check if session is valid
	IsValid() bool

	// Release session
	Destroy() error
}
