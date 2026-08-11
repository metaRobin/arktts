package onnxruntime

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"

	ort "github.com/yalue/onnxruntime_go"
)

// LibraryLoader handles platform detection and library loading
type LibraryLoader struct {
	basePath string
}

// NewLibraryLoader creates a new library loader
func NewLibraryLoader(basePath string) *LibraryLoader {
	return &LibraryLoader{
		basePath: basePath,
	}
}

// DetectPlatform detects the current platform and returns platform information
func (l *LibraryLoader) DetectPlatform() PlatformInfo {
	osName := runtime.GOOS
	arch := runtime.GOARCH

	var libraryName string
	var supported bool

	switch {
	case osName == "linux" && arch == "amd64":
		libraryName = "libonnxruntime_amd64.so"
		supported = true
	case osName == "linux" && arch == "arm64":
		libraryName = "libonnxruntime_aarch64.so"
		supported = true
	case osName == "darwin" && arch == "arm64":
		libraryName = "libonnxruntime_arm64.dylib"
		supported = true
	default:
		libraryName = ""
		supported = false
	}

	libraryPath := ""
	// Check if basePath already points to a specific library file
	if info, err := os.Stat(l.basePath); err == nil && info.Mode().IsRegular() {
		// basePath is already a file, use it directly
		libraryPath = l.basePath
	} else if libraryName != "" {
		// basePath is a directory, append the library name
		libraryPath = filepath.Join(l.basePath, libraryName)
	}

	return PlatformInfo{
		OS:           osName,
		Architecture: arch,
		LibraryPath:  libraryPath,
		Supported:    supported,
	}
}

// LoadLibrary loads the ONNX runtime library for the given platform
func (l *LibraryLoader) LoadLibrary(platform PlatformInfo) error {
	// Check if platform is supported
	if !platform.Supported {
		return NewPlatformError(
			"UNSUPPORTED_PLATFORM",
			fmt.Sprintf("unsupported platform: %s/%s", platform.OS, platform.Architecture),
			nil,
		).WithContext("os", platform.OS).WithContext("architecture", platform.Architecture)
	}

	if platform.LibraryPath == "" {
		return NewValidationError(
			"EMPTY_LIBRARY_PATH",
			fmt.Sprintf("empty library path for platform: %s/%s", platform.OS, platform.Architecture),
			nil,
		).WithContext("os", platform.OS).WithContext("architecture", platform.Architecture)
	}

	// Validate library path
	if err := l.validateLibraryPath(platform.LibraryPath); err != nil {
		return NewLibraryLoadError(
			"LIBRARY_VALIDATION_FAILED",
			"library validation failed",
			err,
		).WithContext("library_path", platform.LibraryPath)
	}

	slog.Info("SetSharedLibraryPath", slog.String("file", getFileLine()), slog.String("library_path", platform.LibraryPath),
		slog.String("os", platform.OS), slog.String("architecture", platform.Architecture))

	// Set the shared library path
	ort.SetSharedLibraryPath(platform.LibraryPath)
	return nil
}

// validateLibraryPath validates that the library file exists and is accessible
func (l *LibraryLoader) validateLibraryPath(libraryPath string) error {
	// Check if library file exists
	info, err := os.Stat(libraryPath)
	if os.IsNotExist(err) {
		return NewValidationError(
			"LIBRARY_NOT_FOUND",
			fmt.Sprintf("ONNX runtime library not found: %s", libraryPath),
			err,
		)
	}
	if err != nil {
		return NewValidationError(
			"LIBRARY_ACCESS_FAILED",
			fmt.Sprintf("failed to access library file %s", libraryPath),
			err,
		)
	}

	// Check if it's a regular file
	if !info.Mode().IsRegular() {
		return NewValidationError(
			"LIBRARY_NOT_REGULAR_FILE",
			fmt.Sprintf("library path is not a regular file: %s", libraryPath),
			nil,
		)
	}

	// Check file permissions (readable)
	file, err := os.Open(libraryPath)
	if err != nil {
		return NewValidationError(
			"LIBRARY_NOT_READABLE",
			fmt.Sprintf("library file is not readable: %s", libraryPath),
			err,
		)
	}
	file.Close()

	return nil
}

// ResolvePath resolves the library path, supporting both absolute and relative paths
func (l *LibraryLoader) ResolvePath(path string) (string, error) {
	if filepath.IsAbs(path) {
		return path, nil
	}

	// For relative paths, resolve from current working directory
	wd, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("failed to get working directory: %w", err)
	}

	resolved := filepath.Join(wd, path)
	return filepath.Clean(resolved), nil
}

// GetSupportedPlatforms returns a list of all supported platform combinations
func GetSupportedPlatforms() []PlatformInfo {
	return []PlatformInfo{
		{OS: "linux", Architecture: "amd64", Supported: true},
		{OS: "linux", Architecture: "arm64", Supported: true},
		{OS: "darwin", Architecture: "arm64", Supported: true},
		{OS: "darwin", Architecture: "amd64", Supported: true},
	}
}

// IsPlatformSupported checks if the given OS/architecture combination is supported
func IsPlatformSupported(osName, arch string) bool {
	supportedPlatforms := GetSupportedPlatforms()
	for _, platform := range supportedPlatforms {
		if platform.OS == osName && platform.Architecture == arch {
			return true
		}
	}
	return false
}

// GetLibraryNameForPlatform returns the library filename for the given platform
func GetLibraryNameForPlatform(osName, arch string) string {
	switch {
	case osName == "linux" && arch == "amd64":
		return "libonnxruntime_amd64.so"
	case osName == "linux" && arch == "arm64":
		return "libonnxruntime_aarch64.so"
	case osName == "darwin" && (arch == "arm64" || arch == "amd64"):
		return "libonnxruntime_arm64.dylib"
	default:
		return ""
	}
}

// ValidateBasePath validates that the base library path exists and contains expected structure
func (l *LibraryLoader) ValidateBasePath() error {
	resolvedPath, err := l.ResolvePath(l.basePath)
	if err != nil {
		return NewValidationError(
			"BASE_PATH_RESOLVE_FAILED",
			"failed to resolve base path",
			err,
		).WithContext("base_path", l.basePath)
	}

	// Check if base path exists
	info, err := os.Stat(resolvedPath)
	if os.IsNotExist(err) {
		return NewValidationError(
			"BASE_PATH_NOT_FOUND",
			fmt.Sprintf("base library path does not exist: %s", resolvedPath),
			err,
		).WithContext("resolved_path", resolvedPath)
	}
	if err != nil {
		return NewValidationError(
			"BASE_PATH_ACCESS_FAILED",
			fmt.Sprintf("failed to access base library path: %s", resolvedPath),
			err,
		).WithContext("resolved_path", resolvedPath)
	}

	// If base path is a file, we're done
	if info.Mode().IsRegular() {
		return nil
	}

	return nil
}
