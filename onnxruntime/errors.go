package onnxruntime

import "fmt"

// ErrorType represents different types of errors that can occur
type ErrorType int

const (
	ErrorTypePlatform ErrorType = iota
	ErrorTypeLibraryLoad
	ErrorTypeValidation
	ErrorTypeRuntime
	ErrorTypeModel
	ErrorTypeConfig
)

// Error represents a structured error with type and context
type Error struct {
	Type    ErrorType
	Code    string
	Message string
	Cause   error
	Context map[string]interface{}
}

// Error implements the error interface
func (e *Error) Error() string {
	if e.Cause != nil {
		return fmt.Sprintf("%s: %v", e.Message, e.Cause)
	}
	return e.Message
}

// Unwrap returns the underlying cause error
func (e *Error) Unwrap() error {
	return e.Cause
}

// NewPlatformError creates a new platform-related error
func NewPlatformError(code, message string, cause error) *Error {
	return &Error{
		Type:    ErrorTypePlatform,
		Code:    code,
		Message: message,
		Cause:   cause,
		Context: make(map[string]interface{}),
	}
}

// NewLibraryLoadError creates a new library loading error
func NewLibraryLoadError(code, message string, cause error) *Error {
	return &Error{
		Type:    ErrorTypeLibraryLoad,
		Code:    code,
		Message: message,
		Cause:   cause,
		Context: make(map[string]interface{}),
	}
}

// NewValidationError creates a new validation error
func NewValidationError(code, message string, cause error) *Error {
	return &Error{
		Type:    ErrorTypeValidation,
		Code:    code,
		Message: message,
		Cause:   cause,
		Context: make(map[string]interface{}),
	}
}

// WithContext adds context information to the error
func (e *Error) WithContext(key string, value interface{}) *Error {
	e.Context[key] = value
	return e
}

// NewRuntimeError creates a new runtime-related error
func NewRuntimeError(code, message string, cause error) *Error {
	return &Error{
		Type:    ErrorTypeRuntime,
		Code:    code,
		Message: message,
		Cause:   cause,
		Context: make(map[string]interface{}),
	}
}

// NewModelError creates a new model-related error
func NewModelError(code, message string, cause error) *Error {
	return &Error{
		Type:    ErrorTypeModel,
		Code:    code,
		Message: message,
		Cause:   cause,
		Context: make(map[string]interface{}),
	}
}

// NewConfigError creates a new configuration-related error
func NewConfigError(code, message string, cause error) *Error {
	return &Error{
		Type:    ErrorTypeConfig,
		Code:    code,
		Message: message,
		Cause:   cause,
		Context: make(map[string]interface{}),
	}
}
