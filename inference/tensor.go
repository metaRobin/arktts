package inference

import (
	"encoding/binary"
	"fmt"
	"unsafe"

	ort "github.com/yalue/onnxruntime_go"
)

// NewFloat16Tensor creates a float16 ONNX tensor from a []uint16 buffer
// (where each uint16 holds IEEE 754 half-precision bits).
// The tensor references the underlying data; the caller must not modify
// the slice while the tensor is alive and must call Destroy when done.
func NewFloat16Tensor(shape ort.Shape, data []uint16) (*ort.CustomDataTensor, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("float16 tensor requires non-empty data")
	}
	// Reinterpret []uint16 as []byte without copying.
	// unsafe.SliceData requires Go 1.20+; we use the classic pointer approach.
	var dataBytes []byte
	if len(data) > 0 {
		//nolint:gocritic // unavoidable unsafe for zero-copy reinterpretation
		dataBytes = unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), len(data)*2)
	}
	return ort.NewCustomDataTensor(shape, dataBytes, ort.TensorElementDataTypeFloat16)
}

// NewFloat16TensorFromFloat32 creates a float16 ONNX tensor from []float32.
// The conversion allocates a []uint16 buffer; the caller must call Destroy.
func NewFloat16TensorFromFloat32(shape ort.Shape, data []float32) (*ort.CustomDataTensor, error) {
	buf := F32To16Slice(data)
	return NewFloat16Tensor(shape, buf)
}

// ReadFloat16Tensor reads a float16 output tensor and returns []float32.
// Handles both *ort.CustomDataTensor (float16 via raw bytes) and
// *ort.Tensor[float32] (when the model outputs float32 directly).
func ReadFloat16Tensor(v ort.Value) ([]float32, error) {
	if v == nil {
		return nil, fmt.Errorf("nil output tensor")
	}
	switch t := v.(type) {
	case *ort.CustomDataTensor:
		raw := t.GetData()
		if len(raw)%2 != 0 {
			return nil, fmt.Errorf("float16 tensor has odd byte length %d", len(raw))
		}
		n := len(raw) / 2
		result := make([]float32, n)
		// Zero-copy reinterpret raw bytes as []uint16, then convert.
		//nolint:gocritic
		u16s := unsafe.Slice((*uint16)(unsafe.Pointer(&raw[0])), n)
		for i, h := range u16s {
			result[i] = Float16ToFloat32(h)
		}
		return result, nil
	case *ort.Tensor[float32]:
		// Model outputs float32 directly.
		return t.GetData(), nil
	default:
		return nil, fmt.Errorf("unexpected output tensor type %T for float16 read", v)
	}
}

// ReadFloat32Tensor reads a float32 output tensor.
func ReadFloat32Tensor(v ort.Value) ([]float32, error) {
	if v == nil {
		return nil, fmt.Errorf("nil output tensor")
	}
	switch t := v.(type) {
	case *ort.Tensor[float32]:
		return t.GetData(), nil
	case *ort.CustomDataTensor:
		// Fall back to float16 interpretation for custom data tensors.
		return ReadFloat16Tensor(v)
	default:
		return nil, fmt.Errorf("unexpected output tensor type %T for float32 read", v)
	}
}

// ReadInt64Tensor reads an int64 output tensor and returns []int64.
func ReadInt64Tensor(v ort.Value) ([]int64, error) {
	if v == nil {
		return nil, fmt.Errorf("nil output tensor")
	}
	switch t := v.(type) {
	case *ort.Tensor[int64]:
		return t.GetData(), nil
	default:
		return nil, fmt.Errorf("unexpected output tensor type %T for int64 read", v)
	}
}

// Uint16ToBytes reinterprets []uint16 as []byte (zero-copy).
func Uint16ToBytes(data []uint16) []byte {
	if len(data) == 0 {
		return nil
	}
	//nolint:gocritic
	return unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), len(data)*2)
}

// BytesToUint16 reinterprets []byte as []uint16 (zero-copy).
// The byte length must be even.
func BytesToUint16(data []byte) ([]uint16, error) {
	if len(data) == 0 {
		return nil, nil
	}
	if len(data)%2 != 0 {
		return nil, fmt.Errorf("odd byte length %d for uint16 reinterpretation", len(data))
	}
	//nolint:gocritic
	return unsafe.Slice((*uint16)(unsafe.Pointer(&data[0])), len(data)/2), nil
}

// WriteFloat16AtPosition writes delta (float16 bytes) into a cache buffer
// at the given positions along the sequence dimension.
//
// cacheShape is [batch, heads, seqLen, headDim] (C-contiguous).
// deltaShape is [batch, heads, len(positions), headDim].
// Only batch=1 is supported.
//
// For each position p in positions and head h:
//   cacheOffset = (h * seqLen + p) * headDim * 2  (bytes)
//   deltaOffset = (h * len(positions) + i) * headDim * 2  (bytes)
//   copy cache[cacheOffset:cacheOffset+headDim*2] = delta[deltaOffset:deltaOffset+headDim*2]
func WriteFloat16AtPosition(cache []byte, heads, seqLen, headDim int, positions []int, delta []byte) error {
	elementBytes := headDim * 2 // float16 = 2 bytes
	expectedDeltaLen := heads * len(positions) * elementBytes
	if len(delta) < expectedDeltaLen {
		return fmt.Errorf("delta too short: got %d bytes, need %d", len(delta), expectedDeltaLen)
	}
	for i, pos := range positions {
		if pos < 0 || pos >= seqLen {
			return fmt.Errorf("position %d out of range [0, %d)", pos, seqLen)
		}
		for h := 0; h < heads; h++ {
			cacheOff := (h*seqLen + pos) * elementBytes
			deltaOff := (h*len(positions) + i) * elementBytes
			copy(cache[cacheOff:cacheOff+elementBytes], delta[deltaOff:deltaOff+elementBytes])
		}
	}
	return nil
}

// WriteFloat16SliceAtPosition writes a []uint16 delta into a []uint16 cache
// at the given positions. Same layout as WriteFloat16AtPosition but typed.
func WriteFloat16SliceAtPosition(cache []uint16, heads, seqLen, headDim int, positions []int, delta []uint16) error {
	elementLen := headDim
	expectedDeltaLen := heads * len(positions) * elementLen
	if len(delta) < expectedDeltaLen {
		return fmt.Errorf("delta too short: got %d elements, need %d", len(delta), expectedDeltaLen)
	}
	for i, pos := range positions {
		if pos < 0 || pos >= seqLen {
			return fmt.Errorf("position %d out of range [0, %d)", pos, seqLen)
		}
		for h := 0; h < heads; h++ {
			cacheOff := (h*seqLen + pos) * elementLen
			deltaOff := (h*len(positions) + i) * elementLen
			copy(cache[cacheOff:cacheOff+elementLen], delta[deltaOff:deltaOff+elementLen])
		}
	}
	return nil
}

// NativeEndianUint16 returns the uint16 value from a 2-byte little-endian slice.
// ONNX Runtime on x86/arm64 uses native (little-endian) byte order.
func NativeEndianUint16(b []byte) uint16 {
	return binary.LittleEndian.Uint16(b)
}
