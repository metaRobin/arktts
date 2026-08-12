package inference

import (
	"fmt"
	"path/filepath"

	"github.com/metaRobin/arktts/onnxruntime"
	ort "github.com/yalue/onnxruntime_go"
)

// Encoder 是 codec encoder 的 ONNX 推理封装。
// 与 Engine 的三个持久 session 不同，Encoder 按需创建、用完即销毁，
// 因为注册是低频操作，不需要常驻内存。
type Encoder struct {
	sess   *onnxruntime.SessionImpl
	useF16 bool // 模型输入是否为 float16
}

// NewEncoder 从 encoder ONNX 模型创建 session。
// modelPath 通常是 registration/codec_encoder_fp16.onnx。
// ortRuntime 必须已初始化。
func NewEncoder(ortRuntime *onnxruntime.RuntimeImpl, modelPath string) (*Encoder, error) {
	sess, err := ortRuntime.CreateSession(modelPath)
	if err != nil {
		return nil, fmt.Errorf("create encoder session: %w", err)
	}

	sessImpl := sess.(*onnxruntime.SessionImpl)

	// 检查输入类型以决定 float16 还是 float32
	inputs, _, err := sessImpl.GetModelInputOutputInfo()
	if err != nil {
		sessImpl.Destroy()
		return nil, fmt.Errorf("get encoder model info: %w", err)
	}

	useF16 := false
	if len(inputs) > 0 {
		useF16 = inputs[0].DataType == ort.TensorElementDataTypeFloat16
	}

	return &Encoder{sess: sessImpl, useF16: useF16}, nil
}

// Encode 执行 codec 编码：audio [1, 1, N] float32 → codes [num_codebooks, T] int64。
// 若模型接受 float16 输入，内部会做 float32→float16 转换。
func (e *Encoder) Encode(audio []float32) ([][]int64, error) {
	shape := ort.Shape{1, 1, int64(len(audio))}

	var audioTensor ort.Value
	var err error
	if e.useF16 {
		audioTensor, err = NewFloat16TensorFromFloat32(shape, audio)
	} else {
		audioTensor, err = ort.NewTensor(shape, audio)
	}
	if err != nil {
		return nil, fmt.Errorf("create encoder input tensor: %w", err)
	}
	defer audioTensor.Destroy()

	outputs := make([]ort.Value, 1)
	if err := e.sess.RunWithValues([]ort.Value{audioTensor}, outputs); err != nil {
		return nil, fmt.Errorf("encoder inference: %w", err)
	}
	defer func() {
		if outputs[0] != nil {
			outputs[0].Destroy()
		}
	}()

	codesFlat, err := ReadInt64Tensor(outputs[0])
	if err != nil {
		return nil, fmt.Errorf("read encoder output: %w", err)
	}

	// 输出形状应为 [1, num_codebooks, T] 或 [num_codebooks, T]
	// 从 InputOutputInfo 获取输出形状
	_, outputInfos, err := e.sess.GetModelInputOutputInfo()
	if err == nil && len(outputInfos) > 0 {
		shape := outputInfos[0].Dimensions
		if len(shape) == 3 {
			// [1, num_codebooks, T]
			numCodebooks := int(shape[1])
			t := int(shape[2])
			if t <= 0 {
				t = len(codesFlat) / numCodebooks
			}
			return reshapeCodes(codesFlat, numCodebooks, t), nil
		}
		if len(shape) == 2 {
			// [num_codebooks, T]
			numCodebooks := int(shape[0])
			t := int(shape[1])
			if t <= 0 {
				t = len(codesFlat) / numCodebooks
			}
			return reshapeCodes(codesFlat, numCodebooks, t), nil
		}
	}

	// 回退：无法获取形状，假设 [num_codebooks, T] 且 T = len/numCodebooks
	// 尝试从 codes 长度推断（需要知道 num_codebooks，此处无法获取）
	return nil, fmt.Errorf("cannot determine encoder output shape, flat length=%d", len(codesFlat))
}

// Close 销毁 encoder session，释放 ONNX Runtime 内存。
func (e *Encoder) Close() {
	if e.sess != nil {
		e.sess.Destroy()
	}
}

// reshapeCodes 将扁平的 int64 切片重塑为 [numCodebooks][T] 二维切片。
// 布局：row-major（codebook 维在外，时间维在内），与 Python np.stack(axis=1) 一致。
func reshapeCodes(flat []int64, numCodebooks, t int) [][]int64 {
	result := make([][]int64, numCodebooks)
	for cb := 0; cb < numCodebooks; cb++ {
		result[cb] = make([]int64, t)
		for i := 0; i < t; i++ {
			result[cb][i] = flat[cb*t+i]
		}
	}
	return result
}

// EncoderModelPath 返回 encoder 模型文件的完整路径。
// registrationDir 是注册目录（通常是 model/registration）。
func EncoderModelPath(registrationDir string) string {
	return filepath.Join(registrationDir, "codec_encoder_fp16.onnx")
}
