package audio

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

const (
	testSampleRate = 44100
	testFreq       = 440.0
	testDuration   = 1.0
)

// genSine 生成 1 秒 440Hz 正弦波（44100Hz 采样率）。
func genSine() []float32 {
	numSamples := int(float64(testSampleRate) * testDuration)
	samples := make([]float32, numSamples)
	for i := 0; i < numSamples; i++ {
		t := float64(i) / float64(testSampleRate)
		samples[i] = float32(math.Sin(2 * math.Pi * testFreq * t))
	}
	return samples
}

// expectedInt16 计算单个样本期望的 int16 值（与 WriteWAV 逻辑一致）。
// 与 libsndfile PCM_16 对齐：int16 = clip(floor(x * 32768), -32768, 32767)。
func expectedInt16(s float32) int16 {
	v := math.Floor(float64(s) * 32768.0)
	if v > 32767 {
		v = 32767
	} else if v < -32768 {
		v = -32768
	}
	return int16(v)
}

func TestWriteWAV_SineWave(t *testing.T) {
	samples := genSine()
	numSamples := len(samples)

	// 写入临时文件
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "sine_440.wav")
	if err := WriteWAVFile(path, samples, testSampleRate); err != nil {
		t.Fatalf("WriteWAVFile failed: %v", err)
	}

	// 验证文件大小 = 44 + samples*2
	expectedSize := int64(headerSize + numSamples*bytesPerSample)
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("os.Stat failed: %v", err)
	}
	if info.Size() != expectedSize {
		t.Errorf("file size = %d, want %d", info.Size(), expectedSize)
	}

	// 读回文件
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("os.ReadFile failed: %v", err)
	}

	// 验证 RIFF 头
	if string(data[0:4]) != "RIFF" {
		t.Errorf("RIFF marker = %q, want \"RIFF\"", string(data[0:4]))
	}
	chunkSize := binary.LittleEndian.Uint32(data[4:8])
	if chunkSize != uint32(36+numSamples*2) {
		t.Errorf("chunk size = %d, want %d", chunkSize, 36+numSamples*2)
	}
	if string(data[8:12]) != "WAVE" {
		t.Errorf("WAVE marker = %q, want \"WAVE\"", string(data[8:12]))
	}

	// 验证 fmt 子块
	if string(data[12:16]) != "fmt " {
		t.Errorf("fmt marker = %q, want \"fmt \"", string(data[12:16]))
	}
	if binary.LittleEndian.Uint32(data[16:20]) != 16 {
		t.Errorf("subchunk1 size = %d, want 16", binary.LittleEndian.Uint32(data[16:20]))
	}
	if binary.LittleEndian.Uint16(data[20:22]) != 1 {
		t.Errorf("audio format = %d, want 1 (PCM)", binary.LittleEndian.Uint16(data[20:22]))
	}
	if binary.LittleEndian.Uint16(data[22:24]) != 1 {
		t.Errorf("num channels = %d, want 1", binary.LittleEndian.Uint16(data[22:24]))
	}
	if binary.LittleEndian.Uint32(data[24:28]) != uint32(testSampleRate) {
		t.Errorf("sample rate = %d, want %d", binary.LittleEndian.Uint32(data[24:28]), testSampleRate)
	}
	// byte rate = 44100 * 1 * 2 = 88200
	if binary.LittleEndian.Uint32(data[28:32]) != uint32(testSampleRate*2) {
		t.Errorf("byte rate = %d, want %d", binary.LittleEndian.Uint32(data[28:32]), testSampleRate*2)
	}
	// block align = 1 * 2 = 2
	if binary.LittleEndian.Uint16(data[32:34]) != 2 {
		t.Errorf("block align = %d, want 2", binary.LittleEndian.Uint16(data[32:34]))
	}
	if binary.LittleEndian.Uint16(data[34:36]) != 16 {
		t.Errorf("bits per sample = %d, want 16", binary.LittleEndian.Uint16(data[34:36]))
	}

	// 验证 data 子块
	if string(data[36:40]) != "data" {
		t.Errorf("data marker = %q, want \"data\"", string(data[36:40]))
	}
	subchunk2Size := binary.LittleEndian.Uint32(data[40:44])
	if subchunk2Size != uint32(numSamples*2) {
		t.Errorf("subchunk2 size = %d, want %d", subchunk2Size, numSamples*2)
	}

	// 读回并验证部分采样点
	checkIndices := []int{0, 1, 100, testSampleRate / 4, testSampleRate / 2, numSamples - 1}
	for _, idx := range checkIndices {
		if idx < 0 || idx >= numSamples {
			continue
		}
		offset := headerSize + idx*bytesPerSample
		got := int16(binary.LittleEndian.Uint16(data[offset : offset+2]))
		want := expectedInt16(samples[idx])
		if got != want {
			t.Errorf("sample[%d] = %d, want %d", idx, got, want)
		}
	}
}

func TestWriteWAV_Empty(t *testing.T) {
	// 空样本应仍能写入有效头部（0 字节数据）。
	var buf bytes.Buffer
	if err := WriteWAV(&buf, nil, testSampleRate); err != nil {
		t.Fatalf("WriteWAV failed: %v", err)
	}
	if buf.Len() != headerSize {
		t.Errorf("output size = %d, want %d (header only)", buf.Len(), headerSize)
	}
	if string(buf.Bytes()[36:40]) != "data" {
		t.Errorf("data marker = %q, want \"data\"", string(buf.Bytes()[36:40]))
	}
	if binary.LittleEndian.Uint32(buf.Bytes()[40:44]) != 0 {
		t.Error("subchunk2 size should be 0 for empty samples")
	}
}

func TestWriteWAV_Clamping(t *testing.T) {
	samples := []float32{-2.0, -1.0, 0.0, 1.0, 2.0}
	var buf bytes.Buffer
	if err := WriteWAV(&buf, samples, testSampleRate); err != nil {
		t.Fatalf("WriteWAV failed: %v", err)
	}

	data := buf.Bytes()
	// -2.0 -> floor(-2.0*32768)=-65536 -> clip -32768
	if got := int16(binary.LittleEndian.Uint16(data[44:46])); got != -32768 {
		t.Errorf("sample -2.0 -> %d, want -32768", got)
	}
	// -1.0 -> floor(-32768)=-32768
	if got := int16(binary.LittleEndian.Uint16(data[46:48])); got != -32768 {
		t.Errorf("sample -1.0 -> %d, want -32768", got)
	}
	// 0.0 -> floor(0)=0
	if got := int16(binary.LittleEndian.Uint16(data[48:50])); got != 0 {
		t.Errorf("sample 0.0 -> %d, want 0", got)
	}
	// 1.0 -> floor(32768)=32768 -> clip 32767
	if got := int16(binary.LittleEndian.Uint16(data[50:52])); got != 32767 {
		t.Errorf("sample 1.0 -> %d, want 32767", got)
	}
	// 2.0 -> floor(65536) -> clip 32767
	if got := int16(binary.LittleEndian.Uint16(data[52:54])); got != 32767 {
		t.Errorf("sample 2.0 -> %d, want 32767", got)
	}
}
