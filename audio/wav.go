package audio

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

// wavHeader 是标准 44 字节 PCM WAV 文件头。
// 使用 encoding/binary.Write 按字段顺序写入，不含结构体内存对齐填充。
type wavHeader struct {
	RiffMark      [4]byte // "RIFF"
	ChunkSize     uint32  // 36 + data size
	WaveMark      [4]byte // "WAVE"
	FmtMark       [4]byte // "fmt "
	Subchunk1Size uint32  // 16 (PCM)
	AudioFormat   uint16  // 1 (PCM)
	NumChannels   uint16  // 1 (mono)
	SampleRate    uint32  // 采样率，通常 44100
	ByteRate      uint32  // sampleRate * numChannels * bitsPerSample/8
	BlockAlign    uint16  // numChannels * bitsPerSample/8
	BitsPerSample uint16  // 16
	DataMark      [4]byte // "data"
	Subchunk2Size uint32  // len(samples) * 2
}

const (
	numChannels   = 1
	bitsPerSample = 16
	bytesPerSample = bitsPerSample / 8 // 2
	headerSize    = 44
)

// WriteWAV writes float32 audio samples as a 16-bit PCM WAV file.
// sampleRate is typically 44100. samples is mono (1 channel) float32 in range [-1.0, 1.0].
// 输出与 Python soundfile.write（libsndfile PCM_16）字节级一致：
//
//	int16 = clip(floor(x * 32768), -32768, 32767)
//
// floor 而非 round：[-1.0, 1.0) 映射到 [-32768, 32767]，1.0 经
// floor(32768) 裁剪到 32767，超范围值裁剪到边界。
func WriteWAV(w io.Writer, samples []float32, sampleRate int) error {
	dataSize := len(samples) * bytesPerSample

	hdr := wavHeader{
		RiffMark:      [4]byte{'R', 'I', 'F', 'F'},
		ChunkSize:     uint32(36 + dataSize),
		WaveMark:      [4]byte{'W', 'A', 'V', 'E'},
		FmtMark:       [4]byte{'f', 'm', 't', ' '},
		Subchunk1Size: 16,
		AudioFormat:   1,
		NumChannels:   numChannels,
		SampleRate:    uint32(sampleRate),
		ByteRate:      uint32(sampleRate * numChannels * bytesPerSample),
		BlockAlign:    uint16(numChannels * bytesPerSample),
		BitsPerSample: bitsPerSample,
		DataMark:      [4]byte{'d', 'a', 't', 'a'},
		Subchunk2Size: uint32(dataSize),
	}

	if err := binary.Write(w, binary.LittleEndian, &hdr); err != nil {
		return fmt.Errorf("write wav header: %w", err)
	}

	// 复用 PCM 转换逻辑，保证 WAV 与裸 PCM 输出字节级一致。
	if _, err := w.Write(Float32ToS16LE(samples)); err != nil {
		return fmt.Errorf("write wav data: %w", err)
	}

	return nil
}

// WriteWAVFile writes a WAV file to the given path.
func WriteWAVFile(path string, samples []float32, sampleRate int) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create wav file %q: %w", path, err)
	}
	defer f.Close()
	return WriteWAV(f, samples, sampleRate)
}
