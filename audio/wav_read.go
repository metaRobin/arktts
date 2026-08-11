package audio

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

// ReadWAV 从 WAV 文件读取音频，返回 mono float32 样本和采样率。
// 支持：
//   - PCM 格式（audioFormat=1）：16/24/32-bit signed int
//   - IEEE float 格式（audioFormat=3）：32-bit float
//   - 多声道 → mono（取均值）
//   - 跳过未知 chunk（如 LIST/fact），自动定位 data chunk
func ReadWAV(r io.Reader) (samples []float32, sampleRate int, err error) {
	// 读 RIFF header
	var riff [12]byte
	if _, err := io.ReadFull(r, riff[:]); err != nil {
		return nil, 0, fmt.Errorf("read riff header: %w", err)
	}
	if string(riff[:4]) != "RIFF" || string(riff[8:12]) != "WAVE" {
		return nil, 0, fmt.Errorf("not a RIFF/WAVE file")
	}

	var (
		audioFormat   uint16
		numChannels   uint16
		bitsPerSample uint16
		sr            uint32
		dataBytes     []byte
	)

	// 逐 chunk 解析
	for {
		var chunkHdr [8]byte
		if _, err := io.ReadFull(r, chunkHdr[:]); err != nil {
			if err == io.EOF {
				break
			}
			return nil, 0, fmt.Errorf("read chunk header: %w", err)
		}
		chunkID := string(chunkHdr[:4])
		chunkSize := binary.LittleEndian.Uint32(chunkHdr[4:8])

		if chunkID == "fmt " {
			if chunkSize < 16 {
				return nil, 0, fmt.Errorf("fmt chunk too short: %d", chunkSize)
			}
			fmtData := make([]byte, chunkSize)
			if _, err := io.ReadFull(r, fmtData); err != nil {
				return nil, 0, fmt.Errorf("read fmt chunk: %w", err)
			}
			audioFormat = binary.LittleEndian.Uint16(fmtData[0:2])
			numChannels = binary.LittleEndian.Uint16(fmtData[2:4])
			sr = binary.LittleEndian.Uint32(fmtData[4:8])
			bitsPerSample = binary.LittleEndian.Uint16(fmtData[14:16])
		} else if chunkID == "data" {
			dataBytes = make([]byte, chunkSize)
			if _, err := io.ReadFull(r, dataBytes); err != nil {
				return nil, 0, fmt.Errorf("read data chunk: %w", err)
			}
			break // data chunk 后通常无更多有用内容
		} else {
			// 跳过未知 chunk
			if _, err := io.CopyN(io.Discard, r, int64(chunkSize)); err != nil {
				return nil, 0, fmt.Errorf("skip chunk %q: %w", chunkID, err)
			}
		}
	}

	if dataBytes == nil {
		return nil, 0, fmt.Errorf("no data chunk found")
	}
	if audioFormat == 0 {
		return nil, 0, fmt.Errorf("no fmt chunk found")
	}

	samples, err = decodePCMSamples(dataBytes, audioFormat, bitsPerSample, int(numChannels))
	if err != nil {
		return nil, 0, err
	}

	return samples, int(sr), nil
}

// ReadWAVFile 从文件路径读取 WAV。
func ReadWAVFile(path string) ([]float32, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, fmt.Errorf("open wav %q: %w", path, err)
	}
	defer f.Close()
	return ReadWAV(f)
}

// decodePCMSamples 将原始 PCM 字节解码为 mono float32 样本。
// 多声道取均值，对齐 Python soundfile.read(mono=True) 行为。
func decodePCMSamples(data []byte, audioFormat, bitsPerSample uint16, numChannels int) ([]float32, error) {
	if numChannels <= 0 {
		return nil, fmt.Errorf("invalid channel count: %d", numChannels)
	}

	var samples []float32

	switch {
	case audioFormat == 1 && bitsPerSample == 16:
		frameSize := 2 * numChannels
		nFrames := len(data) / frameSize
		samples = make([]float32, nFrames)
		for i := 0; i < nFrames; i++ {
			var sum float64
			for ch := 0; ch < numChannels; ch++ {
				v := int16(binary.LittleEndian.Uint16(data[i*frameSize+ch*2:]))
				sum += float64(v) / 32768.0
			}
			samples[i] = float32(sum / float64(numChannels))
		}

	case audioFormat == 1 && bitsPerSample == 24:
		frameSize := 3 * numChannels
		nFrames := len(data) / frameSize
		samples = make([]float32, nFrames)
		for i := 0; i < nFrames; i++ {
			var sum float64
			for ch := 0; ch < numChannels; ch++ {
				off := i*frameSize + ch*3
				v := int32(data[off]) | int32(data[off+1])<<8 | int32(data[off+2])<<16
				if v&0x800000 != 0 {
					v |= ^0xFFFFFF // 符号扩展
				}
				sum += float64(v) / 8388608.0
			}
			samples[i] = float32(sum / float64(numChannels))
		}

	case audioFormat == 1 && bitsPerSample == 32:
		frameSize := 4 * numChannels
		nFrames := len(data) / frameSize
		samples = make([]float32, nFrames)
		for i := 0; i < nFrames; i++ {
			var sum float64
			for ch := 0; ch < numChannels; ch++ {
				v := int32(binary.LittleEndian.Uint32(data[i*frameSize+ch*4:]))
				sum += float64(v) / 2147483648.0
			}
			samples[i] = float32(sum / float64(numChannels))
		}

	case audioFormat == 3 && bitsPerSample == 32:
		// IEEE 754 float32
		frameSize := 4 * numChannels
		nFrames := len(data) / frameSize
		samples = make([]float32, nFrames)
		for i := 0; i < nFrames; i++ {
			var sum float64
			for ch := 0; ch < numChannels; ch++ {
				v := math.Float32frombits(binary.LittleEndian.Uint32(data[i*frameSize+ch*4:]))
				sum += float64(v)
			}
			samples[i] = float32(sum / float64(numChannels))
		}

	default:
		return nil, fmt.Errorf("unsupported WAV format: audioFormat=%d bitsPerSample=%d", audioFormat, bitsPerSample)
	}

	return samples, nil
}
