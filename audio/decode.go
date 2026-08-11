package audio

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os"
	"os/exec"
	"strings"
)

// DecodeAudio 从文件字节解码音频，返回 mono float32 样本和原始采样率。
//
// WAV 文件用纯 Go 解码；其他格式（mp3/flac/m4a 等）回退到 ffmpeg 子进程。
// 对齐 Python soundfile.read(always_2d=False, dtype="float32") + mono 转换。
func DecodeAudio(data []byte) (samples []float32, sampleRate int, err error) {
	// 检测 WAV 魔数
	if len(data) >= 12 && string(data[:4]) == "RIFF" && string(data[8:12]) == "WAVE" {
		return ReadWAV(bytes.NewReader(data))
	}

	// 非 WAV：用 ffmpeg 转码为 WAV 后读取
	return decodeWithFFmpeg(data)
}

// decodeWithFFmpeg 调用 ffmpeg 子进程将任意音频格式转为 WAV，
// 再用纯 Go WAV 解码器读取。
// 与 Python soundfile 行为对齐：输出 float32 mono。
func decodeWithFFmpeg(data []byte) ([]float32, int, error) {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		return nil, 0, fmt.Errorf("audio format not WAV and ffmpeg not found: %w", err)
	}

	cmd := exec.Command("ffmpeg",
		"-i", "pipe:0",       // 从 stdin 读
		"-f", "wav",           // 输出 WAV
		"-acodec", "pcm_s16le",
		"-ac", "1",            // 强制 mono
		"-ar", "44100",        // 不改变采样率→用原始，但 ffmpeg 需要指定，这里保持源
		"-vn",                 // 无视频
		"-y",                  // 覆盖
		"pipe:1",              // 输出到 stdout
	)
	// 不指定 -ar，让 ffmpeg 保持原始采样率
	cmd.Args = []string{
		"ffmpeg", "-i", "pipe:0",
		"-f", "wav", "-acodec", "pcm_s16le",
		"-ac", "1", "-vn", "-y",
		"pipe:1",
	}
	cmd.Stdin = bytes.NewReader(data)
	var out bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		stderrStr := strings.TrimSpace(stderr.String())
		if len(stderrStr) > 200 {
			stderrStr = stderrStr[len(stderrStr)-200:]
		}
		return nil, 0, fmt.Errorf("ffmpeg decode failed: %w: %s", err, stderrStr)
	}

	return ReadWAV(bytes.NewReader(out.Bytes()))
}

// DecodeAudioFile 从文件路径读取并解码音频。
func DecodeAudioFile(path string) ([]float32, int, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, 0, fmt.Errorf("read audio file %q: %w", path, err)
	}
	return DecodeAudio(data)
}

// PadToMultiple 将样本末尾补零至 multiple 的倍数。
// 对齐 Python: audio = np.pad(audio, (0, pad_len))
func PadToMultiple(samples []float32, multiple int) []float32 {
	if multiple <= 0 {
		return samples
	}
	rem := len(samples) % multiple
	if rem == 0 {
		return samples
	}
	padLen := multiple - rem
	result := make([]float32, len(samples)+padLen)
	copy(result, samples)
	return result
}

// HasFFmpeg 返回系统是否安装了 ffmpeg。
func HasFFmpeg() bool {
	_, err := exec.LookPath("ffmpeg")
	return err == nil
}

// 编译期断言：binary 已使用
var _ = binary.LittleEndian
