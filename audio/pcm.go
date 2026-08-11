package audio

import (
	"encoding/binary"
	"math"
)

// Float32ToS16LE 将 float32 单声道样本（范围 [-1.0, 1.0]）转换为 16-bit 小端序 PCM 字节。
//
// 与 Python soundfile / libsndfile PCM_16 字节级一致：
//
//	int16 = clip(floor(x * 32768), -32768, 32767)
//
// floor 而非 round：[-1.0, 1.0) 映射到 [-32768, 32767]，1.0 经
// floor(32768) 裁剪到 32767，超范围值裁剪到边界。
//
// 输出长度为 len(samples) * 2 字节。
func Float32ToS16LE(samples []float32) []byte {
	buf := make([]byte, len(samples)*2)
	for i, s := range samples {
		v := math.Floor(float64(s) * 32768.0)
		if v > 32767 {
			v = 32767
		} else if v < -32768 {
			v = -32768
		}
		binary.LittleEndian.PutUint16(buf[i*2:], uint16(int16(v)))
	}
	return buf
}
