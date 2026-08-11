// Package inference 提供 ONNX 模型推理所需的底层工具，包括 float16 转换和采样逻辑。
//
// ONNX 模型的 hidden states 与 KV cache 张量使用 float16 存储，
// 需要在 Go 的 float32 与 ONNX 的 uint16（float16）之间进行无损或有损转换。
package inference

import "math"

// Float32ToFloat16 将 float32 转换为 IEEE 754 半精度浮点数（uint16）。
// 采用就近偶数舍入（round-to-nearest-even），与 IEEE 754 默认舍入模式一致。
// 正确处理正常范围、次正常（subnormal）、无穷大和 NaN。
func Float32ToFloat16(f float32) uint16 {
	x := math.Float32bits(f)
	sign := uint16((x >> 16) & 0x8000)
	exp := int((x >> 23) & 0xff)
	mant := x & 0x7fffff

	// 特殊值：无穷大或 NaN
	if exp == 0xff {
		if mant == 0 {
			return sign | 0x7c00 // 无穷大
		}
		// NaN：取 float32 尾数的高 10 位，确保至少 1 位为 1（保持 NaN 语义）
		h := uint16(mant >> 13)
		if h == 0 {
			h = 1
		}
		return sign | 0x7c00 | h
	}

	// 将 float32 指数偏置（127）转换为 float16 指数偏置（15）
	exp = exp - 127 + 15

	if exp >= 31 {
		// 溢出：返回无穷大
		return sign | 0x7c00
	}

	if exp <= 0 {
		// 次正常或下溢为零
		if exp < -10 {
			// 值太小，下溢为零
			return sign
		}
		// 次正常：添加隐含的前导 1，然后右移并舍入
		mant |= 0x800000
		shift := uint(14 - exp)
		result := mant >> shift
		remainder := mant & ((1 << shift) - 1)
		halfBit := uint32(1 << (shift - 1))
		// 就近偶数舍入
		if remainder > halfBit || (remainder == halfBit && result&1 == 1) {
			result++
		}
		// 次正常舍入进位到最小正常值时，位布局自动连续（bit 10 = 指数 LSB）
		return sign | uint16(result)
	}

	// 正常范围：将 23 位尾数舍入为 10 位
	result := mant >> 13
	remainder := mant & 0x1fff
	halfBit := uint32(1 << 12)
	if remainder > halfBit || (remainder == halfBit && result&1 == 1) {
		result++
		// 尾数进位溢出（0x3FF → 0x400），进位到指数
		if result >= 0x400 {
			result = 0
			exp++
		}
	}

	if exp >= 31 {
		// 舍入后溢出为无穷大
		return sign | 0x7c00
	}

	return sign | uint16(exp<<10) | uint16(result)
}

// Float16ToFloat32 将 IEEE 754 半精度浮点数（uint16）转换为 float32。
// 正确处理正常范围、次正常、无穷大和 NaN。
func Float16ToFloat32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h & 0x3ff)

	if exp == 0 {
		if mant == 0 {
			// 零（正零或负零）
			return math.Float32frombits(sign)
		}
		// 次正常：通过左移归一化，找到隐含的前导 1
		e := int32(1)
		for mant&0x400 == 0 {
			mant <<= 1
			e--
		}
		mant &= 0x3ff // 清除隐含的 1
		f32Exp := uint32(e + 112)
		return math.Float32frombits(sign | (f32Exp << 23) | (mant << 13))
	}

	if exp == 0x1f {
		// 无穷大或 NaN
		if mant == 0 {
			return math.Float32frombits(sign | 0x7f800000) // 无穷大
		}
		return math.Float32frombits(sign | 0x7f800000 | (mant << 13)) // NaN
	}

	// 正常范围：重新偏置指数（15 → 127）
	f32Exp := exp + 112 // 127 - 15 = 112
	return math.Float32frombits(sign | (f32Exp << 23) | (mant << 13))
}

// F32To16Slice 将 []float32 切片转换为 []uint16（float16 位表示）。
// 预分配目标切片以避免扩容开销。
func F32To16Slice(src []float32) []uint16 {
	dst := make([]uint16, len(src))
	for i, v := range src {
		dst[i] = Float32ToFloat16(v)
	}
	return dst
}

// F16To32Slice 将 []uint16（float16 位表示）切片转换为 []float32。
// 预分配目标切片以避免扩容开销。
func F16To32Slice(src []uint16) []float32 {
	dst := make([]float32, len(src))
	for i, v := range src {
		dst[i] = Float16ToFloat32(v)
	}
	return dst
}
