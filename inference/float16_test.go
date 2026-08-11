package inference

import (
	"math"
	"testing"
)

func TestFloat32ToFloat16_KnownValues(t *testing.T) {
	tests := []struct {
		name string
		f    float32
		want uint16
	}{
		{"零", 0.0, 0x0000},
		{"正一", 1.0, 0x3C00},
		{"负一", -1.0, 0xBC00},
		{"二", 2.0, 0x4000},
		{"二分之一", 0.5, 0x3800},
		{"float16最大值65504", 65504.0, 0x7BFF},
		{"正无穷", float32(math.Inf(1)), 0x7C00},
		{"负无穷", float32(math.Inf(-1)), 0xFC00},
		{"最小正常值2^-14", 2.0 / 32768.0, 0x0400}, // exp16=1, mant=0
		{"最小次正常值2^-24", float32(math.Ldexp(1.0, -24)), 0x0001},
		{"负零", float32(math.Copysign(0.0, -1.0)), 0x8000},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Float32ToFloat16(tt.f)
			if got != tt.want {
				t.Errorf("Float32ToFloat16(%v) = 0x%04X, want 0x%04X", tt.f, got, tt.want)
			}
		})
	}
}

func TestFloat16ToFloat32_KnownValues(t *testing.T) {
	tests := []struct {
		name string
		h    uint16
		want float32
	}{
		{"零", 0x0000, 0.0},
		{"正一", 0x3C00, 1.0},
		{"负一", 0xBC00, -1.0},
		{"二", 0x4000, 2.0},
		{"二分之一", 0x3800, 0.5},
		{"float16最大值", 0x7BFF, 65504.0},
		{"正无穷", 0x7C00, float32(math.Inf(1))},
		{"负无穷", 0xFC00, float32(math.Inf(-1))},
		{"最小正常值", 0x0400, float32(math.Ldexp(1.0, -14))},
		{"最小次正常值", 0x0001, float32(math.Ldexp(1.0, -24))},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Float16ToFloat32(tt.h)
			if math.Float32bits(got) != math.Float32bits(tt.want) {
				// 对 NaN 使用 IsNaN 判断
				if math.IsNaN(float64(tt.want)) && math.IsNaN(float64(got)) {
					return
				}
				t.Errorf("Float16ToFloat32(0x%04X) = %v, want %v", tt.h, got, tt.want)
			}
		})
	}
}

func TestFloat32ToFloat16_NaN(t *testing.T) {
	// NaN 转换后应仍为 NaN
	nanF32 := float32(math.NaN())
	h := Float32ToFloat16(nanF32)
	// 指数全 1 且尾数非零 → NaN
	if (h & 0x7c00) != 0x7c00 {
		t.Fatalf("NaN 转换后指数不为全1: 0x%04X", h)
	}
	if h&0x03ff == 0 {
		t.Fatalf("NaN 转换后尾数为零（会变成无穷大）: 0x%04X", h)
	}
	result := Float16ToFloat32(h)
	if !math.IsNaN(float64(result)) {
		t.Errorf("NaN 往返后不是 NaN: %v", result)
	}
}

func TestFloat16RoundTrip(t *testing.T) {
	// 这些值在 float16 中可精确表示，f32→f16→f32 应无损
	values := []float32{
		0.0,
		1.0,
		-1.0,
		2.0,
		0.5,
		-0.5,
		65504.0,
		-65504.0,
		float32(math.Ldexp(1.0, -14)),  // 最小正常值
		float32(math.Ldexp(1.0, -24)),  // 最小次正常值
		float32(math.Ldexp(1.0, -23)),  // 2 * 2^-24 = 2^-23（次正常值）
		float32(math.Ldexp(1.0, 15)),   // 2^15 = 32768
		float32(math.Ldexp(3.0, 10)),   // 3072
		2048.0,
		1024.0,
		3.140625, // 11.001001 binary = 1.1001001 * 2^1，10 位尾数可精确表示
	}

	for _, v := range values {
		h := Float32ToFloat16(v)
		roundTrip := Float16ToFloat32(h)
		if math.Float32bits(roundTrip) != math.Float32bits(v) {
			t.Errorf("往返失败: %v → 0x%04X → %v", v, h, roundTrip)
		}
	}
}

func TestFloat32ToFloat16_Rounding(t *testing.T) {
	// 65520.0 超过 float16 最大值（65504），舍入后应溢出为无穷大
	h := Float32ToFloat16(65520.0)
	if h != 0x7C00 {
		t.Errorf("65520.0 应溢出为无穷大(0x7C00), got 0x%04X", h)
	}

	// 65536.0 = 2^16，明确超出 float16 范围
	h = Float32ToFloat16(65536.0)
	if h != 0x7C00 {
		t.Errorf("65536.0 应溢出为无穷大(0x7C00), got 0x%04X", h)
	}
}

func TestF32To16Slice(t *testing.T) {
	src := []float32{0.0, 1.0, -1.0, 2.0, 0.5, 65504.0}
	want := []uint16{0x0000, 0x3C00, 0xBC00, 0x4000, 0x3800, 0x7BFF}

	dst := F32To16Slice(src)
	if len(dst) != len(want) {
		t.Fatalf("切片长度 %d, want %d", len(dst), len(want))
	}
	for i, v := range want {
		if dst[i] != v {
			t.Errorf("dst[%d] = 0x%04X, want 0x%04X", i, dst[i], v)
		}
	}
}

func TestF16To32Slice(t *testing.T) {
	src := []uint16{0x0000, 0x3C00, 0xBC00, 0x4000, 0x3800, 0x7BFF}
	want := []float32{0.0, 1.0, -1.0, 2.0, 0.5, 65504.0}

	dst := F16To32Slice(src)
	if len(dst) != len(want) {
		t.Fatalf("切片长度 %d, want %d", len(dst), len(want))
	}
	for i, v := range want {
		if math.Float32bits(dst[i]) != math.Float32bits(v) {
			t.Errorf("dst[%d] = %v, want %v", i, dst[i], v)
		}
	}
}

func TestSliceRoundTrip(t *testing.T) {
	// 切片往返：f32 → f16 → f32 对可表示值应无损
	src := []float32{0.0, 1.0, -1.0, 2.0, 0.5, 65504.0, -65504.0}
	f16 := F32To16Slice(src)
	dst := F16To32Slice(f16)

	for i, v := range src {
		if math.Float32bits(dst[i]) != math.Float32bits(v) {
			t.Errorf("切片往返失败 [%d]: %v → %v", i, v, dst[i])
		}
	}
}
