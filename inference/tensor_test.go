package inference

import (
	"bytes"
	"testing"
)

// TestWriteFloat16SliceAtPosition_SinglePosition 验证 KV cache 按 position/head
// 写入 delta 的布局正确性。cache 布局 [1, heads, seqLen, headDim] C-contiguous。
func TestWriteFloat16SliceAtPosition_SinglePosition(t *testing.T) {
	const heads, seqLen, headDim = 2, 4, 3
	cache := make([]uint16, heads*seqLen*headDim) // 全零
	// delta: head0=[10,11,12], head1=[20,21,22]
	delta := []uint16{10, 11, 12, 20, 21, 22}

	if err := WriteFloat16SliceAtPosition(cache, heads, seqLen, headDim, []int{2}, delta); err != nil {
		t.Fatalf("写入失败: %v", err)
	}

	// head0 pos2 → cache[(0*4+2)*3 : +3] = cache[6:9]
	if got := cache[6:9]; !eqU16(got, []uint16{10, 11, 12}) {
		t.Errorf("head0 pos2 = %v, want [10 11 12]", got)
	}
	// head1 pos2 → cache[(1*4+2)*3 : +3] = cache[18:21]
	if got := cache[18:21]; !eqU16(got, []uint16{20, 21, 22}) {
		t.Errorf("head1 pos2 = %v, want [20 21 22]", got)
	}
	// 其它位置保持零
	if cache[0] != 0 || cache[5] != 0 || cache[23] != 0 {
		t.Errorf("未写入位置被污染: cache[0]=%d cache[5]=%d cache[23]=%d", cache[0], cache[5], cache[23])
	}
}

// TestWriteFloat16SliceAtPosition_MultiplePositions 验证多 position 写入的 delta 偏移。
func TestWriteFloat16SliceAtPosition_MultiplePositions(t *testing.T) {
	const heads, seqLen, headDim = 2, 4, 3
	cache := make([]uint16, heads*seqLen*headDim)
	// positions=[1,3]; delta 布局 [heads, len(positions), headDim]
	// head0_pos1, head0_pos3, head1_pos1, head1_pos3
	delta := []uint16{11, 12, 13, 31, 32, 33, 41, 42, 43, 51, 52, 53}

	if err := WriteFloat16SliceAtPosition(cache, heads, seqLen, headDim, []int{1, 3}, delta); err != nil {
		t.Fatalf("写入失败: %v", err)
	}

	want := map[int][]uint16{
		3:  {11, 12, 13},  // head0 pos1: (0*4+1)*3=3
		9:  {31, 32, 33},  // head0 pos3: (0*4+3)*3=9
		15: {41, 42, 43},  // head1 pos1: (1*4+1)*3=15
		21: {51, 52, 53},  // head1 pos3: (1*4+3)*3=21
	}
	for off, w := range want {
		if got := cache[off : off+headDim]; !eqU16(got, w) {
			t.Errorf("cache[%d:%d] = %v, want %v", off, off+headDim, got, w)
		}
	}
}

// TestWriteFloat16SliceAtPosition_OutOfRange 越界 position 应返回 error。
func TestWriteFloat16SliceAtPosition_OutOfRange(t *testing.T) {
	cache := make([]uint16, 2*4*3)
	delta := make([]uint16, 2*1*3)

	if err := WriteFloat16SliceAtPosition(cache, 2, 4, 3, []int{4}, delta); err == nil {
		t.Error("position==seqLen 应返回错误，got nil")
	}
	if err := WriteFloat16SliceAtPosition(cache, 2, 4, 3, []int{-1}, delta); err == nil {
		t.Error("position=-1 应返回错误，got nil")
	}
}

// TestWriteFloat16SliceAtPosition_DeltaTooShort delta 长度不足应返回 error。
func TestWriteFloat16SliceAtPosition_DeltaTooShort(t *testing.T) {
	cache := make([]uint16, 2*4*3)
	delta := make([]uint16, 3) // 需要 2*1*3=6，只给 3

	if err := WriteFloat16SliceAtPosition(cache, 2, 4, 3, []int{0}, delta); err == nil {
		t.Error("delta 过短应返回错误，got nil")
	}
}

// TestWriteFloat16AtPosition_Bytes 字节版本写入，与 uint16 版本结果一致。
func TestWriteFloat16AtPosition_Bytes(t *testing.T) {
	const heads, seqLen, headDim = 1, 4, 2
	cache := make([]byte, heads*seqLen*headDim*2) // float16=2 bytes
	// delta: head0=[0x1234, 0x5678]
	delta := []byte{0x34, 0x12, 0x78, 0x56}

	if err := WriteFloat16AtPosition(cache, heads, seqLen, headDim, []int{1}, delta); err != nil {
		t.Fatalf("写入失败: %v", err)
	}
	// head0 pos1: offset=(0*4+1)*2*2=4 bytes
	if got := cache[4:8]; !bytes.Equal(got, delta) {
		t.Errorf("cache[4:8] = %v, want %v", got, delta)
	}
}

// TestBytesToUint16 验证小端字节到 uint16 的零拷贝重解释。
func TestBytesToUint16(t *testing.T) {
	// 0x1234 小端存储为 [0x34, 0x12]
	data := []byte{0x34, 0x12, 0x78, 0x56}
	got, err := BytesToUint16(data)
	if err != nil {
		t.Fatalf("错误: %v", err)
	}
	want := []uint16{0x1234, 0x5678}
	for i, w := range want {
		if got[i] != w {
			t.Errorf("[%d] = 0x%04X, want 0x%04X", i, got[i], w)
		}
	}
}

// TestBytesToUint16_OddLength 奇数字节长度应返回 error。
func TestBytesToUint16_OddLength(t *testing.T) {
	if _, err := BytesToUint16([]byte{0x01, 0x02, 0x03}); err == nil {
		t.Error("奇数字节长度应返回错误")
	}
}

// TestBytesToUint16_Empty 空输入返回 nil,nil。
func TestBytesToUint16_Empty(t *testing.T) {
	got, err := BytesToUint16(nil)
	if err != nil {
		t.Fatalf("空输入不应报错: %v", err)
	}
	if got != nil {
		t.Errorf("空输入应返回 nil，got %v", got)
	}
}

// TestUint16ToBytes 验证 uint16 到小端字节的转换。
func TestUint16ToBytes(t *testing.T) {
	data := []uint16{0x1234, 0x5678, 0xABCD}
	got := Uint16ToBytes(data)
	want := []byte{0x34, 0x12, 0x78, 0x56, 0xCD, 0xAB}
	if !bytes.Equal(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

// TestUint16BytesRoundTrip Uint16ToBytes → BytesToUint16 往返无损。
func TestUint16BytesRoundTrip(t *testing.T) {
	orig := []uint16{0x0000, 0xFFFF, 0x1234, 0xABCD, 0x00FF}
	b := Uint16ToBytes(orig)
	back, err := BytesToUint16(b)
	if err != nil {
		t.Fatalf("往返错误: %v", err)
	}
	for i, v := range orig {
		if back[i] != v {
			t.Errorf("[%d] = 0x%04X, want 0x%04X", i, back[i], v)
		}
	}
}

// TestUint16ToBytes_Empty 空输入返回 nil。
func TestUint16ToBytes_Empty(t *testing.T) {
	if got := Uint16ToBytes(nil); got != nil {
		t.Errorf("空输入应返回 nil，got %v", got)
	}
}

// TestNativeEndianUint16 验证小端字节解析。
func TestNativeEndianUint16(t *testing.T) {
	cases := []struct {
		bytes []byte
		want  uint16
	}{
		{[]byte{0x34, 0x12}, 0x1234},
		{[]byte{0x00, 0x00}, 0x0000},
		{[]byte{0xFF, 0xFF}, 0xFFFF},
		{[]byte{0x01, 0x00}, 0x0001},
	}
	for _, c := range cases {
		if got := NativeEndianUint16(c.bytes); got != c.want {
			t.Errorf("NativeEndianUint16(%v) = 0x%04X, want 0x%04X", c.bytes, got, c.want)
		}
	}
}

// eqU16 比较两个 uint16 切片是否相等。
func eqU16(a, b []uint16) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
