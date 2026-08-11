package voices

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// findVoicesDir 查找测试用的 voices 目录。
func findVoicesDir() string {
	candidates := []string{
		filepath.Join("..", "voices"),       // arktts_go/ -> onnx_runtime/voices
		filepath.Join("..", "..", "voices"), // 子目录 -> onnx_runtime/voices
		filepath.Join("..", "..", "..", "onnx_runtime", "voices"),
	}
	for _, p := range candidates {
		if _, err := os.Stat(filepath.Join(p, "speaker_a", "codes.npy")); err == nil {
			abs, _ := filepath.Abs(p)
			return abs
		}
	}
	return ""
}

func TestLoadWithCache(t *testing.T) {
	dir := findVoicesDir()
	if dir == "" {
		t.Skip("voices 目录不存在，跳过")
	}

	s := New(dir, 10)

	// 第一次加载：磁盘 I/O
	codes1, meta1, err := s.Load("speaker_a")
	if err != nil {
		t.Fatalf("首次加载失败: %v", err)
	}
	if meta1.Name == "" {
		t.Fatal("meta.Name 为空")
	}

	// 第二次加载：应命中缓存
	codes2, meta2, err := s.Load("speaker_a")
	if err != nil {
		t.Fatalf("缓存加载失败: %v", err)
	}

	// 验证缓存返回相同的指针（零拷贝）
	if len(codes1) != len(codes2) {
		t.Errorf("codes 行数不一致: %d vs %d", len(codes1), len(codes2))
	}
	if &codes1[0][0] != &codes2[0][0] {
		t.Error("缓存未返回同一底层数组")
	}
	if meta1.Name != meta2.Name {
		t.Errorf("meta 不一致: %s vs %s", meta1.Name, meta2.Name)
	}
}

func TestInvalidate(t *testing.T) {
	dir := findVoicesDir()
	if dir == "" {
		t.Skip("voices 目录不存在，跳过")
	}

	s := New(dir, 10)

	// 加载到缓存
	if _, _, err := s.Load("speaker_a"); err != nil {
		t.Fatalf("加载失败: %v", err)
	}

	// 检查缓存存在
	s.mu.RLock()
	_, cached := s.cache["speaker_a"]
	s.mu.RUnlock()
	if !cached {
		t.Fatal("缓存写入失败")
	}

	// 失效
	s.Invalidate("speaker_a")

	s.mu.RLock()
	_, cached = s.cache["speaker_a"]
	s.mu.RUnlock()
	if cached {
		t.Fatal("Invalidate 后缓存仍存在")
	}
}

func TestInvalidName(t *testing.T) {
	s := New("/tmp", 10)

	tests := []string{"", ".", "..", "../etc", "foo/bar"}
	for _, name := range tests {
		_, _, err := s.Load(name)
		if !errors.Is(err, ErrInvalidName) {
			t.Errorf("Load(%q) 应返回 ErrInvalidName, got %v", name, err)
		}
	}
}

// --- 基准测试 ---

func BenchmarkLoad_Disk(b *testing.B) {
	dir := findVoicesDir()
	if dir == "" {
		b.Skip("voices 目录不存在")
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s := New(dir, 10) // 每次新建 Store，无缓存
		if _, _, err := s.Load("speaker_a"); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLoad_Cache(b *testing.B) {
	dir := findVoicesDir()
	if dir == "" {
		b.Skip("voices 目录不存在")
	}

	s := New(dir, 10)
	// 预热缓存
	if _, _, err := s.Load("speaker_a"); err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := s.Load("speaker_a"); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkParseNpy(b *testing.B) {
	dir := findVoicesDir()
	if dir == "" {
		b.Skip("voices 目录不存在")
	}

	data, err := os.ReadFile(filepath.Join(dir, "speaker_a", "codes.npy"))
	if err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := parseNpy(data); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkParseNpy_Large 构造一个 10×2048 的合成 npy 测试大数据量下的扩展性。
func BenchmarkParseNpy_Large(b *testing.B) {
	const rows, cols = 10, 2048
	// 构造合成的 npy 数据
	hdr := "{'descr': '<u2', 'fortran_order': False, 'shape': (" + itoa(rows) + ", " + itoa(cols) + "), }"
	// 对齐到 64 字节
	for len(hdr)%64 != 63 { // 63 = 64 - 1 (newline)
		hdr += " "
	}
	hdr += "\n"

	hdrLen := len(hdr)
	data := make([]byte, 0, 10+4+hdrLen+rows*cols*2)
	data = append(data, npyMagic...)
	data = append(data, 1, 0) // version 1.0
	data = append(data, byte(hdrLen), byte(hdrLen>>8))
	data = append(data, hdr...)
	// 填充数据：每元素 = row*cols + col (mod 65536)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			v := uint16(r*cols + c)
			data = append(data, byte(v), byte(v>>8))
		}
	}

	b.SetBytes(int64(len(data)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result, err := parseNpy(data)
		if err != nil {
			b.Fatal(err)
		}
		if len(result) != rows || len(result[0]) != cols {
			b.Fatalf("shape mismatch: %d×%d", len(result), len(result[0]))
		}
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	pos := len(buf)
	for n > 0 {
		pos--
		buf[pos] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[pos:])
}

// --- parseDims 安全性测试 ---

func TestParseDimsSafety(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		wantErr bool
	}{
		{"normal", "10, 216", false},
		{"with_spaces", "  10, 216  ", false},
		{"zero_dim", "0, 216", true},
		{"zero_dim2", "10, 0", true},
		{"overflow", "99999999999999999999999999, 1", true},
		{"too_few", "10", true},
		{"too_many", "10, 20, 30", true},
		{"empty", "", true},
		{"max_ok", "1073741823, 1", false},  // 2^30 - 1
		{"over_max", "1073741824, 1", true}, // 2^30
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r, c, err := parseDims([]byte(tt.input))
			if tt.wantErr {
				if err == nil {
					t.Errorf("parseDims(%q) = (%d, %d, nil), want error", tt.input, r, c)
				}
			} else {
				if err != nil {
					t.Errorf("parseDims(%q) error: %v", tt.input, err)
				}
			}
		})
	}
}

// TestParseNpyOverflow 验证恶意 npy 不会导致 panic。
func TestParseNpyOverflow(t *testing.T) {
	// 构造一个声明 shape=(2000000000, 2000000000) 但数据为空的 npy
	hdr := "{'descr': '<u2', 'fortran_order': False, 'shape': (2000000000, 2000000000), }"
	for len(hdr)%64 != 63 {
		hdr += " "
	}
	hdr += "\n"
	hdrLen := len(hdr)

	data := make([]byte, 0, 10+4+hdrLen)
	data = append(data, npyMagic...)
	data = append(data, 1, 0)
	data = append(data, byte(hdrLen), byte(hdrLen>>8))
	data = append(data, hdr...)

	_, err := parseNpy(data)
	if err == nil {
		t.Fatal("expected overflow error, got nil")
	}
}
