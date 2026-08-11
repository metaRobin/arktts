// Package voices 管理已注册的 TTS voice。
//
// voice 目录结构:
//
//	voices/
//	  speaker_a/
//	    meta.json    # voice 元数据
//	    codes.npy    # uint16, shape [num_codebooks, T]
package voices

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sync"
	"unsafe"
)

// isLittleEndian 在包初始化时检测字节序。
// 所有主流平台（amd64/arm64）均为小端。
var isLittleEndian = func() bool {
	var x uint16 = 1
	b := (*[2]byte)(unsafe.Pointer(&x))
	return b[0] == 1
}()

// 哨兵错误。
var (
	ErrNotFound    = errors.New("voice not found")
	ErrInvalidName = errors.New("invalid voice name")
)

// Meta 是 voice 的元数据。
type Meta struct {
	Name             string `json:"name"`
	ReferenceText    string `json:"reference_text"`
	Shape            []int  `json:"shape"`
	Dtype            string `json:"dtype"`
	SampleRate       int    `json:"sample_rate"`
	SourceAudio      string `json:"source_audio"`
	SourceSampleRate int    `json:"source_sample_rate"`
	SourceSHA256     string `json:"source_sha256"`
	ModelFingerprint string `json:"model_fingerprint"`
	CreatedAt        string `json:"created_at"`
	SourceKind       string `json:"source_kind"`
}

// Store 管理已注册的 voice 集合，内置内存缓存。
// voice 注册后不可变，缓存无需失效策略。
type Store struct {
	root         string
	numCodebooks int

	mu    sync.RWMutex
	cache map[string]cachedVoice
}

type cachedVoice struct {
	codes [][]int64
	meta  Meta
}

// New 创建 Store。
func New(root string, numCodebooks int) *Store {
	return &Store{root: root, numCodebooks: numCodebooks}
}

// Root 返回 voices 根目录路径。
func (s *Store) Root() string { return s.root }

// List 并发扫描所有已注册 voice 的元数据。
func (s *Store) List() ([]Meta, error) {
	entries, err := os.ReadDir(s.root)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("read voices dir: %w", err)
	}

	// 筛选目录
	var dirs []string
	for _, e := range entries {
		if e.IsDir() {
			dirs = append(dirs, e.Name())
		}
	}
	if len(dirs) == 0 {
		return nil, nil
	}

	// 并发加载 meta.json
	metas := make([]Meta, len(dirs))
	var wg sync.WaitGroup
	for i, name := range dirs {
		wg.Add(1)
		go func(idx int, n string) {
			defer wg.Done()
			if m, err := s.loadMeta(n); err == nil {
				metas[idx] = m
			}
		}(i, name)
	}
	wg.Wait()

	// 过滤加载失败的
	result := make([]Meta, 0, len(dirs))
	for _, m := range metas {
		if m.Name != "" {
			result = append(result, m)
		}
	}
	return result, nil
}

// Load 加载指定 voice 的 codes 和 meta，优先从缓存读取。
// 返回的 codes 切片为只读，调用方不应修改。
func (s *Store) Load(name string) ([][]int64, Meta, error) {
	if !isValidName(name) {
		return nil, Meta{}, fmt.Errorf("%w: %q", ErrInvalidName, name)
	}

	// 快路径：读缓存
	s.mu.RLock()
	if v, ok := s.cache[name]; ok {
		s.mu.RUnlock()
		return v.codes, v.meta, nil
	}
	s.mu.RUnlock()

	// 慢路径：从磁盘加载
	codes, meta, err := s.loadFromDisk(name)
	if err != nil {
		return nil, Meta{}, err
	}

	// 写缓存（double-check 防止重复加载）
	s.mu.Lock()
	if s.cache == nil {
		s.cache = make(map[string]cachedVoice)
	}
	s.cache[name] = cachedVoice{codes: codes, meta: meta}
	s.mu.Unlock()

	return codes, meta, nil
}

// Reload 强制从磁盘重新加载，更新缓存。
func (s *Store) Reload(name string) ([][]int64, Meta, error) {
	if !isValidName(name) {
		return nil, Meta{}, fmt.Errorf("%w: %q", ErrInvalidName, name)
	}

	codes, meta, err := s.loadFromDisk(name)
	if err != nil {
		return nil, Meta{}, err
	}

	s.mu.Lock()
	if s.cache == nil {
		s.cache = make(map[string]cachedVoice)
	}
	s.cache[name] = cachedVoice{codes: codes, meta: meta}
	s.mu.Unlock()

	return codes, meta, nil
}

// Invalidate 从缓存中移除指定 voice。
func (s *Store) Invalidate(name string) {
	s.mu.Lock()
	delete(s.cache, name)
	s.mu.Unlock()
}

// ClearCache 清空整个 voice 缓存，用于 reload 场景。
func (s *Store) ClearCache() {
	s.mu.Lock()
	s.cache = nil
	s.mu.Unlock()
}

func (s *Store) loadFromDisk(name string) ([][]int64, Meta, error) {
	// 并发读取 meta.json 和 codes.npy
	var (
		meta  Meta
		codes [][]int64
		mErr  error
		cErr  error
		wg    sync.WaitGroup
	)

	wg.Add(2)
	go func() {
		defer wg.Done()
		meta, mErr = s.loadMeta(name)
	}()
	go func() {
		defer wg.Done()
		codes, cErr = loadNpy(filepath.Join(s.root, name, "codes.npy"))
	}()
	wg.Wait()

	if mErr != nil {
		return nil, Meta{}, mErr
	}
	if meta.ReferenceText == "" {
		return nil, Meta{}, fmt.Errorf("voice %s has no reference_text", name)
	}
	if cErr != nil {
		return nil, Meta{}, fmt.Errorf("voice %s: %w", name, cErr)
	}

	if len(codes) != s.numCodebooks {
		return nil, Meta{}, fmt.Errorf("voice %s: expected %d codebooks, got %d",
			name, s.numCodebooks, len(codes))
	}
	if len(codes[0]) == 0 {
		return nil, Meta{}, fmt.Errorf("voice %s: empty codes", name)
	}

	return codes, meta, nil
}

func (s *Store) loadMeta(name string) (Meta, error) {
	data, err := os.ReadFile(filepath.Join(s.root, name, "meta.json"))
	if err != nil {
		if os.IsNotExist(err) {
			return Meta{}, fmt.Errorf("%w: %s", ErrNotFound, name)
		}
		return Meta{}, err
	}
	var m Meta
	if err := json.Unmarshal(data, &m); err != nil {
		return Meta{}, fmt.Errorf("parse meta for %s: %w", name, err)
	}
	return m, nil
}

func isValidName(name string) bool {
	if name == "" || name == "." || name == ".." {
		return false
	}
	return filepath.Base(name) == name
}

// --- npy 解析（零 regex、零 header 字符串分配） ---

var npyMagic = []byte{0x93, 'N', 'U', 'M', 'P', 'Y'}

// npyHeader 全部值类型，在栈上传递，无逃逸。
type npyHeader struct {
	dtype    string
	elemSize int
	rows     int
	cols     int
}

// loadNpy 读取 .npy 文件并返回二维 int64 切片。
func loadNpy(path string) ([][]int64, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return parseNpy(data)
}

func parseNpy(data []byte) ([][]int64, error) {
	if len(data) < 8 {
		return nil, errors.New("npy: file too short")
	}
	if !bytes.Equal(data[:len(npyMagic)], npyMagic) {
		return nil, errors.New("npy: bad magic")
	}

	pos := len(npyMagic)
	verMajor := data[pos]
	pos += 2 // skip version minor

	var hdrLen int
	switch verMajor {
	case 1:
		hdrLen = int(binary.LittleEndian.Uint16(data[pos : pos+2]))
		pos += 2
	case 2:
		hdrLen = int(binary.LittleEndian.Uint32(data[pos : pos+4]))
		pos += 4
	default:
		return nil, fmt.Errorf("npy: unsupported version %d", verMajor)
	}

	if len(data) < pos+hdrLen {
		return nil, errors.New("npy: truncated header")
	}

	// 直接在 []byte 上解析 header，不转 string、不用 regex
	hdr, err := parseNpyHeader(data[pos : pos+hdrLen])
	if err != nil {
		return nil, err
	}
	pos += hdrLen

	raw := data[pos:]

	// 溢出检查：rows*cols 和 rows*cols*elemSize
	totalElems := hdr.rows * hdr.cols
	if hdr.cols != 0 && totalElems/hdr.cols != hdr.rows {
		return nil, fmt.Errorf("npy: shape overflow (%d, %d)", hdr.rows, hdr.cols)
	}
	totalBytes := totalElems * hdr.elemSize
	if hdr.elemSize != 0 && totalBytes/hdr.elemSize != totalElems {
		return nil, fmt.Errorf("npy: size overflow (%d elements × %d bytes)", totalElems, hdr.elemSize)
	}
	if len(raw) < totalBytes {
		return nil, fmt.Errorf("npy: data too short: have %d bytes, need %d", len(raw), totalBytes)
	}

	// 唯一必要分配：int64 数据缓冲区
	flat := make([]int64, hdr.rows*hdr.cols)
	if err := decodeRaw(raw, hdr.dtype, flat); err != nil {
		return nil, err
	}

	// 零拷贝 reshape：行切片共享 flat 底层数组
	result := make([][]int64, hdr.rows)
	for i := range result {
		result[i] = flat[i*hdr.cols : (i+1)*hdr.cols]
	}
	return result, nil
}

// parseNpyHeader 用 bytes.Index 手写解析，替代 3 个 regex。
// header 格式: {'descr': '<u2', 'fortran_order': False, 'shape': (10, 216), }
func parseNpyHeader(data []byte) (npyHeader, error) {
	var hdr npyHeader

	// 提取 dtype: 'descr': '<u2'
	dtype, ok := extractQuoted(data, []byte("'descr'"))
	if !ok {
		return hdr, errors.New("npy: descr not found")
	}
	hdr.dtype = dtype
	hdr.elemSize = dtypeElemSize(dtype)
	if hdr.elemSize == 0 {
		return hdr, fmt.Errorf("npy: unsupported dtype %q", dtype)
	}

	// 提取 shape: 'shape': (10, 216)
	shapeBytes, ok := extractParen(data, []byte("'shape'"))
	if !ok {
		return hdr, errors.New("npy: shape not found")
	}
	rows, cols, err := parseDims(shapeBytes)
	if err != nil {
		return hdr, err
	}
	hdr.rows = rows
	hdr.cols = cols
	return hdr, nil
}

// extractQuoted 查找 key': 'value' 并返回 value。
func extractQuoted(data, key []byte) (string, bool) {
	idx := bytes.Index(data, key)
	if idx == -1 {
		return "", false
	}
	rest := data[idx+len(key):]
	// 跳过 : 和空白
	colon := bytes.IndexByte(rest, ':')
	if colon == -1 {
		return "", false
	}
	rest = rest[colon+1:]
	for len(rest) > 0 && (rest[0] == ' ' || rest[0] == '\t') {
		rest = rest[1:]
	}
	if len(rest) == 0 || rest[0] != '\'' {
		return "", false
	}
	rest = rest[1:]
	end := bytes.IndexByte(rest, '\'')
	if end == -1 {
		return "", false
	}
	return string(rest[:end]), true
}

// extractParen 查找 key': (...) 并返回括号内的内容。
func extractParen(data, key []byte) ([]byte, bool) {
	idx := bytes.Index(data, key)
	if idx == -1 {
		return nil, false
	}
	rest := data[idx+len(key):]
	colon := bytes.IndexByte(rest, ':')
	if colon == -1 {
		return nil, false
	}
	rest = rest[colon+1:]
	for len(rest) > 0 && (rest[0] == ' ' || rest[0] == '\t') {
		rest = rest[1:]
	}
	if len(rest) == 0 || rest[0] != '(' {
		return nil, false
	}
	rest = rest[1:]
	end := bytes.IndexByte(rest, ')')
	if end == -1 {
		return nil, false
	}
	return rest[:end], true
}

// parseDims 解析逗号分隔的维度（如 "10, 216"），返回栈上的两个 int。
// 安全性：防止整数溢出、零/负维度。不会 panic 或死循环。
func parseDims(data []byte) (int, int, error) {
	var dims [2]int
	count := 0
	num := 0
	hasNum := false
	const maxDim = 1 << 30 // 10 亿，合理上限

	for _, b := range data {
		if b >= '0' && b <= '9' {
			digit := int(b - '0')
			// 溢出检查：num*10+digit >= maxDim
			if num > maxDim/10 || (num == maxDim/10 && digit >= maxDim%10) {
				return 0, 0, fmt.Errorf("npy: dimension overflow")
			}
			num = num*10 + digit
			hasNum = true
		} else if hasNum {
			if count < 2 {
				dims[count] = num
			}
			count++
			num = 0
			hasNum = false
		}
	}
	if hasNum {
		if count < 2 {
			dims[count] = num
		}
		count++
	}

	if count != 2 {
		return 0, 0, fmt.Errorf("npy: expected 2D shape, got %dD", count)
	}
	if dims[0] <= 0 || dims[1] <= 0 {
		return 0, 0, fmt.Errorf("npy: invalid shape (%d, %d), dimensions must be positive", dims[0], dims[1])
	}
	return dims[0], dims[1], nil
}

// dtypeElemSize 返回 dtype 每元素字节数，0 表示不支持。
func dtypeElemSize(dtype string) int {
	switch dtype {
	case "<u1", "|u1", ">u1", "<i1", "|i1", ">i1":
		return 1
	case "<u2", "|u2", ">u2", "<i2", "|i2", ">i2":
		return 2
	case "<u4", "|u4", ">u4", "<i4", "|i4", ">i4", "<f4", "|f4", ">f4":
		return 4
	case "<u8", "|u8", ">u8", "<i8", "|i8", ">i8", "<f8", "|f8", ">f8":
		return 8
	default:
		return 0
	}
}

// decodeRaw 将原始字节解码为 int64。
//
// 小端平台（amd64/arm64）使用 unsafe.Slice 零拷贝重解释，消除逐元素位运算。
// 大端平台回退到安全路径。
func decodeRaw(data []byte, dtype string, out []int64) error {
	// 小端快速路径：直接将 []byte 重解释为目标类型切片
	if isLittleEndian && len(data) > 0 {
		switch dtype {
		case "<u2", "|u2":
			src := unsafe.Slice((*uint16)(unsafe.Pointer(&data[0])), len(out))
			for i, v := range src {
				out[i] = int64(v)
			}
			return nil
		case "<u4", "|u4":
			src := unsafe.Slice((*uint32)(unsafe.Pointer(&data[0])), len(out))
			for i, v := range src {
				out[i] = int64(v)
			}
			return nil
		case "<u8", "|u8":
			src := unsafe.Slice((*uint64)(unsafe.Pointer(&data[0])), len(out))
			for i, v := range src {
				out[i] = int64(v)
			}
			return nil
		case "<i2", "|i2":
			src := unsafe.Slice((*int16)(unsafe.Pointer(&data[0])), len(out))
			for i, v := range src {
				out[i] = int64(v)
			}
			return nil
		case "<i4", "|i4":
			src := unsafe.Slice((*int32)(unsafe.Pointer(&data[0])), len(out))
			for i, v := range src {
				out[i] = int64(v)
			}
			return nil
		case "<i8", "|i8":
			// int64 → int64：直接 copy，零转换
			src := unsafe.Slice((*int64)(unsafe.Pointer(&data[0])), len(out))
			copy(out, src)
			return nil
		case "<f4", "|f4":
			src := unsafe.Slice((*float32)(unsafe.Pointer(&data[0])), len(out))
			for i, v := range src {
				out[i] = int64(v)
			}
			return nil
		case "<f8", "|f8":
			src := unsafe.Slice((*float64)(unsafe.Pointer(&data[0])), len(out))
			for i, v := range src {
				out[i] = int64(v)
			}
			return nil
		}
	}

	// 安全回退：逐元素位运算（大端或非对齐）
	switch dtype {
	case "<u2", "|u2":
		for i := range out {
			j := i * 2
			out[i] = int64(uint16(data[j]) | uint16(data[j+1])<<8)
		}
	case ">u2":
		for i := range out {
			j := i * 2
			out[i] = int64(uint16(data[j])<<8 | uint16(data[j+1]))
		}
	case "<u1", "|u1", ">u1":
		for i := range out {
			out[i] = int64(data[i])
		}
	case "<u4", "|u4":
		for i := range out {
			out[i] = int64(binary.LittleEndian.Uint32(data[i*4:]))
		}
	case ">u4":
		for i := range out {
			j := i * 4
			out[i] = int64(data[j])<<24 | int64(data[j+1])<<16 | int64(data[j+2])<<8 | int64(data[j+3])
		}
	case "<u8", "|u8":
		for i := range out {
			out[i] = int64(binary.LittleEndian.Uint64(data[i*8:]))
		}
	case ">u8":
		for i := range out {
			j := i * 8
			out[i] = int64(data[j])<<56 | int64(data[j+1])<<48 | int64(data[j+2])<<40 | int64(data[j+3])<<32 |
				int64(data[j+4])<<24 | int64(data[j+5])<<16 | int64(data[j+6])<<8 | int64(data[j+7])
		}
	case "<i1", "|i1", ">i1":
		for i := range out {
			out[i] = int64(int8(data[i]))
		}
	case "<i2", "|i2":
		for i := range out {
			j := i * 2
			out[i] = int64(int16(uint16(data[j]) | uint16(data[j+1])<<8))
		}
	case ">i2":
		for i := range out {
			j := i * 2
			out[i] = int64(int16(uint16(data[j])<<8 | uint16(data[j+1])))
		}
	case "<i4", "|i4":
		for i := range out {
			out[i] = int64(int32(binary.LittleEndian.Uint32(data[i*4:])))
		}
	case ">i4":
		for i := range out {
			j := i * 4
			out[i] = int64(int32(data[j])<<24 | int32(data[j+1])<<16 | int32(data[j+2])<<8 | int32(data[j+3]))
		}
	case "<i8", "|i8":
		for i := range out {
			out[i] = int64(binary.LittleEndian.Uint64(data[i*8:]))
		}
	case ">i8":
		for i := range out {
			j := i * 8
			out[i] = int64(data[j])<<56 | int64(data[j+1])<<48 | int64(data[j+2])<<40 | int64(data[j+3])<<32 |
				int64(data[j+4])<<24 | int64(data[j+5])<<16 | int64(data[j+6])<<8 | int64(data[j+7])
		}
	case "<f4", "|f4":
		for i := range out {
			out[i] = int64(math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:])))
		}
	case ">f4":
		for i := range out {
			j := i * 4
			bits := uint32(data[j])<<24 | uint32(data[j+1])<<16 | uint32(data[j+2])<<8 | uint32(data[j+3])
			out[i] = int64(math.Float32frombits(bits))
		}
	case "<f8", "|f8":
		for i := range out {
			out[i] = int64(math.Float64frombits(binary.LittleEndian.Uint64(data[i*8:])))
		}
	case ">f8":
		for i := range out {
			j := i * 8
			bits := uint64(data[j])<<56 | uint64(data[j+1])<<48 | uint64(data[j+2])<<40 | uint64(data[j+3])<<32 |
				uint64(data[j+4])<<24 | uint64(data[j+5])<<16 | uint64(data[j+6])<<8 | uint64(data[j+7])
			out[i] = int64(math.Float64frombits(bits))
		}
	default:
		return fmt.Errorf("npy: unsupported dtype %q", dtype)
	}
	return nil
}
