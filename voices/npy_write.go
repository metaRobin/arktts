package voices

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

// WriteNpyUint16 将二维 uint16 数据写入 .npy 文件。
// 布局：row-major（shape[0] 行 × shape[1] 列），与 numpy save 默认一致。
// 对齐 Python np.save(path, codes.astype(np.uint16))。
func WriteNpyUint16(w io.Writer, data [][]uint16) error {
	if len(data) == 0 || len(data[0]) == 0 {
		return fmt.Errorf("empty data for npy write")
	}
	rows := len(data)
	cols := len(data[0])

	// 构建 header 字典
	// {'descr': '<u2', 'fortran_order': False, 'shape': (rows, cols), }
	headerStr := fmt.Sprintf("{'descr': '<u2', 'fortran_order': False, 'shape': (%d, %d), }", rows, cols)

	// header 需填充到 64 字节对齐（含 \n）
	// 总 header = magic(6) + version(2) + header_len(2) + header_str + padding + \n
	// 即 10 + len(header_str) + padding + 1，需要对齐到 64
	prefixLen := 10 // magic + version + header_len_field
	totalLen := prefixLen + len(headerStr) + 1 // +1 for \n
	// 计算填充到 64 字节倍数
	padLen := (64 - totalLen%64) % 64
	paddedHeader := headerStr
	for i := 0; i < padLen; i++ {
		paddedHeader += " "
	}
	paddedHeader += "\n"

	headerBytes := []byte(paddedHeader)
	headerLen := uint16(len(headerBytes))
	if int(headerLen) != len(headerBytes) {
		return fmt.Errorf("npy header too long: %d", len(headerBytes))
	}

	// 写入
	// 1. magic
	magic := []byte{0x93, 'N', 'U', 'M', 'P', 'Y'}
	if _, err := w.Write(magic); err != nil {
		return fmt.Errorf("write npy magic: %w", err)
	}

	// 2. version 1.0
	if _, err := w.Write([]byte{1, 0}); err != nil {
		return fmt.Errorf("write npy version: %w", err)
	}

	// 3. header length (uint16 LE)
	if err := binary.Write(w, binary.LittleEndian, headerLen); err != nil {
		return fmt.Errorf("write npy header len: %w", err)
	}

	// 4. header string
	if _, err := w.Write(headerBytes); err != nil {
		return fmt.Errorf("write npy header: %w", err)
	}

	// 5. data (row-major uint16 LE)
	buf := make([]byte, 2*cols)
	for row := 0; row < rows; row++ {
		if len(data[row]) != cols {
			return fmt.Errorf("ragged data: row %d has %d cols, expected %d", row, len(data[row]), cols)
		}
		for col := 0; col < cols; col++ {
			binary.LittleEndian.PutUint16(buf[col*2:], data[row][col])
		}
		if _, err := w.Write(buf); err != nil {
			return fmt.Errorf("write npy data row %d: %w", row, err)
		}
	}

	return nil
}

// WriteNpyUint16File 将二维 uint16 数据写入 .npy 文件。
func WriteNpyUint16File(path string, data [][]uint16) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create npy file %q: %w", path, err)
	}
	defer f.Close()
	return WriteNpyUint16(f, data)
}
