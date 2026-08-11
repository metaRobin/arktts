package inference

import (
	"math"
	"testing"
)

// TestFlattenPromptMatrix 验证 [1, rows, cols] 矩阵按 C-contiguous 行优先
// 展平为 []int64，与 ONNX 张量布局一致。
func TestFlattenPromptMatrix(t *testing.T) {
	// 构造 [1, 3, 4] 矩阵，每行每列填入可区分的值
	rows, cols := 3, 4
	matrix := make([][][]int64, 1)
	matrix[0] = make([][]int64, rows)
	for r := 0; r < rows; r++ {
		matrix[0][r] = make([]int64, cols)
		for c := 0; c < cols; c++ {
			matrix[0][r][c] = int64(r*10 + c)
		}
	}

	flat := flattenPromptMatrix(matrix, rows, cols)
	if len(flat) != rows*cols {
		t.Fatalf("长度 = %d, want %d", len(flat), rows*cols)
	}

	// 期望行优先：row0 col0..3, row1 col0..3, row2 col0..3
	want := []int64{0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23}
	for i, v := range want {
		if flat[i] != v {
			t.Errorf("flat[%d] = %d, want %d", i, flat[i], v)
		}
	}
}

// TestFlattenPromptMatrix_SingleRow 单行矩阵展平。
func TestFlattenPromptMatrix_SingleRow(t *testing.T) {
	matrix := [][][]int64{{{7, 8, 9}}}
	flat := flattenPromptMatrix(matrix, 1, 3)
	want := []int64{7, 8, 9}
	for i, v := range want {
		if flat[i] != v {
			t.Errorf("flat[%d] = %d, want %d", i, flat[i], v)
		}
	}
}

// TestArgmax 验证返回最大值索引，平局时返回首个（最低索引）。
func TestArgmax(t *testing.T) {
	cases := []struct {
		name   string
		logits []float32
		want   int
	}{
		{"empty_returns_0", []float32{}, 0},
		{"single", []float32{5.0}, 0},
		{"max_at_start", []float32{9.0, 1.0, 2.0}, 0},
		{"max_at_middle", []float32{1.0, 9.0, 2.0}, 1},
		{"max_at_end", []float32{1.0, 2.0, 9.0}, 2},
		{"negative_values", []float32{-5.0, -1.0, -3.0}, 1},
		// 平局：严格 > 比较，保留首个最大值（最低索引）
		{"tie_returns_lowest_index", []float32{5.0, 5.0, 5.0}, 0},
		{"tie_first_occurrence", []float32{1.0, 5.0, 5.0}, 1},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := Argmax(c.logits)
			if got != c.want {
				t.Errorf("Argmax(%v) = %d, want %d", c.logits, got, c.want)
			}
		})
	}
}

// TestArgmax_InfAndNaN 边界值：-Inf 不应被选为最大；+Inf 应被选。
func TestArgmax_InfAndNaN(t *testing.T) {
	negInf := float32(math.Inf(-1))
	posInf := float32(math.Inf(1))

	if got := Argmax([]float32{negInf, 1.0, negInf}); got != 1 {
		t.Errorf("-Inf 中间夹正常值: Argmax = %d, want 1", got)
	}
	if got := Argmax([]float32{1.0, posInf, 2.0}); got != 1 {
		t.Errorf("+Inf 应被选为最大: Argmax = %d, want 1", got)
	}
}
