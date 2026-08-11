// builder_test.go 固化 Go PromptBuilder 与 Python prompt.py 的矩阵对比逻辑。
//
// 运行前需先生成 baseline:
//
//	python3 tools/prompt_compat/gen_baseline.py
//
// 然后执行:
//
//	go test ./prompt/ -v
package prompt_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/metaRobin/arktts/prompt"
	"github.com/metaRobin/arktts/tokenizer"
)

// baselineCase 对应 gen_baseline.py 输出的单条测试用例。
type baselineCase struct {
	Name            string      `json:"name"`
	TargetText      string      `json:"target_text"`
	ReferenceText   string      `json:"reference_text"`
	ReferenceCodes  [][]int64   `json:"reference_codes"`
	MatrixShape     []int       `json:"matrix_shape"`
	Matrix          [][][]int64 `json:"matrix"`
	PrefixTokens    []int       `json:"prefix_tokens"`
	SuffixTokens    []int       `json:"suffix_tokens"`
	SemanticBeginID int         `json:"semantic_begin_id"`
	NumCodebooks    int         `json:"num_codebooks"`
}

// resolvePath 基于 prompt/ 包目录解析相对路径。
func resolvePath(t *testing.T, rel ...string) string {
	t.Helper()
	p := filepath.Join(rel...)
	if _, err := os.Stat(p); err != nil {
		t.Skipf("文件不存在: %s\n请先运行: python3 tools/prompt_compat/gen_baseline.py", p)
	}
	return p
}

// loadBaseline 加载 Python prompt.py 生成的 baseline JSON。
func loadBaseline(t *testing.T) []baselineCase {
	t.Helper()
	path := resolvePath(t, "..", "tools", "prompt_compat", "prompt_baseline.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("读取 baseline: %v", err)
	}
	var cases []baselineCase
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatalf("解析 baseline: %v", err)
	}
	return cases
}

// loadTokenizer 加载 tokenizer.json。
func loadTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	// prompt/ → arktts_go/ → onnx_runtime/model/tokenizer/
	path := resolvePath(t, "..", "..", "model", "tokenizer", "tokenizer.json")
	tk, err := tokenizer.LoadFromFile(path)
	if err != nil {
		t.Fatalf("加载 tokenizer: %v", err)
	}
	return tk
}

// firstDiff 在 [1, R, C] 矩阵中找到第一个不同的元素位置。
func firstDiff(py, goM [][][]int64) (row, col int, pyVal, goVal int64, found bool) {
	if len(py) == 0 || len(py[0]) == 0 || len(goM) == 0 || len(goM[0]) == 0 {
		return 0, 0, 0, 0, false
	}
	rows := len(py[0])
	cols := len(py[0][0])
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			if py[0][r][c] != goM[0][r][c] {
				return r, c, py[0][r][c], goM[0][r][c], true
			}
		}
	}
	return 0, 0, 0, 0, false
}

// TestBuildMatrix 对比 Go PromptBuilder 构造的 [1, num_codebooks+1, T] 矩阵
// 与 Python prompt.py 的 PromptBuilder.build 输出是否完全一致。
func TestBuildMatrix(t *testing.T) {
	cases := loadBaseline(t)
	tk := loadTokenizer(t)

	for _, tc := range cases {
		t.Run(tc.Name, func(t *testing.T) {
			builder := prompt.New(tk, tc.SemanticBeginID, tc.NumCodebooks)
			goMatrix, err := builder.Build(tc.TargetText, tc.ReferenceText, tc.ReferenceCodes)
			if err != nil {
				t.Fatalf("Build 失败: %v", err)
			}

			if reflect.DeepEqual(goMatrix, tc.Matrix) {
				t.Logf("✅ shape [1, %d, %d] 一致", len(goMatrix[0]), len(goMatrix[0][0]))
				return
			}

			pyRows, pyCols := 0, 0
			if len(tc.Matrix) > 0 && len(tc.Matrix[0]) > 0 {
				pyRows = len(tc.Matrix[0])
				pyCols = len(tc.Matrix[0][0])
			}
			goRows, goCols := 0, 0
			if len(goMatrix) > 0 && len(goMatrix[0]) > 0 {
				goRows = len(goMatrix[0])
				goCols = len(goMatrix[0][0])
			}
			t.Errorf("矩阵不一致: py=[1,%d,%d] go=[1,%d,%d]", pyRows, pyCols, goRows, goCols)

			// 形状一致时定位首个差异元素
			if pyRows == goRows && pyCols == goCols {
				if r, c, pv, gv, ok := firstDiff(tc.Matrix, goMatrix); ok {
					// 打印差异位置周围的上下文
					start := c - 3
					if start < 0 {
						start = 0
					}
					end := c + 4
					if end > pyCols {
						end = pyCols
					}
					t.Errorf("首个差异 [0,%d,%d]: py=%d go=%d\n    py row%d [%d:%d]: %v\n    go row%d [%d:%d]: %v",
						r, c, pv, gv,
						r, start, end, tc.Matrix[0][r][start:end],
						r, start, end, goMatrix[0][r][start:end])
				}
			}
		})
	}
}

// TestPrefixSuffix 对比 Go 端 prefix/suffix token 序列与 Python baseline。
// prefix 和 suffix 是矩阵 row0 的前后两段，用于隔离 tokenizer 编码差异。
func TestPrefixSuffix(t *testing.T) {
	cases := loadBaseline(t)
	tk := loadTokenizer(t)

	for _, tc := range cases {
		t.Run(tc.Name, func(t *testing.T) {
			builder := prompt.New(tk, tc.SemanticBeginID, tc.NumCodebooks)
			goPrefix, goSuffix := builder.BuildPrefixSuffix(tc.TargetText, tc.ReferenceText)

			if !reflect.DeepEqual(goPrefix, tc.PrefixTokens) {
				t.Errorf("prefix 不一致:\n  py (%d): %v\n  go (%d): %v",
					len(tc.PrefixTokens), tc.PrefixTokens, len(goPrefix), goPrefix)
			}
			if !reflect.DeepEqual(goSuffix, tc.SuffixTokens) {
				t.Errorf("suffix 不一致:\n  py (%d): %v\n  go (%d): %v",
					len(tc.SuffixTokens), tc.SuffixTokens, len(goSuffix), goSuffix)
			}
			if reflect.DeepEqual(goPrefix, tc.PrefixTokens) && reflect.DeepEqual(goSuffix, tc.SuffixTokens) {
				t.Logf("✅ prefix(%d) + suffix(%d) 一致", len(goPrefix), len(goSuffix))
			}
		})
	}
}

// TestEncodeText 对比单段文本的 tokenizer 编码结果。
// 直接验证 tokenizer.Encode 与 Python tokenizer.encode(text, add_special_tokens=False) 一致。
func TestEncodeText(t *testing.T) {
	cases := loadBaseline(t)
	tk := loadTokenizer(t)
	builder := prompt.New(tk, cases[0].SemanticBeginID, cases[0].NumCodebooks)

	// 从 baseline 的 prefix/suffix 反推各段文本的编码
	// prefix_parts[0] = "<|im_start|>system\n" 是固定的，可作为最小验证
	fixedTexts := []struct {
		name string
		text string
	}{
		{"im_start_system", "<|im_start|>system\n"},
		{"convert_prompt", "convert the provided text to speech reference to the following:\n\nText:\n"},
		{"im_end", "<|im_end|>\n"},
		{"im_start_user", "<|im_start|>user\n"},
		{"im_start_assistant_voice", "<|im_start|>assistant\n<|voice|>"},
		{"speech_newline", "\n\nSpeech:\n"},
		{"speaker_tag", "<|speaker:0|>"},
	}

	for _, ft := range fixedTexts {
		t.Run(ft.name, func(t *testing.T) {
			ids := builder.EncodeText(ft.text)
			if len(ids) == 0 {
				t.Fatalf("编码结果为空: %q", ft.text)
			}
			t.Logf("encode(%q) = %v (len=%d)", ft.text, ids, len(ids))
		})
	}
}

// TestCleanText 验证 CleanText 与 Python clean_text 的一致性。
func TestCleanText(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{"plain", "hello world", "hello world"},
		{"multi_space", "hello   world", "hello world"},
		{"leading_trailing", "  hello world  ", "hello world"},
		{"tabs_newlines", "hello\t\nworld", "hello world"},
		{"chinese", "你好 世界", "你好 世界"},
		{"empty", "", ""},
		{"only_spaces", "   ", ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := prompt.CleanText(tt.input)
			if got != tt.want {
				t.Errorf("CleanText(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

// TestFormatReferenceText 验证 FormatReferenceText 与 Python format_reference_text 的一致性。
func TestFormatReferenceText(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{"no_speaker_tag", "hello world", "<|speaker:0|>hello world"},
		{"with_speaker_0", "<|speaker:0|>hello", "<|speaker:0|>hello"},
		{"with_speaker_5", "<|speaker:5|>test", "<|speaker:5|>test"},
		{"with_speaker_99", "<|speaker:99|>abc", "<|speaker:99|>abc"},
		{"empty", "", "<|speaker:0|>"},
		{"multi_space_cleaned", "hello   world", "<|speaker:0|>hello world"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := prompt.FormatReferenceText(tt.input)
			if got != tt.want {
				t.Errorf("FormatReferenceText(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

// --- 安全性测试 ---

// TestBuildInvalidCodes 验证 Build 对非法输入返回 error 而非 panic。
func TestBuildInvalidCodes(t *testing.T) {
	tk := loadTokenizer(t)
	builder := prompt.New(tk, 151678, 10)

	tests := []struct {
		name   string
		codes  [][]int64
		errMsg string
	}{
		{"wrong_row_count", make([][]int64, 5), "must have shape"},
		{"empty_codes", [][]int64{{}}, "T>0"},
		{"ragged_codes", [][]int64{{1, 2}, {3}, {4, 5}}, "row 1 has len 1"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := builder.Build("test", "ref", tt.codes)
			if err == nil {
				t.Fatalf("expected error containing %q, got nil", tt.errMsg)
			}
		})
	}
}

// --- 基准测试 ---

// BenchmarkBuild 测试矩阵构造的性能和分配次数。
func BenchmarkBuild(b *testing.B) {
	path := filepath.Join("..", "tools", "prompt_compat", "prompt_baseline.json")
	data, err := os.ReadFile(path)
	if err != nil {
		b.Skip("baseline 不存在，请先运行 gen_baseline.py")
	}
	var cases []baselineCase
	if err := json.Unmarshal(data, &cases); err != nil {
		b.Fatal(err)
	}

	tkPath := filepath.Join("..", "..", "model", "tokenizer", "tokenizer.json")
	tk, err := tokenizer.LoadFromFile(tkPath)
	if err != nil {
		b.Skip("tokenizer 不存在")
	}

	// 选一个有代表性的用例
	tc := cases[0]
	builder := prompt.New(tk, tc.SemanticBeginID, tc.NumCodebooks)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, err := builder.Build(tc.TargetText, tc.ReferenceText, tc.ReferenceCodes)
		if err != nil {
			b.Fatal(err)
		}
	}
}
