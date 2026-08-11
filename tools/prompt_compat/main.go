// prompt_compat 用 Go 端 PromptBuilder 构造矩阵，与 Python baseline 对比。
//
// 用法:
//
//	cd onnx_runtime/arktts_go
//	python3 tools/prompt_compat/gen_baseline.py   # 先生成 baseline
//	go run ./tools/prompt_compat/                 # 再跑对比
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"reflect"

	"github.com/metaRobin/arktts/prompt"
	"github.com/metaRobin/arktts/tokenizer"
)

type testCase struct {
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

func main() {
	defaultTokenizer := filepath.Join("..", "model", "tokenizer", "tokenizer.json")
	defaultBaseline := filepath.Join("tools", "prompt_compat", "prompt_baseline.json")
	tkPath := flag.String("tokenizer", defaultTokenizer, "path to tokenizer.json")
	blPath := flag.String("baseline", defaultBaseline, "path to prompt_baseline.json")
	flag.Parse()

	// 加载 baseline
	raw, err := os.ReadFile(*blPath)
	must(err, "read baseline")
	var cases []testCase
	must(json.Unmarshal(raw, &cases), "parse baseline")

	// 加载 tokenizer
	fmt.Fprintf(os.Stderr, "loading tokenizer: %s\n", *tkPath)
	tk, err := tokenizer.LoadFromFile(*tkPath)
	must(err, "load tokenizer")
	fmt.Fprintf(os.Stderr, "vocab size: %d\n\n", tk.VocabSize())

	// 用第一个 case 的配置创建 PromptBuilder
	builder := prompt.New(tk, cases[0].SemanticBeginID, cases[0].NumCodebooks)

	pass, fail := 0, 0
	for _, tc := range cases {
		// 用 Go 端构造矩阵
		goMatrix, err := builder.Build(tc.TargetText, tc.ReferenceText, tc.ReferenceCodes)
		must(err, "build prompt")

		// 对比矩阵
		if reflect.DeepEqual(goMatrix, tc.Matrix) {
			pass++
			fmt.Fprintf(os.Stderr, "  ✅ %s: shape %v\n", tc.Name, shapeOf(goMatrix))
		} else {
			fail++
			fmt.Fprintf(os.Stderr, "  ❌ %s: MISMATCH\n", tc.Name)
			printMatrixDiff(os.Stderr, tc.Matrix, goMatrix, tc.Name)

			// 对比 prefix/suffix
			goPrefix, goSuffix := builder.BuildPrefixSuffix(tc.TargetText, tc.ReferenceText)
			if !reflect.DeepEqual(goPrefix, tc.PrefixTokens) {
				fmt.Fprintf(os.Stderr, "    prefix diff:\n")
				fmt.Fprintf(os.Stderr, "      py (%d): %v\n", len(tc.PrefixTokens), tc.PrefixTokens)
				fmt.Fprintf(os.Stderr, "      go (%d): %v\n", len(goPrefix), goPrefix)
			}
			if !reflect.DeepEqual(goSuffix, tc.SuffixTokens) {
				fmt.Fprintf(os.Stderr, "    suffix diff:\n")
				fmt.Fprintf(os.Stderr, "      py (%d): %v\n", len(tc.SuffixTokens), tc.SuffixTokens)
				fmt.Fprintf(os.Stderr, "      go (%d): %v\n", len(goSuffix), goSuffix)
			}
		}
	}

	fmt.Fprintf(os.Stderr, "\n=== PromptBuilder 矩阵对比结果 ===\n")
	fmt.Fprintf(os.Stderr, "总用例: %d  通过: %d  失败: %d\n", len(cases), pass, fail)
	if fail == 0 {
		fmt.Fprintf(os.Stderr, "✅ ALL PASSED\n")
	} else {
		fmt.Fprintf(os.Stderr, "❌ MISMATCH\n")
		os.Exit(1)
	}
}

func shapeOf(m [][][]int64) []int {
	if len(m) == 0 {
		return []int{0}
	}
	s := []int{len(m), len(m[0])}
	if len(m[0]) > 0 {
		s = append(s, len(m[0][0]))
	}
	return s
}

func printMatrixDiff(w *os.File, py, goM [][][]int64, name string) {
	pyShape := shapeOf(py)
	goShape := shapeOf(goM)
	fmt.Fprintf(w, "    py shape: %v\n", pyShape)
	fmt.Fprintf(w, "    go shape: %v\n", goShape)

	if pyShape[1] != goShape[1] || pyShape[2] != goShape[2] {
		fmt.Fprintf(w, "    shape mismatch, skipping element diff\n")
		return
	}

	// 找第一个不同的元素
	for row := 0; row < pyShape[1]; row++ {
		for col := 0; col < pyShape[2]; col++ {
			pyVal := py[0][row][col]
			goVal := goM[0][row][col]
			if pyVal != goVal {
				fmt.Fprintf(w, "    first diff at [0,%d,%d]: py=%d go=%d\n", row, col, pyVal, goVal)
				// 打印该位置周围 5 个元素
				start := col - 2
				if start < 0 {
					start = 0
				}
				end := col + 3
				if end > pyShape[2] {
					end = pyShape[2]
				}
				fmt.Fprintf(w, "    py row %d [%d:%d]: %v\n", row, start, end, py[0][row][start:end])
				fmt.Fprintf(w, "    go row %d [%d:%d]: %v\n", row, start, end, goM[0][row][start:end])
				return
			}
		}
	}
}

func must(err error, msg string) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "error %s: %v\n", msg, err)
		os.Exit(2)
	}
}
