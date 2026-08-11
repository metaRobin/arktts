// 用自实现的 tokenizer 与 baseline.json 做 100 条严格对比。
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/metaRobin/arktts/tokenizer"
)

type baselineItem struct {
	Index int      `json:"index"`
	Text  string   `json:"text"`
	IDs   []int    `json:"ids"`
	Tokens []string `json:"tokens"`
}

type baselineFile struct {
	Samples []baselineItem `json:"samples"`
}

func main() {
	defaultTokenizer := filepath.Join("..", "model", "tokenizer", "tokenizer.json")
	defaultBaseline := filepath.Join("tools", "tokenizer_compat", "baseline.json")
	tkPath := flag.String("tokenizer", defaultTokenizer, "path to tokenizer.json")
	blPath := flag.String("baseline", defaultBaseline, "path to baseline.json")
	flag.Parse()

	raw, err := os.ReadFile(*blPath)
	must(err, "read baseline")
	var bl baselineFile
	must(json.Unmarshal(raw, &bl), "parse baseline")

	fmt.Fprintf(os.Stderr, "loading tokenizer: %s\n", *tkPath)
	tk, err := tokenizer.LoadFromFile(*tkPath)
	must(err, "load tokenizer")
	fmt.Fprintf(os.Stderr, "vocab size: %d\n", tk.VocabSize())

	pass, fail := 0, 0
	var diffs []baselineItem
	for _, item := range bl.Samples {
		goIDs := tk.Encode(item.Text, false)
		if idsEqual(goIDs, item.IDs) {
			pass++
		} else {
			fail++
			diffs = append(diffs, item)
			// 打印前 10 个失败
			if len(diffs) <= 10 {
				fmt.Fprintf(os.Stderr, "\n[FAIL] index=%d text=%q\n", item.Index, trunc(item.Text, 80))
				fmt.Fprintf(os.Stderr, "  py ids (%d): %v\n", len(item.IDs), item.IDs)
				fmt.Fprintf(os.Stderr, "  go ids (%d): %v\n", len(goIDs), goIDs)
			}
		}
	}

	fmt.Fprintf(os.Stderr, "\n=== 自实现 tokenizer 结果 ===\n")
	fmt.Fprintf(os.Stderr, "总样本: %d  通过: %d  失败: %d\n", len(bl.Samples), pass, fail)
	if fail == 0 {
		fmt.Fprintf(os.Stderr, "✅ ALL PASSED\n")
	} else {
		fmt.Fprintf(os.Stderr, "❌ MISMATCH\n")
		os.Exit(1)
	}
}

func idsEqual(a, b []int) bool {
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

func must(err error, msg string) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "error %s: %v\n", msg, err)
		os.Exit(2)
	}
}

func trunc(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
