// Package prompt 实现了 PromptBuilder，构造 [1, num_codebooks+1, T] 的输入矩阵。
//
// 与 arktts_runtime/prompt.py 的 PromptBuilder.build 逻辑完全一致。
package prompt

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/metaRobin/arktts/tokenizer"
)

// PromptBuilder 构造 TTS 模型的输入矩阵。
type PromptBuilder struct {
	tk              *tokenizer.Tokenizer
	semanticBeginID int
	numCodebooks    int
}

// New 创建 PromptBuilder。
func New(tk *tokenizer.Tokenizer, semanticBeginID, numCodebooks int) *PromptBuilder {
	return &PromptBuilder{
		tk:              tk,
		semanticBeginID: semanticBeginID,
		numCodebooks:    numCodebooks,
	}
}

// CleanText 等价于 prompt.py 的 clean_text：strip + 多空格合并为单空格。
func CleanText(text string) string {
	return strings.Join(strings.Fields(text), " ")
}

var speakerTagRe = regexp.MustCompile(`<\|speaker:\d+\|>`)

// FormatReferenceText 等价于 prompt.py 的 format_reference_text。
func FormatReferenceText(text string) string {
	text = CleanText(text)
	if speakerTagRe.MatchString(text) {
		return text
	}
	return "<|speaker:0|>" + text
}

// EncodeText 等价于 prompt.py 的 encode_text：add_special_tokens=False 编码。
func (b *PromptBuilder) EncodeText(text string) []int {
	return b.tk.Encode(text, false)
}

// Build 构造 [1, num_codebooks+1, T] 的输入矩阵。
//
// 与 prompt.py 的 PromptBuilder.build 完全一致：
//   - row 0 = prefix_tokens + (codes[0] + semantic_begin_id) + suffix_tokens
//   - row 1..N = 0（prefix 段），codes[1..N]（semantic 段）
//
// 优化：所有行共享一个连续 flat 缓冲区，仅 2 次分配（flat + 行头）。
func (b *PromptBuilder) Build(targetText, referenceText string, referenceCodes [][]int64) ([][][]int64, error) {
	// 校验 reference_codes
	if len(referenceCodes) != b.numCodebooks {
		return nil, fmt.Errorf("reference codes must have shape [%d, T>0], got %d rows",
			b.numCodebooks, len(referenceCodes))
	}
	codesT := len(referenceCodes[0])
	if codesT == 0 {
		return nil, fmt.Errorf("reference codes must have T>0")
	}
	for i, row := range referenceCodes {
		if len(row) != codesT {
			return nil, fmt.Errorf("reference codes row %d has len %d, expected %d", i, len(row), codesT)
		}
	}

	// 构造 prefix
	prefixParts := []string{
		"<|im_start|>system\n",
		"convert the provided text to speech reference to the following:\n\nText:\n",
		FormatReferenceText(referenceText),
		"\n\nSpeech:\n",
	}
	var prefix []int
	for _, part := range prefixParts {
		prefix = append(prefix, b.EncodeText(part)...)
	}

	// 构造 suffix
	suffixParts := []string{
		"<|im_end|>\n",
		"<|im_start|>user\n",
		CleanText(targetText),
		"<|im_end|>\n",
		"<|im_start|>assistant\n<|voice|>",
	}
	var suffix []int
	for _, part := range suffixParts {
		suffix = append(suffix, b.EncodeText(part)...)
	}

	// 单一 flat 分配：所有行共享一个连续缓冲区
	row0Len := len(prefix) + codesT + len(suffix)
	numRows := b.numCodebooks + 1
	flat := make([]int64, numRows*row0Len)

	// 行视图：零拷贝切片
	values := make([][]int64, numRows)
	for i := range values {
		values[i] = flat[i*row0Len : (i+1)*row0Len]
	}

	// 写 row 0：prefix + semantic_ids + suffix
	row0 := values[0]
	idx := 0
	for _, id := range prefix {
		row0[idx] = int64(id)
		idx++
	}
	// semantic_ids = codes[0] + semantic_begin_id（直接写入，省掉中间数组）
	semanticBase := int64(b.semanticBeginID)
	for i := 0; i < codesT; i++ {
		row0[idx] = referenceCodes[0][i] + semanticBase
		idx++
	}
	for _, id := range suffix {
		row0[idx] = int64(id)
		idx++
	}

	// 写 codes 到 row 1..N：用 copy() 替代逐元素循环
	begin := len(prefix)
	for row := 1; row <= b.numCodebooks; row++ {
		copy(values[row][begin:begin+codesT], referenceCodes[row-1])
	}

	// 返回 [1, num_codebooks+1, T]
	return [][][]int64{values}, nil
}

// BuildPrefixSuffix 返回 prefix 和 suffix 的 token ids，用于调试对比。
func (b *PromptBuilder) BuildPrefixSuffix(targetText, referenceText string) (prefix, suffix []int) {
	prefixParts := []string{
		"<|im_start|>system\n",
		"convert the provided text to speech reference to the following:\n\nText:\n",
		FormatReferenceText(referenceText),
		"\n\nSpeech:\n",
	}
	for _, part := range prefixParts {
		prefix = append(prefix, b.EncodeText(part)...)
	}

	suffixParts := []string{
		"<|im_end|>\n",
		"<|im_start|>user\n",
		CleanText(targetText),
		"<|im_end|>\n",
		"<|im_start|>assistant\n<|voice|>",
	}
	for _, part := range suffixParts {
		suffix = append(suffix, b.EncodeText(part)...)
	}
	return prefix, suffix
}
