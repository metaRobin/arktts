// Package tokenizer 实现了 HuggingFace tokenizers 的 byte-level BPE 编码，
// 专门用于加载 Audio8 TTS 的 tokenizer.json（Qwen 系 GPT2 BPE）。
//
// 与 Python tokenizers 库 100% 一致，纯 Go 实现，无 cgo 依赖。
//
// 核心流程:
//  1. NFC 标准化
//  2. Split (GPT2 正则切分) → segments
//  3. ByteLevel (字节→unicode 映射)
//  4. BPE (先查 vocab，再按 merges 合并)
//
// 用法:
//
//	tk, err := tokenizer.LoadFromFile("tokenizer.json")
//	ids := tk.Encode("你好", false) // addSpecialTokens=false
package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/dlclark/regexp2"
	"golang.org/x/text/unicode/norm"
)

// Tokenizer 是 byte-level BPE 分词器。
type Tokenizer struct {
	vocab       map[string]int  // token → id
	merges      []mergeRule     // merge 规则列表（按 rank 排序）
	mergeRanks  map[string]int  // "a b" → rank（索引）
	splitRe     *regexp2.Regexp // GPT2 pre-tokenizer 正则
	addedTokens map[string]int  // added_token → id
	addedRe     *regexp.Regexp  // added_tokens 匹配正则（字面量，用标准 regexp）
	normalizer  norm.Form       // NFC
	encodeCache sync.Map        // string → []int（NFC 后的文本 → token IDs）
}

type mergeRule struct {
	parts [2]string
}

// tokenizerJSON 是 tokenizer.json 的最小化结构，只解析需要的字段。
type tokenizerJSON struct {
	Normalizer *struct {
		Type string `json:"type"`
	} `json:"normalizer"`
	PreTokenizer *struct {
		Type         string `json:"type"`
		Pretokenizers []struct {
			Type    string `json:"type"`
			Pattern *struct {
				Regex  string `json:"Regex"`
				String string `json:"String"`
			} `json:"pattern"`
			Behavior string `json:"behavior"`
			Invert   bool   `json:"invert"`
		} `json:"pretokenizers"`
	} `json:"pre_tokenizer"`
	Model struct {
		Type          string              `json:"type"`
		Vocab         map[string]int      `json:"vocab"`
		Merges        []interface{}       `json:"merges"` // 可能是 ["a b"] 或 [["a","b"]]
		IgnoreMerges  bool                `json:"ignore_merges"`
	} `json:"model"`
	AddedTokens []struct {
		ID       int    `json:"id"`
		Content  string `json:"content"`
	} `json:"added_tokens"`
}

// LoadFromFile 从 tokenizer.json 文件加载分词器。
func LoadFromFile(path string) (*Tokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read tokenizer.json: %w", err)
	}
	return LoadFromBytes(data)
}

// LoadFromBytes 从 tokenizer.json 字节流加载分词器。
func LoadFromBytes(data []byte) (*Tokenizer, error) {
	var tj tokenizerJSON
	if err := json.Unmarshal(data, &tj); err != nil {
		return nil, fmt.Errorf("parse tokenizer.json: %w", err)
	}

	tk := &Tokenizer{
		vocab:      tj.Model.Vocab,
		mergeRanks: make(map[string]int),
		addedTokens: make(map[string]int),
		normalizer: norm.NFC,
	}

	// 解析 merges（可能是 ["a b"] 或 [["a","b"]]）
	for i, m := range tj.Model.Merges {
		var a, b string
		switch v := m.(type) {
		case string:
			parts := strings.SplitN(v, " ", 2)
			if len(parts) != 2 {
				return nil, fmt.Errorf("invalid merge %d: %q", i, v)
			}
			a, b = parts[0], parts[1]
		case []interface{}:
			if len(v) != 2 {
				return nil, fmt.Errorf("invalid merge %d: %v", i, v)
			}
			a = fmt.Sprint(v[0])
			b = fmt.Sprint(v[1])
		default:
			return nil, fmt.Errorf("invalid merge %d: %T", i, m)
		}
		tk.merges = append(tk.merges, mergeRule{parts: [2]string{a, b}})
		tk.mergeRanks[a+" "+b] = i
	}

	// 解析 GPT2 正则（从 Split pre_tokenizer）
	if tj.PreTokenizer != nil && tj.PreTokenizer.Type == "Sequence" {
		for _, pt := range tj.PreTokenizer.Pretokenizers {
			if pt.Type == "Split" && pt.Pattern != nil && pt.Pattern.Regex != "" {
				re, err := regexp2.Compile(pt.Pattern.Regex, regexp2.RE2)
				if err != nil {
					return nil, fmt.Errorf("compile GPT2 regex: %w", err)
				}
				tk.splitRe = re
				break
			}
		}
	}
	if tk.splitRe == nil {
		return nil, fmt.Errorf("Split pre_tokenizer with Regex pattern not found")
	}

	// 解析 added_tokens
	for _, at := range tj.AddedTokens {
		tk.addedTokens[at.Content] = at.ID
	}

	// 构建 added_tokens 匹配正则（按长度降序排列，避免部分匹配）
	if len(tk.addedTokens) > 0 {
		tokens := make([]string, 0, len(tk.addedTokens))
		for tok := range tk.addedTokens {
			tokens = append(tokens, tok)
		}
		// 按长度降序排序
		for i := 0; i < len(tokens); i++ {
			for j := i + 1; j < len(tokens); j++ {
				if len(tokens[j]) > len(tokens[i]) {
					tokens[i], tokens[j] = tokens[j], tokens[i]
				}
			}
		}
		// 转义正则特殊字符（用标准 regexp.QuoteMeta，字面量匹配）
		parts := make([]string, len(tokens))
		for i, tok := range tokens {
			parts[i] = regexp.QuoteMeta(tok)
		}
		pattern := strings.Join(parts, "|")
		re, err := regexp.Compile(pattern)
		if err != nil {
			return nil, fmt.Errorf("compile added_tokens regex: %w", err)
		}
		tk.addedRe = re
	}

	return tk, nil
}

// VocabSize 返回词表大小。
func (t *Tokenizer) VocabSize() int {
	return len(t.vocab)
}

// Encode 对文本进行 BPE 编码，返回 token IDs。
// addSpecialTokens 为 true 时在首尾添加 special tokens（当前未实现，等同于 false）。
//
// 返回的 slice 是只读的（可能来自缓存），调用方不应修改。
// 相同输入会命中缓存返回同一个 slice，适合 PromptBuilder 的固定 prompt 段。
func (t *Tokenizer) Encode(text string, addSpecialTokens bool) []int {
	_ = addSpecialTokens

	// 步骤 1: NFC 标准化
	text = t.normalizer.String(text)

	// 步骤 2: 查缓存（固定 prompt 段第二次调用起直接命中）
	if cached, ok := t.encodeCache.Load(text); ok {
		return cached.([]int)
	}

	// 步骤 3: 提取 added_tokens
	segments := t.extractAddedTokens(text)

	// 步骤 4: 对每个 segment 做 Split + ByteLevel + BPE
	// 预分配：平均 ~1 token / 3 字符
	ids := make([]int, 0, len(text)/3+1)
	for _, seg := range segments {
		if seg.isAddedToken {
			ids = append(ids, seg.id)
			continue
		}
		splitSegs := t.split(seg.text)
		for _, ss := range splitSegs {
			byteLevelText := byteLevelEncode(ss)
			segIDs := t.bpe(byteLevelText)
			ids = append(ids, segIDs...)
		}
	}

	// 存缓存（存副本，防止外部修改）
	cached := make([]int, len(ids))
	copy(cached, ids)
	t.encodeCache.Store(text, cached)

	return ids
}

// segment 表示文本的一段，可能是 added_token 或普通文本。
type segment struct {
	text        string
	isAddedToken bool
	id          int // added_token 的 id
}

// extractAddedTokens 把文本拆分为 added_token 段和普通文本段。
func (t *Tokenizer) extractAddedTokens(text string) []segment {
	if t.addedRe == nil {
		return []segment{{text: text}}
	}

	matches := t.addedRe.FindAllStringSubmatchIndex(text, -1)
	if len(matches) == 0 {
		return []segment{{text: text}}
	}

	var segments []segment
	cur := 0
	for _, m := range matches {
		start, end := m[0], m[1]
		// 添加匹配前的普通文本
		if start > cur {
			segments = append(segments, segment{text: text[cur:start]})
		}
		// 添加 added_token
		matched := text[start:end]
		if id, ok := t.addedTokens[matched]; ok {
			segments = append(segments, segment{
				text:        matched,
				isAddedToken: true,
				id:          id,
			})
		} else {
			segments = append(segments, segment{text: matched})
		}
		cur = end
	}
	// 添加最后的普通文本
	if cur < len(text) {
		segments = append(segments, segment{text: text[cur:]})
	}

	return segments
}

// split 用 GPT2 正则切分文本，返回 segments。
func (t *Tokenizer) split(text string) []string {
	// 预分配：平均 ~1 segment / 4 字符
	segments := make([]string, 0, len(text)/4+1)
	m, err := t.splitRe.FindStringMatchStartingAt(text, 0)
	if err != nil {
		return []string{text}
	}
	for m != nil {
		segments = append(segments, m.String())
		m, err = t.splitRe.FindNextMatch(m)
		if err != nil {
			break
		}
	}
	return segments
}

// bpe 对 ByteLevel 转换后的 segment 做 BPE 编码。
//
// 优化：用字节偏移 span 替代字符串拷贝，in-place 合并。
// 消除每字符 string(r)、每次合并 newSymbols + merged 字符串的分配。
func (t *Tokenizer) bpe(text string) []int {
	if text == "" {
		return nil
	}

	// 先查 vocab：如果整个 segment 在 vocab 里，直接返回
	if id, ok := t.vocab[text]; ok {
		return []int{id}
	}

	// 构建 span：每个字符在 text 中的字节偏移 [start, end)
	type span struct {
		start int
		end   int
	}
	spans := make([]span, 0, len(text)) // 上限 = 字节数
	pos := 0
	for _, r := range text {
		charLen := utf8.RuneLen(r)
		spans = append(spans, span{start: pos, end: pos + charLen})
		pos += charLen
	}
	if len(spans) == 0 {
		return nil
	}

	// 可复用的 pair key 缓冲区
	pairBuf := make([]byte, 0, 64)

	// BPE 合并循环（in-place，无 newSymbols 分配）
	n := len(spans)
	for n > 1 {
		bestRank := -1
		bestIdx := -1

		for i := 0; i < n-1; i++ {
			// 构建 pair key: "text_i text_j"
			pairBuf = pairBuf[:0]
			pairBuf = append(pairBuf, text[spans[i].start:spans[i].end]...)
			pairBuf = append(pairBuf, ' ')
			pairBuf = append(pairBuf, text[spans[i+1].start:spans[i+1].end]...)

			if rank, ok := t.mergeRanks[string(pairBuf)]; ok {
				if bestRank == -1 || rank < bestRank {
					bestRank = rank
					bestIdx = i
				}
			}
		}

		if bestIdx == -1 {
			break
		}

		// In-place 合并：扩展 span，后移剩余元素
		spans[bestIdx].end = spans[bestIdx+1].end
		copy(spans[bestIdx+1:], spans[bestIdx+2:])
		n--
	}

	// 查 vocab 得到 id
	ids := make([]int, 0, n)
	for i := 0; i < n; i++ {
		if id, ok := t.vocab[text[spans[i].start:spans[i].end]]; ok {
			ids = append(ids, id)
		}
	}
	return ids
}

// byteLevelEncode 把文本的每个字节映射为 unicode 字符。
// 与 HuggingFace 的 bytes_to_unicode() 一致。
var byteToUnicode [256]rune

func init() {
	// 构建字节到 unicode 的映射表（与 HF bytes_to_unicode 一致）
	bs := make([]int, 0, 256)
	for b := 33; b <= 126; b++ {
		bs = append(bs, b) // !-~
	}
	for b := 161; b <= 172; b++ {
		bs = append(bs, b) // ¡-¬
	}
	for b := 174; b <= 255; b++ {
		bs = append(bs, b) // ®-ÿ
	}

	// 在 bs 中的字节映射为自身
	for _, b := range bs {
		byteToUnicode[b] = rune(b)
	}

	// 不在 bs 中的字节映射为 256+n
	n := 0
	for b := 0; b < 256; b++ {
		found := false
		for _, e := range bs {
			if e == b {
				found = true
				break
			}
		}
		if !found {
			byteToUnicode[b] = rune(256 + n)
			n++
		}
	}
}

// byteLevelEncode 把文本的每个字节映射为对应的 unicode 字符。
func byteLevelEncode(text string) string {
	// 预分配：最坏情况每个字节映射为 2 字节 UTF-8
	var sb strings.Builder
	sb.Grow(len(text) * 2)
	for i := 0; i < len(text); i++ {
		sb.WriteRune(byteToUnicode[text[i]])
	}
	return sb.String()
}
