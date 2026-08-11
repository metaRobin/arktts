package inference

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/metaRobin/arktts/config"
)

// ─── Golden JSON 类型定义 ───

type p0Case struct {
	Logits      []float32 `json:"logits"`
	Temperature float64   `json:"temperature"`
	TopP        float64   `json:"top_p"`
	TopK        int       `json:"top_k"`
	Expected    int       `json:"expected"`
}

type p1Case struct {
	Logits         []float32 `json:"logits"`
	Previous       []int     `json:"previous"`
	Temperature    float64   `json:"temperature"`
	TopP           float64   `json:"top_p"`
	TopK           int       `json:"top_k"`
	ExpectedNormal int       `json:"expected_normal"`
	ExpectedHigh   int       `json:"expected_high"`
	ExpectedFinal  int       `json:"expected_final"`
}

type goldenData struct {
	P0 struct {
		Seed  int64    `json:"seed"`
		Cases []p0Case `json:"cases"`
	} `json:"p0_sample_sequence"`
	P1 struct {
		Seed           int64    `json:"seed"`
		Begin          int      `json:"begin"`
		End            int      `json:"end"`
		Stop           int      `json:"stop"`
		SlowLogitsSize int      `json:"slow_logits_size"`
		Cases          []p1Case `json:"cases"`
	} `json:"p1_sample_semantic"`
}

func loadSamplerGolden(t *testing.T) *goldenData {
	t.Helper()
	data, err := os.ReadFile("sampler_golden.json")
	if err != nil {
		t.Fatalf("read sampler_golden.json: %v", err)
	}
	var g goldenData
	if err := json.Unmarshal(data, &g); err != nil {
		t.Fatalf("parse sampler_golden.json: %v", err)
	}
	return &g
}

func intInSlice(slice []int, v int) bool {
	for _, s := range slice {
		if s == v {
			return true
		}
	}
	return false
}

// ─── P0: Sample 序列回归测试 ───
//
// 用同一 PCG64(42) 顺序调用 15 次 Sample，每次的 logits 大小/参数不同。
// golden 值由 Python numpy 生成（/tmp/gen_sampler_golden.py）。
//
// 此测试是 P0 优化（Sample 缓冲区复用）的回归守护：
//   - 如果复用的缓冲区未正确清零 → 残留数据影响排序/softmax → 结果发散
//   - 如果缓冲区未正确 resize → 越界或 stale data → 结果发散
//   - 如果 RNG 消费次数变化 → 后续所有调用结果全部发散
func TestP0_SampleGoldenSequence(t *testing.T) {
	g := loadSamplerGolden(t)
	rng := NewPCG64(g.P0.Seed)

	for i, c := range g.P0.Cases {
		result := Sample(c.Logits, c.Temperature, c.TopP, c.TopK, rng)
		if result != c.Expected {
			t.Errorf("P0 case %d: got %d, want %d (size=%d, temp=%.3f, top_p=%.2f, top_k=%d)",
				i+1, result, c.Expected, len(c.Logits), c.Temperature, c.TopP, c.TopK)
		}
	}
}

// TestP0_SampleSequenceReproducible 验证同一序列两次运行结果完全一致。
// 如果缓冲区复用引入了不确定性（如未初始化内存），两次运行可能不同。
func TestP0_SampleSequenceReproducible(t *testing.T) {
	g := loadSamplerGolden(t)

	runOnce := func() []int {
		rng := NewPCG64(g.P0.Seed)
		results := make([]int, len(g.P0.Cases))
		for i, c := range g.P0.Cases {
			results[i] = Sample(c.Logits, c.Temperature, c.TopP, c.TopK, rng)
		}
		return results
	}

	first := runOnce()
	second := runOnce()
	for i := range first {
		if first[i] != second[i] {
			t.Errorf("P0 reproducible: run1[%d]=%d != run2[%d]=%d", i, first[i], i, second[i])
		}
	}
}

// ─── P1: SampleSemantic 回归测试 ───
//
// 16 组场景，覆盖：
//   - 空 previous / 非空 previous
//   - repeat penalty 触发（normal 在 previous 中 → 返回 high）
//   - repeat penalty 不触发（normal 不在 previous 中 → 返回 normal）
//   - normal != high 的场景（case 13/14/15，真正验证回退逻辑）
//   - 边界值：normal == begin / end / stop
//   - 超长 previous（>10，验证不截断行为）
//
// golden 值由 Python numpy 生成。此测试是 P1 优化（预计算 allowedIDs）的回归守护。
func TestP1_SampleSemanticGolden(t *testing.T) {
	g := loadSamplerGolden(t)

	m := &config.RuntimeManifest{
		SlowLogitsLayout: "semantic_then_eos",
		SlowLogitsSize:   g.P1.SlowLogitsSize,
		SemanticBeginID:  g.P1.Begin,
		SemanticEndID:    g.P1.End,
		ImEndID:          g.P1.Stop,
	}
	e := &Engine{manifest: m}
	// 手动初始化预计算的 allowedIDs（模拟 NewEngine 中的逻辑）
	allowedRange := m.SemanticEndID - m.SemanticBeginID + 1
	e.semanticAllowedIDs = make([]int, allowedRange+1)
	for i := 0; i < allowedRange; i++ {
		e.semanticAllowedIDs[i] = m.SemanticBeginID + i
	}
	e.semanticAllowedIDs[allowedRange] = m.ImEndID

	for i, c := range g.P1.Cases {
		rng := NewPCG64(g.P1.Seed)
		result := e.SampleSemantic(c.Logits, c.Previous, c.Temperature, c.TopP, c.TopK, rng)

		prevHasNormal := intInSlice(c.Previous, c.ExpectedNormal)
		if result != c.ExpectedFinal {
			t.Errorf("P1 case %d: got %d, want %d (normal=%d, high=%d, prev_has_normal=%v, prev_len=%d)",
				i+1, result, c.ExpectedFinal,
				c.ExpectedNormal, c.ExpectedHigh, prevHasNormal, len(c.Previous))
		}

		// 额外验证：当 normal != high 时，repeat penalty 逻辑可被区分
		if c.ExpectedNormal != c.ExpectedHigh {
			if prevHasNormal && c.ExpectedFinal != c.ExpectedHigh {
				t.Errorf("P1 case %d: normal in previous but final != high (normal=%d, high=%d, final=%d)",
					i+1, c.ExpectedNormal, c.ExpectedHigh, c.ExpectedFinal)
			}
			if !prevHasNormal && c.ExpectedFinal != c.ExpectedNormal {
				t.Errorf("P1 case %d: normal not in previous but final != normal (normal=%d, high=%d, final=%d)",
					i+1, c.ExpectedNormal, c.ExpectedHigh, c.ExpectedFinal)
			}
		}
	}
}

// TestP1_SampleSemanticAllowedIDsMapping 验证 allowedIDs 映射正确。
// 使用 top_k=1（贪心），normal 和 high 都确定性地返回 argmax 位置。
// 如果预计算的 allowedIDs 顺序/内容错误，映射出的 token ID 会不对。
func TestP1_SampleSemanticAllowedIDsMapping(t *testing.T) {
	g := loadSamplerGolden(t)

	m := &config.RuntimeManifest{
		SlowLogitsLayout: "semantic_then_eos",
		SlowLogitsSize:   g.P1.SlowLogitsSize,
		SemanticBeginID:  g.P1.Begin,
		SemanticEndID:    g.P1.End,
		ImEndID:          g.P1.Stop,
	}
	e := &Engine{manifest: m}
	// 手动初始化预计算的 allowedIDs（模拟 NewEngine 中的逻辑）
	allowedRange := m.SemanticEndID - m.SemanticBeginID + 1
	e.semanticAllowedIDs = make([]int, allowedRange+1)
	for i := 0; i < allowedRange; i++ {
		e.semanticAllowedIDs[i] = m.SemanticBeginID + i
	}
	e.semanticAllowedIDs[allowedRange] = m.ImEndID

	// 构造 logits：峰值在 index 0（对应 begin）、index end-begin（对应 end）、
	// index end-begin+1（对应 stop），验证 ID 映射
	peaks := []struct {
		peakIdx    int
		expectedID int
		desc       string
	}{
		{0, g.P1.Begin, "begin"},
		{g.P1.End - g.P1.Begin, g.P1.End, "end"},
		{g.P1.SlowLogitsSize - 1, g.P1.Stop, "stop"},
	}

	for _, pk := range peaks {
		logits := make([]float32, g.P1.SlowLogitsSize)
		for i := range logits {
			logits[i] = 1.0
		}
		logits[pk.peakIdx] = 10.0

		rng := NewPCG64(g.P1.Seed)
		// top_k=1 → 贪心，normal 和 high 都返回 peakIdx
		result := e.SampleSemantic(logits, nil, 0.7, 0.9, 1, rng)
		if result != pk.expectedID {
			t.Errorf("allowedIDs mapping %s: peakIdx=%d, got %d, want %d",
				pk.desc, pk.peakIdx, result, pk.expectedID)
		}
	}
}
