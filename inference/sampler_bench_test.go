package inference

import (
	"testing"

	"github.com/metaRobin/arktts/config"
)

// BenchmarkSample 量化采样器每次调用的分配次数。
func BenchmarkSample(b *testing.B) {
	// 模拟 codebook size 的 logits（典型 1024）
	logits := make([]float32, 1024)
	for i := range logits {
		logits[i] = float32(i) * 0.001
	}
	rng := NewPCG64(42)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = Sample(logits, 0.7, 0.9, 50, rng)
	}
}

// BenchmarkSampleSemantic 量化语义采样的分配（含 allowedIDs 构造 + 2 次 Sample）。
func BenchmarkSampleSemantic(b *testing.B) {
	m := &config.RuntimeManifest{
		SlowLogitsLayout: "semantic_then_eos",
		SlowLogitsSize:   384, // end-begin+1+1 (stop)
		SemanticBeginID:  151552,
		SemanticEndID:    151934,
		ImEndID:          151643,
	}
	e := &Engine{manifest: m}
	// 手动初始化预计算的 allowedIDs（模拟 NewEngine 中的逻辑）
	allowedRange := m.SemanticEndID - m.SemanticBeginID + 1
	e.semanticAllowedIDs = make([]int, allowedRange+1)
	for i := 0; i < allowedRange; i++ {
		e.semanticAllowedIDs[i] = m.SemanticBeginID + i
	}
	e.semanticAllowedIDs[allowedRange] = m.ImEndID
	logits := make([]float32, m.SlowLogitsSize)
	for i := range logits {
		logits[i] = float32(i) * 0.00001
	}
	previous := make([]int, 0, 10)
	rng := NewPCG64(42)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = e.SampleSemantic(logits, previous, 0.7, 0.9, 50, rng)
	}
}
