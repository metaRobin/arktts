package inference

import (
	"math"
	"testing"
)

func TestSample_SingleToken(t *testing.T) {
	// 单个 token：无论参数如何，始终返回 0
	rng := NewPCG64(42)
	logits := []float32{5.0}
	result := Sample(logits, 0.7, 0.9, 50, rng)
	if result != 0 {
		t.Errorf("单 token 采样结果 = %d, want 0", result)
	}
}

func TestSample_TopK1_Greedy(t *testing.T) {
	// top_k=1 时只保留 top-1 token，相当于贪心解码，结果确定
	rng := NewPCG64(42)
	logits := []float32{1.0, 3.0, 2.0, 0.5}
	// 最大值在索引 1
	result := Sample(logits, 0.7, 0.9, 1, rng)
	if result != 1 {
		t.Errorf("top_k=1 贪心采样结果 = %d, want 1 (argmax)", result)
	}
}

func TestSample_TopP0_Greedy(t *testing.T) {
	// top_p=0 时，除 top-1 外所有 token 的累积概率 > 0 被移除
	// 等效于贪心解码
	rng := NewPCG64(42)
	logits := []float32{1.0, 3.0, 2.0, 0.5}
	result := Sample(logits, 0.7, 0.0, 50, rng)
	if result != 1 {
		t.Errorf("top_p=0 贪心采样结果 = %d, want 1 (argmax)", result)
	}
}

func TestSample_Deterministic(t *testing.T) {
	// 相同种子 + 相同输入应产生相同输出
	logits := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}

	rng1 := NewPCG64(123)
	result1 := Sample(logits, 0.7, 0.9, 50, rng1)

	rng2 := NewPCG64(123)
	result2 := Sample(logits, 0.7, 0.9, 50, rng2)

	if result1 != result2 {
		t.Errorf("相同种子采样不一致: %d vs %d", result1, result2)
	}
	if result1 < 0 || result1 >= len(logits) {
		t.Errorf("采样结果越界: %d", result1)
	}
}

func TestSample_AllSameLogits(t *testing.T) {
	// 所有 logits 相同：未移除的 token 概率均等，结果应在有效范围内
	rng := NewPCG64(42)
	logits := []float32{2.0, 2.0, 2.0, 2.0, 2.0}
	result := Sample(logits, 1.0, 1.0, 10, rng)
	if result < 0 || result >= len(logits) {
		t.Errorf("采样结果越界: %d", result)
	}
}

func TestSample_Temperature1_TopP1(t *testing.T) {
	// temperature=1.0, top_p=1.0：不过滤，正常采样
	rng := NewPCG64(42)
	logits := []float32{1.0, 2.0, 3.0, 4.0, 5.0}
	result := Sample(logits, 1.0, 1.0, 10, rng)
	if result < 0 || result >= len(logits) {
		t.Errorf("采样结果越界: %d", result)
	}
}

func TestSample_TopK2_OnlyTopTwo(t *testing.T) {
	// top_k=2：只保留前 2 个 token，结果必须是 argmax 或次大的索引
	// 用大量不同种子验证结果始终在 top-2 集合中
	logits := []float32{1.0, 5.0, 2.0, 4.0, 0.5}
	// 排序后：index 1 (5.0), index 3 (4.0), 其余被移除
	validResults := map[int]bool{1: true, 3: true}

	for seed := int64(0); seed < 100; seed++ {
		rng := NewPCG64(seed)
		result := Sample(logits, 1.0, 1.0, 2, rng)
		if !validResults[result] {
			t.Errorf("seed=%d: top_k=2 采样结果 = %d, 应在 {1, 3} 中", seed, result)
		}
	}
}

func TestSample_RemovedTokenNeverSelected(t *testing.T) {
	// top_k=1 时，只有 top-1 被保留，其余概率为 0
	// 用大量种子验证被移除的 token 永远不会被采样到
	logits := []float32{1.0, 5.0, 2.0, 4.0}
	for seed := int64(0); seed < 200; seed++ {
		rng := NewPCG64(seed)
		result := Sample(logits, 0.7, 0.9, 1, rng)
		if result != 1 { // 5.0 在索引 1，是唯一的 argmax
			t.Fatalf("seed=%d: top_k=1 应始终返回 argmax(1), got %d", seed, result)
		}
	}
}

func TestSample_LowTemperatureGreedy(t *testing.T) {
	// 极低 temperature 使分布极度尖锐，趋近于贪心
	rng := NewPCG64(42)
	logits := []float32{1.0, 2.0, 10.0, 3.0, 4.0}
	result := Sample(logits, 0.001, 1.0, 50, rng)
	if result != 2 { // 10.0 在索引 2
		t.Errorf("极低温度应趋近贪心, got %d, want 2", result)
	}
}

func TestSample_NegativeLogits(t *testing.T) {
	// 负 logits 值测试
	rng := NewPCG64(42)
	logits := []float32{-5.0, -1.0, -3.0, -2.0}
	result := Sample(logits, 1.0, 1.0, 10, rng)
	if result < 0 || result >= len(logits) {
		t.Errorf("采样结果越界: %d", result)
	}
	// top_k=1 时应返回最大值索引（-1.0 在索引 1）
	rng2 := NewPCG64(42)
	result2 := Sample(logits, 1.0, 1.0, 1, rng2)
	if result2 != 1 {
		t.Errorf("top_k=1 负 logits 采样 = %d, want 1", result2)
	}
}

func TestSample_InfLogits(t *testing.T) {
	// 包含 -Inf 的 logits（模拟被屏蔽的 token）
	negInf := float32(math.Inf(-1))
	logits := []float32{1.0, negInf, 3.0, negInf}
	// -Inf token 概率为 0，不应被采样
	for seed := int64(0); seed < 50; seed++ {
		r := NewPCG64(seed)
		result := Sample(logits, 1.0, 1.0, 10, r)
		if result == 1 || result == 3 {
			t.Errorf("seed=%d: -Inf token 不应被采样, got %d", seed, result)
		}
	}
}

// TestSample_Golden 用固定 logits + 固定 PCG64 种子验证采样结果与 Python
// runtime.py 的 _sample 完全一致。期望值由 numpy 离线生成，是算法一致性的
// 核心回归测试：任何 sampler 实现偏移都会导致此处失败。
func TestSample_Golden(t *testing.T) {
	negInf := float32(math.Inf(-1))
	cases := []struct {
		name        string
		logits      []float32
		temperature float64
		topP        float64
		topK        int
		seed        int64
		want        int
	}{
		{"distinct_42", []float32{1, 2, 3, 4, 5}, 0.7, 0.9, 50, 42, 3},
		{"distinct_123", []float32{1, 2, 3, 4, 5}, 0.7, 0.9, 50, 123, 4},
		{"distinct_0", []float32{1, 2, 3, 4, 5}, 0.7, 0.9, 50, 0, 4},
		{"negative_logits", []float32{-5, -1, -3, -2}, 1.0, 1.0, 10, 42, 1},
		{"topk2_restrict", []float32{1, 5, 2, 4, 0.5}, 1.0, 1.0, 2, 7, 1},
		{"tight_topp", []float32{10, 1, 1, 1, 1}, 0.7, 0.5, 50, 42, 0},
		{"all_equal", []float32{2, 2, 2, 2}, 1.0, 1.0, 10, 99, 3},
		{"vocab20", []float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}, 0.8, 0.95, 10, 42, 19},
		{"inf_masked", []float32{1, negInf, 3, negInf}, 1.0, 1.0, 10, 42, 2},
		{"low_temp_greedy", []float32{1, 2, 10, 3, 4}, 0.001, 1.0, 50, 42, 2},
		// 边界：两个相等 logit，top_p=0.5。cumulative[0]==0.5 不满足 >0.5，
		// 故 top-1 保留；平局时高索引优先 → idx1 被保护，idx0 被移除。
		{"topp_boundary_equal", []float32{1, 1}, 1.0, 0.5, 10, 42, 1},
		// 平局 + top_k=2：只保留两个最高索引 {2,3}，RNG 选出 2。
		{"tie_topk2", []float32{1, 1, 1, 1}, 1.0, 1.0, 2, 42, 2},
		// top_p=1.0 边界：cumulative 永不 >1.0，top_p 不移除任何 token。
		{"topp1_boundary", []float32{3, 1, 2}, 1.0, 1.0, 50, 42, 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			rng := NewPCG64(c.seed)
			got := Sample(c.logits, c.temperature, c.topP, c.topK, rng)
			if got != c.want {
				t.Errorf("Sample(logits=%v, temp=%v, topP=%v, topK=%d, seed=%d) = %d, want %d",
					c.logits, c.temperature, c.topP, c.topK, c.seed, got, c.want)
			}
		})
	}
}

// TestSample_TieBreaking_HighIndexFirst 显式验证平局时高索引优先的策略。
// 4 个相等 logit + top_k=1：order 降序排列后 order[0] 应为最高索引 3，
// 仅保留它，结果恒为 3（与种子无关）。
func TestSample_TieBreaking_HighIndexFirst(t *testing.T) {
	logits := []float32{5, 5, 5, 5}
	for seed := int64(0); seed < 100; seed++ {
		rng := NewPCG64(seed)
		got := Sample(logits, 1.0, 1.0, 1, rng)
		if got != 3 {
			t.Errorf("seed=%d: 平局+top_k=1 应返回最高索引 3, got %d", seed, got)
		}
	}
}

// TestSample_TieBreaking_TopK2KeepsHighestIndices 验证平局时 top_k=2 仅保留
// 两个最高索引。对 4 个相等 logit，结果必须 ∈ {2, 3}。
func TestSample_TieBreaking_TopK2KeepsHighestIndices(t *testing.T) {
	logits := []float32{1, 1, 1, 1}
	valid := map[int]bool{2: true, 3: true}
	for seed := int64(0); seed < 200; seed++ {
		rng := NewPCG64(seed)
		got := Sample(logits, 1.0, 1.0, 2, rng)
		if !valid[got] {
			t.Fatalf("seed=%d: 平局+top_k=2 结果 %d 不在 {2,3} 中", seed, got)
		}
	}
}

// TestSample_RNGDrawCount 验证每次 Sample 恰好消费 n 次 Float64（n=len(logits)）。
// 这对与 Python numpy 的 RNG 序列对齐至关重要：消费次数偏移会导致后续采样全部发散。
// 方法：p1 跑 Sample 后取下一值；p2 手动跳过 n 次后取下一值，两者应相等。
func TestSample_RNGDrawCount(t *testing.T) {
	logits := []float32{1, 2, 3, 4, 5} // n=5
	const seed = int64(42)

	p1 := NewPCG64(seed)
	_ = Sample(logits, 0.7, 0.9, 50, p1)
	after1 := p1.Float64()

	p2 := NewPCG64(seed)
	for i := 0; i < len(logits); i++ { // 跳过 n 次
		_ = p2.Float64()
	}
	after2 := p2.Float64()

	if after1 != after2 {
		t.Errorf("RNG 消费次数不匹配: Sample 后下一值=%.17g, 手动跳过 %d 次后=%.17g",
			after1, len(logits), after2)
	}
}
