package inference

import (
	"math"
	"sort"
)

// Sample 执行 top-p/top-k 采样，精确复刻 Python 端 _sample 函数的逻辑。
//
// 步骤：
//  1. 按 logits 值降序排序索引（平局时高索引优先，匹配 np.argsort[::-1]）
//  2. 对排序后的值计算 softmax
//  3. 计算累积概率
//  4. 标记移除：累积概率 > topP 或排名 >= topK（但始终保留 top-1）
//  5. 将被移除 token 的 logit 设为 -Inf
//  6. 按 temperature 缩放（带 max(temperature, 1e-5) 保护）
//  7. 对缩放后的 logits 计算 softmax
//  8. 使用 Gumbel-max 技巧采样：argmax(probs / (-log(clip(uniform, 1e-12, 1.0))))
//
// 返回采样的 token 索引。
//
// 缓冲区复用优化：原始 8 个 make（order/sortedValues/base/cumulative/removed/masked/scaled/probs）
// 合并为 3 个：order（排序索引）、work1（base→cumulative）、work2（masked→scaled→probs）。
// sortedValues 省略（直接用 logits[order[i]]），removed 省略（直接在构造 masked 时判断）。
func Sample(logits []float32, temperature, topP float64, topK int, rng *PCG64) int {
	n := len(logits)
	if n == 0 {
		return 0
	}

	// 步骤 1：按 logit 值降序排序索引
	// 平局时高索引优先，精确匹配 Python np.argsort(values)[::-1] 的行为
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(i, j int) bool {
		li := float64(logits[order[i]])
		lj := float64(logits[order[j]])
		if li != lj {
			return li > lj
		}
		return order[i] > order[j]
	})

	// work1: base → cumulative（同一缓冲区复用）
	work1 := make([]float64, n)
	// work2: masked → scaled → probs（同一缓冲区复用）
	work2 := make([]float64, n)

	// 步骤 2：base = softmax(sortedValues)
	// 省略 sortedValues 切片，直接通过 order[i] 索引 logits
	maxVal := float64(logits[order[0]]) // order[0] 是最大值（降序排序）
	var sum float64
	for i, idx := range order {
		work1[i] = math.Exp(float64(logits[idx]) - maxVal)
		sum += work1[i]
	}
	for i := range work1 {
		work1[i] /= sum
	}

	// 步骤 3：cumulative（原地覆盖 base，base 之后不再需要）
	for i := 1; i < n; i++ {
		work1[i] += work1[i-1]
	}

	// 步骤 4+5：masked = values.copy()，被移除的设为 -Inf
	// 省略 removed []bool，直接在构造 masked 时用 cumulative 判断
	for i := 0; i < n; i++ {
		work2[i] = float64(logits[i])
	}
	for i := 1; i < n; i++ { // i=0 始终不移除
		if work1[i] > topP || i >= topK {
			work2[order[i]] = math.Inf(-1)
		}
	}

	// 步骤 6：scaled = masked / temp（原地覆盖 masked）
	temp := math.Max(temperature, 1e-5)
	for i := range work2 {
		work2[i] /= temp
	}

	// 步骤 7：probs = softmax(scaled)（原地覆盖 scaled）
	maxScaled := math.Inf(-1)
	for _, v := range work2 {
		if v > maxScaled {
			maxScaled = v
		}
	}
	var probSum float64
	for i := range work2 {
		work2[i] = math.Exp(work2[i] - maxScaled)
		probSum += work2[i]
	}
	for i := range work2 {
		work2[i] /= probSum
	}

	// 步骤 8：Gumbel-max 采样
	// noise = -log(clip(rng.random(), 1e-12, 1.0))
	// result = argmax(probs / noise)
	// 注意：被移除 token 的 probs 为 0，score = 0/noise = 0，不会被选中
	bestIdx := 0
	bestScore := math.Inf(-1)
	for i := 0; i < n; i++ {
		u := rng.Float64()
		if u < 1e-12 {
			u = 1e-12
		}
		// rng.Float64() 返回 [0, 1)，不会超过 1.0，无需上裁剪
		noise := -math.Log(u)
		score := work2[i] / noise
		if score > bestScore {
			bestScore = score
			bestIdx = i
		}
	}

	return bestIdx
}
