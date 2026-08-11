package inference

import (
	"math/bits"
)

// PCG64 是 numpy.random.default_rng 的纯 Go 实现，使用 PCG64 (XSL-RR 128/64) 算法。
//
// 完全复刻 numpy 的 SeedSequence + PCG64 管线：
//   1. SeedSequence 将整数 seed 混淆为 4 个 uint32 pool
//   2. generate_state 生成 8 个 uint32（4 个 uint64）初始化字
//   3. PCG64 使用 (inc<<1)|1 和两步 LCG 初始化状态
//   4. 每次输出：step → XSL-RR 输出 → float64 = (uint64>>11) / 2^53
//
// 确保 Go 端随机数序列与 Python np.random.default_rng(seed) 完全一致。
type PCG64 struct {
	stateHi uint64
	stateLo uint64
	incHi   uint64
	incLo   uint64
}

// PCG64 LCG 乘法常数 (128-bit): 0x2360ed051fc65da44385df649fccf645
const (
	pcgMultHi uint64 = 0x2360ed051fc65da4
	pcgMultLo uint64 = 0x4385df649fccf645
)

// SeedSequence 常量
const (
	ssInitA    uint32 = 0x43b0d7e5
	ssMultA    uint32 = 0x931e8875
	ssInitB    uint32 = 0x8b51f9dd
	ssMultB    uint32 = 0x58f38ded
	ssMixMultL uint32 = 0xca01f9dd
	ssMixMultR uint32 = 0x4973f715
	ssXshift   uint   = 16
)

// NewPCG64 创建一个与 numpy.random.default_rng(seed) 输出完全一致的 PCG64 随机数生成器。
func NewPCG64(seed int64) *PCG64 {
	// 1. 将 seed 转换为 uint32 数组（与 numpy _int_to_uint32_array 一致）
	entropy := intToUint32Array(uint64(seed))

	// 2. 混淆熵到 pool（pool_size=4）
	pool := make([]uint32, 4)
	mixEntropy(pool, entropy)

	// 3. generate_state(4, dtype=uint64) → 8 个 uint32 → 4 个 uint64
	state32 := generateState(pool, 8)
	words := make([]uint64, 4)
	for i := 0; i < 4; i++ {
		words[i] = uint64(state32[i*2]) | (uint64(state32[i*2+1]) << 32)
	}

	// 4. 组合为 128-bit seed_state 和 seed_inc（大端字序：words[0] 是高位）
	seedStateHi := words[0]
	seedStateLo := words[1]
	seedIncHi := words[2]
	seedIncLo := words[3]

	p := &PCG64{}

	// 5. PCG64 初始化
	// state = 0
	p.stateHi = 0
	p.stateLo = 0

	// inc = (seed_inc << 1) | 1 （保证奇数）
	p.incLo = (seedIncLo << 1) | 1
	p.incHi = (seedIncHi << 1) | (seedIncLo >> 63)

	// step: state = 0 * MULT + inc = inc
	p.step()

	// state += seed_state
	newLo, carry := bits.Add64(p.stateLo, seedStateLo, 0)
	newHi, _ := bits.Add64(p.stateHi, seedStateHi, carry)
	p.stateLo = newLo
	p.stateHi = newHi

	// step again: state = (inc + seed_state) * MULT + inc
	p.step()

	return p
}

// step 执行 PCG64 LCG 步进：state = state * MULT + inc (mod 2^128)
func (p *PCG64) step() {
	// 128-bit 乘法：state * MULT（只保留低 128 位）
	hi1, lo1 := bits.Mul64(p.stateLo, pcgMultLo)
	_, lo2 := bits.Mul64(p.stateLo, pcgMultHi)
	_, lo3 := bits.Mul64(p.stateHi, pcgMultLo)

	newLo := lo1
	// 三个 64-bit 值相加：hi1 + lo2 + lo3
	// 注意：不能将第一次加法的 carry 传入第二次，否则会重复计算
	s1, c1 := bits.Add64(hi1, lo2, 0)
	newHi, c2 := bits.Add64(s1, lo3, 0)
	_ = c1 + c2 // 超过 128 位的进位丢弃

	// 加 inc
	newLo, carry := bits.Add64(newLo, p.incLo, 0)
	newHi, _ = bits.Add64(newHi, p.incHi, carry)

	p.stateHi = newHi
	p.stateLo = newLo
}

// Uint64 返回下一个 64 位随机整数。
// PCG64 使用 "advance-then-output" 模式：先步进，再从新状态输出。
func (p *PCG64) Uint64() uint64 {
	p.step()
	// XSL-RR 输出
	xorshifted := p.stateHi ^ p.stateLo
	rot := p.stateHi >> 58 // state >> 122 的低 6 位
	return bits.RotateLeft64(xorshifted, -int(rot))
}

// Float64 返回 [0, 1) 范围的随机浮点数，与 numpy Generator.random() 一致。
// 转换方式：(uint64 >> 11) * (1.0 / 2^53)
func (p *PCG64) Float64() float64 {
	u := p.Uint64()
	return float64(u>>11) * (1.0 / 9007199254740992.0)
}

// intToUint32Array 将非负整数拆分为 uint32 数组（低位在前），与 numpy _int_to_uint32_array 一致。
func intToUint32Array(n uint64) []uint32 {
	if n == 0 {
		return []uint32{0}
	}
	var arr []uint32
	for n > 0 {
		arr = append(arr, uint32(n&0xFFFFFFFF))
		n >>= 32
	}
	return arr
}

// hashmix 是 SeedSequence 的哈希混合函数。
// 注意：hashConst 是输入输出参数（会被修改）。
func hashmix(value uint32, hashConst *uint32) uint32 {
	value ^= *hashConst
	*hashConst *= ssMultA
	value *= *hashConst
	value ^= value >> ssXshift
	return value
}

// mix 是 SeedSequence 的交叉混合函数。
func mix(x, y uint32) uint32 {
	result := ssMixMultL * x - ssMixMultR * y
	result ^= result >> ssXshift
	return result
}

// mixEntropy 将熵混合到 pool 中，完全复刻 numpy SeedSequence.mix_entropy。
func mixEntropy(pool []uint32, entropy []uint32) {
	hashConst := ssInitA

	// Phase 1: 将熵添加到 pool（每个元素 hashmix）
	for i := range pool {
		if i < len(entropy) {
			pool[i] = hashmix(entropy[i], &hashConst)
		} else {
			pool[i] = hashmix(0, &hashConst)
		}
	}

	// Phase 2: 交叉混合（每个元素影响其他所有元素）
	for iSrc := range pool {
		for iDst := range pool {
			if iSrc != iDst {
				pool[iDst] = mix(pool[iDst], hashmix(pool[iSrc], &hashConst))
			}
		}
	}

	// Phase 3: 处理剩余的熵（pool 装不下的部分）
	for iSrc := len(pool); iSrc < len(entropy); iSrc++ {
		for iDst := range pool {
			pool[iDst] = mix(pool[iDst], hashmix(entropy[iSrc], &hashConst))
		}
	}
}

// generateState 从 pool 生成 nWords 个 uint32 状态字，完全复刻 numpy SeedSequence.generate_state。
func generateState(pool []uint32, nWords int) []uint32 {
	state := make([]uint32, nWords)
	hashConst := ssInitB
	poolSize := len(pool)

	for iDst := 0; iDst < nWords; iDst++ {
		dataVal := pool[iDst%poolSize]
		dataVal ^= hashConst
		hashConst *= ssMultB
		dataVal *= hashConst
		dataVal ^= dataVal >> ssXshift
		state[iDst] = dataVal
	}

	return state
}
