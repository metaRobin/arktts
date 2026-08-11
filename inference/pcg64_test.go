package inference

import (
	"fmt"
	"testing"
)

func TestPCG64_SeedSequence(t *testing.T) {
	// 验证 SeedSequence(42) 的 pool 值
	entropy := intToUint32Array(42)
	pool := make([]uint32, 4)
	mixEntropy(pool, entropy)

	// Python 已知值: [1662858758, 128880814, 1875164712, 753753205]
	expected := []uint32{1662858758, 128880814, 1875164712, 753753205}
	for i, v := range expected {
		if pool[i] != v {
			t.Errorf("pool[%d] = %d, want %d", i, pool[i], v)
		}
	}
}

func TestPCG64_GenerateState(t *testing.T) {
	// 验证 generate_state 输出
	entropy := intToUint32Array(42)
	pool := make([]uint32, 4)
	mixEntropy(pool, entropy)
	state32 := generateState(pool, 8)

	// Python 已知 uint32 值
	expected32 := []uint32{3444837047, 2669555309, 2046530742, 3581440988,
		1691623607, 2099784219, 1184028159, 862288241}
	for i, v := range expected32 {
		if state32[i] != v {
			t.Errorf("state32[%d] = %d, want %d", i, state32[i], v)
		}
	}

	// 验证 uint64 组合
	words := make([]uint64, 4)
	for i := 0; i < 4; i++ {
		words[i] = uint64(state32[i*2]) | (uint64(state32[i*2+1]) << 32)
	}
	expected64 := []uint64{11465652750463011511, 15382171918060459190,
		9018504550953525431, 3703499796004394495}
	for i, v := range expected64 {
		if words[i] != v {
			t.Errorf("words[%d] = %d, want %d", i, words[i], v)
		}
	}
}

func TestPCG64_StateAndInc(t *testing.T) {
	// 验证 PCG64 初始化后的 state 和 inc
	p := NewPCG64(42)

	// Python 已知值:
	// state = 274674114334540486603088602300644985544
	// inc = 332724090758049132448979897138935081983
	expectedStateHi := uint64(0xcea44f6798798f2a)
	expectedStateLo := uint64(0xacbc7c9d68860ac8)
	expectedIncHi := uint64(0xfa505436c9a8416e)
	expectedIncLo := uint64(0x66caf2e28d25abff)

	if p.stateHi != expectedStateHi {
		t.Errorf("stateHi = 0x%016x, want 0x%016x", p.stateHi, expectedStateHi)
	}
	if p.stateLo != expectedStateLo {
		t.Errorf("stateLo = 0x%016x, want 0x%016x", p.stateLo, expectedStateLo)
	}
	if p.incHi != expectedIncHi {
		t.Errorf("incHi = 0x%016x, want 0x%016x", p.incHi, expectedIncHi)
	}
	if p.incLo != expectedIncLo {
		t.Errorf("incLo = 0x%016x, want 0x%016x", p.incLo, expectedIncLo)
	}
}

func TestPCG64_Uint64(t *testing.T) {
	// 验证前 5 个 uint64 输出
	p := NewPCG64(42)

	// Python 已知值
	expected := []uint64{
		14276969152011380360, // 0xc621fbcd16d92688
		8095878257575067585,  // 0x705a5661a791ffc1
		15838336090824644132, // 0xdbcd12c26eda1624
		12864169557245331597, // 0xb286b60e1600888d
		1737265434024182251,  // 0x181c01b5339381eb
	}

	for i, want := range expected {
		got := p.Uint64()
		if got != want {
			t.Errorf("uint64[%d] = %d (0x%016x), want %d (0x%016x)",
				i, got, got, want, want)
		}
	}
}

func TestPCG64_Float64(t *testing.T) {
	// 验证前 5 个 float64 输出
	p := NewPCG64(42)

	// Python 已知值: np.random.default_rng(42).random(5)
	expected := []float64{
		0.7739560485559633,
		0.4388784397520523,
		0.8585979199113825,
		0.6973680290593639,
		0.09417734788764953,
	}

	for i, want := range expected {
		got := p.Float64()
		if got != want {
			t.Errorf("float64[%d] = %.17g, want %.17g", i, got, want)
		}
	}
}

// TestPCG64_Float64_MultiSeed 验证多个种子的前 5 个 float64 输出与 numpy 一致。
// 覆盖 seed=0/1/100/999999/12345，确保 SeedSequence+PCG64 在不同熵输入下都正确。
func TestPCG64_Float64_MultiSeed(t *testing.T) {
	cases := []struct {
		seed int64
		want []float64
	}{
		{0, []float64{0.63696168732145431, 0.26978671376387031, 0.040973523936194689,
			0.016527635528529094, 0.81327023920027242}},
		{1, []float64{0.51182162470025672, 0.9504636963259353, 0.14415961271963373,
			0.94864944713724386, 0.31183145201048545}},
		{100, []float64{0.83498163050200891, 0.59655402696788729, 0.28886324169120359,
			0.042951570694211405, 0.97365439510621421}},
		{999999, []float64{0.62257801174698335, 0.50321986129929841, 0.61278110245884587,
			0.97442236943003546, 0.54176893874297283}},
		{12345, []float64{0.22733602246716966, 0.31675833970975287, 0.79736545733273412,
			0.67625467075097456, 0.391109550601909}},
	}
	for _, c := range cases {
		t.Run(fmt.Sprintf("seed_%d", c.seed), func(t *testing.T) {
			p := NewPCG64(c.seed)
			for i, want := range c.want {
				got := p.Float64()
				if got != want {
					t.Errorf("seed=%d float64[%d] = %.17g, want %.17g", c.seed, i, got, want)
				}
			}
		})
	}
}

// TestPCG64_IntToUint32Array 验证整数到 uint32 小端字数组的拆分。
// 与 numpy _int_to_uint32_array 对齐（非负整数，低位在前）。
func TestPCG64_IntToUint32Array(t *testing.T) {
	cases := []struct {
		name string
		n    uint64
		want []uint32
	}{
		{"零", 0, []uint32{0}},
		{"小整数", 42, []uint32{42}},
		{"uint32最大值", 0xFFFFFFFF, []uint32{0xFFFFFFFF}},
		{"跨32位边界", 0x100000000, []uint32{0, 1}},
		{"64位值", 0x123456789ABCDEF0, []uint32{0x9ABCDEF0, 0x12345678}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := intToUint32Array(c.n)
			if len(got) != len(c.want) {
				t.Fatalf("len = %d, want %d (got %v)", len(got), len(c.want), got)
			}
			for i := range c.want {
				if got[i] != c.want[i] {
					t.Errorf("[%d] = 0x%08X, want 0x%08X", i, got[i], c.want[i])
				}
			}
		})
	}
}

// TestPCG64_InstanceIndependence 验证两个独立实例互不干扰。
// 同种子创建两个生成器，交替调用应产生相同序列。
func TestPCG64_InstanceIndependence(t *testing.T) {
	p1 := NewPCG64(42)
	p2 := NewPCG64(42)

	// 交替调用，验证状态独立
	for i := 0; i < 10; i++ {
		v1 := p1.Float64()
		v2 := p2.Float64()
		if v1 != v2 {
			t.Fatalf("第 %d 次调用不一致: p1=%.17g p2=%.17g", i, v1, v2)
		}
	}

	// p1 多调用几次后，p2 不受影响
	_ = p1.Float64()
	_ = p1.Float64()
	v2Next := p2.Float64() // p2 的第 11 次输出

	// 新建 p3，调用 10 次对齐 p2 的前 10 次，第 11 次应等于 v2Next
	p3 := NewPCG64(42)
	for i := 0; i < 10; i++ {
		_ = p3.Float64()
	}
	if p3.Float64() != v2Next {
		t.Errorf("实例状态被共享：p3 第11次 != p2 第11次")
	}
}

// TestPCG64_NegativeSeedNoPanic 验证负数种子不 panic 且输出稳定。
// 注意：numpy default_rng 拒绝负数种子，这是 Go 端的扩展行为（按 uint64 重解释），
// 因此无 Python 对照，仅验证不崩溃且可复现。
func TestPCG64_NegativeSeedNoPanic(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("负数种子 panic: %v", r)
		}
	}()

	p1 := NewPCG64(-1)
	p2 := NewPCG64(-1)
	for i := 0; i < 5; i++ {
		v1 := p1.Float64()
		v2 := p2.Float64()
		if v1 != v2 {
			t.Errorf("负种子不可复现：第 %d 次 %.17g vs %.17g", i, v1, v2)
		}
		if v1 < 0 || v1 >= 1.0 {
			t.Errorf("负种子输出越界 [0,1): %.17g", v1)
		}
	}
}

// TestPCG64_Float64Range 验证大量输出的范围始终在 [0, 1)。
func TestPCG64_Float64Range(t *testing.T) {
	p := NewPCG64(2024)
	for i := 0; i < 10000; i++ {
		v := p.Float64()
		if v < 0 || v >= 1.0 {
			t.Fatalf("第 %d 次输出越界 [0,1): %.17g", i, v)
		}
	}
}
