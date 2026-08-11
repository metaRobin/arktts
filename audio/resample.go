package audio

import "math"

// ResamplePoly 对音频样本进行多相 FIR 重采样，对齐 scipy.signal.resample_poly。
//
// 将输入从原始采样率重采样到目标采样率。
// 内部流程：设计 Kaiser 窗 FIR 低通滤波器 → polyphase upfirdn → 裁剪到期望长度。
//
// up = targetRate, down = srcRate，内部先用 gcd 化简。
// 滤波器参数与 scipy 默认一致：half_len = 10*max(up,down), Kaiser beta=5.0。
func ResamplePoly(data []float32, srcRate, dstRate int) []float32 {
	if srcRate == dstRate || len(data) == 0 {
		out := make([]float32, len(data))
		copy(out, data)
		return out
	}

	up, down := srcRate, dstRate
	// gcd 化简
	g := gcd(up, down)
	up /= g
	down /= g

	// 设计 FIR 低通滤波器，与 scipy._design_resample_poly 一致
	taps := designResampleFilter(up, down)

	// polyphase upfirdn
	return polyphaseUpfirdn(data, taps, up, down)
}

// designResampleFilter 设计重采样 FIR 低通滤波器。
// 与 scipy.signal._design_resample_poly 一致：
//
//	f_c = 1/max(up, down)  （归一化到上采样后的 Nyquist）
//	half_len = 10 * max(up, down)
//	窗 = Kaiser(beta=5.0)
//	h[n] = sinc(2*f_c*(n - N/2)) * w[n]
func designResampleFilter(up, down int) []float32 {
	maxRate := up
	if down > maxRate {
		maxRate = down
	}
	fc := 1.0 / float64(maxRate) // 归一化截止频率
	halfLen := 10 * maxRate
	N := 2*halfLen + 1

	beta := 5.0
	i0Beta := besselI0(beta)

	taps := make([]float32, N)
	mid := float64(N-1) / 2.0

	for n := 0; n < N; n++ {
		t := float64(n) - mid
		// 理想低通：2*fc*sinc(2*fc*t)
		x := 2 * fc * t
		var sincVal float64
		if x == 0 {
			sincVal = 1.0
		} else {
			sincVal = math.Sin(math.Pi*x) / (math.Pi * x)
		}
		// Kaiser 窗
		val := t / mid
		if val < -1 {
			val = -1
		} else if val > 1 {
			val = 1
		}
		w := besselI0(beta*math.Sqrt(1-val*val)) / i0Beta

		taps[n] = float32(2 * fc * sincVal * w)
	}

	return taps
}

// polyphaseUpfirdn 执行 polyphase upfirdn：
// 1. 上采样（插入 up-1 个零）
// 2. 与 taps 卷积
// 3. 下采样（每 down 取一）
// 4. 裁剪到 ceil(len(data)*up/down)
func polyphaseUpfirdn(data []float32, taps []float32, up, down int) []float32 {
	nIn := len(data)
	nOut := (nIn*up + down - 1) / down // ceil(nIn*up/down)
	out := make([]float32, nOut)

	tapLen := len(taps)
	// upfirdn 输出索引：y[m] = sum(taps[down*k + phase] * x[...])
	// 但直接实现更清晰：对每个输出样本，计算对应输入窗口
	for m := 0; m < nOut; m++ {
		// 输出 m 对应上采样后的索引 m*down
		// 卷积输出 = sum(taps[j] * x_up[m*down - j])
		// x_up[i] = x[i/up] if i%up==0 else 0
		var sum float64
		// j 从 0 到 tapLen-1，但只有 (m*down - j) % up == 0 的项非零
		// 即 j ≡ m*down (mod up)
		startJ := (m * down) % up
		if startJ >= tapLen {
			// 跳过到下一个对齐的 j
			startJ = startJ - ((startJ / up) * up)
		}
		for j := startJ; j < tapLen; j += up {
			idxUp := m*down - j // 上采样域索引
			if idxUp < 0 {
				continue
			}
			if idxUp%up != 0 {
				continue
			}
			xIdx := idxUp / up
			if xIdx >= nIn {
				continue
			}
			sum += float64(taps[j]) * float64(data[xIdx])
		}
		out[m] = float32(sum)
	}

	return out
}

// gcd 返回 a 和 b 的最大公约数。
func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

// besselI0 计算第一类零阶修正贝塞尔函数 I0(x)。
// 使用级数展开，收敛快（50 项内达到 1e-12 精度）。
func besselI0(x float64) float64 {
	sum := 1.0
	term := 1.0
	x2 := x * x / 4.0
	for k := 1; k < 50; k++ {
		term *= x2 / float64(k*k)
		sum += term
		if term < 1e-12*sum {
			break
		}
	}
	return sum
}
