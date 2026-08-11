package runtime

import (
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
)

// findDirs 定位测试用的 model 和 voices 目录。
func findDirs() (modelDir, voicesDir string) {
	candidates := []string{
		filepath.Join(".."),       // runtime/ -> onnx_runtime/
		filepath.Join("..", ".."), // 子目录
	}
	for _, base := range candidates {
		m := filepath.Join(base, "model")
		v := filepath.Join(base, "voices")
		if _, err := os.Stat(filepath.Join(v, "speaker_a", "codes.npy")); err == nil {
			if _, err := os.Stat(filepath.Join(m, "runtime_manifest.json")); err == nil {
				absM, _ := filepath.Abs(m)
				absV, _ := filepath.Abs(v)
				return absM, absV
			}
		}
	}
	return "", ""
}

func newTestRuntime(t testing.TB) *Runtime {
	modelDir, voicesDir := findDirs()
	if modelDir == "" {
		t.Skip("model/voices 目录不存在，跳过")
	}
	rt, err := New(modelDir, voicesDir)
	if err != nil {
		t.Fatalf("创建 Runtime 失败: %v", err)
	}
	return rt
}

// --- 正确性测试 ---

// TestConcurrentBuildPrompt 并发调用 BuildPrompt，验证结果一致且无 data race。
func TestConcurrentBuildPrompt(t *testing.T) {
	rt := newTestRuntime(t)

	const goroutines = 100
	const iterations = 50

	var wg sync.WaitGroup
	var failures atomic.Int64

	// 预热缓存
	if _, err := rt.BuildPrompt("预热", "speaker_a"); err != nil {
		t.Fatalf("预热失败: %v", err)
	}

	wg.Add(goroutines)
	for g := 0; g < goroutines; g++ {
		go func(id int) {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				matrix, err := rt.BuildPrompt("并发测试文本", "speaker_a")
				if err != nil {
					failures.Add(1)
					return
				}
				// 验证矩阵形状
				if len(matrix) != 1 || len(matrix[0]) != 11 {
					failures.Add(1)
					return
				}
			}
		}(g)
	}
	wg.Wait()

	if f := failures.Load(); f > 0 {
		t.Fatalf("并发测试失败: %d/%d", f, goroutines*iterations)
	}
}

// TestConcurrentMixedReadWrite 并发读 + Invalidate + Reload，验证锁竞争下不 panic 且数据一致。
func TestConcurrentMixedReadWrite(t *testing.T) {
	rt := newTestRuntime(t)

	const readers = 80
	const writers = 20
	const iterations = 30

	var wg sync.WaitGroup
	var failures atomic.Int64

	// 预热缓存
	if _, err := rt.BuildPrompt("预热", "speaker_a"); err != nil {
		t.Fatalf("预热失败: %v", err)
	}

	// 读 goroutine
	wg.Add(readers)
	for g := 0; g < readers; g++ {
		go func() {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				matrix, err := rt.BuildPrompt("读", "speaker_a")
				if err != nil {
					failures.Add(1)
					return
				}
				if len(matrix[0]) != 11 {
					failures.Add(1)
					return
				}
			}
		}()
	}

	// 写 goroutine：交替 Invalidate 和 Reload
	wg.Add(writers)
	for g := 0; g < writers; g++ {
		go func() {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				rt.InvalidateVoice("speaker_a")
				if err := rt.ReloadVoice("speaker_a"); err != nil {
					failures.Add(1)
					return
				}
			}
		}()
	}

	wg.Wait()

	if f := failures.Load(); f > 0 {
		t.Fatalf("混合读写失败: %d", f)
	}
}

// TestConcurrentColdCache 冷缓存下高并发，验证多 goroutine 同时 miss 不会重复加载或 panic。
func TestConcurrentColdCache(t *testing.T) {
	rt := newTestRuntime(t)

	const goroutines = 50

	var wg sync.WaitGroup
	var successes atomic.Int64

	// 不预热，直接并发
	wg.Add(goroutines)
	for g := 0; g < goroutines; g++ {
		go func() {
			defer wg.Done()
			_, err := rt.BuildPrompt("冷启动", "speaker_a")
			if err == nil {
				successes.Add(1)
			}
		}()
	}
	wg.Wait()

	if s := successes.Load(); s != goroutines {
		t.Fatalf("冷缓存并发: 成功 %d/%d", s, goroutines)
	}
}

// TestConcurrentListAndLoad 并发 List + Load，验证 List 不加锁不干扰 Load。
func TestConcurrentListAndLoad(t *testing.T) {
	rt := newTestRuntime(t)

	const goroutines = 50
	const iterations = 20

	var wg sync.WaitGroup
	var failures atomic.Int64

	wg.Add(goroutines * 2)

	// 一半做 List
	for g := 0; g < goroutines; g++ {
		go func() {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				metas, err := rt.ListVoices()
				if err != nil {
					failures.Add(1)
					return
				}
				if len(metas) == 0 {
					failures.Add(1)
					return
				}
			}
		}()
	}

	// 一半做 Load
	for g := 0; g < goroutines; g++ {
		go func() {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				if _, err := rt.BuildPrompt("并发", "speaker_a"); err != nil {
					failures.Add(1)
					return
				}
			}
		}()
	}

	wg.Wait()

	if f := failures.Load(); f > 0 {
		t.Fatalf("List+Load 并发失败: %d", f)
	}
}

// --- 并发基准测试 ---

// BenchmarkBuildPrompt_Concurrent 并发 BuildPrompt，观察 RWMutex 读锁在多核下的扩展性。
func BenchmarkBuildPrompt_Concurrent(b *testing.B) {
	rt := newTestRuntime(b)

	// 预热缓存
	if _, err := rt.BuildPrompt("预热", "speaker_a"); err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			if _, err := rt.BuildPrompt("并发压测", "speaker_a"); err != nil {
				b.Fatal(err)
			}
		}
	})
}

// BenchmarkPromptLen_Concurrent 并发 PromptLen（不构造矩阵，纯缓存命中 + tokenizer）。
func BenchmarkPromptLen_Concurrent(b *testing.B) {
	rt := newTestRuntime(b)

	if _, err := rt.PromptLen("预热", "speaker_a"); err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			if _, err := rt.PromptLen("并发压测", "speaker_a"); err != nil {
				b.Fatal(err)
			}
		}
	})
}

// BenchmarkMixedReadWrite 并发读 + 偶尔 Invalidate，模拟真实场景：
// 大量 TTS 请求 + 偶尔 voice 刷新。
func BenchmarkMixedReadWrite(b *testing.B) {
	rt := newTestRuntime(b)

	if _, err := rt.BuildPrompt("预热", "speaker_a"); err != nil {
		b.Fatal(err)
	}

	var invalidate atomic.Uint64

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			// 每 1000 次读做 1 次 Invalidate
			if invalidate.Add(1)%1000 == 0 {
				rt.InvalidateVoice("speaker_a")
			}
			if _, err := rt.BuildPrompt("混合压测", "speaker_a"); err != nil {
				b.Fatal(err)
			}
		}
	})
}

// BenchmarkColdCache_Concurrent 冷缓存并发，观察首次加载的锁竞争。
func BenchmarkColdCache_Concurrent(b *testing.B) {
	rt := newTestRuntime(b)

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			// 每次迭代先失效，强制冷缓存
			rt.InvalidateVoice("speaker_a")
			if _, err := rt.BuildPrompt("冷缓存", "speaker_a"); err != nil {
				b.Fatal(err)
			}
		}
	})
}
