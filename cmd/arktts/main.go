// arktts 是 Audio8 TTS ONNX Runtime 的 Go 实现。
//
// 功能：
//   - prompt 构造（tokenizer + PromptBuilder + VoiceStore）
//   - ONNX 推理（slow AR + fast AR + codec decoder）
//   - WAV 音频输出
//
// 用法:
//
//	cd arktts_go
//	go run ./cmd/arktts --list                                   # 列出已注册 voice
//	go run ./cmd/arktts --text "你好" --voice speaker_a           # 构造 prompt 矩阵
//	go run ./cmd/arktts --synth --text "你好" --voice speaker_a   # 完整 TTS 推理，输出 WAV
package main

import (
	"flag"
	"log"
	"os"
	"path/filepath"
	stdruntime "runtime"

	"github.com/metaRobin/arktts/audio"
	"github.com/metaRobin/arktts/inference"
	"github.com/metaRobin/arktts/runtime"
)

func main() {
	log.SetFlags(0)

	exeDir := exeDir()
	defaultModelDir := envOr("ARKTTS_MODEL_DIR", filepath.Join(exeDir, "model"))
	defaultVoicesDir := envOr("ARKTTS_VOICES_DIR", filepath.Join(exeDir, "reference_voices"))
	defaultLibPath := envOr("ARKTTS_ONNX_LIB", filepath.Join(exeDir, defaultOnnxLibPath()))
	defaultOutput := "output.wav"

	modelDir := flag.String("model-dir", defaultModelDir, "模型目录")
	voicesDir := flag.String("voices-dir", defaultVoicesDir, "voice 目录")
	libPath := flag.String("lib", defaultLibPath, "ONNX Runtime 动态库路径")
	text := flag.String("text", "", "要合成的文本")
	voice := flag.String("voice", "", "voice 名称")
	list := flag.Bool("list", false, "列出已注册 voice")
	synth := flag.Bool("synth", false, "执行完整 TTS 推理（生成 WAV）")
	output := flag.String("output", defaultOutput, "输出 WAV 文件路径")
	maxNewTokens := flag.Int("max-new-tokens", 1024, "最大生成 token 数")
	temperature := flag.Float64("temperature", 0.7, "采样温度")
	topP := flag.Float64("top-p", 0.9, "top-p 采样")
	topK := flag.Int("top-k", 50, "top-k 采样")
	seed := flag.Int64("seed", 42, "随机种子")
	threads := flag.Int("threads", 5, "ONNX Runtime 线程数")
	flag.Parse()

	rt, err := runtime.New(*modelDir, *voicesDir)
	if err != nil {
		log.Fatalf("初始化失败: %v", err)
	}
	defer rt.Close()

	if *list {
		metas, err := rt.ListVoices()
		if err != nil {
			log.Fatalf("列出 voice 失败: %v", err)
		}
		if len(metas) == 0 {
			log.Println("无已注册 voice")
			return
		}
		log.Printf("已注册 voice (%d):\n", len(metas))
		for _, m := range metas {
			log.Printf("  %s  ref=%q  shape=%v", m.Name, m.ReferenceText, m.Shape)
		}
		return
	}

	if *text == "" || *voice == "" {
		log.Println("用法: arktts --text <文本> --voice <voice名称> [--synth]")
		log.Println("      arktts --list")
		log.Println()
		log.Println("参数:")
		flag.PrintDefaults()
		os.Exit(1)
	}

	m := rt.Manifest()
	log.Printf("模型: %s  vocab=%d  codebooks=%d  max_seq_len=%d",
		m.ModelFamily, rt.Tokenizer().VocabSize(), m.NumCodebooks, m.MaxSeqLen)
	log.Printf("semantic_begin_id=%d  sample_rate=%d", m.SemanticBeginID, m.SampleRate)

	promptLen, err := rt.PromptLen(*text, *voice)
	if err != nil {
		log.Fatalf("加载 voice 失败: %v", err)
	}
	log.Printf("voice: %s  prompt 长度: %d (max %d)", *voice, promptLen, m.MaxSeqLen)
	if promptLen >= m.MaxSeqLen {
		log.Fatalf("prompt 长度 %d 超过最大序列长度 %d", promptLen, m.MaxSeqLen)
	}

	if !*synth {
		// 仅构造 prompt 矩阵
		matrix, err := rt.BuildPrompt(*text, *voice)
		if err != nil {
			log.Fatalf("构造 prompt 失败: %v", err)
		}
		rows, cols := len(matrix[0]), len(matrix[0][0])
		log.Printf("prompt 矩阵: [1, %d, %d]", rows, cols)
		preview := min(cols, 20)
		log.Printf("row 0 前 %d tokens: %v", preview, matrix[0][0][:preview])
		log.Printf("row 0 后 10 tokens: %v", matrix[0][0][cols-10:])
		log.Println()
		log.Println("✅ prompt 构造成功（使用 --synth 执行完整推理）")
		return
	}

	// 完整 TTS 推理
	log.Println("初始化 ONNX 推理引擎...")
	if err := rt.InitEngine(*libPath, *threads); err != nil {
		log.Fatalf("初始化推理引擎失败: %v", err)
	}

	opts := inference.GenerateOptions{
		MaxNewTokens: *maxNewTokens,
		Temperature:  *temperature,
		TopP:         *topP,
		TopK:         *topK,
		Seed:         *seed,
	}

	log.Printf("开始推理: text=%q  voice=%s  max_new_tokens=%d  temp=%.1f  top_p=%.1f  top_k=%d  seed=%d",
		*text, *voice, opts.MaxNewTokens, opts.Temperature, opts.TopP, opts.TopK, opts.Seed)

	audioSamples, codes, err := rt.Synthesize(*text, *voice, opts)
	if err != nil {
		log.Fatalf("推理失败: %v", err)
	}

	log.Printf("推理完成: %d 音频样本, %d codec frames", len(audioSamples), len(codes[0]))

	// 写 WAV 文件
	if err := audio.WriteWAVFile(*output, audioSamples, m.SampleRate); err != nil {
		log.Fatalf("写 WAV 失败: %v", err)
	}
	log.Printf("✅ 已保存 %s (%.2f 秒, %d Hz)", *output, float64(len(audioSamples))/float64(m.SampleRate), m.SampleRate)

}


// exeDir 返回可执行文件所在目录，用于解析默认的 model/voices/lib 相对路径，
// 使 arktts-cli 可从任意工作目录运行（只要 model/voices/onnxruntime 与二进制同目录）。
func exeDir() string {
	exe, err := os.Executable()
	if err != nil {
		return "."
	}
	return filepath.Dir(exe)
}

// envOr 读取环境变量，未设置时返回默认值。
func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// defaultOnnxLibPath 返回当前平台的 ONNX Runtime 库默认路径。
func defaultOnnxLibPath() string {
	switch stdruntime.GOOS {
	case "darwin":
		if stdruntime.GOARCH == "arm64" {
			return filepath.Join("onnxruntime-osx-arm64-1.28.0", "lib", "libonnxruntime.dylib")
		}
	case "linux":
		if stdruntime.GOARCH == "amd64" {
			return filepath.Join("onnxruntime-linux-amd64-1.28.0", "lib", "libonnxruntime.so")
		}
		if stdruntime.GOARCH == "arm64" {
			return filepath.Join("onnxruntime-linux-aarch64-1.28.0", "lib", "libonnxruntime.so")
		}
	}
	return ""
}
