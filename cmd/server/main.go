// Command server 启动 arktts HTTP 服务，对应 Python start_server.sh / service.py。
//
// 环境变量（与 Python 版一致）：
//
//	ARKTTS_MODEL_DIR        模型目录（默认 model）
//	ARKTTS_VOICES_DIR       voice 目录（默认 voices）
//	ARKTTS_REGISTRATION_DIR 注册目录（默认 $MODEL_DIR/registration）
//	ARKTTS_THREADS          ONNX Runtime CPU 线程数（默认 5）
//	ARKTTS_ONNX_LIB         ONNX Runtime 动态库路径（默认自动检测）
//	HOST                    监听地址（默认 127.0.0.1）
//	PORT                    监听端口（默认 8024）
package main

import (
	"fmt"
	"log"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	stdruntime "runtime"

	"github.com/metaRobin/arktts/runtime"
	"github.com/metaRobin/arktts/service"
)

func main() {
	modelDir := envOr("ARKTTS_MODEL_DIR", "model")
	voicesDir := envOr("ARKTTS_VOICES_DIR", "voices")
	registrationDir := envOr("ARKTTS_REGISTRATION_DIR", filepath.Join(modelDir, "registration"))
	threads := envIntOr("ARKTTS_THREADS", 5)
	libPath := envOr("ARKTTS_ONNX_LIB", defaultOnnxLibPath())
	host := envOr("HOST", "127.0.0.1")
	port := envOr("PORT", "8024")

	slog.Info("arktts server starting",
		slog.String("model_dir", modelDir),
		slog.String("voices_dir", voicesDir),
		slog.String("registration_dir", registrationDir),
		slog.String("lib", libPath),
		slog.Int("threads", threads),
		slog.String("host", host),
		slog.String("port", port))

	rt, err := runtime.New(modelDir, voicesDir)
	if err != nil {
		log.Fatalf("init runtime: %v", err)
	}
	defer rt.Close()

	if err := rt.InitEngine(libPath, threads); err != nil {
		log.Fatalf("init engine: %v", err)
	}

	srv := service.New(rt, threads, registrationDir)
	addr := fmt.Sprintf("%s:%s", host, port)
	slog.Info("listening", slog.String("addr", addr))
	if err := http.ListenAndServe(addr, srv.Routes()); err != nil {
		log.Fatalf("server: %v", err)
	}
}

// envOr 读取环境变量，未设置时返回默认值。
func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// envIntOr 读取环境变量为 int，未设置或非法时返回默认值。
func envIntOr(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

// defaultOnnxLibPath 返回当前平台预置的 ONNX Runtime 库路径。
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
