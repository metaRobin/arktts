package onnxruntime

import (
	"fmt"
	"log/slog"
	"os"
	"runtime"
)

func init() {
	// slog json 格式输出
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stderr, &slog.HandlerOptions{
		Level: slog.LevelDebug,
	})))
}

func SetLogLevel(level slog.Level) {
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stderr, &slog.HandlerOptions{
		Level: level,
	})))
}

func getFileLine() string {
	_, file, line, _ := runtime.Caller(1)
	return fmt.Sprintf("%s:%d", file, line)
}
