//工具：打印 ONNX 模型的输入/输出信息（名称、形状、数据类型）。
package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/metaRobin/arktts/onnxruntime"
)

func main() {
	if len(os.Args) < 3 {
		fmt.Println("用法: model_info <library_path> <model_path> [model_path...]")
		os.Exit(1)
	}
	libPath := os.Args[1]
	modelPaths := os.Args[2:]

	rt := onnxruntime.NewRuntime()
	if err := rt.Initialize(onnxruntime.RuntimeConfig{
		LibraryPath:    libPath,
		IntraOpThreads: 1,
	}); err != nil {
		fmt.Fprintf(os.Stderr, "init failed: %v\n", err)
		os.Exit(1)
	}
	defer rt.Cleanup()

	for _, mp := range modelPaths {
		abs, _ := filepath.Abs(mp)
		fmt.Printf("\n=== %s ===\n", abs)
		sess, err := rt.CreateSession(mp)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  create session failed: %v\n", err)
			continue
		}
		si := sess.(*onnxruntime.SessionImpl)
		inputs, outputs, err := si.GetModelInputOutputInfo()
		if err != nil {
			fmt.Fprintf(os.Stderr, "  get info failed: %v\n", err)
			continue
		}
		fmt.Printf("  Inputs (%d):\n", len(inputs))
		for _, in := range inputs {
			fmt.Printf("    %s  shape=%v  dtype=%v\n", in.Name, in.Dimensions, in.DataType)
		}
		fmt.Printf("  Outputs (%d):\n", len(outputs))
		for _, out := range outputs {
			fmt.Printf("    %s  shape=%v  dtype=%v\n", out.Name, out.Dimensions, out.DataType)
		}
		sess.Destroy()
	}
}
