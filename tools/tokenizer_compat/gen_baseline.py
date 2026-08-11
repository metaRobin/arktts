"""生成 tokenizer 编码基准数据。

用 HuggingFace `tokenizers` 库（与 arktts_runtime/prompt.py 同一来源）加载
tokenizer.json，对 100 条覆盖多场景的样本用 add_special_tokens=False 编码，
结果写入 baseline.json，供 Go 端做严格对比。

用法:
    cd onnx_runtime/arktts_go/tools/tokenizer_compat
    python3 gen_baseline.py
    # 或指定 tokenizer 路径
    python3 gen_baseline.py --tokenizer /path/to/tokenizer.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

# 100 条覆盖多场景的样本
# 注：prompt.py 中 add_special_tokens=False，所以即使样本里包含
# <|im_start|> 等字符串，tokenizer 也应该按 added_tokens 表精确匹配。
SAMPLES: list[str] = [
    # ---- 简单英文 (1-10) ----
    "Hello",
    "Hello, world!",
    "Welcome to Audio8 TTS ONNX Runtime.",
    "The quick brown fox jumps over the lazy dog.",
    "a",
    "A",
    "ab",
    "yes",
    "no",
    "ok",
    # ---- 简单中文 (11-20) ----
    "你好",
    "你好，世界！",
    "你好，这是 Audio8 TTS 的本地语音合成测试。",
    "今天天气不错。",
    "中",
    "国",
    "中文测试",
    "语音合成",
    "苹果香蕉橘子",
    "上海北京广州",
    # ---- 中英混合 (21-30) ----
    "Hello 你好",
    "Audio8 TTS 测试",
    "GPT-4 是一个模型。",
    "用 ONNX Runtime 推理",
    "Python 对比 Go",
    "TTS=Text to Speech",
    "Audio8 AI",
    "model 目录",
    "voice=speaker_a",
    "CPU 执行器",
    # ---- 数字 (31-40) ----
    "0 1 2 3 4 5 6 7 8 9",
    "100, 200, 300",
    "3.14159265",
    "1e-5",
    "2024-01-15",
    "1234567890",
    "0.5 到 30 秒",
    "50 MiB",
    "44100 Hz",
    "2048 帧",
    # ---- 标点 (41-50) ----
    "...",
    "???!!!",
    "「中」『日』",
    "“引号”",
    "tab\there",
    "newline\nhere",
    "comma,comma,comma",
    "dot.dot.dot",
    "slash/a/b",
    "back\\slash",
    # ---- prompt.py 实际片段 (51-60) ----
    "<|im_start|>system\n",
    "convert the provided text to speech reference to the following:\n\nText:\n",
    "<|speaker:0|>This is a test.\n\nSpeech:\n",
    "<|im_end|>\n",
    "<|im_start|>user\n",
    "<|im_start|>assistant\n<|voice|>",
    "<|speaker:0|>",
    "<|speaker:5|>reference text here",
    "<|voice|>",
    "system\nconvert the provided text to speech reference to the following:\n\nText:\n<|speaker:0|>reference\n\nSpeech:\n<|im_end|>\n<|im_start|>user\ntarget<|im_end|>\n<|im_start|>assistant\n<|voice|>",
    # ---- 长文本 (61-70) ----
    ("Audio8 TTS 是一个基于 ONNX Runtime 的 CPU 推理引擎，"
     "支持 CLI 推理、Web 与 HTTP 服务、流式 PCM 输出和参考音色注册。"
     "运行时不依赖 PyTorch、Transformers 或 Hugging Face Hub。"),
    ("Lorem ipsum dolor sit amet, consectetur adipiscing elit, "
     "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
     "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris."),
    "苹果香蕉橘子葡萄西瓜草莓蓝莓树莓黑莓柠檬橙子柚子",
    ("The reference transcript must match the spoken content. "
     "Noisy, long, or incorrectly transcribed references can reduce "
     "stability and speaker similarity."),
    "1" * 100,
    "字" * 50,
    "A" * 100,
    ("混合 mixed 文本 text with 各种 various tokens 标点!@#$%^&*() "
     "数字 12345 中文标点，。！？"),
    "    leading spaces and trailing    ",
    "\n\n\nleading newlines",
    # ---- 特殊字符 (71-80) ----
    "α β γ δ ε",
    "😊🎉👍❤️",
    "café",
    "naïve",
    "Résumé",
    "Üeber",
    "日本語テスト",
    "한국어 시험",
    "Русский текст",
    "Ελληνικά",
    # ---- 代码片段 (81-90) ----
    "func main() { fmt.Println(\"hi\") }",
    "import \"fmt\"",
    "x := []int{1, 2, 3}",
    "// comment line",
    "if err != nil { return err }",
    "type ArkTtsRuntime struct {}",
    "fmt.Sprintf(\"%d\", n)",
    "go run ./cmd/infer",
    "bash run_infer.sh --voice speaker_a",
    "docker run -it --rm alpine",
    # ---- 边界与空字符 (91-100) ----
    "",
    " ",
    "  ",
    "\t",
    "\n",
    " \t\n ",
    "a b",
    " a ",
    "非常长的一段没有任何标点符号的中文文本用于测试 tokenizer 在没有自然分隔符的情况下如何切分中文字符这段话会持续很长很长很长很长很长",
    "TheEnd",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    default_tokenizer = here.parent.parent.parent / "model" / "tokenizer" / "tokenizer.json"
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=default_tokenizer,
        help=f"path to tokenizer.json (default: {default_tokenizer})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "baseline.json",
        help="output baseline json path (default: baseline.json next to this script)",
    )
    args = parser.parse_args()

    tokenizer_path: Path = args.tokenizer
    if not tokenizer_path.is_file():
        print(f"error: tokenizer.json not found at {tokenizer_path}", file=sys.stderr)
        return 2

    print(f"loading tokenizer: {tokenizer_path}")
    tk = Tokenizer.from_file(str(tokenizer_path))
    print(f"tokenizer vocab size: {tk.get_vocab_size()}")

    results: list[dict] = []
    failures: list[dict] = []
    for idx, text in enumerate(SAMPLES):
        try:
            enc = tk.encode(text, add_special_tokens=False)
            results.append({
                "index": idx,
                "text": text,
                "ids": list(enc.ids),
                "tokens": list(enc.tokens),
                "attention_mask": list(enc.attention_mask),
            })
        except Exception as exc:  # noqa: BLE001
            failures.append({"index": idx, "text": text, "error": str(exc)})

    payload = {
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_sha256": _sha256(tokenizer_path),
        "sample_count": len(SAMPLES),
        "success_count": len(results),
        "failure_count": len(failures),
        "samples": results,
        "failures": failures,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote baseline: {args.output}")
    print(f"  samples: {len(results)} ok, {len(failures)} failed")
    if failures:
        return 1
    return 0


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    sys.exit(main())
