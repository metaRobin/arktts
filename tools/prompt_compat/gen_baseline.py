"""生成 prompt 矩阵 baseline，供 Go 端对比。

用 arktts_runtime.prompt.PromptBuilder 构造 [1, num_codebooks+1, T] 矩阵，
覆盖多种 reference_text / target_text / reference_codes 组合。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

# 直接复用 arktts_runtime 的逻辑
HERE = Path(__file__).resolve().parent
ONNX_ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ONNX_ROOT))

from arktts_runtime.prompt import PromptBuilder, clean_text, format_reference_text

SEMANTIC_BEGIN_ID = 151678
NUM_CODEBOOKS = 10
TOKENIZER_DIR = ONNX_ROOT / "model" / "tokenizer"

# 测试用例：(target_text, reference_text, reference_codes_shape_desc)
TEST_CASES = [
    {
        "name": "simple_english",
        "target_text": "Hello, world!",
        "reference_text": "This is a test.",
        "codes_shape": (10, 50),
    },
    {
        "name": "chinese",
        "target_text": "你好，世界！",
        "reference_text": "这是一个语音合成测试。",
        "codes_shape": (10, 80),
    },
    {
        "name": "mixed",
        "target_text": "Audio8 TTS 测试 2024。",
        "reference_text": "Hello 你好, this is 混合文本.",
        "codes_shape": (10, 60),
    },
    {
        "name": "with_speaker_tag",
        "target_text": "使用自定义 speaker 标签。",
        "reference_text": "<|speaker:5|>自定义参考音色文本。",
        "codes_shape": (10, 40),
    },
    {
        "name": "long_text",
        "target_text": (
            "这是一段较长的目标文本，用于测试 tokenizer 在处理长文本时的正确性。"
            "包含中文、English、数字 12345、标点！@#￥%……&*（）、换行\n"
            "以及 special token 字符串如 <|im_start|> 等。"
        ),
        "reference_text": (
            "参考文本同样较长，包含各种字符。The quick brown fox jumps over the lazy dog. "
            "苹果香蕉橘子 100, 200, 300。"
        ),
        "codes_shape": (10, 120),
    },
    {
        "name": "minimal_codes",
        "target_text": "最小 codes 测试。",
        "reference_text": "参考。",
        "codes_shape": (10, 1),
    },
    {
        "name": "empty_reference_text",
        "target_text": "空参考文本测试。",
        "reference_text": "",
        "codes_shape": (10, 30),
    },
]


def main() -> int:
    builder = PromptBuilder(TOKENIZER_DIR, SEMANTIC_BEGIN_ID, NUM_CODEBOOKS)

    results = []
    for tc in TEST_CASES:
        # 生成确定性 reference_codes（用 index 作为种子，确保 Python/Go 一致）
        np.random.seed(hash(tc["name"]) & 0xFFFFFFFF)
        rows, cols = tc["codes_shape"]
        codes = np.random.randint(0, 4096, size=(rows, cols), dtype=np.int64)

        matrix = builder.build(tc["target_text"], tc["reference_text"], codes)

        # 同时记录中间数据，便于 Go 端调试
        prefix_parts = [
            "<|im_start|>system\n",
            "convert the provided text to speech reference to the following:\n\nText:\n",
            format_reference_text(tc["reference_text"]),
            "\n\nSpeech:\n",
        ]
        suffix_parts = [
            "<|im_end|>\n",
            "<|im_start|>user\n",
            clean_text(tc["target_text"]),
            "<|im_end|>\n",
            "<|im_start|>assistant\n<|voice|>",
        ]
        prefix_tokens = [t for p in prefix_parts for t in builder.encode_text(p)]
        suffix_tokens = [t for p in suffix_parts for t in builder.encode_text(p)]

        results.append({
            "name": tc["name"],
            "target_text": tc["target_text"],
            "reference_text": tc["reference_text"],
            "reference_codes": codes.tolist(),
            "matrix_shape": list(matrix.shape),
            "matrix": matrix.tolist(),
            # 中间数据
            "prefix_tokens": prefix_tokens,
            "suffix_tokens": suffix_tokens,
            "semantic_begin_id": SEMANTIC_BEGIN_ID,
            "num_codebooks": NUM_CODEBOOKS,
        })
        print(f"  {tc['name']}: matrix shape {matrix.shape}")

    output = HERE / "prompt_baseline.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {len(results)} test cases to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
