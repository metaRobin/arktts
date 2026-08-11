"""分析 gomlx 4 个失败案例在 Python tokenizers 中的具体处理逻辑。"""
import json
from pathlib import Path
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
TK_PATH = HERE.parent.parent.parent / "model" / "tokenizer" / "tokenizer.json"

# 加载 tokenizer.json 原始配置
config = json.loads(TK_PATH.read_text(encoding="utf-8"))
merges_raw = config["model"]["merges"]
vocab = config["model"]["vocab"]

# merges 是嵌套列表 [['Ġ', 'Ġ'], ...]
merge_strs = set()
for m in merges_raw:
    if isinstance(m, list):
        merge_strs.add(" ".join(m))
    else:
        merge_strs.add(m)

print("=== tokenizer.json 配置 ===")
print(f"model.type: {config['model']['type']}")
print(f"model.ignore_merges: {config['model']['ignore_merges']}")
print(f"model.byte_fallback: {config['model']['byte_fallback']}")
print(f"model.unk_token: {config['model']['unk_token']}")
print(f"vocab size: {len(vocab)}")
print(f"merges count: {len(merges_raw)}")
print(f"pre_tokenizer:")
print(json.dumps(config['pre_tokenizer'], indent=2, ensure_ascii=False))
print()

# 加载 tokenizer
tk = Tokenizer.from_file(str(TK_PATH))

# 4 个失败案例
cases = [
    ("#1", "<|speaker:0|>This is a test.\n\nSpeech:\n"),
    ("#2", "    leading spaces and trailing    "),
    ("#3", "go run ./cmd/infer"),
    ("#4", "  "),
]

for label, text in cases:
    print("=" * 70)
    print(f"{label} text: {text!r}")
    enc = tk.encode(text, add_special_tokens=False)

    print(f"  ids:    {enc.ids}")
    print(f"  tokens: {enc.tokens}")
    print()

    # 逐 token 分析
    for i, (tid, tok) in enumerate(zip(enc.ids, enc.tokens)):
        print(f"    [{i}] id={tid:>6} token={tok!r}")

    # 针对性检查 merges
    print()
    print("  --- merge / vocab 检查 ---")
    if label == "#1":
        print(f"  '> This' in merges: {'> This' in merge_strs}")
        print(f"  '>This' in vocab: {'>This' in vocab} (id={vocab.get('>This')})")
        print(f"  '>' in vocab: id={vocab.get('>')}")
        print(f"  'This' in vocab: id={vocab.get('This')}")
    elif label == "#2":
        for target in ["Ġ Ġ", "ĠĠ Ġ", "ĠĠĠ Ġ", "ĠĠĠĠ Ġ"]:
            print(f"  '{target}' in merges: {target in merge_strs}")
        for n in range(1, 6):
            tok = "Ġ" * n
            print(f"  '{tok}' in vocab: id={vocab.get(tok)}")
    elif label == "#3":
        for target in ["Ġ /", "Ġ. /", "Ġ ./", ". /", "Ġ. Ġ/"]:
            print(f"  '{target}' in merges: {target in merge_strs}")
        print(f"  'Ġ./' in vocab: id={vocab.get('Ġ./')}")
        print(f"  'Ġ.' in vocab: id={vocab.get('Ġ.')}")
        print(f"  '/' in vocab: id={vocab.get('/')}")
    elif label == "#4":
        print(f"  'Ġ Ġ' in merges: {'Ġ Ġ' in merge_strs}")
        print(f"  'ĠĠ' in vocab: id={vocab.get('ĠĠ')}")
        print(f"  'Ġ' in vocab: id={vocab.get('Ġ')}")
    print()
