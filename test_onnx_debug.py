"""
Debug ONNX inference: dump exact logits values so we can compare with JS.
Also test different input formats to see if int64 vs int32 matters.
"""
import json
import numpy as np

# Use onnxruntime
try:
    import onnxruntime as ort
except ImportError:
    print("Installing onnxruntime...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "onnxruntime", "-q"])
    import onnxruntime as ort

MODEL_PATH = "docs/assets/phase2_model.onnx"
TOKENIZER_PATH = "docs/assets/tokenizer_english.json"
VOCAB_SIZE = 1024

# Load tokenizer
with open(TOKENIZER_PATH) as f:
    tok_data = json.load(f)

vocab = tok_data["model"]["vocab"]
vocab_inv = {v: k for k, v in vocab.items()}
merges = tok_data["model"]["merges"]
merge_rank = {}
for i, m in enumerate(merges):
    merge_rank[f"{m[0]} {m[1]}"] = i

# GPT-2 bytes_to_unicode
def bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

b2u = bytes_to_unicode()
u2b = {v: k for k, v in b2u.items()}

def text_to_byte_level(text):
    return ''.join(b2u[b] for b in text.encode('utf-8'))

import re
def pre_tokenize(text):
    pattern = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    # Python re doesn't support \p{L}, use regex module or manual
    # Actually let's just use the same approach as our working Python test
    import regex
    return regex.findall(pattern, text)

def apply_bpe(tokens):
    if len(tokens) < 2:
        return tokens
    word = list(tokens)
    while len(word) >= 2:
        best_pair = None
        best_rank = float('inf')
        for i in range(len(word) - 1):
            key = f"{word[i]} {word[i+1]}"
            rank = merge_rank.get(key)
            if rank is not None and rank < best_rank:
                best_rank = rank
                best_pair = (word[i], word[i+1])
        if best_pair is None:
            break
        merged = best_pair[0] + best_pair[1]
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == best_pair[0] and word[i+1] == best_pair[1]:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        word = new_word
    return word

NEWLINE_PLACEHOLDER = '\u0126'

def encode(text):
    text = text.replace('\n', NEWLINE_PLACEHOLDER)
    try:
        import regex
        pattern = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        words = regex.findall(pattern, text)
    except ImportError:
        # Fallback: simple split
        words = text.split()

    ids = []
    for word in words:
        bl = text_to_byte_level(word)
        chars = list(bl)
        merged = apply_bpe(chars)
        for token in merged:
            tid = vocab.get(token)
            if tid is not None:
                ids.append(tid)
    return ids

def decode(ids):
    text = ''.join(vocab_inv.get(i, '') for i in ids)
    bs = bytearray()
    for ch in text:
        b = u2b.get(ch)
        if b is not None:
            bs.append(b)
    result = bs.decode('utf-8', errors='replace')
    return result.replace(NEWLINE_PLACEHOLDER, '\n')

# Load ONNX model
print("Loading ONNX model...")
session = ort.InferenceSession(MODEL_PATH)
print(f"Inputs: {[(i.name, i.type, i.shape) for i in session.get_inputs()]}")
print(f"Outputs: {[(o.name, o.type, o.shape) for o in session.get_outputs()]}")

# Encode prompt
prompt = 'Alice went to the garden.\nBob went to the kitchen.\nWhere is Alice?'
token_ids = encode(prompt)
print(f"\nPrompt: {prompt!r}")
print(f"Token IDs: {token_ids}")
print(f"Num tokens: {len(token_ids)}")

# Test with int64 input (same as JS BigInt64Array)
print("\n=== Test with int64 input ===")
input_ids_64 = np.array([token_ids], dtype=np.int64)
print(f"Input shape: {input_ids_64.shape}, dtype: {input_ids_64.dtype}")

results = session.run(None, {"input_ids": input_ids_64})
logits = results[0]
print(f"Output shape: {logits.shape}, dtype: {logits.dtype}")

last_pos = len(token_ids) - 1
last_logits = logits[0, last_pos, :]
print(f"Last position ({last_pos}) logit range: {last_logits.min():.4f} to {last_logits.max():.4f}")

# Top 10
top_indices = np.argsort(last_logits)[::-1][:10]
print("\nTop 10 tokens (greedy):")
for rank, idx in enumerate(top_indices):
    token = vocab_inv.get(idx, '???')
    # Decode
    bs = bytearray()
    for ch in token:
        b = u2b.get(ch)
        if b is not None:
            bs.append(b)
    try:
        decoded = bs.decode('utf-8')
    except:
        decoded = token
    print(f"  {rank}: id={idx} token={token!r} decoded={decoded!r} logit={last_logits[idx]:.4f}")

# Now dump the first 20 logits as a fingerprint to compare with JS
print(f"\nFirst 20 logits at last position: {last_logits[:20].tolist()}")

# Generate 20 tokens with temperature=0.8, top-k=40 (deterministic with seed)
print("\n=== Autoregressive generation (greedy, 20 tokens) ===")
gen_ids = list(token_ids)
prompt_len = len(gen_ids)
for i in range(20):
    inp = np.array([gen_ids], dtype=np.int64)
    res = session.run(None, {"input_ids": inp})
    lgt = res[0][0, len(gen_ids) - 1, :]
    next_token = int(np.argmax(lgt))
    gen_ids.append(next_token)
    if len(gen_ids) > 256:
        gen_ids = gen_ids[-256:]

generated = decode(gen_ids[prompt_len:])
print(f"Greedy generated: {generated!r}")

# Also test: what happens if logits output is [1, seq_len, vocab] but JS reads it flat?
print("\n=== Flat array indexing check ===")
flat_logits = logits.flatten()
print(f"Flat logits length: {len(flat_logits)}")
print(f"Expected: {logits.shape[0] * logits.shape[1] * logits.shape[2]}")
# In JS: offset = lastPos * vocabSize
# This assumes shape is [1, seq_len, vocab] laid out as batch*seq*vocab
# For batch=1: flat[lastPos * vocab + v] should equal logits[0, lastPos, v]
js_offset = last_pos * VOCAB_SIZE
for v in [0, 1, 2, 100, 500, 1023]:
    py_val = logits[0, last_pos, v]
    flat_val = flat_logits[js_offset + v]
    match = "OK" if abs(py_val - flat_val) < 1e-6 else "MISMATCH!"
    print(f"  v={v}: flat[{js_offset+v}]={flat_val:.6f} vs logits[0,{last_pos},{v}]={py_val:.6f} {match}")

print("\nDone.")
