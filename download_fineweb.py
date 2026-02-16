"""Download a Chinchilla-optimal slice of FineWeb-Edu sample-10BT via streaming."""

import json
import os
from datasets import load_dataset

TARGET_TOKENS_APPROX = 15_200_000  # 758K params * 20 (Chinchilla optimal)
# FineWeb-Edu avg ~0.5 BPE tokens per char with vocab=1024,
# but with standard tokenizers it's ~0.25-0.3 tokens/char.
# Be conservative: assume ~4 chars/token, so we need ~60M chars.
TARGET_CHARS = TARGET_TOKENS_APPROX * 4  # ~60M chars

OUTPUT_DIR = "/Users/billy/repo/lojban_experiment/data/fineweb_edu"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "train.txt")
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Target: ~{TARGET_TOKENS_APPROX:,} tokens (~{TARGET_CHARS:,} chars)")
print("Streaming FineWeb-Edu sample-10BT...")

ds = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    streaming=True,
)

total_chars = 0
total_docs = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for example in ds:
        text = example["text"]
        if not text or len(text.strip()) < 100:
            continue  # skip very short docs

        f.write(text)
        f.write("\n\n")  # document separator

        total_chars += len(text)
        total_docs += 1

        if total_docs % 1000 == 0:
            print(f"  {total_docs:,} docs, {total_chars:,} chars ({total_chars/TARGET_CHARS*100:.1f}%)")

        if total_chars >= TARGET_CHARS:
            break

print(f"\nDone: {total_docs:,} docs, {total_chars:,} chars")
print(f"Saved to {OUTPUT_FILE}")

# Save metadata
metadata = {
    "source": "HuggingFaceFW/fineweb-edu (sample-10BT)",
    "total_docs": total_docs,
    "total_chars": total_chars,
    "target_tokens_approx": TARGET_TOKENS_APPROX,
    "chars_per_token_assumption": 4,
}
with open(METADATA_FILE, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Metadata saved to {METADATA_FILE}")

# File size
size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
print(f"File size: {size_mb:.1f} MB")
