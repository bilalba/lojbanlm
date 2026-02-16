#!/usr/bin/env python3
"""
Lojban vs English Small Model Coherence Experiment
===================================================
Tests whether Lojban's regular grammar helps small language models
produce more coherent output than English at equal parameter count.

Character-level GPT (~10.8M params) trained on Alice in Wonderland
in both languages, evaluated via grammar checkers + structural metrics.
"""

import json
import math
import os
import random
import subprocess
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Configuration ───────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 42
CTX_LEN = 256
D_MODEL = 384
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.1

BATCH_SIZE = 64
MAX_STEPS = 5000
LR = 3e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
WARMUP_STEPS = 100
LOG_INTERVAL = 250

NUM_SAMPLES = 100
PROMPT_LEN = 32
GEN_LEN = 256
TEMPERATURE = 0.8
TOP_K = 40

torch.manual_seed(SEED)
random.seed(SEED)
torch.set_num_threads(8)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")


# ─── Data Preparation ────────────────────────────────────────────────────────

def load_english(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    start = raw.index("*** START OF")
    start = raw.index("\n", start) + 1
    end = raw.index("*** END OF")
    text = raw[start:end].strip()
    return text


def load_lojban(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    text = text.replace("\u0096", "-")
    text = text.replace("\u0097", "--")
    return text.strip()


class CharTokenizer:
    def __init__(self, text: str, name: str = ""):
        self.name = name
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for i, c in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        return [self.char_to_idx[c] for c in text]

    def decode(self, indices: list[int]) -> str:
        return "".join(self.idx_to_char[i] for i in indices)


def prepare_data(text: str, tokenizer: CharTokenizer):
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split = int(len(data) * 0.9)
    return data[:split], data[split:]


def get_batch(data: torch.Tensor, batch_size: int, ctx_len: int):
    ix = torch.randint(len(data) - ctx_len - 1, (batch_size,))
    x = torch.stack([data[i:i + ctx_len] for i in ix])
    y = torch.stack([data[i + 1:i + ctx_len + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


# ─── Model ────────────────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj_drop(self.proj(out))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, ctx_len: int, d_model: int,
                 n_head: int, n_layer: int, dropout: float):
        super().__init__()
        self.ctx_len = ctx_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(ctx_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_head, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # Tie embeddings
        self.head.weight = self.tok_emb.weight

        # Causal mask
        mask = torch.tril(torch.ones(ctx_len, ctx_len)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("mask", mask)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok + pos)
        for block in self.blocks:
            x = block(x, self.mask)
        x = self.ln_f(x)
        return self.head(x)

    def count_params(self) -> int:
        # Don't double-count tied embeddings
        return sum(p.numel() for p in self.parameters())


# ─── Training ─────────────────────────────────────────────────────────────────

def train_model(name: str, train_data: torch.Tensor, val_data: torch.Tensor,
                vocab_size: int) -> GPT:
    print(f"\n{'='*60}")
    print(f"Training {name} model (vocab_size={vocab_size})")
    print(f"{'='*60}")

    model = GPT(vocab_size, CTX_LEN, D_MODEL, N_HEAD, N_LAYER, DROPOUT).to(DEVICE)
    n_params = model.count_params()
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def lr_schedule(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    model.train()
    t0 = time.time()
    train_losses = []

    for step in range(1, MAX_STEPS + 1):
        x, y = get_batch(train_data, BATCH_SIZE, CTX_LEN)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        if step % LOG_INTERVAL == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch(val_data, BATCH_SIZE, CTX_LEN)
                val_loss = F.cross_entropy(
                    model(vx).view(-1, vocab_size), vy.view(-1)
                ).item()
            model.train()
            elapsed = time.time() - t0
            print(f"  step {step:5d} | train {loss.item():.4f} | "
                  f"val {val_loss:.4f} | lr {scheduler.get_last_lr()[0]:.6f} | "
                  f"{elapsed:.0f}s")
            train_losses.append({
                "step": step, "train_loss": loss.item(), "val_loss": val_loss
            })

    total_time = time.time() - t0
    print(f"Training complete in {total_time:.0f}s")
    return model


# ─── Generation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(model: GPT, tokenizer: CharTokenizer, train_data: torch.Tensor,
             num_samples: int = NUM_SAMPLES) -> list[dict]:
    model.eval()
    samples = []

    for i in range(num_samples):
        # Random prompt from training data
        start = random.randint(0, len(train_data) - PROMPT_LEN - 1)
        prompt_ids = train_data[start:start + PROMPT_LEN].tolist()
        prompt_text = tokenizer.decode(prompt_ids)

        # Generate
        context = list(prompt_ids)
        generated_ids = []
        for _ in range(GEN_LEN):
            ctx = context[-model.ctx_len:]
            x = torch.tensor([ctx], dtype=torch.long, device=DEVICE)
            logits = model(x)[0, -1, :] / TEMPERATURE

            # Top-k filtering
            if TOP_K > 0:
                topk_vals, topk_idx = torch.topk(logits, TOP_K)
                logits = torch.full_like(logits, float("-inf"))
                logits.scatter_(0, topk_idx, topk_vals)

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            context.append(next_id)
            generated_ids.append(next_id)

        generated_text = tokenizer.decode(generated_ids)
        samples.append({
            "prompt": prompt_text,
            "generated": generated_text,
            "full": prompt_text + generated_text
        })

    return samples


# ─── Evaluation ───────────────────────────────────────────────────────────────

def eval_lojban_grammar(samples: list[dict]) -> dict:
    """Evaluate Lojban grammaticality using camxes parser."""
    camxes_path = BASE_DIR / "ilmentufa" / "run_camxes.js"
    total_sentences = 0
    parseable = 0
    errors = []

    for sample in samples:
        text = sample["generated"]
        # Split on .i (Lojban sentence separator)
        sentences = [s.strip() for s in text.split(".i") if s.strip()]

        for sent in sentences:
            # Clean: remove leading/trailing punctuation, skip very short
            sent = sent.strip(" .-")
            if len(sent) < 5:
                continue
            total_sentences += 1

            try:
                result = subprocess.run(
                    ["node", str(camxes_path), sent],
                    capture_output=True, text=True, timeout=10
                )
                # camxes outputs a parse tree on success, error message on failure
                output = result.stdout.strip()
                if output and "error" not in output.lower() and result.returncode == 0:
                    parseable += 1
                else:
                    errors.append({"sentence": sent[:80], "error": output[:100]})
            except (subprocess.TimeoutExpired, Exception) as e:
                errors.append({"sentence": sent[:80], "error": str(e)[:100]})

    rate = parseable / total_sentences if total_sentences > 0 else 0.0
    print(f"  Lojban: {parseable}/{total_sentences} parseable ({rate:.1%})")
    return {
        "total_sentences": total_sentences,
        "parseable": parseable,
        "grammaticality_rate": round(rate, 4),
        "sample_errors": errors[:10]
    }


def eval_english_grammar(samples: list[dict]) -> dict:
    """Evaluate English grammaticality using LanguageTool."""
    import language_tool_python
    tool = language_tool_python.LanguageTool("en-US")

    total_sentences = 0
    error_free = 0
    total_errors = 0

    import re
    for sample in samples:
        text = sample["generated"]
        sentences = re.split(r'(?<=[.!?])\s+', text)

        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 10:
                continue
            total_sentences += 1

            matches = tool.check(sent)
            # Only count grammar errors, not spelling or style
            grammar_errors = [
                m for m in matches
                if m.ruleIssueType == "grammar"
            ]
            if len(grammar_errors) == 0:
                error_free += 1
            total_errors += len(grammar_errors)

    tool.close()
    rate = error_free / total_sentences if total_sentences > 0 else 0.0
    avg_errors = total_errors / total_sentences if total_sentences > 0 else 0.0
    print(f"  English: {error_free}/{total_sentences} error-free ({rate:.1%})")
    return {
        "total_sentences": total_sentences,
        "error_free": error_free,
        "grammaticality_rate": round(rate, 4),
        "avg_grammar_errors": round(avg_errors, 4)
    }


def compute_structural_metrics(samples: list[dict], train_text: str) -> dict:
    """Language-agnostic structural quality metrics."""
    gen_text = " ".join(s["generated"] for s in samples)

    # 1. Character entropy KL divergence
    def char_dist(text):
        counts = Counter(text)
        total = sum(counts.values())
        return {c: n / total for c, n in counts.items()}

    train_dist = char_dist(train_text)
    gen_dist = char_dist(gen_text)
    all_chars = set(train_dist) | set(gen_dist)
    eps = 1e-10
    kl_div = sum(
        gen_dist.get(c, eps) * math.log(gen_dist.get(c, eps) / train_dist.get(c, eps))
        for c in all_chars
    )

    # 2. N-gram repetition (unique n-grams / total n-grams)
    def ngram_diversity(text, n):
        ngrams = [text[i:i + n] for i in range(len(text) - n + 1)]
        if not ngrams:
            return 1.0
        return len(set(ngrams)) / len(ngrams)

    diversity_3 = ngram_diversity(gen_text, 3)
    diversity_4 = ngram_diversity(gen_text, 4)
    diversity_5 = ngram_diversity(gen_text, 5)

    # 3. Word-length distribution similarity
    def word_len_dist(text):
        words = text.split()
        if not words:
            return {}
        counts = Counter(len(w) for w in words)
        total = sum(counts.values())
        return {l: n / total for l, n in counts.items()}

    train_wl = word_len_dist(train_text)
    gen_wl = word_len_dist(gen_text)
    all_lens = set(train_wl) | set(gen_wl)
    wl_similarity = 1.0 - 0.5 * sum(
        abs(train_wl.get(l, 0) - gen_wl.get(l, 0)) for l in all_lens
    )

    return {
        "char_kl_divergence": round(kl_div, 6),
        "ngram_diversity_3": round(diversity_3, 4),
        "ngram_diversity_4": round(diversity_4, 4),
        "ngram_diversity_5": round(diversity_5, 4),
        "word_length_similarity": round(wl_similarity, 4)
    }


# ─── Sanity Checks ───────────────────────────────────────────────────────────

def run_sanity_checks(eng_tok, loj_tok, eng_text, loj_text,
                      eng_model, loj_model):
    print("\n" + "=" * 60)
    print("Sanity Checks")
    print("=" * 60)
    ok = True

    # Roundtrip encoding
    for tok, text, name in [(eng_tok, eng_text, "English"), (loj_tok, loj_text, "Lojban")]:
        decoded = tok.decode(tok.encode(text))
        if decoded == text:
            print(f"  [PASS] {name} encode/decode roundtrip")
        else:
            print(f"  [FAIL] {name} encode/decode roundtrip")
            ok = False

    # Parameter count comparison
    eng_params = eng_model.count_params()
    loj_params = loj_model.count_params()
    diff_pct = abs(eng_params - loj_params) / max(eng_params, loj_params) * 100
    if diff_pct < 0.2:
        print(f"  [PASS] Param count diff: {diff_pct:.3f}% "
              f"(eng={eng_params:,} loj={loj_params:,})")
    else:
        print(f"  [WARN] Param count diff: {diff_pct:.3f}% "
              f"(eng={eng_params:,} loj={loj_params:,})")

    # camxes check
    camxes_path = BASE_DIR / "ilmentufa" / "run_camxes.js"
    try:
        good = subprocess.run(
            ["node", str(camxes_path), "mi klama lo zarci"],
            capture_output=True, text=True, timeout=10
        )
        if good.returncode == 0 and "error" not in good.stdout.lower():
            print("  [PASS] camxes parses valid Lojban")
        else:
            print(f"  [FAIL] camxes valid: {good.stdout[:80]}")
            ok = False

        bad = subprocess.run(
            ["node", str(camxes_path), "xyzzy bloop glork"],
            capture_output=True, text=True, timeout=10
        )
        # For truly invalid input, camxes may still try to parse partially
        print(f"  [INFO] camxes on gibberish: {bad.stdout[:80]}")
    except Exception as e:
        print(f"  [FAIL] camxes: {e}")
        ok = False

    # LanguageTool check
    try:
        import language_tool_python
        lt = language_tool_python.LanguageTool("en-US")
        good_matches = [m for m in lt.check("The cat sat on the mat.")
                        if m.ruleIssueType == "grammar"]
        if len(good_matches) == 0:
            print("  [PASS] LanguageTool passes valid English")
        else:
            print(f"  [FAIL] LanguageTool valid: {len(good_matches)} errors")
            ok = False

        bad_matches = [m for m in lt.check("Cat the sat mat on the.")
                       if m.ruleIssueType == "grammar"]
        if len(bad_matches) > 0:
            print(f"  [PASS] LanguageTool flags bad English ({len(bad_matches)} errors)")
        else:
            print("  [WARN] LanguageTool did not flag bad English")
        lt.close()
    except Exception as e:
        print(f"  [FAIL] LanguageTool: {e}")
        ok = False

    return ok


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    # Load data
    print("Loading data...")
    eng_text = load_english(BASE_DIR / "alice_english.txt")
    loj_text = load_lojban(BASE_DIR / "alice_lojban.txt")
    print(f"  English: {len(eng_text):,} chars")
    print(f"  Lojban:  {len(loj_text):,} chars")

    # Build tokenizers
    eng_tok = CharTokenizer(eng_text, "English")
    loj_tok = CharTokenizer(loj_text, "Lojban")
    print(f"  English vocab: {eng_tok.vocab_size}")
    print(f"  Lojban vocab:  {loj_tok.vocab_size}")

    # Prepare data
    eng_train, eng_val = prepare_data(eng_text, eng_tok)
    loj_train, loj_val = prepare_data(loj_text, loj_tok)
    print(f"  English train/val: {len(eng_train):,}/{len(eng_val):,}")
    print(f"  Lojban train/val:  {len(loj_train):,}/{len(loj_val):,}")

    # Train models
    eng_model = train_model("English", eng_train, eng_val, eng_tok.vocab_size)
    loj_model = train_model("Lojban", loj_train, loj_val, loj_tok.vocab_size)

    # Sanity checks
    run_sanity_checks(eng_tok, loj_tok, eng_text, loj_text, eng_model, loj_model)

    # Generate samples
    print("\nGenerating English samples...")
    eng_samples = generate(eng_model, eng_tok, eng_train)
    print(f"  Generated {len(eng_samples)} samples")

    print("Generating Lojban samples...")
    loj_samples = generate(loj_model, loj_tok, loj_train)
    print(f"  Generated {len(loj_samples)} samples")

    # Save samples
    with open(RESULTS_DIR / "english_samples.json", "w") as f:
        json.dump(eng_samples, f, indent=2, ensure_ascii=False)
    with open(RESULTS_DIR / "lojban_samples.json", "w") as f:
        json.dump(loj_samples, f, indent=2, ensure_ascii=False)

    # Evaluate
    print("\nEvaluating grammaticality...")
    print("  (Lojban evaluation via camxes — this may take a few minutes)")
    loj_grammar = eval_lojban_grammar(loj_samples)

    print("  (English evaluation via LanguageTool)")
    eng_grammar = eval_english_grammar(eng_samples)

    print("\nComputing structural metrics...")
    eng_structural = compute_structural_metrics(eng_samples, eng_text)
    loj_structural = compute_structural_metrics(loj_samples, loj_text)

    # Compile results
    results = {
        "english": {
            "grammar": eng_grammar,
            "structural": eng_structural,
            "params": eng_model.count_params(),
            "vocab_size": eng_tok.vocab_size
        },
        "lojban": {
            "grammar": loj_grammar,
            "structural": loj_structural,
            "params": loj_model.count_params(),
            "vocab_size": loj_tok.vocab_size
        }
    }

    with open(RESULTS_DIR / "evaluation.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY (total time: {elapsed/60:.1f} min)")
    print(f"{'='*60}")
    print(f"\nEnglish ({eng_model.count_params():,} params, vocab={eng_tok.vocab_size}):")
    print(f"  Grammaticality: {eng_grammar['grammaticality_rate']:.1%} "
          f"({eng_grammar['error_free']}/{eng_grammar['total_sentences']} sentences)")
    print(f"  Char KL div:    {eng_structural['char_kl_divergence']:.6f}")
    print(f"  3/4/5-gram div: {eng_structural['ngram_diversity_3']:.4f} / "
          f"{eng_structural['ngram_diversity_4']:.4f} / "
          f"{eng_structural['ngram_diversity_5']:.4f}")
    print(f"  Word-len sim:   {eng_structural['word_length_similarity']:.4f}")

    print(f"\nLojban ({loj_model.count_params():,} params, vocab={loj_tok.vocab_size}):")
    print(f"  Grammaticality: {loj_grammar['grammaticality_rate']:.1%} "
          f"({loj_grammar['parseable']}/{loj_grammar['total_sentences']} sentences)")
    print(f"  Char KL div:    {loj_structural['char_kl_divergence']:.6f}")
    print(f"  3/4/5-gram div: {loj_structural['ngram_diversity_3']:.4f} / "
          f"{loj_structural['ngram_diversity_4']:.4f} / "
          f"{loj_structural['ngram_diversity_5']:.4f}")
    print(f"  Word-len sim:   {loj_structural['word_length_similarity']:.4f}")

    print(f"\nResults saved to {RESULTS_DIR}/evaluation.json")


if __name__ == "__main__":
    main()
