#!/usr/bin/env python3
"""
Experiment V2: Lojban vs English Small Model Coherence
======================================================
Multi-corpus, multi-size, multi-seed experiment suite.

Changes from v1:
- 4 training books (~400K chars) instead of 1 (~130K)
- 3 model sizes (Small/Medium/Large) to test scaling
- 3 seeds for confidence intervals
- 90/5/5 train/val/prompt split
- Held-out test book (Metamorphosis) for BPC evaluation
- Early stopping with patience=1500 steps
- Fixed validation batches
- Best-checkpoint saving by val loss
- BPC metric (cross-entropy / ln(2)), normalized across vocab sizes
- Memorization detection via longest common substring
- Grammaticality evaluated on novel (non-memorized) text only
- Tatoeba ground-truth calibration for grammar checkers
- Repetition rate metrics (n=10,20,50)
"""

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Constants ───────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
CORPUS_DIR = BASE_DIR / "corpus"
RESULTS_DIR = BASE_DIR / "results" / "v2"

SEEDS = [42, 137, 2024]

MODEL_CONFIGS = {
    "small":  {"d_model": 128, "n_layer": 4, "n_head": 2, "dropout": 0.15},
    "medium": {"d_model": 256, "n_layer": 4, "n_head": 4, "dropout": 0.20},
    "large":  {"d_model": 384, "n_layer": 6, "n_head": 6, "dropout": 0.30},
}

TRAIN_BOOKS = ["alice_in_wonderland", "wizard_of_oz", "esther", "in_a_grove"]
TEST_BOOK = "metamorphosis"

BOOK_FILES = {
    "alice_in_wonderland": {
        "english": "alice_in_wonderland/alice_english.txt",
        "lojban":  "alice_in_wonderland/alice_lojban.txt",
    },
    "wizard_of_oz": {
        "english": "wizard_of_oz/wizard_of_oz_english.txt",
        "lojban":  "wizard_of_oz/wizard_of_oz_lojban.txt",
    },
    "esther": {
        "english": "esther/esther_english.txt",
        "lojban":  "esther/esther_lojban.txt",
    },
    "in_a_grove": {
        "english": "in_a_grove/in_a_grove_english.txt",
        "lojban":  "in_a_grove/in_a_grove_lojban.txt",
    },
    "metamorphosis": {
        "english": "metamorphosis/metamorphosis_english.txt",
        "lojban":  "metamorphosis/metamorphosis_lojban.txt",
    },
}


# ─── Configuration ───────────────────────────────────────────────────────────

@dataclass
class RunConfig:
    size: str
    language: str
    seed: int
    d_model: int
    n_layer: int
    n_head: int
    dropout: float
    ctx_len: int = 256
    batch_size: int = 64
    max_steps: int = 15000
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 100
    patience: int = 1500
    eval_interval: int = 100
    num_samples: int = 100
    prompt_len: int = 32
    gen_len: int = 256
    temperature: float = 0.8
    top_k: int = 40
    lcs_threshold: int = 50


# ─── Data ────────────────────────────────────────────────────────────────────

class CharTokenizer:
    def __init__(self, text: str, name: str = ""):
        self.name = name
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for i, c in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        return [self.char_to_idx[c] for c in text if c in self.char_to_idx]

    def decode(self, indices: list[int]) -> str:
        return "".join(self.idx_to_char[i] for i in indices)

    def coverage(self, text: str) -> float:
        if not text:
            return 0.0
        return sum(1 for c in text if c in self.char_to_idx) / len(text)


def load_book_text(book: str, language: str) -> str:
    rel_path = BOOK_FILES[book][language]
    path = CORPUS_DIR / rel_path
    text = path.read_text(encoding="utf-8")

    # Strip BOM
    text = text.lstrip("\ufeff")

    # Strip Gutenberg headers from Alice English
    if book == "alice_in_wonderland" and language == "english":
        if "*** START OF" in text:
            start = text.index("*** START OF")
            start = text.index("\n", start) + 1
            end = text.index("*** END OF")
            text = text[start:end]

    # Clean unicode oddities in Lojban texts
    if language == "lojban":
        text = text.replace("\u0096", "-")
        text = text.replace("\u0097", "--")

    return text.strip()


def load_corpus(books: list[str], language: str) -> str:
    texts = []
    for book in books:
        text = load_book_text(book, language)
        texts.append(text)
        print(f"    {book}: {len(text):,} chars")
    return "\n\n".join(texts)


def load_tatoeba() -> list[dict]:
    path = CORPUS_DIR / "tatoeba" / "jbo_eng_parallel.tsv"
    pairs = []
    with open(path, encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                pairs.append({"lojban": parts[0].strip(),
                              "english": parts[1].strip()})
    return pairs


def prepare_splits(text: str, tokenizer: CharTokenizer):
    """90% train / 5% val / 5% prompt split."""
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = len(data)
    t1 = int(n * 0.90)
    t2 = int(n * 0.95)
    return data[:t1], data[t1:t2], data[t2:]


# ─── Model ───────────────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj_drop(self.proj(out))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, dropout):
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

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, ctx_len, d_model, n_head, n_layer, dropout):
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
        self.head.weight = self.tok_emb.weight  # tied embeddings

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

    def forward(self, idx):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok + pos)
        for block in self.blocks:
            x = block(x, self.mask)
        x = self.ln_f(x)
        return self.head(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ─── Training ────────────────────────────────────────────────────────────────

def get_batch(data, batch_size, ctx_len, device):
    ix = torch.randint(len(data) - ctx_len - 1, (batch_size,))
    x = torch.stack([data[i:i + ctx_len] for i in ix])
    y = torch.stack([data[i + 1:i + ctx_len + 1] for i in ix])
    return x.to(device), y.to(device)


def make_fixed_val_batches(val_data, batch_size, ctx_len, device, n_batches=10):
    """Pre-compute fixed validation batches for consistent eval."""
    batches = []
    for _ in range(n_batches):
        x, y = get_batch(val_data, batch_size, ctx_len, device)
        batches.append((x, y))
    return batches


def eval_val_loss(model, val_batches, vocab_size):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in val_batches:
            logits = model(x)
            total += F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)).item()
    return total / len(val_batches)


def train_model(config, train_data, val_data, vocab_size, device):
    print(f"\n{'='*60}")
    print(f"Training {config.language} model "
          f"({config.size}, seed={config.seed})")
    print(f"  d={config.d_model} L={config.n_layer} H={config.n_head} "
          f"drop={config.dropout}")
    print(f"{'='*60}")

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    model = GPT(vocab_size, config.ctx_len, config.d_model, config.n_head,
                config.n_layer, config.dropout).to(device)
    n_params = model.count_params()
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                   weight_decay=config.weight_decay)

    def lr_schedule(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        progress = (step - config.warmup_steps) / max(
            config.max_steps - config.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Fixed validation batches (seeded for reproducibility)
    val_batches = make_fixed_val_batches(val_data, config.batch_size,
                                         config.ctx_len, device, n_batches=10)

    model.train()
    t0 = time.time()
    log = []
    best_val_loss = float("inf")
    best_state = None
    best_step = 0
    no_improve = 0

    bpc_thresholds = [3.0, 2.5, 2.0, 1.5]
    bpc_reached = {}
    final_step = config.max_steps

    for step in range(1, config.max_steps + 1):
        x, y = get_batch(train_data, config.batch_size, config.ctx_len, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()

        if step % config.eval_interval == 0 or step == 1:
            val_loss = eval_val_loss(model, val_batches, vocab_size)
            val_bpc = val_loss / math.log(2)
            train_bpc = loss.item() / math.log(2)
            model.train()

            elapsed = time.time() - t0
            print(f"  step {step:5d} | train_bpc {train_bpc:.3f} | "
                  f"val_bpc {val_bpc:.3f} | "
                  f"lr {scheduler.get_last_lr()[0]:.6f} | {elapsed:.0f}s")

            log.append({
                "step": step,
                "train_loss": round(loss.item(), 4),
                "val_loss": round(val_loss, 4),
                "train_bpc": round(train_bpc, 4),
                "val_bpc": round(val_bpc, 4),
            })

            # Track BPC convergence speed
            for t in bpc_thresholds:
                if t not in bpc_reached and val_bpc <= t:
                    bpc_reached[t] = step
                    print(f"    >> Reached val BPC {t} at step {step}")

            # Best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
                best_step = step
                no_improve = 0
            else:
                no_improve += config.eval_interval

            # Early stopping
            if no_improve >= config.patience:
                print(f"  Early stopping at step {step} "
                      f"(best at step {best_step})")
                final_step = step
                break
    else:
        final_step = config.max_steps

    total_time = time.time() - t0
    best_val_bpc = best_val_loss / math.log(2)
    print(f"  Done in {total_time:.0f}s | best val_bpc {best_val_bpc:.3f} "
          f"at step {best_step}")

    # Load best checkpoint
    if best_state is not None:
        model.load_state_dict({k: v.to(device)
                               for k, v in best_state.items()})

    info = {
        "n_params": n_params,
        "vocab_size": vocab_size,
        "total_time_s": round(total_time, 1),
        "total_steps": final_step,
        "best_step": best_step,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_bpc": round(best_val_bpc, 4),
        "final_train_loss": round(loss.item(), 4),
        "early_stopped": no_improve >= config.patience,
        "bpc_thresholds_reached": {str(k): v for k, v in bpc_reached.items()},
        "log": log,
    }
    return model, info


# ─── Generation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_samples(model, tokenizer, prompt_data, test_ids, config, device):
    """Generate text samples from in-domain (prompt split) and out-of-domain
    (Metamorphosis test) prompts."""
    model.eval()
    samples = []

    n_in = int(config.num_samples * 0.8)
    n_out = config.num_samples - n_in

    for i in range(config.num_samples):
        if i < n_in:
            source = "in_domain"
            data = prompt_data
        else:
            source = "out_of_domain"
            data = test_ids

        if len(data) < config.prompt_len + 1:
            continue

        start = random.randint(0, len(data) - config.prompt_len - 1)
        prompt_ids = data[start:start + config.prompt_len].tolist()
        prompt_text = tokenizer.decode(prompt_ids)

        context = list(prompt_ids)
        generated_ids = []
        for _ in range(config.gen_len):
            ctx = context[-model.ctx_len:]
            x = torch.tensor([ctx], dtype=torch.long, device=device)
            logits = model(x)[0, -1, :] / config.temperature

            if config.top_k > 0:
                k = min(config.top_k, logits.size(-1))
                topk_vals, topk_idx = torch.topk(logits, k)
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
            "full": prompt_text + generated_text,
            "source": source,
        })

    return samples


# ─── Memorization Detection ─────────────────────────────────────────────────

def compute_lcs_length(generated, train_text):
    """Binary search for length of longest contiguous match."""
    lo, hi = 0, len(generated)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        found = any(
            generated[i:i + mid] in train_text
            for i in range(len(generated) - mid + 1)
        )
        if found:
            lo = mid
        else:
            hi = mid - 1
    return lo


def tag_memorization(samples, train_text, threshold=50):
    """Compute LCS for each sample and flag as memorized if above threshold."""
    print(f"  Computing memorization (LCS threshold={threshold})...")
    t0 = time.time()
    for i, s in enumerate(samples):
        lcs = compute_lcs_length(s["generated"], train_text)
        s["lcs_length"] = lcs
        s["is_memorized"] = lcs >= threshold
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(samples)} samples processed...")

    elapsed = time.time() - t0
    n_memorized = sum(1 for s in samples if s["is_memorized"])
    n_novel = len(samples) - n_memorized
    avg_lcs = (sum(s["lcs_length"] for s in samples) / len(samples)
               if samples else 0)
    print(f"    Done in {elapsed:.0f}s | "
          f"Memorized: {n_memorized}/{len(samples)} | "
          f"Novel: {n_novel}/{len(samples)} | "
          f"Avg LCS: {avg_lcs:.0f} chars")
    return {
        "n_memorized": n_memorized,
        "n_novel": n_novel,
        "avg_lcs": round(avg_lcs, 1),
        "threshold": threshold,
    }


# ─── BPC on Test Set ────────────────────────────────────────────────────────

@torch.no_grad()
def compute_test_bpc(model, test_ids, tokenizer, ctx_len, device):
    """Compute bits-per-character on test data using non-overlapping windows."""
    model.eval()
    total_loss = 0.0
    total_chars = 0

    for start in range(0, len(test_ids) - ctx_len - 1, ctx_len):
        x = test_ids[start:start + ctx_len].unsqueeze(0).to(device)
        y = test_ids[start + 1:start + ctx_len + 1].unsqueeze(0).to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size),
                                y.view(-1), reduction="sum")
        total_loss += loss.item()
        total_chars += ctx_len

    avg_loss = total_loss / total_chars if total_chars > 0 else 0.0
    bpc = avg_loss / math.log(2)
    return {
        "test_bpc": round(bpc, 4),
        "test_chars_evaluated": total_chars,
    }


# ─── Grammar Evaluation ─────────────────────────────────────────────────────

def eval_lojban_grammar(samples, novel_only=True):
    """Evaluate Lojban grammaticality via camxes, on novel samples only."""
    camxes_path = BASE_DIR / "ilmentufa" / "run_camxes.js"
    if not camxes_path.exists():
        print("  SKIP: camxes not found")
        return {"skipped": True, "reason": "camxes not found"}

    subset = ([s for s in samples if not s.get("is_memorized", False)]
              if novel_only else samples)
    if not subset:
        return {"total_sentences": 0, "note": "no novel samples"}

    total = 0
    parseable = 0
    errors = []

    for sample in subset:
        text = sample["generated"]
        sentences = [s.strip() for s in text.split(".i") if s.strip()]
        for sent in sentences:
            sent = sent.strip(" .-")
            if len(sent) < 5:
                continue
            total += 1
            try:
                result = subprocess.run(
                    ["node", str(camxes_path), sent],
                    capture_output=True, text=True, timeout=10
                )
                if (result.returncode == 0
                        and "error" not in result.stdout.lower()):
                    parseable += 1
                else:
                    errors.append({"sentence": sent[:80],
                                   "error": result.stdout[:100]})
            except Exception as e:
                errors.append({"sentence": sent[:80], "error": str(e)[:100]})

    rate = parseable / total if total > 0 else 0.0
    print(f"  Lojban grammar: {parseable}/{total} parseable ({rate:.1%})")
    return {
        "total_sentences": total,
        "parseable": parseable,
        "grammaticality_rate": round(rate, 4),
        "novel_only": novel_only,
        "n_samples_evaluated": len(subset),
        "sample_errors": errors[:10],
    }


def eval_english_grammar(samples, novel_only=True):
    """Evaluate English grammaticality via LanguageTool, on novel samples only."""
    try:
        import language_tool_python
    except ImportError:
        print("  SKIP: language_tool_python not available")
        return {"skipped": True, "reason": "language_tool_python not available"}

    subset = ([s for s in samples if not s.get("is_memorized", False)]
              if novel_only else samples)
    if not subset:
        return {"total_sentences": 0, "note": "no novel samples"}

    tool = language_tool_python.LanguageTool("en-US")
    total = 0
    error_free = 0
    total_errors = 0

    for sample in subset:
        text = sample["generated"]
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 10:
                continue
            total += 1
            matches = tool.check(sent)
            grammar_errors = [m for m in matches
                              if m.rule_issue_type == "grammar"]
            if not grammar_errors:
                error_free += 1
            total_errors += len(grammar_errors)

    tool.close()
    rate = error_free / total if total > 0 else 0.0
    avg = total_errors / total if total > 0 else 0.0
    print(f"  English grammar: {error_free}/{total} error-free ({rate:.1%})")
    return {
        "total_sentences": total,
        "error_free": error_free,
        "grammaticality_rate": round(rate, 4),
        "avg_grammar_errors": round(avg, 4),
        "novel_only": novel_only,
        "n_samples_evaluated": len(subset),
    }


# ─── Structural Metrics ─────────────────────────────────────────────────────

def compute_structural_metrics(samples, train_text):
    gen_text = " ".join(s["generated"] for s in samples)

    # Character KL divergence
    def char_dist(text):
        counts = Counter(text)
        total = sum(counts.values())
        return {c: n / total for c, n in counts.items()}

    train_d = char_dist(train_text)
    gen_d = char_dist(gen_text)
    all_chars = set(train_d) | set(gen_d)
    eps = 1e-10
    kl = sum(
        gen_d.get(c, eps) * math.log(gen_d.get(c, eps) / train_d.get(c, eps))
        for c in all_chars
    )

    # N-gram diversity (unique / total)
    def ngram_div(text, n):
        ngrams = [text[i:i + n] for i in range(len(text) - n + 1)]
        return len(set(ngrams)) / len(ngrams) if ngrams else 1.0

    # Repetition rate (1 - diversity, for longer n-grams)
    def rep_rate(text, n):
        return 1.0 - ngram_div(text, n)

    # Word-length distribution similarity
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
    wl_sim = 1.0 - 0.5 * sum(
        abs(train_wl.get(l, 0) - gen_wl.get(l, 0)) for l in all_lens
    )

    return {
        "char_kl_divergence": round(kl, 6),
        "ngram_diversity_3": round(ngram_div(gen_text, 3), 4),
        "ngram_diversity_4": round(ngram_div(gen_text, 4), 4),
        "ngram_diversity_5": round(ngram_div(gen_text, 5), 4),
        "repetition_rate_10": round(rep_rate(gen_text, 10), 4),
        "repetition_rate_20": round(rep_rate(gen_text, 20), 4),
        "repetition_rate_50": round(rep_rate(gen_text, 50), 4),
        "word_length_similarity": round(wl_sim, 4),
    }


# ─── Tatoeba Calibration ────────────────────────────────────────────────────

def calibrate_grammar_checkers(tatoeba, n_samples=200):
    """Run grammar checkers on known-good Tatoeba sentences to establish
    baseline pass rates. This calibrates how strict each checker is."""
    sample = random.sample(tatoeba, min(n_samples, len(tatoeba)))
    result = {"n_sampled": len(sample)}

    # Lojban via camxes
    camxes_path = BASE_DIR / "ilmentufa" / "run_camxes.js"
    if camxes_path.exists():
        loj_total = 0
        loj_pass = 0
        for pair in sample:
            sent = pair["lojban"].strip()
            if len(sent) < 3:
                continue
            loj_total += 1
            try:
                r = subprocess.run(
                    ["node", str(camxes_path), sent],
                    capture_output=True, text=True, timeout=10
                )
                if r.returncode == 0 and "error" not in r.stdout.lower():
                    loj_pass += 1
            except Exception:
                pass
        rate = loj_pass / loj_total if loj_total > 0 else 0.0
        result["camxes_total"] = loj_total
        result["camxes_pass"] = loj_pass
        result["camxes_pass_rate"] = round(rate, 4)
        print(f"    camxes: {loj_pass}/{loj_total} pass ({rate:.1%})")
    else:
        result["camxes"] = "not_available"

    # English via LanguageTool
    try:
        import language_tool_python
        tool = language_tool_python.LanguageTool("en-US")
        eng_total = 0
        eng_pass = 0
        for pair in sample:
            sent = pair["english"].strip()
            if len(sent) < 5:
                continue
            eng_total += 1
            matches = tool.check(sent)
            grammar_errors = [m for m in matches
                              if m.rule_issue_type == "grammar"]
            if not grammar_errors:
                eng_pass += 1
        tool.close()
        rate = eng_pass / eng_total if eng_total > 0 else 0.0
        result["lt_total"] = eng_total
        result["lt_pass"] = eng_pass
        result["lt_pass_rate"] = round(rate, 4)
        print(f"    LanguageTool: {eng_pass}/{eng_total} pass ({rate:.1%})")
    except ImportError:
        result["language_tool"] = "not_available"

    return result


# ─── Single Run ──────────────────────────────────────────────────────────────

def run_single(config, train_data, val_data, prompt_data, tokenizer,
               test_ids, test_text, train_text, device,
               skip_grammar=False):
    """Execute one complete experiment run: train, generate, evaluate."""
    run_dir = (RESULTS_DIR / config.size
               / f"{config.language}_seed{config.seed}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already completed
    result_path = run_dir / "result.json"
    if result_path.exists():
        print(f"\n  SKIP: {result_path} already exists")
        return None

    # Train
    model, train_info = train_model(config, train_data, val_data,
                                     tokenizer.vocab_size, device)

    # BPC on test set
    print("  Computing test BPC...")
    test_bpc_info = compute_test_bpc(model, test_ids, tokenizer,
                                      config.ctx_len, device)
    print(f"    Test BPC: {test_bpc_info['test_bpc']}")

    # BPC on val set (for comparison)
    val_bpc = train_info["best_val_bpc"]

    # Generate samples
    print("  Generating samples...")
    random.seed(config.seed)  # reset for reproducible generation
    samples = generate_samples(model, tokenizer, prompt_data, test_ids,
                                config, device)
    print(f"    Generated {len(samples)} samples")

    # Memorization detection
    mem_info = tag_memorization(samples, train_text, config.lcs_threshold)

    # Grammar evaluation (novel samples only)
    if not skip_grammar:
        print("  Evaluating grammar (novel samples only)...")
        if config.language == "lojban":
            grammar = eval_lojban_grammar(samples, novel_only=True)
        else:
            grammar = eval_english_grammar(samples, novel_only=True)
    else:
        grammar = {"skipped": True, "reason": "skip_grammar flag"}

    # Structural metrics
    print("  Computing structural metrics...")
    structural = compute_structural_metrics(samples, train_text)

    # Compile results
    result = {
        "config": asdict(config),
        "training": train_info,
        "test_bpc": test_bpc_info,
        "val_bpc": val_bpc,
        "memorization": mem_info,
        "grammar": grammar,
        "structural": structural,
    }

    # Save
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    with open(run_dir / "samples.json", "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"  Results saved to {run_dir}/")

    # Free GPU memory
    del model
    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass

    return result


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Lojban vs English experiment v2: "
                    "multi-corpus, multi-size, multi-seed")
    parser.add_argument("--size",
                        choices=["small", "medium", "large", "all"],
                        default="all",
                        help="Model size to run (default: all)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Single seed (default: all 3)")
    parser.add_argument("--language",
                        choices=["english", "lojban", "all"],
                        default="all",
                        help="Language to run (default: all)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick sanity check: 500 steps, 20 samples, "
                             "1 seed")
    parser.add_argument("--skip-grammar", action="store_true",
                        help="Skip grammar evaluation")
    parser.add_argument("--skip-calibration", action="store_true",
                        help="Skip Tatoeba calibration")
    args = parser.parse_args()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Device: CUDA")
    else:
        device = torch.device("cpu")
        print("Device: CPU")
    torch.set_num_threads(8)

    t_start = time.time()

    # ─── Load Data ───────────────────────────────────────────────────────

    print("\nLoading training corpus (4 books)...")
    print("  English:")
    eng_train_text = load_corpus(TRAIN_BOOKS, "english")
    print("  Lojban:")
    loj_train_text = load_corpus(TRAIN_BOOKS, "lojban")

    # Truncate to equal character count
    min_len = min(len(eng_train_text), len(loj_train_text))
    eng_train_text = eng_train_text[:min_len]
    loj_train_text = loj_train_text[:min_len]
    print(f"\nTruncated to {min_len:,} chars each")

    # Build tokenizers from training text only
    eng_tok = CharTokenizer(eng_train_text, "English")
    loj_tok = CharTokenizer(loj_train_text, "Lojban")
    print(f"English vocab: {eng_tok.vocab_size} | "
          f"Lojban vocab: {loj_tok.vocab_size}")

    # 90/5/5 split
    eng_train, eng_val, eng_prompt = prepare_splits(eng_train_text, eng_tok)
    loj_train, loj_val, loj_prompt = prepare_splits(loj_train_text, loj_tok)
    print(f"English split: {len(eng_train):,} / {len(eng_val):,} / "
          f"{len(eng_prompt):,} (train/val/prompt)")
    print(f"Lojban split:  {len(loj_train):,} / {len(loj_val):,} / "
          f"{len(loj_prompt):,} (train/val/prompt)")

    # Load test book
    print("\nLoading test book (Metamorphosis)...")
    eng_test_text = load_book_text(TEST_BOOK, "english")
    loj_test_text = load_book_text(TEST_BOOK, "lojban")

    # Pre-encode test data
    eng_test_ids = torch.tensor(eng_tok.encode(eng_test_text), dtype=torch.long)
    loj_test_ids = torch.tensor(loj_tok.encode(loj_test_text), dtype=torch.long)
    print(f"  English: {len(eng_test_text):,} chars -> "
          f"{len(eng_test_ids):,} tokens "
          f"(coverage: {eng_tok.coverage(eng_test_text):.1%})")
    print(f"  Lojban:  {len(loj_test_text):,} chars -> "
          f"{len(loj_test_ids):,} tokens "
          f"(coverage: {loj_tok.coverage(loj_test_text):.1%})")

    # ─── Tatoeba Calibration ─────────────────────────────────────────────

    if not args.skip_calibration:
        print("\nCalibrating grammar checkers on Tatoeba...")
        random.seed(42)  # fixed seed for calibration
        tatoeba = load_tatoeba()
        print(f"  Loaded {len(tatoeba)} parallel pairs")
        calibration = calibrate_grammar_checkers(tatoeba)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / "calibration.json", "w") as f:
            json.dump(calibration, f, indent=2)
        print(f"  Saved to {RESULTS_DIR / 'calibration.json'}")

    # ─── Save corpus info ────────────────────────────────────────────────

    corpus_info = {
        "train_books": TRAIN_BOOKS,
        "test_book": TEST_BOOK,
        "train_chars_per_language": min_len,
        "english_vocab_size": eng_tok.vocab_size,
        "lojban_vocab_size": loj_tok.vocab_size,
        "english_train_tokens": len(eng_train),
        "english_val_tokens": len(eng_val),
        "english_prompt_tokens": len(eng_prompt),
        "lojban_train_tokens": len(loj_train),
        "lojban_val_tokens": len(loj_val),
        "lojban_prompt_tokens": len(loj_prompt),
        "english_test_chars": len(eng_test_text),
        "english_test_tokens": len(eng_test_ids),
        "english_test_coverage": round(eng_tok.coverage(eng_test_text), 4),
        "lojban_test_chars": len(loj_test_text),
        "lojban_test_tokens": len(loj_test_ids),
        "lojban_test_coverage": round(loj_tok.coverage(loj_test_text), 4),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "corpus_info.json", "w") as f:
        json.dump(corpus_info, f, indent=2)

    # ─── Run Experiments ─────────────────────────────────────────────────

    if args.quick:
        sizes = [args.size] if args.size != "all" else ["small"]
        seeds = [42]
        languages = (["english", "lojban"] if args.language == "all"
                     else [args.language])
    else:
        sizes = ([args.size] if args.size != "all"
                 else ["small", "medium", "large"])
        seeds = [args.seed] if args.seed is not None else SEEDS
        languages = ([args.language] if args.language != "all"
                     else ["english", "lojban"])

    total_runs = len(sizes) * len(seeds) * len(languages)
    run_idx = 0

    for size in sizes:
        mc = MODEL_CONFIGS[size]
        for seed in seeds:
            for language in languages:
                run_idx += 1
                config = RunConfig(
                    size=size, language=language, seed=seed, **mc
                )
                if args.quick:
                    config.max_steps = 500
                    config.patience = 200
                    config.num_samples = 20
                    config.eval_interval = 50

                print(f"\n{'#'*60}")
                print(f"# Run {run_idx}/{total_runs}: "
                      f"{size} / {language} / seed={seed}")
                print(f"{'#'*60}")

                if language == "english":
                    run_single(
                        config, eng_train, eng_val, eng_prompt,
                        eng_tok, eng_test_ids, eng_test_text,
                        eng_train_text, device,
                        skip_grammar=args.skip_grammar,
                    )
                else:
                    run_single(
                        config, loj_train, loj_val, loj_prompt,
                        loj_tok, loj_test_ids, loj_test_text,
                        loj_train_text, device,
                        skip_grammar=args.skip_grammar,
                    )

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"All {total_runs} runs complete in {elapsed/60:.1f} min")
    print(f"Results in {RESULTS_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
