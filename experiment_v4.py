#!/usr/bin/env python3
"""
Experiment V4: Lojban vs English — BPE Tokenization
=====================================================
Tests whether BPE tokenization equalizes training dynamics between English
and Lojban by normalizing information per token, removing the character-level
entropy confound that biased V3's early stopping.

Changes from V3:
- BPE tokenization (vocab=1024) instead of character-level
- 2 model sizes (medium ~570K, large ~758K) with tied embeddings
- Fixed 10K steps by default (no early stopping)
- BPC computed over actual characters (not token count)
- Token-based generation (128 tokens ~= 256+ chars)
"""

import argparse
import json
import math
import random
import re
import subprocess
import time
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from bpe_tokenizer import BPETokenizerWrapper, DEFAULT_VOCAB_SIZE

# ─── Constants ───────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
CORPUS_DIR = BASE_DIR / "corpus"
BABI_DIR = CORPUS_DIR / "babi"
RESULTS_DIR = BASE_DIR / "results" / "v4"

SEEDS = [42, 137, 2024]

BPE_VOCAB_SIZE = DEFAULT_VOCAB_SIZE

MODEL_CONFIGS = {
    "medium": {"d_model": 96,  "n_layer": 4, "n_head": 4, "dropout": 0.15, "ctx_len": 256},
    "large":  {"d_model": 128, "n_layer": 3, "n_head": 4, "dropout": 0.15, "ctx_len": 256},
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

BABI_TASK_NAMES = {
    1: "single_supporting_fact",
    2: "two_supporting_facts",
    3: "three_supporting_facts",
    4: "two_argument_relations",
    5: "three_argument_relations",
    6: "yes_no_questions",
    7: "counting",
    8: "lists_sets",
    9: "simple_negation",
    10: "indefinite_knowledge",
    11: "basic_coreference",
    12: "conjunction",
    13: "compound_coreference",
    14: "time_reasoning",
    15: "basic_deduction",
    16: "basic_induction",
    17: "positional_reasoning",
    18: "size_reasoning",
    19: "path_finding",
    20: "agents_motivations",
}

ALL_BABI_TASKS = list(range(1, 21))


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
    max_steps: int = 10000
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 100
    patience: int = 10001  # > max_steps: no early stopping by default
    eval_interval: int = 100
    num_samples: int = 50
    prompt_len: int = 32     # in tokens
    gen_len: int = 128       # in tokens (~256+ chars with BPE)
    temperature: float = 0.8
    top_k: int = 40
    lcs_threshold: int = 50


# ─── Data ────────────────────────────────────────────────────────────────────

def load_book_text(book, language):
    rel_path = BOOK_FILES[book][language]
    path = CORPUS_DIR / rel_path
    text = path.read_text(encoding="utf-8")

    text = text.lstrip("\ufeff")

    if book == "alice_in_wonderland" and language == "english":
        if "*** START OF" in text:
            start = text.index("*** START OF")
            start = text.index("\n", start) + 1
            end = text.index("*** END OF")
            text = text[start:end]

    if language == "lojban":
        text = text.replace("\u0096", "-")
        text = text.replace("\u0097", "--")

    return text.strip()


def load_corpus(books, language):
    texts = []
    for book in books:
        text = load_book_text(book, language)
        texts.append(text)
        print(f"    {book}: {len(text):,} chars")
    return "\n\n".join(texts)


def load_tatoeba():
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


def prepare_splits(text, tokenizer):
    """90% train / 5% val / 5% prompt split (in token space)."""
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = len(data)
    t1 = int(n * 0.90)
    t2 = int(n * 0.95)
    return data[:t1], data[t1:t2], data[t2:]


# ─── bAbI Data ───────────────────────────────────────────────────────────────

def load_babi_examples(task_id, split, language):
    """Load bAbI examples for one task/split/language."""
    task_name = BABI_TASK_NAMES[task_id]
    suffix = "en" if language == "english" else "lj"
    path = BABI_DIR / f"task{task_id:02d}_{task_name}" / f"{split}.{suffix}.txt"
    content = path.read_text(encoding="utf-8")
    raw_examples = [ex.strip() for ex in content.strip().split("\n\n") if ex.strip()]

    examples = []
    for raw in raw_examples:
        lines = raw.split("\n")
        last_line = lines[-1]
        q_idx = last_line.rfind("? ")
        if q_idx == -1:
            continue
        question_with_marker = last_line[:q_idx + 1]  # end at "?" not "? "
        answer = last_line[q_idx + 2:].strip()

        if len(lines) > 1:
            context = "\n".join(lines[:-1]) + "\n" + question_with_marker
        else:
            context = question_with_marker

        examples.append({
            "text": raw,
            "context": context,
            "answer": answer,
            "task_id": task_id,
        })
    return examples


def load_babi_train_text(language, task_ids):
    """Load all bAbI training examples as a single text string."""
    all_texts = []
    for task_id in task_ids:
        examples = load_babi_examples(task_id, "train", language)
        for ex in examples:
            all_texts.append(ex["text"])
    return "\n\n".join(all_texts)


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
          f"drop={config.dropout} ctx={config.ctx_len}")
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

            for t in bpc_thresholds:
                if t not in bpc_reached and val_bpc <= t:
                    bpc_reached[t] = step
                    print(f"    >> Reached val BPC {t} at step {step}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
                best_step = step
                no_improve = 0
            else:
                no_improve += config.eval_interval

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
    """Generate text samples from in-domain and out-of-domain prompts."""
    model.eval()
    samples = []

    n_in = int(config.num_samples * 0.8)

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
    """Compute bits-per-character on test data.

    With BPE, each token may represent multiple characters. We compute
    cross-entropy loss per token, then normalize by the actual character
    count those tokens represent.
    """
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
        # Count actual characters the target tokens represent
        y_ids = test_ids[start + 1:start + ctx_len + 1].tolist()
        chars_in_window = len(tokenizer.decode(y_ids))
        total_chars += chars_in_window

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

    def ngram_div(text, n):
        ngrams = [text[i:i + n] for i in range(len(text) - n + 1)]
        return len(set(ngrams)) / len(ngrams) if ngrams else 1.0

    def rep_rate(text, n):
        return 1.0 - ngram_div(text, n)

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
    """Run grammar checkers on known-good Tatoeba sentences."""
    sample = random.sample(tatoeba, min(n_samples, len(tatoeba)))
    result = {"n_sampled": len(sample)}

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


# ─── bAbI Evaluation ─────────────────────────────────────────────────────────

@torch.no_grad()
def generate_babi_answer(model, tokenizer, prompt_text, max_answer_tokens,
                          device, ctx_len):
    """Feed prompt, greedily decode answer until newline or max tokens.

    With BPE, a single token may contain a newline, so we decode all
    generated tokens so far and check for newline in the decoded text.
    """
    model.eval()

    prompt_ids = tokenizer.encode(prompt_text)
    # Left-truncate if prompt exceeds ctx_len (preserves question at end)
    if len(prompt_ids) > ctx_len - 1:
        prompt_ids = prompt_ids[-(ctx_len - 1):]

    context = list(prompt_ids)
    generated_ids = []

    for _ in range(max_answer_tokens):
        ctx = context[-ctx_len:]
        x = torch.tensor([ctx], dtype=torch.long, device=device)
        logits = model(x)[0, -1, :]  # greedy: no temperature
        next_id = logits.argmax().item()

        generated_ids.append(next_id)
        context.append(next_id)

        # Decode all generated tokens so far, check for newline
        text_so_far = tokenizer.decode(generated_ids)
        if "\n" in text_so_far:
            text_so_far = text_so_far[:text_so_far.index("\n")]
            return text_so_far.strip()

    return tokenizer.decode(generated_ids).strip()


def eval_babi_accuracy(model, tokenizer, examples, device, ctx_len):
    """Evaluate exact-match accuracy on a list of bAbI examples."""
    correct = 0
    total = len(examples)
    results_per_example = []

    for ex in examples:
        predicted = generate_babi_answer(
            model, tokenizer, ex["context"],
            max_answer_tokens=20, device=device, ctx_len=ctx_len
        )
        expected = ex["answer"]
        is_correct = (predicted == expected)
        correct += int(is_correct)
        results_per_example.append({
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
            "task_id": ex["task_id"],
        })

    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "examples": results_per_example,
    }


def eval_all_babi(model, tokenizer, language, device, ctx_len,
                   task_ids=None, splits=("test_seen", "test_unseen")):
    """Run bAbI evaluation across all tasks and splits."""
    if task_ids is None:
        task_ids = ALL_BABI_TASKS

    results = {}
    all_predictions = []

    for split in splits:
        split_results = {}
        all_correct = 0
        all_total = 0

        for task_id in task_ids:
            examples = load_babi_examples(task_id, split, language)
            task_result = eval_babi_accuracy(
                model, tokenizer, examples, device, ctx_len
            )
            task_key = f"task{task_id:02d}"
            split_results[task_key] = {
                "accuracy": task_result["accuracy"],
                "correct": task_result["correct"],
                "total": task_result["total"],
            }
            all_correct += task_result["correct"]
            all_total += task_result["total"]

            for pred in task_result["examples"]:
                pred["split"] = split
                pred["task_key"] = task_key
                all_predictions.append(pred)

            print(f"    {split}/{task_key}: "
                  f"{task_result['correct']}/{task_result['total']} "
                  f"({task_result['accuracy']:.1%})")

        split_results["overall"] = {
            "accuracy": round(all_correct / all_total, 4) if all_total > 0 else 0.0,
            "correct": all_correct,
            "total": all_total,
        }
        print(f"  {split} overall: {all_correct}/{all_total} "
              f"({split_results['overall']['accuracy']:.1%})")
        results[split] = split_results

    return results, all_predictions


# ─── Single Run ──────────────────────────────────────────────────────────────

def run_single(config, train_data, val_data, prompt_data, tokenizer,
               test_ids, test_text, narrative_train_text, device,
               babi_task_ids, skip_grammar=False, skip_narrative_eval=False):
    """Execute one complete experiment run: train, evaluate bAbI + narrative."""
    run_dir = (RESULTS_DIR / config.size
               / f"{config.language}_seed{config.seed}")
    run_dir.mkdir(parents=True, exist_ok=True)

    result_path = run_dir / "result.json"
    if result_path.exists():
        print(f"\n  SKIP: {result_path} already exists")
        return None

    # Train
    model, train_info = train_model(config, train_data, val_data,
                                     tokenizer.vocab_size, device)

    result = {
        "config": asdict(config),
        "training": train_info,
    }

    # ─── bAbI Evaluation ─────────────────────────────────────────────
    print("  Evaluating bAbI reasoning tasks...")
    babi_results, babi_predictions = eval_all_babi(
        model, tokenizer, config.language, device, config.ctx_len,
        task_ids=babi_task_ids
    )
    result["babi"] = babi_results

    with open(run_dir / "babi_predictions.json", "w") as f:
        json.dump(babi_predictions, f, indent=2, ensure_ascii=False)

    # ─── Narrative Evaluation ────────────────────────────────────────
    if not skip_narrative_eval:
        # Test BPC
        print("  Computing test BPC...")
        test_bpc_info = compute_test_bpc(model, test_ids, tokenizer,
                                          config.ctx_len, device)
        print(f"    Test BPC: {test_bpc_info['test_bpc']}")
        result["test_bpc"] = test_bpc_info
        result["val_bpc"] = train_info["best_val_bpc"]

        # Generate narrative samples
        print("  Generating narrative samples...")
        random.seed(config.seed)
        samples = generate_samples(model, tokenizer, prompt_data, test_ids,
                                    config, device)
        print(f"    Generated {len(samples)} samples")

        # Memorization detection
        mem_info = tag_memorization(samples, narrative_train_text,
                                     config.lcs_threshold)
        result["memorization"] = mem_info

        # Grammar evaluation
        if not skip_grammar:
            print("  Evaluating grammar (novel samples only)...")
            if config.language == "lojban":
                grammar = eval_lojban_grammar(samples, novel_only=True)
            else:
                grammar = eval_english_grammar(samples, novel_only=True)
        else:
            grammar = {"skipped": True, "reason": "skip_grammar flag"}
        result["grammar"] = grammar

        # Structural metrics
        print("  Computing structural metrics...")
        structural = compute_structural_metrics(samples, narrative_train_text)
        result["structural"] = structural

        # Save narrative samples
        with open(run_dir / "samples.json", "w") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
    else:
        result["narrative_eval"] = "skipped"

    # Save results
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

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
        description="Lojban vs English experiment v4: "
                    "BPE tokenization + bAbI reasoning tasks")
    parser.add_argument("--size",
                        choices=["medium", "large", "all"],
                        default="all",
                        help="Model size to run (default: all)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Single seed (default: all 3)")
    parser.add_argument("--language",
                        choices=["english", "lojban", "all"],
                        default="all",
                        help="Language to run (default: all)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick sanity check: 500 steps, medium, 1 seed, "
                             "2 bAbI tasks")
    parser.add_argument("--skip-grammar", action="store_true",
                        help="Skip grammar evaluation (no Java/Node needed)")
    parser.add_argument("--skip-calibration", action="store_true",
                        help="Skip Tatoeba calibration")
    parser.add_argument("--skip-narrative-eval", action="store_true",
                        help="Skip narrative eval (BPC, generation, grammar, "
                             "structural)")
    parser.add_argument("--babi-tasks", type=int, nargs="*", default=None,
                        help="Specific bAbI task IDs to evaluate (default: all 20)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override default 10K training steps")
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

    # ─── Determine run grid ──────────────────────────────────────────

    if args.quick:
        sizes = ["medium"]
        seeds = [42]
        languages = (["english", "lojban"] if args.language == "all"
                     else [args.language])
        babi_task_ids = [1, 6]
    else:
        sizes = ([args.size] if args.size != "all"
                 else ["medium", "large"])
        seeds = [args.seed] if args.seed is not None else SEEDS
        languages = ([args.language] if args.language != "all"
                     else ["english", "lojban"])
        babi_task_ids = args.babi_tasks if args.babi_tasks else ALL_BABI_TASKS

    # ─── Load Narrative Data ─────────────────────────────────────────

    print("\nLoading training corpus (4 books)...")
    print("  English:")
    eng_narrative = load_corpus(TRAIN_BOOKS, "english")
    print("  Lojban:")
    loj_narrative = load_corpus(TRAIN_BOOKS, "lojban")

    # Truncate to equal character count
    min_len = min(len(eng_narrative), len(loj_narrative))
    eng_narrative = eng_narrative[:min_len]
    loj_narrative = loj_narrative[:min_len]
    print(f"\nNarrative truncated to {min_len:,} chars each")

    # ─── Load bAbI Training Data ─────────────────────────────────────

    print(f"\nLoading bAbI training data ({len(babi_task_ids)} tasks)...")
    eng_babi_text = load_babi_train_text("english", babi_task_ids)
    loj_babi_text = load_babi_train_text("lojban", babi_task_ids)
    print(f"  English bAbI: {len(eng_babi_text):,} chars")
    print(f"  Lojban bAbI:  {len(loj_babi_text):,} chars")

    # ─── Combine Training Data ───────────────────────────────────────

    eng_combined = eng_narrative + "\n\n" + eng_babi_text
    loj_combined = loj_narrative + "\n\n" + loj_babi_text
    print(f"\nCombined training data:")
    print(f"  English: {len(eng_combined):,} chars "
          f"(narrative {len(eng_narrative):,} + bAbI {len(eng_babi_text):,})")
    print(f"  Lojban:  {len(loj_combined):,} chars "
          f"(narrative {len(loj_narrative):,} + bAbI {len(loj_babi_text):,})")

    # ─── Train or Load BPE Tokenizers ────────────────────────────────

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    eng_tok_path = RESULTS_DIR / "tokenizer_english.json"
    loj_tok_path = RESULTS_DIR / "tokenizer_lojban.json"

    eng_tok = BPETokenizerWrapper("English")
    loj_tok = BPETokenizerWrapper("Lojban")

    if eng_tok_path.exists():
        print(f"\nLoading saved English BPE tokenizer from {eng_tok_path}")
        eng_tok.load(eng_tok_path)
    else:
        print(f"\nTraining English BPE tokenizer (vocab={BPE_VOCAB_SIZE})...")
        eng_tok.train(eng_combined, vocab_size=BPE_VOCAB_SIZE)
        eng_tok.save(eng_tok_path)
        print(f"  Saved to {eng_tok_path}")

    if loj_tok_path.exists():
        print(f"Loading saved Lojban BPE tokenizer from {loj_tok_path}")
        loj_tok.load(loj_tok_path)
    else:
        print(f"Training Lojban BPE tokenizer (vocab={BPE_VOCAB_SIZE})...")
        loj_tok.train(loj_combined, vocab_size=BPE_VOCAB_SIZE)
        loj_tok.save(loj_tok_path)
        print(f"  Saved to {loj_tok_path}")

    print(f"English BPE vocab: {eng_tok.vocab_size} | "
          f"Lojban BPE vocab: {loj_tok.vocab_size}")

    # Compute tokens-per-char ratios
    eng_tokens = eng_tok.encode(eng_combined)
    loj_tokens = loj_tok.encode(loj_combined)
    eng_tpc = len(eng_tokens) / len(eng_combined)
    loj_tpc = len(loj_tokens) / len(loj_combined)
    print(f"English tokens/char: {eng_tpc:.3f} "
          f"({len(eng_tokens):,} tokens / {len(eng_combined):,} chars)")
    print(f"Lojban tokens/char:  {loj_tpc:.3f} "
          f"({len(loj_tokens):,} tokens / {len(loj_combined):,} chars)")

    # 90/5/5 split of combined text (in token space)
    eng_train, eng_val, eng_prompt = prepare_splits(eng_combined, eng_tok)
    loj_train, loj_val, loj_prompt = prepare_splits(loj_combined, loj_tok)
    print(f"English split: {len(eng_train):,} / {len(eng_val):,} / "
          f"{len(eng_prompt):,} (train/val/prompt tokens)")
    print(f"Lojban split:  {len(loj_train):,} / {len(loj_val):,} / "
          f"{len(loj_prompt):,} (train/val/prompt tokens)")

    # Load test book (narrative only)
    print("\nLoading test book (Metamorphosis)...")
    eng_test_text = load_book_text(TEST_BOOK, "english")
    loj_test_text = load_book_text(TEST_BOOK, "lojban")

    eng_test_ids = torch.tensor(eng_tok.encode(eng_test_text), dtype=torch.long)
    loj_test_ids = torch.tensor(loj_tok.encode(loj_test_text), dtype=torch.long)
    eng_test_tpc = len(eng_test_ids) / len(eng_test_text) if eng_test_text else 0
    loj_test_tpc = len(loj_test_ids) / len(loj_test_text) if loj_test_text else 0
    print(f"  English: {len(eng_test_text):,} chars -> "
          f"{len(eng_test_ids):,} tokens "
          f"(tokens/char: {eng_test_tpc:.3f})")
    print(f"  Lojban:  {len(loj_test_text):,} chars -> "
          f"{len(loj_test_ids):,} tokens "
          f"(tokens/char: {loj_test_tpc:.3f})")

    # ─── Tatoeba Calibration ─────────────────────────────────────────

    if not args.skip_calibration and not args.quick:
        print("\nCalibrating grammar checkers on Tatoeba...")
        random.seed(42)
        tatoeba = load_tatoeba()
        print(f"  Loaded {len(tatoeba)} parallel pairs")
        calibration = calibrate_grammar_checkers(tatoeba)
        with open(RESULTS_DIR / "calibration.json", "w") as f:
            json.dump(calibration, f, indent=2)
        print(f"  Saved to {RESULTS_DIR / 'calibration.json'}")

    # ─── Save corpus info ────────────────────────────────────────────

    corpus_info = {
        "tokenizer": "bpe",
        "bpe_vocab_size_target": BPE_VOCAB_SIZE,
        "train_books": TRAIN_BOOKS,
        "test_book": TEST_BOOK,
        "narrative_chars_per_language": min_len,
        "babi_tasks_used": babi_task_ids,
        "english_babi_chars": len(eng_babi_text),
        "lojban_babi_chars": len(loj_babi_text),
        "english_combined_chars": len(eng_combined),
        "lojban_combined_chars": len(loj_combined),
        "english_bpe_vocab_size": eng_tok.vocab_size,
        "lojban_bpe_vocab_size": loj_tok.vocab_size,
        "english_tokens_per_char": round(eng_tpc, 4),
        "lojban_tokens_per_char": round(loj_tpc, 4),
        "english_train_tokens": len(eng_train),
        "english_val_tokens": len(eng_val),
        "english_prompt_tokens": len(eng_prompt),
        "lojban_train_tokens": len(loj_train),
        "lojban_val_tokens": len(loj_val),
        "lojban_prompt_tokens": len(loj_prompt),
        "english_test_chars": len(eng_test_text),
        "english_test_tokens": len(eng_test_ids),
        "lojban_test_chars": len(loj_test_text),
        "lojban_test_tokens": len(loj_test_ids),
    }
    with open(RESULTS_DIR / "corpus_info.json", "w") as f:
        json.dump(corpus_info, f, indent=2)

    # ─── Run Experiments ─────────────────────────────────────────────

    total_runs = len(sizes) * len(seeds) * len(languages)
    run_idx = 0

    for size in sizes:
        mc = MODEL_CONFIGS[size]
        for seed in seeds:
            for language in languages:
                run_idx += 1
                config = RunConfig(
                    size=size, language=language, seed=seed,
                    d_model=mc["d_model"], n_layer=mc["n_layer"],
                    n_head=mc["n_head"], dropout=mc["dropout"],
                    ctx_len=mc["ctx_len"],
                )

                if args.max_steps is not None:
                    config.max_steps = args.max_steps
                    config.patience = args.max_steps + 1

                if args.quick:
                    config.max_steps = 500
                    config.patience = 501
                    config.num_samples = 10
                    config.eval_interval = 50

                if language == "english":
                    tok = eng_tok
                    train_d = eng_train
                    val_d = eng_val
                    prompt_d = eng_prompt
                    test_ids = eng_test_ids
                    test_text = eng_test_text
                    narrative_text = eng_narrative
                    train_chars_info = {
                        "narrative": len(eng_narrative),
                        "babi": len(eng_babi_text),
                        "total": len(eng_combined),
                    }
                else:
                    tok = loj_tok
                    train_d = loj_train
                    val_d = loj_val
                    prompt_d = loj_prompt
                    test_ids = loj_test_ids
                    test_text = loj_test_text
                    narrative_text = loj_narrative
                    train_chars_info = {
                        "narrative": len(loj_narrative),
                        "babi": len(loj_babi_text),
                        "total": len(loj_combined),
                    }

                print(f"\n{'#'*60}")
                print(f"# Run {run_idx}/{total_runs}: "
                      f"{size} / {language} / seed={seed}")
                print(f"{'#'*60}")

                result = run_single(
                    config, train_d, val_d, prompt_d,
                    tok, test_ids, test_text,
                    narrative_text, device,
                    babi_task_ids=babi_task_ids,
                    skip_grammar=args.skip_grammar,
                    skip_narrative_eval=args.skip_narrative_eval,
                )

                # Inject training data size info into saved result
                if result is not None:
                    result["training"]["train_chars"] = train_chars_info
                    result_path = (RESULTS_DIR / config.size
                                   / f"{config.language}_seed{config.seed}"
                                   / "result.json")
                    with open(result_path, "w") as f:
                        json.dump(result, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"All {total_runs} runs complete in {elapsed/60:.1f} min")
    print(f"Results in {RESULTS_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
