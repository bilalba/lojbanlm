"""Data loading, splitting, and batch sampling for V5 experiment."""

import torch

from .config import (
    CORPUS_DIR, BABI_DIR, BOOK_FILES, FINEWEB_FILES,
    TRAIN_BOOKS, BABI_TASK_NAMES,
)


# ─── Book / Corpus Loading ──────────────────────────────────────────────────

def load_book_text(book, language):
    """Load a single book's text."""
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


def load_fineweb_parallel(language):
    """Load FineWeb parallel data for one language."""
    rel_path = FINEWEB_FILES[language]
    path = CORPUS_DIR / rel_path
    text = path.read_text(encoding="utf-8")
    return text.strip()


def load_pretraining_corpus(language):
    """Load all pretraining text: books + FineWeb parallel.

    Returns the raw concatenated text (not yet truncated).
    """
    texts = []
    for book in TRAIN_BOOKS:
        text = load_book_text(book, language)
        texts.append(text)
        print(f"    {book}: {len(text):,} chars")

    fineweb = load_fineweb_parallel(language)
    texts.append(fineweb)
    print(f"    fineweb_parallel: {len(fineweb):,} chars")

    return "\n\n".join(texts)


def load_corpus(books, language):
    """Load and concatenate text from a list of books."""
    texts = []
    for book in books:
        text = load_book_text(book, language)
        texts.append(text)
        print(f"    {book}: {len(text):,} chars")
    return "\n\n".join(texts)


# ─── Tatoeba ────────────────────────────────────────────────────────────────

def load_tatoeba():
    """Load Tatoeba parallel sentences for grammar calibration."""
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


# ─── bAbI Data ──────────────────────────────────────────────────────────────

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
        question_with_marker = last_line[:q_idx + 1]
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


# ─── Splitting & Batching ───────────────────────────────────────────────────

def prepare_splits(text, tokenizer, ratios=(0.90, 0.05, 0.05)):
    """Split tokenized text into train/val/prompt (or train/val for 2-way)."""
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = len(data)
    if len(ratios) == 3:
        t1 = int(n * ratios[0])
        t2 = int(n * (ratios[0] + ratios[1]))
        return data[:t1], data[t1:t2], data[t2:]
    elif len(ratios) == 2:
        t1 = int(n * ratios[0])
        return data[:t1], data[t1:]
    else:
        raise ValueError(f"Expected 2 or 3 ratios, got {len(ratios)}")


def get_batch(data, batch_size, ctx_len, device):
    """Sample a random batch of (input, target) sequences."""
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
