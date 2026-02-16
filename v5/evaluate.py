"""Evaluation: bAbI accuracy, test BPC, generation, grammar, structural, memorization."""

import math
import random
import re
import subprocess
import time
from collections import Counter

import torch
import torch.nn.functional as F

from .config import BASE_DIR, ALL_BABI_TASKS
from .data import load_babi_examples


# ─── bAbI Evaluation ────────────────────────────────────────────────────────

@torch.no_grad()
def generate_babi_answer(model, tokenizer, prompt_text, max_answer_tokens,
                          device, ctx_len):
    """Feed prompt, greedily decode answer until newline or max tokens."""
    model.eval()

    prompt_ids = tokenizer.encode(prompt_text)
    if len(prompt_ids) > ctx_len - 1:
        prompt_ids = prompt_ids[-(ctx_len - 1):]

    context = list(prompt_ids)
    generated_ids = []

    for _ in range(max_answer_tokens):
        ctx = context[-ctx_len:]
        x = torch.tensor([ctx], dtype=torch.long, device=device)
        logits = model(x)[0, -1, :]
        next_id = logits.argmax().item()

        generated_ids.append(next_id)
        context.append(next_id)

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


# ─── Test BPC ───────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_test_bpc(model, test_ids, tokenizer, ctx_len, device):
    """Compute bits-per-character on test data, normalizing by actual chars."""
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
        y_ids = test_ids[start + 1:start + ctx_len + 1].tolist()
        chars_in_window = len(tokenizer.decode(y_ids))
        total_chars += chars_in_window

    avg_loss = total_loss / total_chars if total_chars > 0 else 0.0
    bpc = avg_loss / math.log(2)
    return {
        "test_bpc": round(bpc, 4),
        "test_chars_evaluated": total_chars,
    }


# ─── Sample Generation ─────────────────────────────────────────────────────

@torch.no_grad()
def generate_samples(model, tokenizer, prompt_data, test_ids,
                     num_samples, prompt_len, gen_len,
                     temperature, top_k, device):
    """Generate text samples from in-domain and out-of-domain prompts."""
    model.eval()
    samples = []
    n_in = int(num_samples * 0.8)

    for i in range(num_samples):
        if i < n_in:
            source = "in_domain"
            data = prompt_data
        else:
            source = "out_of_domain"
            data = test_ids

        if len(data) < prompt_len + 1:
            continue

        start = random.randint(0, len(data) - prompt_len - 1)
        prompt_ids = data[start:start + prompt_len].tolist()
        prompt_text = tokenizer.decode(prompt_ids)

        context = list(prompt_ids)
        generated_ids = []
        for _ in range(gen_len):
            ctx = context[-model.ctx_len:]
            x = torch.tensor([ctx], dtype=torch.long, device=device)
            logits = model(x)[0, -1, :] / temperature

            if top_k > 0:
                k = min(top_k, logits.size(-1))
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
    """Character KL, n-gram diversity, repetition, word-length similarity."""
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
