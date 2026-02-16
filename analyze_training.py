"""Analyze how much bAbI data each model actually trained on before early stopping."""
import json
import os
import math

BASE = os.path.expanduser("~/lojban_experiment/results/v3")

def load_result(size, lang, seed):
    f = os.path.join(BASE, size, "{}_seed{}".format(lang, seed), "result.json")
    return json.load(open(f))

# Load corpus info (flat keys, not nested)
corpus = json.load(open(os.path.join(BASE, "corpus_info.json")))

narrative_chars = corpus["narrative_chars_per_language"]  # 407194

DATA = {
    "english": {
        "babi_chars": corpus["english_babi_chars"],
        "combined_chars": corpus["english_combined_chars"],
        "train_tokens": corpus["english_train_tokens"],
        "val_tokens": corpus["english_val_tokens"],
    },
    "lojban": {
        "babi_chars": corpus["lojban_babi_chars"],
        "combined_chars": corpus["lojban_combined_chars"],
        "train_tokens": corpus["lojban_train_tokens"],
        "val_tokens": corpus["lojban_val_tokens"],
    },
}

print("=" * 70)
print("TRAINING DATA COMPOSITION")
print("=" * 70)

for lang in ["english", "lojban"]:
    info = DATA[lang]
    total = info["combined_chars"]
    babi = info["babi_chars"]
    print("\n  {}:".format(lang))
    print("    narrative: {:,} chars ({:.1f}%)".format(narrative_chars, narrative_chars/total*100))
    print("    bAbI:      {:,} chars ({:.1f}%)".format(babi, babi/total*100))
    print("    total:     {:,} chars".format(total))
    print("    train split (90%): {:,} tokens".format(info["train_tokens"]))

# ===== Per-size training exposure =====

print("\n" + "=" * 70)
print("TRAINING EXPOSURE — chars seen before early stop")
print("=" * 70)

SIZES = {
    "nano":  {"ctx_len": 128, "batch_size": 128},
    "micro": {"ctx_len": 128, "batch_size": 128},
    "mini":  {"ctx_len": 256, "batch_size": 64},
    "small": {"ctx_len": 256, "batch_size": 64},
    "base":  {"ctx_len": 256, "batch_size": 64},
}

for size in ["nano", "micro", "mini", "small", "base"]:
    ctx_len = SIZES[size]["ctx_len"]
    batch_size = SIZES[size]["batch_size"]
    chars_per_step = ctx_len * batch_size

    print("\n  --- {} (ctx={}, batch={}, chars/step={:,}) ---".format(
        size, ctx_len, batch_size, chars_per_step))

    print("  {:>10} {:>5} {:>8} {:>10} {:>12} {:>8} {:>8}".format(
        "Lang", "Seed", "BestStep", "TotalStep", "CharsSeenM", "Epochs", "EarlySt"))

    for lang in ["english", "lojban"]:
        train_tokens = DATA[lang]["train_tokens"]
        for seed in [42, 137, 2024]:
            r = load_result(size, lang, seed)
            t = r["training"]
            best_step = t["best_step"]
            total_steps = t["total_steps"]
            early = t["early_stopped"]
            chars_seen = total_steps * chars_per_step
            epochs = chars_seen / train_tokens

            print("  {:>10} {:>5} {:>8} {:>10} {:>11.1f}M {:>7.1f}x {:>8}".format(
                lang, seed, best_step, total_steps,
                chars_seen / 1e6, epochs,
                "Y" if early else "N"))

# ===== Key comparison: bAbI exposure =====

print("\n" + "=" * 70)
print("ESTIMATED bAbI EXPOSURE AT BEST STEP")
print("  bAbI is ~85-87% of training data, so ~85-87% of random chunks are bAbI")
print("=" * 70)

print("\n  {:>6} {:>10} {:>10} {:>10} {:>10} {:>8}".format(
    "Size", "Lang", "AvgBest", "TotalChrs", "bAbIChrs", "bAbIEpochs"))
print("  " + "-" * 62)

for size in ["nano", "micro", "mini", "small", "base"]:
    ctx_len = SIZES[size]["ctx_len"]
    batch_size = SIZES[size]["batch_size"]
    chars_per_step = ctx_len * batch_size

    for lang in ["english", "lojban"]:
        info = DATA[lang]
        babi_frac = info["babi_chars"] / info["combined_chars"]
        babi_train = info["babi_chars"] * 0.9  # 90% split

        best_steps = []
        for seed in [42, 137, 2024]:
            r = load_result(size, lang, seed)
            best_steps.append(r["training"]["best_step"])

        avg_best = sum(best_steps) / 3
        chars_at_best = avg_best * chars_per_step
        babi_chars_at_best = chars_at_best * babi_frac
        babi_epochs = babi_chars_at_best / babi_train

        print("  {:>6} {:>10} {:>10.0f} {:>9.1f}M {:>9.1f}M {:>8.1f}x".format(
            size, lang, avg_best, chars_at_best/1e6,
            babi_chars_at_best/1e6, babi_epochs))
    print()


# ===== Training logs — loss trajectory =====

print("=" * 70)
print("LOSS TRAJECTORY (base, seed42)")
print("=" * 70)

for lang in ["english", "lojban"]:
    r = load_result("base", lang, 42)
    log = r["training"]["log"]
    print("\n  {} base/42 ({} log entries):".format(lang, len(log)))
    for entry in log[:8]:
        print("    step {:>5}: train_bpc={:.3f} val_bpc={:.3f}".format(
            entry["step"], entry["train_bpc"], entry["val_bpc"]))
    print("    ...")
    for entry in log[-5:]:
        print("    step {:>5}: train_bpc={:.3f} val_bpc={:.3f}".format(
            entry["step"], entry["train_bpc"], entry["val_bpc"]))
    print("    best_step={}, total_steps={}, early_stopped={}".format(
        r["training"]["best_step"], r["training"]["total_steps"],
        r["training"]["early_stopped"]))


# ===== Ratio: English trains Nx longer than Lojban =====

print("\n" + "=" * 70)
print("TRAINING DURATION RATIO (English / Lojban)")
print("=" * 70)

for size in ["nano", "micro", "mini", "small", "base"]:
    en_steps = [load_result(size, "english", s)["training"]["best_step"] for s in [42, 137, 2024]]
    lj_steps = [load_result(size, "lojban", s)["training"]["best_step"] for s in [42, 137, 2024]]
    en_avg = sum(en_steps) / 3
    lj_avg = sum(lj_steps) / 3
    ratio = en_avg / lj_avg if lj_avg > 0 else float("inf")
    print("  {}: EN avg {:.0f} steps / LJ avg {:.0f} steps = {:.1f}x longer".format(
        size, en_avg, lj_avg, ratio))


# ===== Full loss log for all sizes (Lojban) to see how fast it converges =====

print("\n" + "=" * 70)
print("LOJBAN CONVERGENCE SPEED — full log (seed42)")
print("=" * 70)

for size in ["nano", "base"]:
    r = load_result(size, "lojban", 42)
    log = r["training"]["log"]
    print("\n  {} lojban/42:".format(size))
    for entry in log:
        print("    step {:>5}: train_bpc={:.3f} val_bpc={:.3f}".format(
            entry["step"], entry["train_bpc"], entry["val_bpc"]))
    print("    best_step={}, total_steps={}".format(
        r["training"]["best_step"], r["training"]["total_steps"]))
