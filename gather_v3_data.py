"""Gather all V3 data needed for the writeup."""
import json
import os
from collections import Counter

BASE = os.path.expanduser("~/lojban_experiment/results/v3")

def load_result(size, lang, seed):
    f = os.path.join(BASE, size, "{}_seed{}".format(lang, seed), "result.json")
    return json.load(open(f))

def load_preds(size, lang, seed):
    f = os.path.join(BASE, size, "{}_seed{}".format(lang, seed), "babi_predictions.json")
    return json.load(open(f))

SEEDS = [42, 137, 2024]
SIZES = ["nano", "micro", "mini", "small", "base"]

# 1. Full BPC table
print("=== BPC TABLE ===")
print("{:<6} {:<8} {:>8} {:>8} {:>8} {:>8}".format(
    "Size", "Lang", "ValBPC", "TestBPC", "GenGap", "BestStep"))
for size in SIZES:
    for lang in ["english", "lojban"]:
        vals, tests, steps = [], [], []
        for seed in SEEDS:
            r = load_result(size, lang, seed)
            vals.append(r["val_bpc"])
            tests.append(r["test_bpc"]["test_bpc"])
            steps.append(r["training"]["best_step"])
        v = sum(vals)/3
        t = sum(tests)/3
        print("{:<6} {:<8} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.0f}".format(
            size, lang, v, t, t - v, sum(steps)/3))
    print()

# 2. Grammar table
print("\n=== GRAMMAR TABLE ===")
for size in SIZES:
    for lang in ["english", "lojban"]:
        rates = []
        for seed in SEEDS:
            r = load_result(size, lang, seed)
            rates.append(r["grammar"]["grammaticality_rate"] * 100)
        print("{:<6} {:<8} seeds: {:.1f}%, {:.1f}%, {:.1f}%  mean={:.1f}%".format(
            size, lang, rates[0], rates[1], rates[2], sum(rates)/3))

# 3. bAbI overall accuracy table
print("\n=== bAbI OVERALL ACCURACY ===")
for size in SIZES:
    for lang in ["english", "lojban"]:
        seen, unseen = [], []
        for seed in SEEDS:
            r = load_result(size, lang, seed)
            seen.append(r["babi"]["test_seen"]["overall"]["accuracy"] * 100)
            unseen.append(r["babi"]["test_unseen"]["overall"]["accuracy"] * 100)
        print("{:<6} {:<8} seen={:.1f}% +/- {:.1f}  unseen={:.1f}% +/- {:.1f}".format(
            size, lang,
            sum(seen)/3, max(seen)-min(seen),
            sum(unseen)/3, max(unseen)-min(unseen)))

# 4. Memorization
print("\n=== MEMORIZATION ===")
for size in SIZES:
    for lang in ["english", "lojban"]:
        mems, lcss = [], []
        for seed in SEEDS:
            r = load_result(size, lang, seed)
            m = r["memorization"]
            mems.append(m["n_memorized"])
            lcss.append(m["avg_lcs"])
        print("{:<6} {:<8} memorized: {},{},{}  avg_lcs: {:.0f},{:.0f},{:.0f}".format(
            size, lang, mems[0], mems[1], mems[2], lcss[0], lcss[1], lcss[2]))

# 5. Structural metrics (base only, for brevity)
print("\n=== STRUCTURAL METRICS (base, avg across seeds) ===")
for lang in ["english", "lojban"]:
    all_metrics = {}
    for seed in SEEDS:
        r = load_result("base", lang, seed)
        s = r["structural"]
        for k, v in s.items():
            if k not in all_metrics:
                all_metrics[k] = []
            all_metrics[k].append(v)
    print("  {}:".format(lang))
    for k in sorted(all_metrics):
        vals = all_metrics[k]
        print("    {}: {:.4f}".format(k, sum(vals)/3))

# 6. Training dynamics
print("\n=== TRAINING DYNAMICS ===")
for size in SIZES:
    for lang in ["english", "lojban"]:
        best_steps, total_steps, times, early = [], [], [], []
        for seed in SEEDS:
            r = load_result(size, lang, seed)
            t = r["training"]
            best_steps.append(t["best_step"])
            total_steps.append(t["total_steps"])
            times.append(t["total_time_s"])
            early.append(t["early_stopped"])
        print("{:<6} {:<8} best={:.0f} total={:.0f} time={:.0f}s early={}".format(
            size, lang,
            sum(best_steps)/3, sum(total_steps)/3, sum(times)/3,
            sum(early)))

# 7. Per-task bAbI (base, avg across seeds)
print("\n=== PER-TASK bAbI (base, mean across seeds, test_seen) ===")
print("{:>8} {:>8} {:>8}".format("Task", "English", "Lojban"))
for task_id in range(1, 21):
    task_key = "task{:02d}".format(task_id)
    en = sum(load_result("base", "english", s)["babi"]["test_seen"][task_key]["accuracy"]
             for s in SEEDS) / 3 * 100
    lj = sum(load_result("base", "lojban", s)["babi"]["test_seen"][task_key]["accuracy"]
             for s in SEEDS) / 3 * 100
    print("{:>8} {:>7.1f}% {:>7.1f}%".format(task_key, en, lj))

# 8. Mode collapse stats (base)
print("\n=== MODE COLLAPSE (base, task01, all seeds) ===")
for lang in ["english", "lojban"]:
    for seed in SEEDS:
        preds = load_preds("base", lang, seed)
        t1 = [p for p in preds if p["task_id"] == 1 and p["split"] == "test_seen"]
        counts = Counter(p["predicted"] for p in t1)
        top = counts.most_common(1)[0]
        print("{:<8} seed={}: {} unique, top={} ({:.0f}%)".format(
            lang, seed, len(counts), repr(top[0]), top[1]/len(t1)*100))
