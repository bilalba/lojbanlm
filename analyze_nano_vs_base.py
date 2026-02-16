"""Compare nano vs base Lojban — nano trained 7500 steps, base only 1300."""
import json
import os
from collections import Counter

BASE = os.path.expanduser("~/lojban_experiment/results/v3")

def load_preds(size, lang, seed):
    f = os.path.join(BASE, size, "{}_seed{}".format(lang, seed), "babi_predictions.json")
    return json.load(open(f))

def load_result(size, lang, seed):
    f = os.path.join(BASE, size, "{}_seed{}".format(lang, seed), "result.json")
    return json.load(open(f))

print("=" * 70)
print("NANO vs BASE LOJBAN — did more training help nano?")
print("  nano trained ~7200 steps avg, base only ~1100 steps avg")
print("=" * 70)

for task_id in [1, 2, 7, 16]:
    task_key = "task{:02d}".format(task_id)
    print("\n  {}:".format(task_key))
    for size in ["nano", "base"]:
        accs = []
        diversities = []
        for seed in [42, 137, 2024]:
            r = load_result(size, "lojban", seed)
            accs.append(r["babi"]["test_seen"][task_key]["accuracy"] * 100)

            preds = load_preds(size, "lojban", seed)
            task_preds = [p for p in preds if p["task_id"] == task_id and p["split"] == "test_seen"]
            diversities.append(len(Counter(p["predicted"] for p in task_preds)))

        print("    {} lojban: acc={:.1f}% +/- {:.1f}, diversity={:.1f} unique".format(
            size,
            sum(accs)/3,
            max(accs) - min(accs),
            sum(diversities)/3))


# Now the real question: does the nano Lojban model show ANY more discrimination?
print("\n" + "=" * 70)
print("DETAILED: nano vs base lojban, task01, seed42 — prediction distribution")
print("=" * 70)

for size in ["nano", "base"]:
    preds = load_preds(size, "lojban", 42)
    task_preds = [p for p in preds if p["task_id"] == 1 and p["split"] == "test_seen"]
    pred_counts = Counter(p["predicted"] for p in task_preds)
    expected_counts = Counter(p["expected"] for p in task_preds)

    print("\n  {} lojban/42 task01:".format(size))
    print("    Expected distribution:")
    for ans, cnt in expected_counts.most_common():
        print("      {}: {}".format(repr(ans), cnt))
    print("    Predicted distribution:")
    for ans, cnt in pred_counts.most_common():
        print("      {}: {}".format(repr(ans), cnt))


# Check: what's happening to val loss AFTER best_step? Is it going UP?
# This would confirm overfitting
print("\n" + "=" * 70)
print("VAL LOSS AFTER BEST STEP — overfitting check (base lojban seed42)")
print("  If val loss rises after best_step, model is overfitting")
print("=" * 70)

r = load_result("base", "lojban", 42)
log = r["training"]["log"]
best_step = r["training"]["best_step"]
best_val = None

for entry in log:
    if entry["step"] == best_step:
        best_val = entry["val_bpc"]
    if entry["step"] >= best_step - 200:
        marker = " <-- BEST" if entry["step"] == best_step else ""
        print("  step {:>5}: val_bpc={:.4f} train_bpc={:.3f}{}".format(
            entry["step"], entry["val_bpc"], entry["train_bpc"], marker))


# ===== The real smoking gun: compare base EN and base LJ at equivalent training steps =====
# Base EN at step 1300 had val_bpc ~1.5 — it was still terrible too
print("\n" + "=" * 70)
print("ENGLISH at step 1300 (same point where Lojban stopped)")
print("=" * 70)

r = load_result("base", "english", 42)
log = r["training"]["log"]
for entry in log:
    if entry["step"] <= 1500:
        print("  step {:>5}: val_bpc={:.3f} train_bpc={:.3f}".format(
            entry["step"], entry["val_bpc"], entry["train_bpc"]))
print("\n  At step 1300, English val_bpc was still ~1.4-1.5")
print("  English kept improving to step 5400 (val_bpc 0.942)")
print("  Lojban STOPPED at 1300 (val_bpc 0.958) and never improved further")
