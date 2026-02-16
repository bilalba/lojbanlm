"""Additional bAbI analysis — training dynamics and English scaling."""
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


# ===== 9. English scaling tasks — which tasks does English actually learn? =====

print("=" * 70)
print("9. ENGLISH SCALING — tasks that improve nano -> base")
print("=" * 70)

tasks_that_scale = []
tasks_flat = []
for task_id in range(1, 21):
    task_key = "task{:02d}".format(task_id)
    nano_acc = sum(load_result("nano", "english", s)["babi"]["test_seen"][task_key]["accuracy"]
                   for s in [42, 137, 2024]) / 3 * 100
    base_acc = sum(load_result("base", "english", s)["babi"]["test_seen"][task_key]["accuracy"]
                   for s in [42, 137, 2024]) / 3 * 100
    delta = base_acc - nano_acc
    if delta > 5:
        tasks_that_scale.append((task_id, nano_acc, base_acc, delta))
    else:
        tasks_flat.append((task_id, nano_acc, base_acc, delta))

print("\n  Tasks that IMPROVE with scale (>5pp gain):")
for t, n, b, d in sorted(tasks_that_scale, key=lambda x: -x[3]):
    print("    task{:02d}: {:.1f}% -> {:.1f}% (+{:.1f}pp)".format(t, n, b, d))

print("\n  Tasks FLAT with scale:")
for t, n, b, d in sorted(tasks_flat, key=lambda x: -x[3]):
    print("    task{:02d}: {:.1f}% -> {:.1f}% ({:+.1f}pp)".format(t, n, b, d))


# ===== 10. Does English mode-collapse too at nano? =====

print("\n" + "=" * 70)
print("10. ENGLISH MODE COLLAPSE CHECK — nano vs base (seed42)")
print("=" * 70)

for task_id in [1, 11, 13, 16]:
    task_key = "task{:02d}".format(task_id)
    print("\n  {}:".format(task_key))
    for size in ["nano", "base"]:
        preds = load_preds(size, "english", 42)
        task_preds = [p for p in preds if p["task_id"] == task_id and p["split"] == "test_seen"]
        pred_counts = Counter(p["predicted"] for p in task_preds)
        correct = sum(1 for p in task_preds if p["correct"])
        top3 = pred_counts.most_common(3)
        top_str = ", ".join("{} ({:.0f}%)".format(repr(v), c/len(task_preds)*100) for v, c in top3)
        print("    EN {}: {}/{} | {} unique | {}".format(
            size, correct, len(task_preds), len(pred_counts), top_str))


# ===== 11. Answer base rates — what's chance-level for each task? =====

print("\n" + "=" * 70)
print("11. CHANCE-LEVEL BASELINES (if always predicting most common answer)")
print("=" * 70)

BABI_DATA = os.path.expanduser("~/lojban_experiment/babi/data")
for task_dir in sorted(os.listdir(BABI_DATA)):
    if not task_dir.startswith("task"):
        continue
    task_num = task_dir.split("_")[0]
    f = os.path.join(BABI_DATA, task_dir, "test_seen.en.txt")
    if not os.path.exists(f):
        continue
    answers = []
    for line in open(f):
        line = line.strip()
        if "?" in line:
            ans = line.split("?")[-1].strip()
            if ans:
                answers.append(ans)
    if not answers:
        continue
    c = Counter(answers)
    majority = c.most_common(1)[0][1]
    n_unique = len(c)
    uniform_chance = 100.0 / n_unique
    majority_chance = majority / len(answers) * 100
    print("  {}: {} answers, {} unique, uniform={:.1f}%, majority={:.1f}%".format(
        task_num, len(answers), n_unique, uniform_chance, majority_chance))


# ===== 12. Lojban yes/no bias — is it always predicting "na go'i"? =====

print("\n" + "=" * 70)
print("12. YES/NO BIAS PATTERN (tasks 6, 9, 17, 18)")
print("    Checking if models just learn the majority class")
print("=" * 70)

for task_id in [6, 9, 17, 18]:
    task_key = "task{:02d}".format(task_id)
    print("\n  {}:".format(task_key))

    # Check actual yes/no distribution in test data
    task_dirs = [d for d in os.listdir(BABI_DATA) if d.startswith(task_key)]
    if task_dirs:
        for lang_ext, label in [("en", "EN"), ("lj", "LJ")]:
            f = os.path.join(BABI_DATA, task_dirs[0], "test_seen.{}.txt".format(lang_ext))
            if not os.path.exists(f):
                continue
            answers = []
            for line in open(f):
                line = line.strip()
                if "?" in line:
                    ans = line.split("?")[-1].strip()
                    if ans:
                        answers.append(ans)
            c = Counter(answers)
            dist = ", ".join("{}: {:.0f}%".format(repr(a), cnt/len(answers)*100) for a, cnt in c.most_common())
            print("    {} test distribution: {}".format(label, dist))

    # Check what models predict
    for lang in ["english", "lojban"]:
        preds = load_preds("base", lang, 42)
        task_preds = [p for p in preds if p["task_id"] == task_id and p["split"] == "test_seen"]
        pred_counts = Counter(p["predicted"] for p in task_preds)
        correct = sum(1 for p in task_preds if p["correct"])
        dist = ", ".join("{}: {:.0f}%".format(repr(a), cnt/len(task_preds)*100)
                         for a, cnt in pred_counts.most_common())
        print("    {} base/42 predicts: {} ({}/{} correct)".format(lang, dist, correct, len(task_preds)))
