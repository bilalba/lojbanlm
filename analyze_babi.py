"""Deep analysis of bAbI prediction behavior in V3 results."""
import json
import os
from collections import Counter

BASE = os.path.expanduser("~/lojban_experiment/results/v3")
BABI_DATA = os.path.expanduser("~/lojban_experiment/babi/data")

def load_preds(size, lang, seed):
    f = os.path.join(BASE, size, "{}_seed{}".format(lang, seed), "babi_predictions.json")
    return json.load(open(f))

def load_result(size, lang, seed):
    f = os.path.join(BASE, size, "{}_seed{}".format(lang, seed), "result.json")
    return json.load(open(f))

# ===== 1. Tasks where Lojban answers are 1 word (task07, task16) =====
# If the multi-word hypothesis is right, these should NOT show mode collapse

print("=" * 70)
print("1. SINGLE-WORD LOJBAN ANSWERS (task07, task16)")
print("   If multi-word answers cause collapse, these should be fine")
print("=" * 70)

for task_id, task_name in [(7, "task07"), (16, "task16")]:
    print("\n  --- {} ---".format(task_name))
    # Show what the answers look like
    for lang_ext, label in [("en", "EN"), ("lj", "LJ")]:
        task_dirs = [d for d in os.listdir(BABI_DATA) if d.startswith(task_name)]
        if task_dirs:
            f = os.path.join(BABI_DATA, task_dirs[0], "train.{}.txt".format(lang_ext))
            answers = set()
            for line in open(f):
                line = line.strip()
                if "?" in line:
                    ans = line.split("?")[-1].strip()
                    if ans:
                        answers.add(ans)
            print("  {} answers: {}".format(label, sorted(answers)))

    print()
    for lang in ["english", "lojban"]:
        preds = load_preds("base", lang, 42)
        task_preds = [p for p in preds if p["task_id"] == task_id and p["split"] == "test_seen"]
        pred_counts = Counter(p["predicted"] for p in task_preds)
        correct = sum(1 for p in task_preds if p["correct"])
        print("  {} base/42: {}/{} correct, {} unique preds".format(
            lang, correct, len(task_preds), len(pred_counts)))
        for ans, cnt in pred_counts.most_common(5):
            print("    {}: {} ({:.0f}%)".format(repr(ans), cnt, cnt/len(task_preds)*100))


# ===== 2. Strip "lo " prefix — does Lojban get the content word right? =====

print("\n" + "=" * 70)
print("2. CONTENT-WORD ACCURACY (strip 'lo ' prefix from Lojban)")
print("   Does the model know the right answer but fail on format?")
print("=" * 70)

for size in ["nano", "base"]:
    print("\n  --- {} ---".format(size))
    for seed in [42]:
        preds = load_preds(size, "lojban", seed)
        for task_id in [1, 2, 6, 15]:
            task_preds = [p for p in preds if p["task_id"] == task_id and p["split"] == "test_seen"]

            exact_correct = sum(1 for p in task_preds if p["correct"])

            # Strip "lo " from both expected and predicted, then compare
            def strip_lo(s):
                if s.startswith("lo "):
                    return s[3:]
                return s

            content_correct = sum(1 for p in task_preds
                                  if strip_lo(p["predicted"]) == strip_lo(p["expected"]))

            # Also check: does predicted even start with "lo "?
            starts_lo = sum(1 for p in task_preds if p["predicted"].startswith("lo "))

            print("  task{:02d} lojban {}/seed{}: exact={}/{}, content_word={}/{}, starts_lo={}/{}".format(
                task_id, size, seed,
                exact_correct, len(task_preds),
                content_correct, len(task_preds),
                starts_lo, len(task_preds)))


# ===== 3. Per-task accuracy comparison (averaged across seeds) =====

print("\n" + "=" * 70)
print("3. PER-TASK ACCURACY BY SIZE (avg across 3 seeds, test_seen)")
print("=" * 70)

for size in ["nano", "micro", "mini", "small", "base"]:
    print("\n  --- {} ---".format(size))
    print("  {:>8} {:>8} {:>8}  {:>8} {:>8} {:>8}".format(
        "Task", "EN_acc", "LJ_acc", "Task", "EN_acc", "LJ_acc"))

    rows = []
    for task_id in range(1, 21):
        en_accs = []
        lj_accs = []
        for seed in [42, 137, 2024]:
            en_r = load_result(size, "english", seed)
            lj_r = load_result(size, "lojban", seed)
            task_key = "task{:02d}".format(task_id)
            en_accs.append(en_r["babi"]["test_seen"][task_key]["accuracy"] * 100)
            lj_accs.append(lj_r["babi"]["test_seen"][task_key]["accuracy"] * 100)
        rows.append((task_id, sum(en_accs)/3, sum(lj_accs)/3))

    # Print in two columns
    for i in range(10):
        t1, en1, lj1 = rows[i]
        t2, en2, lj2 = rows[i + 10]
        marker1 = "*" if lj1 > en1 + 2 else (" " if abs(lj1 - en1) <= 2 else "")
        marker2 = "*" if lj2 > en2 + 2 else (" " if abs(lj2 - en2) <= 2 else "")
        print("  task{:02d}  {:>6.1f}% {:>6.1f}%{} | task{:02d}  {:>6.1f}% {:>6.1f}%{}".format(
            t1, en1, lj1, marker1, t2, en2, lj2, marker2))


# ===== 4. The small/seed42 outlier — what happened? =====

print("\n" + "=" * 70)
print("4. SMALL/SEED42 OUTLIER — Lojban scored 35% (vs ~20% elsewhere)")
print("=" * 70)

# Compare prediction diversity across all 3 seeds for small
for seed in [42, 137, 2024]:
    preds = load_preds("small", "lojban", seed)
    print("\n  small lojban seed={}:".format(seed))
    for task_id in [1, 2, 6]:
        task_preds = [p for p in preds if p["task_id"] == task_id and p["split"] == "test_seen"]
        pred_counts = Counter(p["predicted"] for p in task_preds)
        correct = sum(1 for p in task_preds if p["correct"])
        top = pred_counts.most_common(3)
        top_str = ", ".join("{} ({:.0f}%)".format(repr(v), c/len(task_preds)*100) for v, c in top)
        print("    task{:02d}: {}/{} correct | {} unique | top: {}".format(
            task_id, correct, len(task_preds), len(pred_counts), top_str))


# ===== 5. What is the model actually generating? Show raw predictions =====

print("\n" + "=" * 70)
print("5. RAW PREDICTIONS — English vs Lojban (base, seed42, task01)")
print("   First 15 test_seen examples side by side")
print("=" * 70)

en_preds = load_preds("base", "english", 42)
lj_preds = load_preds("base", "lojban", 42)
en_t1 = [p for p in en_preds if p["task_id"] == 1 and p["split"] == "test_seen"][:15]
lj_t1 = [p for p in lj_preds if p["task_id"] == 1 and p["split"] == "test_seen"][:15]

print("  {:>3} {:>15} {:>15} {:>2}  |  {:>20} {:>20} {:>2}".format(
    "#", "EN_expected", "EN_predicted", "", "LJ_expected", "LJ_predicted", ""))
for i, (en, lj) in enumerate(zip(en_t1, lj_t1)):
    en_mark = "OK" if en["correct"] else "XX"
    lj_mark = "OK" if lj["correct"] else "XX"
    print("  {:>3} {:>15} {:>15} {:>2}  |  {:>20} {:>20} {:>2}".format(
        i+1, en["expected"], en["predicted"], en_mark,
        lj["expected"], lj["predicted"], lj_mark))


# ===== 6. Yes/No tasks (6, 9, 17, 18) — shorter answers, closer comparison =====

print("\n" + "=" * 70)
print("6. YES/NO TASKS (6, 9, 17, 18) — closest to equal answer length")
print("   EN: yes/no (1 word)  LJ: go'i/na go'i (1-2 words)")
print("=" * 70)

for task_id in [6, 9, 17, 18]:
    print("\n  task{:02d}:".format(task_id))
    for size in ["nano", "micro", "mini", "small", "base"]:
        en_accs = []
        lj_accs = []
        for seed in [42, 137, 2024]:
            en_r = load_result(size, "english", seed)
            lj_r = load_result(size, "lojban", seed)
            task_key = "task{:02d}".format(task_id)
            en_accs.append(en_r["babi"]["test_seen"][task_key]["accuracy"] * 100)
            lj_accs.append(lj_r["babi"]["test_seen"][task_key]["accuracy"] * 100)
        print("    {}: EN {:.1f}% vs LJ {:.1f}%".format(size, sum(en_accs)/3, sum(lj_accs)/3))


# ===== 7. Training convergence — did bAbI loss even go down? =====

print("\n" + "=" * 70)
print("7. TRAINING INFO — steps, best val BPC")
print("=" * 70)

for size in ["nano", "micro", "mini", "small", "base"]:
    print("\n  {}:".format(size))
    for lang in ["english", "lojban"]:
        vals = []
        steps = []
        for seed in [42, 137, 2024]:
            r = load_result(size, lang, seed)
            vals.append(r["val_bpc"])
            steps.append(r["training"]["best_step"])
        print("    {}: val_bpc={:.3f} +/- {:.3f}  best_step={:.0f} +/- {:.0f}".format(
            lang,
            sum(vals)/3,
            (sum((v - sum(vals)/3)**2 for v in vals) / 3) ** 0.5,
            sum(steps)/3,
            (sum((s - sum(steps)/3)**2 for s in steps) / 3) ** 0.5,
        ))


# ===== 8. Unseen vocab — does the gap widen? =====

print("\n" + "=" * 70)
print("8. SEEN vs UNSEEN accuracy (base, avg across seeds)")
print("   If model learned reasoning, unseen should not drop much")
print("=" * 70)

for task_id in [1, 2, 6, 7, 15, 16]:
    task_key = "task{:02d}".format(task_id)
    print("\n  {}:".format(task_key))
    for lang in ["english", "lojban"]:
        seen_accs = []
        unseen_accs = []
        for seed in [42, 137, 2024]:
            r = load_result("base", lang, seed)
            seen_accs.append(r["babi"]["test_seen"][task_key]["accuracy"] * 100)
            unseen_accs.append(r["babi"]["test_unseen"][task_key]["accuracy"] * 100)
        seen_avg = sum(seen_accs) / 3
        unseen_avg = sum(unseen_accs) / 3
        drop = seen_avg - unseen_avg
        print("    {}: seen={:.1f}%  unseen={:.1f}%  drop={:.1f}pp".format(
            lang, seen_avg, unseen_avg, drop))
