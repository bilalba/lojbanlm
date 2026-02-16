"""Check answer token lengths for English vs Lojban bAbI tasks."""
import os

base = os.path.expanduser("~/lojban_experiment/babi/data")

for task_dir in sorted(os.listdir(base)):
    if not task_dir.startswith("task"):
        continue
    task_num = task_dir.split("_")[0]

    for lang_ext, label in [("en", "EN"), ("lj", "LJ")]:
        f = os.path.join(base, task_dir, "train.{}.txt".format(lang_ext))
        if not os.path.exists(f):
            continue
        answers = set()
        for line in open(f):
            line = line.strip()
            if "?" in line:
                ans = line.split("?")[-1].strip()
                if ans:
                    answers.add(ans)

        avg_words = sum(len(a.split()) for a in answers) / max(len(answers), 1)
        avg_chars = sum(len(a) for a in answers) / max(len(answers), 1)
        max_words = max(len(a.split()) for a in answers) if answers else 0
        print("{} {} | {} unique answers | avg {:.1f} words, {:.1f} chars | max {} words".format(
            task_num, label, len(answers), avg_words, avg_chars, max_words))
    print()
