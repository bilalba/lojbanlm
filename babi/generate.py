#!/usr/bin/env python3
"""Generate bAbI-style reasoning task data in parallel English/Lojban.

Usage:
    python3 -m babi.generate                    # all 20 tasks
    python3 -m babi.generate --tasks 1 6 9 15   # specific tasks
    python3 -m babi.generate --train-n 500      # smaller dataset
"""

import argparse
import os
import random
import sys

from .vocab import TRAIN_VOCAB, FULL_VOCAB, HELD_OUT_ENGLISH
from .tasks import TASK_GENERATORS


def has_unseen_entity(en_text):
    """Check if English text contains any held-out vocabulary word."""
    for word in HELD_OUT_ENGLISH:
        if word in en_text:
            return True
    return False


def write_examples(path, examples):
    """Write examples to file, separated by blank lines."""
    with open(path, "w") as f:
        for i, text in enumerate(examples):
            f.write(text)
            if i < len(examples) - 1:
                f.write("\n\n")
        f.write("\n")


def generate_task(task_id, task_name, gen_fn, args):
    """Generate all splits for one task."""
    out_dir = os.path.join(args.output_dir, f"task{task_id:02d}_{task_name}")
    os.makedirs(out_dir, exist_ok=True)

    total_train_val = args.train_n + args.val_n
    buffer = max(500, total_train_val)  # extra to account for dedup

    # --- Train + Val (train vocab only) ---
    rng_tv = random.Random(args.seed)
    pool = gen_fn(rng_tv, TRAIN_VOCAB, total_train_val + buffer)

    if len(pool) < total_train_val:
        print(f"  WARNING: task {task_id} generated only {len(pool)} unique "
              f"examples (need {total_train_val} for train+val)")

    train_en = [en for en, _ in pool[:args.train_n]]
    train_lj = [lj for _, lj in pool[:args.train_n]]
    val_en = [en for en, _ in pool[args.train_n:total_train_val]]
    val_lj = [lj for _, lj in pool[args.train_n:total_train_val]]

    # --- Test seen (train vocab, different seed) ---
    rng_ts = random.Random(args.seed + 10000)
    test_seen_pool = gen_fn(rng_ts, TRAIN_VOCAB, args.test_n + buffer)
    # Remove any that overlap with train or val
    train_val_set = set(train_en) | set(val_en)
    test_seen_pool = [(en, lj) for en, lj in test_seen_pool
                      if en not in train_val_set]
    test_seen_en = [en for en, _ in test_seen_pool[:args.test_n]]
    test_seen_lj = [lj for _, lj in test_seen_pool[:args.test_n]]

    # --- Test unseen (full vocab, filter for held-out entities) ---
    rng_tu = random.Random(args.seed + 20000)
    unseen_pool = gen_fn(rng_tu, FULL_VOCAB, args.test_n * 5 + buffer)
    unseen_pool = [(en, lj) for en, lj in unseen_pool
                   if has_unseen_entity(en)]
    test_unseen_en = [en for en, _ in unseen_pool[:args.test_n]]
    test_unseen_lj = [lj for _, lj in unseen_pool[:args.test_n]]

    # --- Write files ---
    write_examples(os.path.join(out_dir, "train.en.txt"), train_en)
    write_examples(os.path.join(out_dir, "train.lj.txt"), train_lj)
    write_examples(os.path.join(out_dir, "val.en.txt"), val_en)
    write_examples(os.path.join(out_dir, "val.lj.txt"), val_lj)
    write_examples(os.path.join(out_dir, "test_seen.en.txt"), test_seen_en)
    write_examples(os.path.join(out_dir, "test_seen.lj.txt"), test_seen_lj)
    write_examples(os.path.join(out_dir, "test_unseen.en.txt"), test_unseen_en)
    write_examples(os.path.join(out_dir, "test_unseen.lj.txt"), test_unseen_lj)

    counts = {
        "train": len(train_en),
        "val": len(val_en),
        "test_seen": len(test_seen_en),
        "test_unseen": len(test_unseen_en),
    }
    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Generate bAbI reasoning tasks in English and Lojban")
    parser.add_argument("--tasks", type=int, nargs="*", default=None,
                        help="Task IDs to generate (default: all)")
    parser.add_argument("--train-n", type=int, default=1000,
                        help="Training examples per task")
    parser.add_argument("--val-n", type=int, default=200,
                        help="Validation examples per task")
    parser.add_argument("--test-n", type=int, default=200,
                        help="Test examples per split per task")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: babi/data)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data")

    task_ids = args.tasks if args.tasks else sorted(TASK_GENERATORS.keys())

    print(f"Generating bAbI tasks: {task_ids}")
    print(f"  train={args.train_n}, val={args.val_n}, test={args.test_n}")
    print(f"  seed={args.seed}, output={args.output_dir}")
    print()

    summary = {}
    for tid in task_ids:
        if tid not in TASK_GENERATORS:
            print(f"  SKIP: unknown task {tid}")
            continue
        name, gen_fn = TASK_GENERATORS[tid]
        print(f"  Task {tid:2d}: {name}...", end=" ", flush=True)
        counts = generate_task(tid, name, gen_fn, args)
        summary[tid] = counts
        parts = [f"{k}={v}" for k, v in counts.items()]
        print(", ".join(parts))

    print(f"\nDone. {len(summary)} tasks generated in {args.output_dir}/")

    # Print any warnings for undersized splits
    for tid, counts in summary.items():
        for split, count in counts.items():
            expected = {"train": args.train_n, "val": args.val_n,
                        "test_seen": args.test_n,
                        "test_unseen": args.test_n}[split]
            if count < expected:
                name = TASK_GENERATORS[tid][0]
                print(f"  WARNING: task{tid:02d}_{name} {split}: "
                      f"only {count}/{expected} examples")


if __name__ == "__main__":
    main()
