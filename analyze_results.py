#!/usr/bin/env python3
"""
Analyze results from experiment_v2.py runs.

Loads all result JSONs, computes mean/std across seeds, runs paired t-tests,
reports Cohen's d effect sizes, prints summary tables, and shows scaling trends.

Usage:
    python3 analyze_results.py                    # analyze all available results
    python3 analyze_results.py --size small       # analyze only small models
    python3 analyze_results.py --format markdown  # output as markdown tables
"""

import argparse
import json
import math
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results" / "v2"

SIZES = ["small", "medium", "large"]
SEEDS = [42, 137, 2024]
LANGUAGES = ["english", "lojban"]


def load_results(sizes=None):
    """Load all available result JSONs, grouped by (size, language, seed)."""
    results = {}
    sizes = sizes or SIZES

    for size in sizes:
        for lang in LANGUAGES:
            for seed in SEEDS:
                path = (RESULTS_DIR / size
                        / f"{lang}_seed{seed}" / "result.json")
                if path.exists():
                    with open(path) as f:
                        data = json.load(f)
                    results[(size, lang, seed)] = data

    return results


def group_by_size(results):
    """Group results by model size, returning {size: {language: [results]}}."""
    grouped = {}
    for (size, lang, seed), data in results.items():
        if size not in grouped:
            grouped[size] = {"english": [], "lojban": []}
        grouped[size][lang].append(data)
    return grouped


def mean_std(values):
    """Compute mean and standard deviation."""
    if not values:
        return 0.0, 0.0
    n = len(values)
    mu = sum(values) / n
    if n < 2:
        return mu, 0.0
    var = sum((x - mu) ** 2 for x in values) / (n - 1)
    return mu, math.sqrt(var)


def cohens_d(vals_a, vals_b):
    """Compute Cohen's d effect size (paired)."""
    if len(vals_a) != len(vals_b) or len(vals_a) < 2:
        return float("nan")
    diffs = [a - b for a, b in zip(vals_a, vals_b)]
    mu = sum(diffs) / len(diffs)
    var = sum((d - mu) ** 2 for d in diffs) / (len(diffs) - 1)
    sd = math.sqrt(var) if var > 0 else 1e-10
    return mu / sd


def paired_t_test(vals_a, vals_b):
    """Paired t-test. Returns (t_stat, p_value).
    vals_a and vals_b should be paired by seed."""
    n = len(vals_a)
    if n != len(vals_b) or n < 2:
        return float("nan"), float("nan")

    diffs = [a - b for a, b in zip(vals_a, vals_b)]
    mu = sum(diffs) / n
    var = sum((d - mu) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n) if var > 0 else 1e-10
    t_stat = mu / se

    # Two-tailed p-value approximation using Student's t distribution
    # For small n, use a rough approximation
    df = n - 1
    # Approximation of p-value from t distribution (Abramowitz & Stegun)
    x = abs(t_stat)
    if df == 1:
        p = 1.0 - (2.0 / math.pi) * math.atan(x)
    elif df == 2:
        p = 1.0 / math.sqrt(1.0 + x * x / 2.0)
    else:
        # General approximation
        a = 1.0 - 1.0 / (4.0 * df) + 1.0 / (32.0 * df * df)
        b = x * (1.0 - 1.0 / (2.0 * df))
        p = 2.0 * (1.0 - _normal_cdf(b * a))

    return t_stat, min(p, 1.0)


def _normal_cdf(x):
    """Standard normal CDF approximation."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def extract_metric(results, size, language, metric_path):
    """Extract a specific metric from results for all seeds of a (size, lang)."""
    values = []
    for seed in SEEDS:
        key = (size, language, seed)
        if key not in results:
            continue
        data = results[key]
        # Navigate nested dict path like "test_bpc.test_bpc"
        obj = data
        for part in metric_path.split("."):
            if isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                obj = None
                break
        if obj is not None and isinstance(obj, (int, float)):
            values.append(obj)
    return values


def print_summary_table(results, sizes, fmt="text"):
    """Print the main summary table."""
    sep = "|" if fmt == "markdown" else "  "

    metrics = [
        ("Test BPC", "test_bpc.test_bpc", True),  # lower is better
        ("Val BPC", "training.best_val_bpc", True),
        ("Grammar %", "grammar.grammaticality_rate", False),
        ("Avg LCS", "memorization.avg_lcs", True),
        ("Rep Rate 20", "structural.repetition_rate_20", True),
        ("3-gram Div", "structural.ngram_diversity_3", False),
    ]

    print("\n" + "=" * 80)
    print("SUMMARY: Mean ± Std across seeds (3 seeds)")
    print("=" * 80)

    for metric_name, metric_path, lower_better in metrics:
        print(f"\n--- {metric_name} ---")
        if fmt == "markdown":
            print(f"| Size | English | Lojban | Diff | Cohen's d | p-value |")
            print(f"|------|---------|--------|------|-----------|---------|")

        for size in sizes:
            eng_vals = extract_metric(results, size, "english", metric_path)
            loj_vals = extract_metric(results, size, "lojban", metric_path)

            if not eng_vals or not loj_vals:
                continue

            eng_mu, eng_sd = mean_std(eng_vals)
            loj_mu, loj_sd = mean_std(loj_vals)
            diff = loj_mu - eng_mu

            # Paired stats (requires same number of seeds)
            if len(eng_vals) == len(loj_vals) and len(eng_vals) >= 2:
                d = cohens_d(loj_vals, eng_vals)
                t_stat, p_val = paired_t_test(loj_vals, eng_vals)
                sig = "*" if p_val < 0.05 else ""
                if p_val < 0.01:
                    sig = "**"
                if p_val < 0.001:
                    sig = "***"
            else:
                d = float("nan")
                p_val = float("nan")
                sig = ""

            # Direction indicator
            if lower_better:
                better = "Loj" if diff < 0 else "Eng"
            else:
                better = "Loj" if diff > 0 else "Eng"

            if fmt == "markdown":
                print(f"| {size:6s} | {eng_mu:.4f}±{eng_sd:.4f} | "
                      f"{loj_mu:.4f}±{loj_sd:.4f} | "
                      f"{diff:+.4f} ({better}) | "
                      f"{d:.2f} | {p_val:.3f}{sig} |")
            else:
                print(f"  {size:6s}  Eng: {eng_mu:.4f}±{eng_sd:.4f}  "
                      f"Loj: {loj_mu:.4f}±{loj_sd:.4f}  "
                      f"Δ={diff:+.4f} ({better})  "
                      f"d={d:.2f}  p={p_val:.3f}{sig}")


def print_scaling_analysis(results, sizes):
    """Analyze how Lojban advantage scales with model size."""
    print("\n" + "=" * 80)
    print("SCALING ANALYSIS: Does Lojban advantage grow as models shrink?")
    print("=" * 80)

    metric = "test_bpc.test_bpc"
    print(f"\nTest BPC (lower = better):")
    print(f"  {'Size':8s}  {'Eng BPC':>10s}  {'Loj BPC':>10s}  "
          f"{'Advantage':>10s}  {'% Better':>10s}")

    advantages = []
    for size in sizes:
        eng_vals = extract_metric(results, size, "english", metric)
        loj_vals = extract_metric(results, size, "lojban", metric)
        if not eng_vals or not loj_vals:
            continue

        eng_mu, _ = mean_std(eng_vals)
        loj_mu, _ = mean_std(loj_vals)
        adv = eng_mu - loj_mu  # positive = Lojban better
        pct = (adv / eng_mu * 100) if eng_mu > 0 else 0

        advantages.append((size, adv, pct))
        print(f"  {size:8s}  {eng_mu:10.4f}  {loj_mu:10.4f}  "
              f"{adv:+10.4f}  {pct:+9.1f}%")

    if len(advantages) >= 2:
        print(f"\n  Scaling trend: ", end="")
        adv_vals = [a[2] for a in advantages]  # percentage advantages
        if adv_vals[0] > adv_vals[-1]:
            print("Lojban advantage GROWS as model size shrinks "
                  "(strongest finding)")
        elif adv_vals[0] < adv_vals[-1]:
            print("Lojban advantage SHRINKS as model size shrinks")
        else:
            print("Lojban advantage roughly constant across sizes")


def print_training_stats(results, sizes):
    """Print training statistics."""
    print("\n" + "=" * 80)
    print("TRAINING STATISTICS")
    print("=" * 80)

    for size in sizes:
        print(f"\n--- {size.upper()} ---")
        for lang in LANGUAGES:
            vals = []
            for seed in SEEDS:
                key = (size, lang, seed)
                if key in results:
                    vals.append(results[key])

            if not vals:
                continue

            params = vals[0]["training"]["n_params"]
            steps = [v["training"]["total_steps"] for v in vals]
            times = [v["training"]["total_time_s"] for v in vals]
            early = sum(1 for v in vals if v["training"]["early_stopped"])

            steps_mu, steps_sd = mean_std(steps)
            time_mu, time_sd = mean_std(times)

            print(f"  {lang:8s}: {params:,} params | "
                  f"steps: {steps_mu:.0f}±{steps_sd:.0f} | "
                  f"time: {time_mu:.0f}±{time_sd:.0f}s | "
                  f"early stopped: {early}/{len(vals)}")


def print_memorization_stats(results, sizes):
    """Print memorization statistics."""
    print("\n" + "=" * 80)
    print("MEMORIZATION STATISTICS")
    print("=" * 80)

    for size in sizes:
        print(f"\n--- {size.upper()} ---")
        for lang in LANGUAGES:
            mem_vals = extract_metric(results, size, lang,
                                      "memorization.n_memorized")
            novel_vals = extract_metric(results, size, lang,
                                        "memorization.n_novel")
            lcs_vals = extract_metric(results, size, lang,
                                      "memorization.avg_lcs")

            if not mem_vals:
                continue

            mem_mu, _ = mean_std(mem_vals)
            novel_mu, _ = mean_std(novel_vals)
            lcs_mu, lcs_sd = mean_std(lcs_vals)

            print(f"  {lang:8s}: memorized={mem_mu:.0f} novel={novel_mu:.0f} "
                  f"avg_lcs={lcs_mu:.0f}±{lcs_sd:.0f}")


def print_per_run_detail(results, sizes):
    """Print detailed per-run results."""
    print("\n" + "=" * 80)
    print("PER-RUN DETAIL")
    print("=" * 80)

    print(f"\n  {'Size':6s}  {'Lang':8s}  {'Seed':>6s}  "
          f"{'Steps':>6s}  {'ValBPC':>7s}  {'TestBPC':>8s}  "
          f"{'Gram%':>6s}  {'Novel':>6s}  {'AvgLCS':>7s}")

    for size in sizes:
        for lang in LANGUAGES:
            for seed in SEEDS:
                key = (size, lang, seed)
                if key not in results:
                    continue
                r = results[key]
                steps = r["training"]["total_steps"]
                val_bpc = r["training"]["best_val_bpc"]
                test_bpc = r["test_bpc"]["test_bpc"]
                gram = r["grammar"].get("grammaticality_rate", -1)
                novel = r["memorization"]["n_novel"]
                avg_lcs = r["memorization"]["avg_lcs"]

                gram_str = f"{gram:.1%}" if gram >= 0 else "skip"
                print(f"  {size:6s}  {lang:8s}  {seed:6d}  "
                      f"{steps:6d}  {val_bpc:7.3f}  {test_bpc:8.3f}  "
                      f"{gram_str:>6s}  {novel:6d}  {avg_lcs:7.1f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment v2 results")
    parser.add_argument("--size", choices=["small", "medium", "large", "all"],
                        default="all")
    parser.add_argument("--format", choices=["text", "markdown"],
                        default="text")
    parser.add_argument("--detail", action="store_true",
                        help="Show per-run detail")
    args = parser.parse_args()

    sizes = [args.size] if args.size != "all" else SIZES

    results = load_results(sizes)
    if not results:
        print(f"No results found in {RESULTS_DIR}")
        print(f"Run experiment_v2.py first.")
        sys.exit(1)

    n = len(results)
    n_sizes = len(set(s for s, _, _ in results.keys()))
    n_seeds = len(set(s for _, _, s in results.keys()))
    print(f"Loaded {n} results ({n_sizes} sizes, {n_seeds} seeds)")

    # Check calibration
    cal_path = RESULTS_DIR / "calibration.json"
    if cal_path.exists():
        with open(cal_path) as f:
            cal = json.load(f)
        print(f"\nTatoeba calibration:")
        if "camxes_pass_rate" in cal:
            print(f"  camxes: {cal['camxes_pass']}/{cal['camxes_total']} "
                  f"({cal['camxes_pass_rate']:.1%})")
        if "lt_pass_rate" in cal:
            print(f"  LanguageTool: {cal['lt_pass']}/{cal['lt_total']} "
                  f"({cal['lt_pass_rate']:.1%})")

    available_sizes = sorted(set(s for s, _, _ in results.keys()),
                             key=lambda x: SIZES.index(x))

    print_summary_table(results, available_sizes, fmt=args.format)
    print_scaling_analysis(results, available_sizes)
    print_training_stats(results, available_sizes)
    print_memorization_stats(results, available_sizes)

    if args.detail:
        print_per_run_detail(results, available_sizes)

    # Corpus info
    info_path = RESULTS_DIR / "corpus_info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        print(f"\n--- Corpus Info ---")
        print(f"  Training chars: {info['train_chars_per_language']:,} per language")
        print(f"  English vocab: {info['english_vocab_size']} | "
              f"Lojban vocab: {info['lojban_vocab_size']}")
        print(f"  Test chars: Eng={info['english_test_chars']:,} "
              f"Loj={info['lojban_test_chars']:,}")


if __name__ == "__main__":
    main()
