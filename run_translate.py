#!/usr/bin/env -S python3 -u
"""
FineWeb English → Lojban Translation Pipeline

Translates FineWeb-Edu English text to Lojban using LLM APIs,
with camxes grammar validation and multi-pass retry.

Usage:
    python3 run_translate.py --max-chars 1000000           # 1M char pilot
    python3 run_translate.py                                # full 60M chars
    python3 run_translate.py --dry-run --max-chars 10000    # preview (no API calls)
    python3 run_translate.py --status                       # check progress/cost
    python3 run_translate.py --budget 50.0                  # cost cap
    python3 run_translate.py --skip-validation              # skip camxes check
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Load .env file
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                os.environ.setdefault(_key.strip(), _val.strip())

from fineweb_translate.config import OUTPUT_DIR, PipelineConfig
from fineweb_translate.cost_tracker import CostTracker
from fineweb_translate.pipeline import run_pipeline
from fineweb_translate.progress import ProgressTracker


def print_status():
    """Print progress and cost summary, then exit."""
    progress = ProgressTracker(
        progress_path=OUTPUT_DIR / "progress.jsonl",
        translations_path=OUTPUT_DIR / "translations.jsonl",
        failures_path=OUTPUT_DIR / "failures.jsonl",
    )
    progress.print_stats()

    cost = CostTracker(log_path=OUTPUT_DIR / "cost_log.jsonl")
    cost.print_summary()

    # Show output file sizes
    for name in ["translations.jsonl", "failures.jsonl", "progress.jsonl", "cost_log.jsonl"]:
        p = OUTPUT_DIR / name
        if p.exists():
            size_kb = p.stat().st_size / 1024
            lines = sum(1 for _ in open(p))
            print(f"  {name}: {lines} lines, {size_kb:.1f} KB")


def main():
    parser = argparse.ArgumentParser(
        description="Translate FineWeb-Edu English to Lojban",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--max-chars", type=int, default=0,
        help="Max English chars to translate (0=all, default=0)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=20,
        help="Chunks per async batch (default: 20)",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=10,
        help="Max concurrent API requests (default: 10)",
    )
    parser.add_argument(
        "--budget", type=float, default=0.0,
        help="Max USD to spend (0=unlimited)",
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Skip camxes grammar validation",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done, no API calls",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print progress/cost summary and exit",
    )
    parser.add_argument(
        "--start-doc", type=int, default=0,
        help="Start from document index (for manual partitioning)",
    )
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    config = PipelineConfig(
        max_chars=args.max_chars,
        batch_size=args.batch_size,
        max_concurrent=args.max_concurrent,
        skip_validation=args.skip_validation,
        dry_run=args.dry_run,
        start_doc=args.start_doc,
        budget_usd=args.budget,
    )

    asyncio.run(run_pipeline(config))


if __name__ == "__main__":
    main()
