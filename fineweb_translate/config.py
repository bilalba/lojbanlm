"""Configuration constants and runtime config for the translation pipeline."""

from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Input
FINEWEB_PATH = BASE_DIR / "data" / "fineweb_edu" / "train.txt"

# Output
OUTPUT_DIR = BASE_DIR / "data" / "fineweb_lojban"
PROGRESS_PATH = OUTPUT_DIR / "progress.jsonl"
TRANSLATIONS_PATH = OUTPUT_DIR / "translations.jsonl"
FAILURES_PATH = OUTPUT_DIR / "failures.jsonl"
COST_LOG_PATH = OUTPUT_DIR / "cost_log.jsonl"

# Dictionary sources
GISMU_DATA_PATH = BASE_DIR / "ilmentufa" / "glosser" / "gismu-data.js"
JBOVLASTE_DIR = BASE_DIR / "data" / "jbovlaste"
JBOVLASTE_XML_PATH = JBOVLASTE_DIR / "en.xml"
JBOVLASTE_URL = "https://raw.githubusercontent.com/lojbanistan/jbovlaste-dicts/master/en.xml"

# Validation
CAMXES_PATH = BASE_DIR / "ilmentufa" / "run_camxes.js"
TATOEBA_PATH = BASE_DIR / "corpus" / "tatoeba" / "jbo_eng_parallel.tsv"

# Translation models by pass (chunk-level, ordered by grammar quality + cost)
MODELS = {
    "pass1": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
    "pass2": {"provider": "google", "model": "gemini/gemini-3-flash-preview"},
    "pass3": {"provider": "openai", "model": "gpt-5.2"},
}

# Model used for sentence-level repair (fast + cheap + best grammar)
REPAIR_MODEL = "gemini/gemini-3-flash-preview"

# Pricing per million tokens (2026)
PRICING = {
    "gemini/gemini-3-flash-preview": {"input": 0.10, "output": 0.40},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-sonnet-4-5-20250929": {"input": 1.50, "output": 7.50},
    "claude-opus-4-6": {"input": 2.50, "output": 12.50},
    "gpt-5.2": {"input": 1.75, "output": 14.00},
}

# Chunking
MAX_CHUNK_CHARS = 2000
MIN_CHUNK_CHARS = 200

# API rate limits
MAX_CONCURRENT_REQUESTS = 10
RATE_LIMIT_RPM_OPENAI = 500
RATE_LIMIT_RPM_ANTHROPIC = 50
REQUEST_TIMEOUT_SECONDS = 60

# Validation thresholds
CAMXES_TIMEOUT_SECONDS = 10
MIN_GRAMMAR_PASS_RATE = 1.0  # require 100% camxes pass
MAX_RETRIES = 2  # chunk-level retries with different models
MAX_SENTENCE_REPAIRS = 3  # per-sentence repair attempts after chunk translation

# Batch processing
BATCH_SIZE = 20

# Dictionary
MAX_HINTS_PER_CHUNK = 40


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime configuration, overridable via CLI."""

    input_path: Path = FINEWEB_PATH
    output_dir: Path = OUTPUT_DIR
    max_chars: int = 0  # 0 = unlimited
    batch_size: int = BATCH_SIZE
    max_concurrent: int = MAX_CONCURRENT_REQUESTS
    skip_validation: bool = False
    dry_run: bool = False
    start_doc: int = 0
    budget_usd: float = 0.0  # 0 = unlimited
