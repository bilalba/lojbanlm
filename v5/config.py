"""Configuration dataclasses and constants for V5 experiment."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "corpus"
BABI_DIR = CORPUS_DIR / "babi"
RESULTS_DIR = BASE_DIR / "results" / "v5"

SEEDS = [42, 137, 2024]
BPE_VOCAB_SIZE = 1024


@dataclass
class ModelConfig:
    d_model: int
    n_layer: int
    n_head: int
    dropout: float
    ctx_len: int = 256


MODEL_CONFIGS = {
    "medium": ModelConfig(d_model=96, n_layer=4, n_head=4, dropout=0.15),
    "large": ModelConfig(d_model=128, n_layer=3, n_head=4, dropout=0.15),
}


@dataclass
class Phase1Config:
    """Pretraining on general text. Checkpoint by val BPC."""
    max_steps: int = 5000
    lr: float = 3e-4
    batch_size: int = 64
    eval_interval: int = 100
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    patience: int = 5001  # > max_steps: no early stopping


@dataclass
class Phase2Config:
    """bAbI SFT. Checkpoint by bAbI val accuracy."""
    max_steps: int = 5000
    lr: float = 3e-5
    batch_size: int = 64
    eval_interval: int = 100
    warmup_steps: int = 50
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    patience: int = 5001


@dataclass
class EvalConfig:
    num_samples: int = 50
    prompt_len: int = 32
    gen_len: int = 128
    temperature: float = 0.8
    top_k: int = 40
    lcs_threshold: int = 50
    skip_grammar: bool = False
    skip_narrative_eval: bool = False
    babi_tasks: Optional[List[int]] = None  # None = all 20


BABI_TASK_NAMES = {
    1: "single_supporting_fact",
    2: "two_supporting_facts",
    3: "three_supporting_facts",
    4: "two_argument_relations",
    5: "three_argument_relations",
    6: "yes_no_questions",
    7: "counting",
    8: "lists_sets",
    9: "simple_negation",
    10: "indefinite_knowledge",
    11: "basic_coreference",
    12: "conjunction",
    13: "compound_coreference",
    14: "time_reasoning",
    15: "basic_deduction",
    16: "basic_induction",
    17: "positional_reasoning",
    18: "size_reasoning",
    19: "path_finding",
    20: "agents_motivations",
}

ALL_BABI_TASKS = list(range(1, 21))

TRAIN_BOOKS = [
    "alice_in_wonderland",
    "wizard_of_oz",
    "esther",
    "in_a_grove",
    "little_prince",
]
TEST_BOOK = "metamorphosis"

BOOK_FILES = {
    "alice_in_wonderland": {
        "english": "alice_in_wonderland/alice_english.txt",
        "lojban": "alice_in_wonderland/alice_lojban.txt",
    },
    "wizard_of_oz": {
        "english": "wizard_of_oz/wizard_of_oz_english.txt",
        "lojban": "wizard_of_oz/wizard_of_oz_lojban.txt",
    },
    "esther": {
        "english": "esther/esther_english.txt",
        "lojban": "esther/esther_lojban.txt",
    },
    "in_a_grove": {
        "english": "in_a_grove/in_a_grove_english.txt",
        "lojban": "in_a_grove/in_a_grove_lojban.txt",
    },
    "little_prince": {
        "english": "little_prince/little_prince_english.txt",
        "lojban": "little_prince/little_prince_lojban.txt",
    },
    "metamorphosis": {
        "english": "metamorphosis/metamorphosis_english.txt",
        "lojban": "metamorphosis/metamorphosis_lojban.txt",
    },
}

# FineWeb parallel data (separate from books)
FINEWEB_FILES = {
    "english": "fineweb_lojban/fineweb_english.txt",
    "lojban": "fineweb_lojban/fineweb_lojban.txt",
}
