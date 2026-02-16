"""BPE tokenizer utilities for V5 experiment."""

import sys
from pathlib import Path

# Ensure parent dir is importable (for bpe_tokenizer.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bpe_tokenizer import BPETokenizerWrapper, DEFAULT_VOCAB_SIZE

__all__ = ["BPETokenizerWrapper", "DEFAULT_VOCAB_SIZE"]
