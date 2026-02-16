"""
BPE Tokenizer wrapper for the Lojban vs English experiment.

Wraps HuggingFace `tokenizers` library to provide byte-level BPE
with the same interface used by CharTokenizer in V3. Trained separately
per language on combined (narrative + bAbI) text.

Usage:
    tok = BPETokenizerWrapper("English")
    tok.train(text, vocab_size=1024)
    tok.save("tokenizer_english.json")

    # Later:
    tok = BPETokenizerWrapper("English")
    tok.load("tokenizer_english.json")

    ids = tok.encode("hello world")
    text = tok.decode(ids)
"""

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


DEFAULT_VOCAB_SIZE = 1024

# Placeholder for newlines: ByteLevel pre-tokenizer drops \n as a word
# boundary, so we replace newlines with a non-whitespace Unicode char
# before tokenization and reverse it after decoding.
_NEWLINE_PLACEHOLDER = "\u0126"  # Ħ — not in English or Lojban text


class BPETokenizerWrapper:
    """Wraps HuggingFace tokenizers ByteLevel BPE for this experiment.

    Key design choices:
    - Byte-level fallback ensures no OOV tokens
    - No special tokens (no PAD, no EOS) — same as V3's CharTokenizer
    - Separate tokenizer trained per language
    - Vocab size 1024 (256 byte base + ~768 merges)
    - Newlines preserved via placeholder substitution (ByteLevel drops \n)
    """

    def __init__(self, name=""):
        self.name = name
        self._tokenizer = None

    def train(self, text, vocab_size=DEFAULT_VOCAB_SIZE):
        """Train byte-level BPE on the given text."""
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=[],
            show_progress=False,
        )

        # Replace newlines with placeholder so they get tokenized
        text = text.replace("\n", _NEWLINE_PLACEHOLDER)

        # Feed paragraphs (split on double-placeholder) to the trainer.
        # Each chunk retains single placeholders so the tokenizer learns them.
        sep = _NEWLINE_PLACEHOLDER + _NEWLINE_PLACEHOLDER
        chunks = [c for c in text.split(sep) if c.strip()]

        tokenizer.train_from_iterator(
            chunks,
            trainer=trainer,
        )

        self._tokenizer = tokenizer

    def save(self, path):
        """Save tokenizer to JSON file."""
        self._tokenizer.save(str(path))

    def load(self, path):
        """Load tokenizer from JSON file."""
        self._tokenizer = Tokenizer.from_file(str(path))

    def encode(self, text):
        """Encode text to list of token IDs."""
        text = text.replace("\n", _NEWLINE_PLACEHOLDER)
        return self._tokenizer.encode(text).ids

    def decode(self, ids):
        """Decode list of token IDs to text."""
        text = self._tokenizer.decode(ids)
        return text.replace(_NEWLINE_PLACEHOLDER, "\n")

    @property
    def vocab_size(self):
        return self._tokenizer.get_vocab_size()
