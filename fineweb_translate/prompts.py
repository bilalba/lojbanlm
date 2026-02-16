"""Translation prompt construction: system prompt, few-shot examples, dictionary hints."""

import csv
import random
from pathlib import Path

from .config import TATOEBA_PATH

SYSTEM_PROMPT = """\
You are an expert Lojban translator. Translate the English text to grammatically correct Lojban.

Rules:
- Output ONLY the Lojban translation. No English, no explanations, no notes.
- Preserve paragraph structure (keep blank lines between paragraphs).
- Use standard Lojban orthography with proper attitudinals and terminators.
- Names: use la + lojbanized form (e.g., "John" -> "la .djan.", "Mary" -> "la .maris.")
- Separate sentences with .i
- Numbers: use Lojban digits (pa, re, ci, vo, mu, xa, ze, bi, so, no)
- Technical terms with no Lojban equivalent: use fu'ivla or le glico valsi be zo'oi TERM
- Preserve the informational content. Exact word-for-word alignment is NOT needed.
- Output must be parseable by the camxes Lojban grammar checker.\
"""

_few_shot_cache: list[tuple[str, str]] | None = None


def load_few_shot_examples(
    path: Path = TATOEBA_PATH,
    n: int = 8,
    min_eng_len: int = 40,
    seed: int = 42,
) -> list[tuple[str, str]]:
    """Select n Tatoeba pairs with English length >= min_eng_len.

    Returns list of (english, lojban) tuples. Cached after first call.
    """
    global _few_shot_cache
    if _few_shot_cache is not None:
        return _few_shot_cache

    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header ("lojban\tenglish")
        for row in reader:
            if len(row) < 2:
                continue
            lojban, english = row[0].strip(), row[1].strip()
            if len(english) >= min_eng_len and lojban and english:
                pairs.append((english, lojban))

    rng = random.Random(seed)
    rng.shuffle(pairs)
    _few_shot_cache = pairs[:n]
    return _few_shot_cache


def build_messages(
    english_text: str,
    few_shot: list[tuple[str, str]] | None = None,
    dictionary_hints: str = "",
    retry_feedback: str = "",
) -> list[dict]:
    """Build the messages array for the API call.

    Returns list of {"role": ..., "content": ...} dicts.
    Compatible with both OpenAI and Anthropic message formats.
    The system message is included as role="system" (caller adapts per provider).
    """
    if few_shot is None:
        few_shot = load_few_shot_examples()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Few-shot examples as alternating user/assistant
    for eng, loj in few_shot:
        messages.append({"role": "user", "content": eng})
        messages.append({"role": "assistant", "content": loj})

    # Dictionary hints (if any)
    user_content = ""
    if dictionary_hints:
        user_content += f"Use these Lojban words where applicable: {dictionary_hints}\n\n"

    # Retry feedback from failed validation
    if retry_feedback:
        user_content += f"Previous translation failed grammar validation. {retry_feedback}\n\n"

    user_content += english_text
    messages.append({"role": "user", "content": user_content})

    return messages
