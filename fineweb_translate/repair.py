"""Sentence-level repair: identify invalid words, replace them, re-validate."""

import asyncio
import re

import litellm

from .config import MAX_SENTENCE_REPAIRS, REPAIR_MODEL
from .dictionary import LojbanDictionary
from .validator import CamxesProcess, split_lojban_sentences

# Known valid cmavo (structure words) — not in gismu-data.js but are real Lojban
# This is a subset; camxes itself is the ultimate authority
KNOWN_CMAVO = frozenset(
    ".i .e .a .o .u cu ko'a ko'e ko'i fo'a fo'e fo'i "
    "lo le la lo'e le'e li lu li'u "
    "be bei be'o pe ne po po'e po'u "
    "ke ke'e ku ku'o kei "
    "nu du'u ka ni jei si'o "
    "fa fe fi fo fu "
    "se te ve xe "
    "na na'e to'e no'e nai "
    "go'i ja'a "
    "xu ma mo "
    "pu ca ba za'a "
    "vi va vu "
    "zo'e zo'oi ce'u "
    "gi'e gi'a gi'o gi'u "
    "je ja jo ju "
    "noi poi voi "
    "do mi ko da de di "
    "ri ra ru "
    "vau".split()
)

REPAIR_SYSTEM = """\
You are a Lojban grammar repair tool. Fix this sentence so it parses with camxes.

CRITICAL: Only use REAL Lojban words. Do NOT invent words.
- If a word is marked as INVALID, you MUST replace it with the suggested alternative or rephrase.
- Use zo'oi WORD for untranslatable foreign terms.
- Names: la .name. (with dots and periods)
- Add terminators (ku, kei, vau, ku'o) where needed.
- Output ONLY the fixed Lojban sentence. No English. No explanation.\
"""

_dict_instance: LojbanDictionary | None = None


def _get_dict() -> LojbanDictionary:
    global _dict_instance
    if _dict_instance is None:
        _dict_instance = LojbanDictionary()
        _dict_instance.load()
    return _dict_instance


def find_invalid_words(sentence: str) -> list[str]:
    """Find words in a Lojban sentence that aren't in any dictionary."""
    d = _get_dict()
    gismu = d._gismu_data or {}
    jbovlaste = d._jbovlaste or {}

    # Tokenize: extract Lojban words (letters and apostrophes)
    tokens = re.findall(r"[a-z']+", sentence.lower())

    invalid = []
    for tok in tokens:
        # Skip known cmavo
        if tok in KNOWN_CMAVO:
            continue
        # Skip single letters (often valid cmavo)
        if len(tok) <= 2:
            continue
        # Skip if in gismu dict
        if tok in gismu:
            continue
        # Skip if in jbovlaste
        if tok in jbovlaste:
            continue
        # Skip common cmavo patterns (CV'V, CVV, etc.)
        if re.match(r"^[bcdfgjklmnprstvxz]?[aeiou]'?[aeiou]$", tok):
            continue
        # Skip if looks like a valid lujvo (contains valid rafsi patterns)
        # Basic heuristic: 6+ chars with vowels = likely lujvo attempt, check jbovlaste
        if len(tok) >= 5 and tok in jbovlaste:
            continue
        invalid.append(tok)

    return list(dict.fromkeys(invalid))  # deduplicate, preserve order


async def repair_sentence(
    sentence: str,
    error: str,
    camxes: CamxesProcess,
    model: str = REPAIR_MODEL,
) -> tuple[str, int, int]:
    """Try to repair a single sentence.

    Returns (fixed_sentence, total_in_tokens, total_out_tokens).
    """
    total_in = 0
    total_out = 0

    # Identify invalid words
    invalid = find_invalid_words(sentence)

    # Build specific repair instructions
    if invalid:
        d = _get_dict()
        word_notes = []
        for w in invalid[:5]:  # cap at 5
            # Try to find what they might have meant
            hints = d.lookup(w, max_hints=3)
            if hints:
                word_notes.append(f'"{w}" is NOT real Lojban. Possible replacements: {hints}')
            else:
                word_notes.append(f'"{w}" is NOT real Lojban. Use zo\'oi {w} or rephrase.')
        invalid_info = "\n".join(word_notes)
    else:
        invalid_info = "All words appear valid. The issue is likely grammar structure (missing terminators, wrong word order)."

    messages = [
        {"role": "system", "content": REPAIR_SYSTEM},
        {"role": "user", "content": (
            f"Broken sentence: {sentence}\n"
            f"Parser error: {error}\n\n"
            f"Word analysis:\n{invalid_info}\n\n"
            f"Output the corrected sentence only."
        )},
    ]

    for attempt in range(MAX_SENTENCE_REPAIRS):
        try:
            response = await litellm.acompletion(
                model=model, messages=messages, max_tokens=512,
            )
            fixed = (response.choices[0].message.content or "").strip()
            # Strip markdown code fences if present
            if fixed.startswith("```"):
                fixed = re.sub(r"```\w*\n?", "", fixed).strip().rstrip("`").strip()
            usage = response.usage
            total_in += usage.prompt_tokens if usage else 0
            total_out += usage.completion_tokens if usage else 0

            if not fixed:
                continue

            passed, new_error = await camxes.check(fixed)
            if passed:
                return fixed, total_in, total_out

            # Append as conversation for next attempt
            messages.append({"role": "assistant", "content": fixed})
            messages.append({"role": "user", "content": f"Still fails: {new_error[:120]}. Try again."})

        except Exception:
            continue

    return sentence, total_in, total_out


async def repair_chunk(
    lojban_text: str,
    camxes: CamxesProcess,
    model: str = REPAIR_MODEL,
) -> tuple[str, bool, float, int, int]:
    """Repair all failing sentences in a chunk. Drop sentences that can't be fixed.

    Returns (repaired_text, all_passed, pass_rate, total_in_tokens, total_out_tokens).
    """
    sentences = split_lojban_sentences(lojban_text)
    if not sentences:
        return lojban_text, False, 0.0, 0, 0

    total_in = 0
    total_out = 0

    # First pass: check all sentences in parallel
    check_results = await asyncio.gather(*(camxes.check(s) for s in sentences))

    # Separate passing and failing sentences
    kept = []
    to_repair: list[tuple[int, str, str]] = []  # (index, sentence, error)
    for i, (sent, (passed, output)) in enumerate(zip(sentences, check_results)):
        if passed:
            kept.append((i, sent))
        else:
            to_repair.append((i, sent, output))

    # Repair failing sentences in parallel
    if to_repair:
        repair_tasks = [
            repair_sentence(sent, error, camxes, model)
            for _, sent, error in to_repair
        ]
        repair_results = await asyncio.gather(*repair_tasks)

        for (idx, _sent, _err), (fixed, in_tok, out_tok) in zip(to_repair, repair_results):
            total_in += in_tok
            total_out += out_tok
            passed, _ = await camxes.check(fixed)
            if passed:
                kept.append((idx, fixed))
            # else: drop the sentence — don't include invalid Lojban

    # Sort by original order, extract text
    kept.sort(key=lambda x: x[0])
    repaired_text = " .i ".join(text for _, text in kept)
    all_passed = len(kept) > 0  # all kept sentences are valid; accept if any survived
    pass_rate = len(kept) / len(sentences) if sentences else 0.0

    return repaired_text, all_passed, pass_rate, total_in, total_out
