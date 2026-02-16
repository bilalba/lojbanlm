"""Lojban grammar validation via persistent camxes subprocess."""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from .config import CAMXES_PATH, CAMXES_TIMEOUT_SECONDS, MIN_GRAMMAR_PASS_RATE


@dataclass
class ValidationResult:
    chunk_id: str
    total_sentences: int
    parsed_ok: int
    parse_errors: list[dict] = field(default_factory=list)
    pass_rate: float = 0.0
    accepted: bool = False


def split_lojban_sentences(text: str) -> list[str]:
    """Split Lojban text into sentences on '.i' boundaries."""
    sentences = []
    for s in text.split(".i"):
        s = s.strip()
        if len(s) >= 5:  # skip trivially short fragments
            sentences.append(s)
    return sentences


class CamxesProcess:
    """Long-lived Node.js camxes process for fast batch validation.

    Uses camxes loop mode (-m L) which reads sentences from stdin
    and writes parse results to stdout, one per line.
    Avoids ~100ms Node.js startup per sentence.
    """

    def __init__(self, camxes_path: Path = CAMXES_PATH):
        self._camxes_path = camxes_path
        self._proc: asyncio.subprocess.Process | None = None

    async def start(self):
        """Spawn the persistent node process."""
        self._proc = await asyncio.create_subprocess_exec(
            "node", str(self._camxes_path), "-m", "L",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def check(self, sentence: str) -> tuple[bool, str]:
        """Send a sentence and read the result line.

        Returns (passed: bool, output: str).
        Errors contain '@' (position marker from camxes SyntaxError).
        """
        if self._proc is None or self._proc.returncode is not None:
            await self.start()

        # Normalize: single line, no internal newlines
        sentence = sentence.replace("\n", " ").strip()
        if not sentence:
            return True, ""

        try:
            self._proc.stdin.write((sentence + "\n").encode("utf-8"))
            await self._proc.stdin.drain()
            line = await asyncio.wait_for(
                self._proc.stdout.readline(),
                timeout=CAMXES_TIMEOUT_SECONDS,
            )
            result = line.decode("utf-8").strip()

            # Also consume the prompt "> " that camxes outputs
            # (readline.createInterface outputs "> " after each result)
            # The prompt is part of stdout; our readline() may have captured it
            # or it may be buffered. We just check the result for errors.

            # Error detection: SyntaxError output contains @offset
            if "@" in result or "SyntaxError" in result:
                return False, result
            if not result:
                return False, "empty response"
            return True, result

        except asyncio.TimeoutError:
            # Kill and restart on timeout
            await self.stop()
            return False, "timeout"
        except Exception as e:
            return False, str(e)

    async def stop(self):
        """Terminate the process."""
        if self._proc and self._proc.returncode is None:
            try:
                self._proc.stdin.close()
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except (asyncio.TimeoutError, ProcessLookupError):
                self._proc.kill()
        self._proc = None

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *exc):
        await self.stop()


async def validate_translation(
    chunk_id: str,
    lojban_text: str,
    camxes: CamxesProcess,
) -> ValidationResult:
    """Validate all sentences in a translation."""
    sentences = split_lojban_sentences(lojban_text)

    if not sentences:
        return ValidationResult(
            chunk_id=chunk_id,
            total_sentences=0,
            parsed_ok=0,
            pass_rate=0.0,
            accepted=False,
        )

    parsed_ok = 0
    errors = []

    for sent in sentences:
        passed, output = await camxes.check(sent)
        if passed:
            parsed_ok += 1
        else:
            errors.append({"sentence": sent[:200], "error": output[:200]})

    pass_rate = parsed_ok / len(sentences) if sentences else 0.0
    accepted = pass_rate >= MIN_GRAMMAR_PASS_RATE

    return ValidationResult(
        chunk_id=chunk_id,
        total_sentences=len(sentences),
        parsed_ok=parsed_ok,
        parse_errors=errors,
        pass_rate=pass_rate,
        accepted=accepted,
    )


async def validate_batch(
    results: list[tuple[str, str]],  # (chunk_id, lojban_text)
    camxes: CamxesProcess,
) -> list[ValidationResult]:
    """Validate a batch of translations sequentially (shared camxes process)."""
    validations = []
    for chunk_id, lojban_text in results:
        v = await validate_translation(chunk_id, lojban_text, camxes)
        validations.append(v)
    return validations
