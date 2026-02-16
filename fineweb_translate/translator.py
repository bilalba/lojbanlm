"""Async API calls via litellm for translation."""

import asyncio
import time
from dataclasses import dataclass

from .config import MAX_CONCURRENT_REQUESTS, MODELS
from .prompts import build_messages


@dataclass
class TranslationResult:
    chunk_id: str
    english: str
    lojban: str
    model: str
    provider: str
    pass_number: int
    input_tokens: int
    output_tokens: int
    latency_ms: int
    error: str | None = None


class Translator:
    """Async translator using litellm (handles OpenAI + Anthropic + others)."""

    def __init__(self, max_concurrent: int = MAX_CONCURRENT_REQUESTS):
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def translate_chunk(
        self,
        chunk_id: str,
        english_text: str,
        pass_number: int,
        few_shot: list[tuple[str, str]],
        dictionary_hints: str = "",
        retry_feedback: str = "",
    ) -> TranslationResult:
        """Translate a single chunk using the model for the given pass number."""
        import litellm

        pass_key = f"pass{pass_number}"
        model_config = MODELS.get(pass_key, MODELS["pass1"])
        provider = model_config["provider"]
        model = model_config["model"]

        messages = build_messages(
            english_text,
            few_shot=few_shot,
            dictionary_hints=dictionary_hints,
            retry_feedback=retry_feedback,
        )

        async with self._semaphore:
            start = time.monotonic()
            try:
                response = await litellm.acompletion(
                    model=model,
                    messages=messages,
                    max_tokens=4096,
                )

                latency_ms = int((time.monotonic() - start) * 1000)
                usage = response.usage

                return TranslationResult(
                    chunk_id=chunk_id,
                    english=english_text,
                    lojban=response.choices[0].message.content or "",
                    model=model,
                    provider=provider,
                    pass_number=pass_number,
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                    latency_ms=latency_ms,
                )

            except Exception as e:
                latency_ms = int((time.monotonic() - start) * 1000)
                return TranslationResult(
                    chunk_id=chunk_id,
                    english=english_text,
                    lojban="",
                    model=model,
                    provider=provider,
                    pass_number=pass_number,
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=latency_ms,
                    error=str(e),
                )

    async def translate_batch(
        self,
        chunks: list[tuple[str, str, int, str, str]],
        # Each: (chunk_id, english_text, pass_number, dict_hints, retry_feedback)
        few_shot: list[tuple[str, str]],
    ) -> list[TranslationResult]:
        """Translate a batch of chunks concurrently."""
        tasks = [
            self.translate_chunk(
                chunk_id=cid,
                english_text=text,
                pass_number=pn,
                few_shot=few_shot,
                dictionary_hints=hints,
                retry_feedback=feedback,
            )
            for cid, text, pn, hints, feedback in chunks
        ]
        return await asyncio.gather(*tasks)
