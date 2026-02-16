"""Main pipeline orchestrator: chunk → translate → validate → repair → accept."""

import asyncio
import time

from .chunker import Chunk, iter_chunks
from .config import BATCH_SIZE, MAX_RETRIES, MODELS, REPAIR_MODEL, PipelineConfig
from .cost_tracker import CostTracker
from .dictionary import LojbanDictionary
from .progress import ProgressTracker
from .prompts import load_few_shot_examples
from .repair import repair_chunk
from .translator import Translator, TranslationResult
from .validator import CamxesProcess, validate_translation


async def _translate_and_repair(
    result: TranslationResult,
    camxes: CamxesProcess | None,
    progress: ProgressTracker,
    cost: CostTracker,
    skip_validation: bool,
) -> bool:
    """Validate a translation, repair failing sentences, record result.

    Returns True if accepted (100% pass rate after repair).
    """
    # Record translation cost
    if result.input_tokens > 0:
        cost.record(result.chunk_id, result.model, result.input_tokens, result.output_tokens)

    # API error → reject
    if result.error:
        progress.record(
            chunk_id=result.chunk_id, status="rejected",
            pass_number=result.pass_number, model=result.model,
            grammar_pass_rate=None, english=result.english, lojban="",
        )
        return False

    # Skip validation if requested
    if skip_validation or camxes is None:
        progress.record(
            chunk_id=result.chunk_id, status="accepted",
            pass_number=result.pass_number, model=result.model,
            grammar_pass_rate=None, english=result.english, lojban=result.lojban,
            input_tokens=result.input_tokens, output_tokens=result.output_tokens,
        )
        return True

    # Validate + repair failing sentences with Gemini Flash
    repaired_text, all_passed, pass_rate, repair_in, repair_out = await repair_chunk(
        result.lojban, camxes
    )

    # Record repair cost
    if repair_in > 0:
        cost.record(result.chunk_id, REPAIR_MODEL, repair_in, repair_out)

    total_in = result.input_tokens + repair_in
    total_out = result.output_tokens + repair_out

    if all_passed:
        progress.record(
            chunk_id=result.chunk_id, status="accepted",
            pass_number=result.pass_number, model=result.model,
            grammar_pass_rate=1.0, english=result.english, lojban=repaired_text,
            input_tokens=total_in, output_tokens=total_out,
        )
        return True

    # Not 100% after repair — record as rejected (will retry with next model)
    progress.record(
        chunk_id=result.chunk_id, status="rejected",
        pass_number=result.pass_number, model=result.model,
        grammar_pass_rate=pass_rate, english=result.english, lojban=repaired_text,
    )
    return False


async def run_pipeline(config: PipelineConfig):
    """Main pipeline entry point."""
    print(f"=== FineWeb English→Lojban Translation Pipeline ===")
    print(f"Input: {config.input_path}")
    print(f"Output: {config.output_dir}")
    print(f"Strategy: translate → validate → sentence repair (100% target)")
    if config.max_chars > 0:
        print(f"Max chars: {config.max_chars:,}")
    if config.budget_usd > 0:
        print(f"Budget: ${config.budget_usd:.2f}")
    if config.dry_run:
        print("DRY RUN — no API calls will be made")
    print()

    # Initialize components
    dictionary = LojbanDictionary()
    dictionary.load()

    few_shot = load_few_shot_examples()
    print(f"Loaded {len(few_shot)} few-shot examples")

    progress = ProgressTracker(
        progress_path=config.output_dir / "progress.jsonl",
        translations_path=config.output_dir / "translations.jsonl",
        failures_path=config.output_dir / "failures.jsonl",
    )
    progress.print_stats()

    cost = CostTracker(
        log_path=config.output_dir / "cost_log.jsonl",
        budget_usd=config.budget_usd,
    )

    translator = Translator(max_concurrent=config.max_concurrent)

    camxes: CamxesProcess | None = None
    if not config.skip_validation:
        camxes = CamxesProcess()
        await camxes.start()
        print(f"Camxes validation: enabled (repair model: {REPAIR_MODEL})")
    else:
        print("Camxes validation: SKIPPED")

    print()

    # Collect chunks into batches
    batch: list[Chunk] = []
    total_chars = 0
    batch_num = 0
    start_time = time.monotonic()

    try:
        for chunk in iter_chunks(config.input_path, config.start_doc, config.max_chars):
            if progress.is_done(chunk.chunk_id):
                continue

            batch.append(chunk)
            total_chars += len(chunk.text)

            if len(batch) >= config.batch_size:
                batch_num += 1
                await _process_batch(
                    batch, batch_num, config, dictionary, few_shot,
                    translator, camxes, progress, cost, start_time, total_chars,
                )
                batch = []

                if not cost.check_budget():
                    print(f"\nBudget exhausted (${config.budget_usd:.2f})")
                    break

        # Flush remaining
        if batch:
            batch_num += 1
            await _process_batch(
                batch, batch_num, config, dictionary, few_shot,
                translator, camxes, progress, cost, start_time, total_chars,
            )

    finally:
        if camxes:
            await camxes.stop()

    # Final summary
    print(f"\n=== Pipeline Complete ===")
    progress.print_stats()
    cost.print_summary()
    elapsed = time.monotonic() - start_time
    print(f"Time: {elapsed:.0f}s ({total_chars:,} chars processed)")


async def _process_batch(
    batch: list[Chunk],
    batch_num: int,
    config: PipelineConfig,
    dictionary: LojbanDictionary,
    few_shot: list[tuple[str, str]],
    translator: Translator,
    camxes: CamxesProcess | None,
    progress: ProgressTracker,
    cost: CostTracker,
    start_time: float,
    total_chars: int,
):
    """Process a batch: for each chunk, try models in order with sentence repair."""
    elapsed = time.monotonic() - start_time

    if config.dry_run:
        print(f"Batch {batch_num}: {len(batch)} chunks (dry run, skipping API calls)")
        for chunk in batch[:3]:
            hints = dictionary.lookup(chunk.text)
            print(f"  {chunk.chunk_id}: {len(chunk.text)} chars, {len(hints.split(', '))} hints")
        if len(batch) > 3:
            print(f"  ... and {len(batch) - 3} more")
        return

    accepted = 0

    for ci, chunk in enumerate(batch):
        hints = dictionary.lookup(chunk.text)
        print(f"  [{ci+1}/{len(batch)}] {chunk.chunk_id} ({len(chunk.text)} chars)...", end=" ", flush=True)

        # Try each model pass: translate → repair → accept or try next model
        chunk_accepted = False
        for pass_num in range(1, MAX_RETRIES + 2):  # passes 1, 2, 3
            pass_key = f"pass{pass_num}"
            if pass_key not in MODELS:
                break

            result = await translator.translate_chunk(
                chunk_id=chunk.chunk_id,
                english_text=chunk.text,
                pass_number=pass_num,
                few_shot=few_shot,
                dictionary_hints=hints,
            )

            ok = await _translate_and_repair(
                result, camxes, progress, cost, config.skip_validation
            )
            if ok:
                chunk_accepted = True
                accepted += 1
                print(f"pass{pass_num} OK", flush=True)
                break
            else:
                print(f"pass{pass_num} fail", end=" → ", flush=True)

        # All passes exhausted
        if not chunk_accepted:
            print("FAILED", flush=True)
            # Mark as failed (last attempt's data is already in progress)
            progress.record(
                chunk_id=chunk.chunk_id, status="failed",
                pass_number=MAX_RETRIES + 1, model=MODELS[f"pass{MAX_RETRIES + 1}"]["model"],
                grammar_pass_rate=0.0, english=chunk.text, lojban="",
            )

    stats = progress.get_stats()
    cost_summary = cost.summary()
    print(
        f"Batch {batch_num}: {accepted}/{len(batch)} accepted (100% grammar) | "
        f"Total: {stats['accepted']} ok, {stats['failed']} fail | "
        f"${cost_summary['total_cost_usd']:.4f} | {elapsed:.0f}s"
    )
