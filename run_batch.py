#!/usr/bin/env -S python3 -u
"""
Batch translation via Anthropic Batch API (50% discount).

Flow:
    1. submit       — submit translation batch
    2. status       — check batch progress
    3. download     — download results, validate with camxes, accept passing chunks
    4. submit-repair — submit repair batch for failing sentences
    5. download-repair — download repair results, validate, accept/drop

Usage:
    python3 run_batch.py submit                          # submit all remaining chunks
    python3 run_batch.py submit --max-chars 5000         # small test batch
    python3 run_batch.py status                          # check all batch statuses
    python3 run_batch.py download                        # download + validate (camxes only, no API)
    python3 run_batch.py submit-repair                   # batch repair failing sentences
    python3 run_batch.py download-repair                 # download repairs + validate
"""

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path

# Load .env
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

import anthropic

from fineweb_translate.chunker import iter_chunks
from fineweb_translate.config import OUTPUT_DIR, FINEWEB_PATH, PRICING
from fineweb_translate.dictionary import LojbanDictionary
from fineweb_translate.progress import ProgressTracker
from fineweb_translate.prompts import build_messages, load_few_shot_examples
from fineweb_translate.repair import REPAIR_SYSTEM, find_invalid_words, _get_dict
from fineweb_translate.validator import CamxesProcess, split_lojban_sentences

BATCH_DIR = OUTPUT_DIR / "batches"
MODEL = "claude-sonnet-4-5-20250929"
REPAIR_MODEL = MODEL  # use same model for repair in batch mode


def _make_progress():
    return ProgressTracker(
        progress_path=OUTPUT_DIR / "progress.jsonl",
        translations_path=OUTPUT_DIR / "translations.jsonl",
        failures_path=OUTPUT_DIR / "failures.jsonl",
    )


def _save_batch_info(batch, batch_type, num_requests, extra=None):
    """Save batch metadata to a JSON file."""
    info = {
        "batch_id": batch.id,
        "type": batch_type,
        "model": MODEL,
        "num_requests": num_requests,
        "created_at": batch.created_at.isoformat() if batch.created_at else None,
        "status": batch.processing_status,
    }
    if extra:
        info.update(extra)
    info_file = BATCH_DIR / f"batch_{batch.id}.json"
    with open(info_file, "w") as f:
        json.dump(info, f, indent=2)
    return info_file


# ── submit ──────────────────────────────────────────────────────────────

def submit(args):
    """Build and submit a batch of translation requests."""
    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    dictionary = LojbanDictionary()
    dictionary.load()
    few_shot = load_few_shot_examples()
    print(f"Loaded {len(few_shot)} few-shot examples")

    progress = _make_progress()
    progress.print_stats()

    # Collect chunks
    requests = []
    english_map = {}
    total_chars = 0
    for chunk in iter_chunks(FINEWEB_PATH, start_doc=args.start_doc, max_chars=args.max_chars):
        if progress.is_done(chunk.chunk_id):
            continue

        hints = dictionary.lookup(chunk.text)
        messages = build_messages(chunk.text, few_shot=few_shot, dictionary_hints=hints)

        system_content = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                user_messages.append(msg)

        requests.append({
            "custom_id": chunk.chunk_id,
            "params": {
                "model": MODEL,
                "max_tokens": 4096,
                "system": system_content,
                "messages": user_messages,
            },
        })
        english_map[chunk.chunk_id] = chunk.text
        total_chars += len(chunk.text)

    if not requests:
        print("No chunks to translate (all done or none found)")
        return

    print(f"\nPrepared {len(requests)} requests ({total_chars:,} chars)")

    # Estimate cost
    pricing = PRICING.get(MODEL, {"input": 1.50, "output": 7.50})
    est_cost = len(requests) * (2250 * pricing["input"] + 750 * pricing["output"]) / 1_000_000 * 0.5
    print(f"Estimated cost: ${est_cost:.2f} (batch 50% discount)")

    # Save english map for later
    eng_file = BATCH_DIR / f"english_map_{int(time.time())}.json"
    with open(eng_file, "w") as f:
        json.dump(english_map, f)

    # Submit
    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=requests)
    info_file = _save_batch_info(batch, "translate", len(requests), {
        "total_chars": total_chars,
        "estimated_cost": est_cost,
        "english_map_file": str(eng_file),
    })

    print(f"\nBatch submitted!")
    print(f"  ID: {batch.id}")
    print(f"  Status: {batch.processing_status}")
    print(f"  Requests: {len(requests)}")
    print(f"  Info: {info_file}")


# ── status ──────────────────────────────────────────────────────────────

def status(args):
    """Check status of all submitted batches."""
    client = anthropic.Anthropic()
    if not BATCH_DIR.exists():
        print("No batches submitted yet")
        return

    for info_file in sorted(BATCH_DIR.glob("batch_msgbatch_*.json")):
        with open(info_file) as f:
            info = json.load(f)
        batch_id = info["batch_id"]
        try:
            batch = client.messages.batches.retrieve(batch_id)
            counts = batch.request_counts
            print(f"Batch: {batch_id} ({info.get('type', '?')})")
            print(f"  Status: {batch.processing_status}")
            print(f"  Succeeded: {counts.succeeded} / Errored: {counts.errored} / Processing: {counts.processing}")
            print()
        except Exception as e:
            print(f"Batch: {batch_id} — error: {e}\n")


# ── download ────────────────────────────────────────────────────────────

def download(args):
    """Download translation batch results, validate with camxes locally."""
    client = anthropic.Anthropic()
    if not BATCH_DIR.exists():
        print("No batches submitted yet")
        return

    # Find translate batches
    info_files = sorted(BATCH_DIR.glob("batch_msgbatch_*.json"))
    translate_batches = []
    for f in info_files:
        with open(f) as fh:
            info = json.load(fh)
        if info.get("type") == "translate":
            translate_batches.append(info)

    if not translate_batches:
        print("No translate batches found")
        return

    if args.batch_id:
        info = next((b for b in translate_batches if b["batch_id"] == args.batch_id), None)
        if not info:
            print(f"Batch {args.batch_id} not found")
            return
    else:
        info = translate_batches[-1]

    batch_id = info["batch_id"]
    batch = client.messages.batches.retrieve(batch_id)
    print(f"Batch: {batch_id}")
    print(f"Status: {batch.processing_status}")

    if batch.processing_status != "ended":
        counts = batch.request_counts
        print(f"  Succeeded: {counts.succeeded} / Processing: {counts.processing}")
        print("Batch not finished yet.")
        return

    # Load english map
    english_map = {}
    eng_file = info.get("english_map_file")
    if eng_file and Path(eng_file).exists():
        with open(eng_file) as f:
            english_map = json.load(f)

    # Download results
    results = []
    errors = 0
    for result in client.messages.batches.results(batch_id):
        if result.result.type == "succeeded":
            msg = result.result.message
            results.append({
                "chunk_id": result.custom_id,
                "lojban": msg.content[0].text if msg.content else "",
                "input_tokens": msg.usage.input_tokens if msg.usage else 0,
                "output_tokens": msg.usage.output_tokens if msg.usage else 0,
                "english": english_map.get(result.custom_id, ""),
            })
        else:
            errors += 1

    print(f"Downloaded {len(results)} results ({errors} errors)")

    # Save raw
    raw_file = BATCH_DIR / f"results_{batch_id}.jsonl"
    with open(raw_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Validate with camxes (local, no API calls)
    asyncio.run(_validate_and_save(results, batch_id))


async def _validate_and_save(results, batch_id):
    """Validate each chunk with camxes. Accept fully-passing, save failures for repair."""
    progress = _make_progress()
    camxes = CamxesProcess()
    await camxes.start()

    accepted = 0
    needs_repair = []  # chunks with some failing sentences
    skipped = 0

    for i, r in enumerate(results):
        if progress.is_done(r["chunk_id"]):
            skipped += 1
            continue

        sentences = split_lojban_sentences(r["lojban"])
        if not sentences:
            progress.record(
                chunk_id=r["chunk_id"], status="failed",
                pass_number=1, model=MODEL,
                grammar_pass_rate=0.0, english=r["english"], lojban="",
            )
            continue

        # Check all sentences (sequential — camxes is a single stdin/stdout process)
        check_results = [await camxes.check(s) for s in sentences]
        passing = [(i, s) for i, (s, (ok, _)) in enumerate(zip(sentences, check_results)) if ok]
        failing = [(i, s, err) for i, (s, (ok, err)) in enumerate(zip(sentences, check_results)) if not ok]

        if not failing:
            # All pass — accept immediately
            progress.record(
                chunk_id=r["chunk_id"], status="accepted",
                pass_number=1, model=MODEL,
                grammar_pass_rate=1.0, english=r["english"], lojban=r["lojban"],
                input_tokens=r["input_tokens"], output_tokens=r["output_tokens"],
            )
            accepted += 1
        elif passing:
            # Some pass, some fail — save for repair
            needs_repair.append({
                "chunk_id": r["chunk_id"],
                "english": r["english"],
                "input_tokens": r["input_tokens"],
                "output_tokens": r["output_tokens"],
                "passing": [(idx, s) for idx, s in passing],
                "failing": [(idx, s, err) for idx, s, err in failing],
                "total_sentences": len(sentences),
            })
        else:
            # Nothing passes — save for repair too (repair might salvage some)
            needs_repair.append({
                "chunk_id": r["chunk_id"],
                "english": r["english"],
                "input_tokens": r["input_tokens"],
                "output_tokens": r["output_tokens"],
                "passing": [],
                "failing": [(idx, s, err) for idx, s, err in failing],
                "total_sentences": len(sentences),
            })

        if (i + 1) % 50 == 0:
            print(f"  Validated {i+1}/{len(results)}... {accepted} accepted, {len(needs_repair)} need repair", flush=True)

    await camxes.stop()

    # Save repair candidates
    repair_file = BATCH_DIR / f"needs_repair_{batch_id}.jsonl"
    with open(repair_file, "w") as f:
        for r in needs_repair:
            f.write(json.dumps(r) + "\n")

    total_failing_sents = sum(len(r["failing"]) for r in needs_repair)
    print(f"\nValidation complete:")
    print(f"  Accepted (100% pass): {accepted}")
    print(f"  Need repair: {len(needs_repair)} chunks ({total_failing_sents} sentences)")
    print(f"  Skipped (already done): {skipped}")
    if needs_repair:
        print(f"\nRepair file: {repair_file}")
        print(f"Next: python3 run_batch.py submit-repair")

    progress.print_stats()


# ── submit-repair ───────────────────────────────────────────────────────

def submit_repair(args):
    """Submit a repair batch for failing sentences."""
    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    # Find the most recent needs_repair file
    repair_files = sorted(BATCH_DIR.glob("needs_repair_*.jsonl"))
    if not repair_files:
        print("No repair file found. Run 'download' first.")
        return

    repair_file = repair_files[-1]
    print(f"Loading: {repair_file}")

    # Load dictionary for word analysis
    d = _get_dict()

    # Build repair requests
    requests = []
    repair_meta = {}  # custom_id → {chunk_id, sentence_index, ...}

    with open(repair_file) as f:
        for line in f:
            chunk = json.loads(line)
            chunk_id = chunk["chunk_id"]

            for idx, sentence, error in chunk["failing"]:
                # Build repair prompt with invalid word analysis
                invalid = find_invalid_words(sentence)
                if invalid:
                    word_notes = []
                    for w in invalid[:5]:
                        hints = d.lookup(w, max_hints=3)
                        if hints:
                            word_notes.append(f'"{w}" is NOT real Lojban. Possible replacements: {hints}')
                        else:
                            word_notes.append(f'"{w}" is NOT real Lojban. Use zo\'oi {w} or rephrase.')
                    invalid_info = "\n".join(word_notes)
                else:
                    invalid_info = "All words appear valid. The issue is likely grammar structure (missing terminators, wrong word order)."

                user_content = (
                    f"Broken sentence: {sentence}\n"
                    f"Parser error: {error}\n\n"
                    f"Word analysis:\n{invalid_info}\n\n"
                    f"Output the corrected sentence only."
                )

                custom_id = f"{chunk_id}__sent{idx}"
                requests.append({
                    "custom_id": custom_id,
                    "params": {
                        "model": MODEL,
                        "max_tokens": 512,
                        "system": REPAIR_SYSTEM,
                        "messages": [{"role": "user", "content": user_content}],
                    },
                })
                repair_meta[custom_id] = {
                    "chunk_id": chunk_id,
                    "sentence_index": idx,
                    "original_sentence": sentence,
                }

    if not requests:
        print("No sentences to repair")
        return

    print(f"Prepared {len(requests)} repair requests")

    # Save metadata
    meta_file = BATCH_DIR / f"repair_meta_{int(time.time())}.json"
    with open(meta_file, "w") as f:
        json.dump(repair_meta, f)

    # Also save the needs_repair data for reassembly
    chunk_data_file = BATCH_DIR / f"repair_chunks_{int(time.time())}.jsonl"
    with open(repair_file) as src, open(chunk_data_file, "w") as dst:
        dst.write(src.read())

    # Submit
    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=requests)
    info_file = _save_batch_info(batch, "repair", len(requests), {
        "repair_meta_file": str(meta_file),
        "chunk_data_file": str(chunk_data_file),
        "source_repair_file": str(repair_file),
    })

    pricing = PRICING.get(MODEL, {"input": 1.50, "output": 7.50})
    est_cost = len(requests) * (500 * pricing["input"] + 100 * pricing["output"]) / 1_000_000 * 0.5
    print(f"\nRepair batch submitted!")
    print(f"  ID: {batch.id}")
    print(f"  Requests: {len(requests)}")
    print(f"  Estimated cost: ${est_cost:.2f}")
    print(f"  Info: {info_file}")


# ── download-repair ─────────────────────────────────────────────────────

def download_repair(args):
    """Download repair results, validate, reassemble chunks."""
    client = anthropic.Anthropic()
    if not BATCH_DIR.exists():
        print("No batches found")
        return

    # Find repair batches
    info_files = sorted(BATCH_DIR.glob("batch_msgbatch_*.json"))
    repair_batches = []
    for f in info_files:
        with open(f) as fh:
            info = json.load(fh)
        if info.get("type") == "repair":
            repair_batches.append(info)

    if not repair_batches:
        print("No repair batches found")
        return

    info = repair_batches[-1]
    batch_id = info["batch_id"]
    batch = client.messages.batches.retrieve(batch_id)
    print(f"Batch: {batch_id}")
    print(f"Status: {batch.processing_status}")

    if batch.processing_status != "ended":
        counts = batch.request_counts
        print(f"  Succeeded: {counts.succeeded} / Processing: {counts.processing}")
        print("Batch not finished yet.")
        return

    # Load metadata
    with open(info["repair_meta_file"]) as f:
        repair_meta = json.load(f)

    # Load chunk data (passing sentences + structure)
    chunk_data = {}
    with open(info["chunk_data_file"]) as f:
        for line in f:
            d = json.loads(line)
            chunk_data[d["chunk_id"]] = d

    # Download repair results
    repairs = {}  # custom_id → fixed_sentence
    for result in client.messages.batches.results(batch_id):
        if result.result.type == "succeeded":
            msg = result.result.message
            fixed = msg.content[0].text.strip() if msg.content else ""
            if fixed.startswith("```"):
                fixed = re.sub(r"```\w*\n?", "", fixed).strip().rstrip("`").strip()
            repairs[result.custom_id] = fixed

    print(f"Downloaded {len(repairs)} repair results")

    # Validate repairs and reassemble chunks
    asyncio.run(_validate_repairs(repairs, repair_meta, chunk_data))


async def _validate_repairs(repairs, repair_meta, chunk_data):
    """Validate repaired sentences, reassemble chunks, record results."""
    progress = _make_progress()
    camxes = CamxesProcess()
    await camxes.start()

    accepted = 0
    failed = 0
    skipped = 0

    # Group repairs by chunk
    chunk_repairs = {}  # chunk_id → {sentence_index: fixed_sentence}
    for custom_id, fixed in repairs.items():
        meta = repair_meta[custom_id]
        cid = meta["chunk_id"]
        if cid not in chunk_repairs:
            chunk_repairs[cid] = {}
        chunk_repairs[cid][meta["sentence_index"]] = fixed

    for chunk_id, data in chunk_data.items():
        if progress.is_done(chunk_id):
            skipped += 1
            continue

        # Start with passing sentences
        kept = [(idx, s) for idx, s in data["passing"]]

        # Check repaired sentences
        chunk_fixed = chunk_repairs.get(chunk_id, {})
        for idx, orig_sent, _err in data["failing"]:
            fixed = chunk_fixed.get(idx, orig_sent)
            ok, _ = await camxes.check(fixed)
            if ok:
                kept.append((idx, fixed))
            # else: drop

        if kept:
            kept.sort(key=lambda x: x[0])
            lojban = " .i ".join(s for _, s in kept)
            pass_rate = len(kept) / data["total_sentences"]
            progress.record(
                chunk_id=chunk_id, status="accepted",
                pass_number=1, model=MODEL,
                grammar_pass_rate=pass_rate,
                english=data["english"], lojban=lojban,
                input_tokens=data["input_tokens"], output_tokens=data["output_tokens"],
            )
            accepted += 1
        else:
            progress.record(
                chunk_id=chunk_id, status="failed",
                pass_number=1, model=MODEL,
                grammar_pass_rate=0.0,
                english=data["english"], lojban="",
            )
            failed += 1

    await camxes.stop()
    print(f"\nRepair complete: {accepted} accepted, {failed} failed, {skipped} skipped")
    progress.print_stats()


# ── main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch translation via Anthropic Batch API")
    sub = parser.add_subparsers(dest="command")

    p_submit = sub.add_parser("submit", help="Submit translation batch")
    p_submit.add_argument("--max-chars", type=int, default=0)
    p_submit.add_argument("--start-doc", type=int, default=0)

    sub.add_parser("status", help="Check batch status")

    p_dl = sub.add_parser("download", help="Download + validate translations")
    p_dl.add_argument("--batch-id", type=str, default=None)

    sub.add_parser("submit-repair", help="Submit repair batch for failing sentences")

    sub.add_parser("download-repair", help="Download + validate repairs")

    args = parser.parse_args()

    if args.command == "submit":
        submit(args)
    elif args.command == "status":
        status(args)
    elif args.command == "download":
        download(args)
    elif args.command == "submit-repair":
        submit_repair(args)
    elif args.command == "download-repair":
        download_repair(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
