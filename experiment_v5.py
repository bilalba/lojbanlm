#!/usr/bin/env python3
"""
Experiment V5: Two-Phase Training (Pretrain + bAbI SFT)
=======================================================
Phase 1: Pretrain on general text (books + FineWeb parallel).
Phase 2: Supervised fine-tune on bAbI, checkpoint by bAbI val accuracy.

Solves the checkpoint selection confound from V3/V4 where checkpoint was
selected by narrative val loss but evaluated on bAbI accuracy.

Changes from V4:
- Two-phase training (pretrain → SFT)
- More pretraining data (~1.7M chars: books + FineWeb parallel + Little Prince)
- Phase 2 checkpoints by bAbI val accuracy (not narrative loss)
- Model state dicts saved to disk
- Optional ONNX export
- Modular code structure (v5/ package)
"""

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch

from v5.config import (
    RESULTS_DIR, SEEDS, BPE_VOCAB_SIZE,
    MODEL_CONFIGS, ModelConfig, Phase1Config, Phase2Config, EvalConfig,
    TRAIN_BOOKS, TEST_BOOK, ALL_BABI_TASKS,
)
from v5.data import (
    load_pretraining_corpus, load_book_text, load_babi_train_text,
    load_babi_examples, load_tatoeba, prepare_splits,
)
from v5.tokenizer import BPETokenizerWrapper
from v5.model import GPT, save_checkpoint, load_checkpoint, export_onnx
from v5.train import train_phase1, train_phase2, _seed_everything
from v5.evaluate import (
    eval_all_babi, compute_test_bpc, generate_samples,
    tag_memorization, eval_lojban_grammar, eval_english_grammar,
    compute_structural_metrics, calibrate_grammar_checkers,
)


def run_single(size, language, seed, model_cfg, phase1_cfg, phase2_cfg,
               eval_cfg, pretrain_train, pretrain_val, pretrain_prompt,
               babi_train_data, babi_val_examples, tokenizer,
               test_ids, test_text, pretrain_text,
               device, export_onnx_flag=False):
    """Execute one complete experiment run: phase 1 + phase 2 + evaluate."""
    run_dir = RESULTS_DIR / size / f"{language}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    result_path = run_dir / "result.json"
    if result_path.exists():
        print(f"\n  SKIP: {result_path} already exists")
        return None

    vocab_size = tokenizer.vocab_size

    # ─── Seed ────────────────────────────────────────────────────────
    _seed_everything(seed)

    # ─── Create model ────────────────────────────────────────────────
    model = GPT(vocab_size, model_cfg.ctx_len, model_cfg.d_model,
                model_cfg.n_head, model_cfg.n_layer, model_cfg.dropout).to(device)
    n_params = model.count_params()
    print(f"  Parameters: {n_params:,}")

    # ─── Phase 1: Pretraining ────────────────────────────────────────
    model, phase1_info = train_phase1(
        model, pretrain_train, pretrain_val, phase1_cfg, vocab_size, device
    )
    save_checkpoint(model, run_dir / "phase1_model.pt")
    print(f"  Phase 1 checkpoint saved to {run_dir / 'phase1_model.pt'}")

    # ─── Phase 2: bAbI SFT ──────────────────────────────────────────
    model, phase2_info = train_phase2(
        model, babi_train_data, babi_val_examples, tokenizer,
        phase2_cfg, vocab_size, device
    )
    save_checkpoint(model, run_dir / "phase2_model.pt")
    print(f"  Phase 2 checkpoint saved to {run_dir / 'phase2_model.pt'}")

    # ─── Optional ONNX export ────────────────────────────────────────
    if export_onnx_flag:
        onnx_path = run_dir / "phase2_model.onnx"
        export_onnx(model, vocab_size, model_cfg.ctx_len, onnx_path)
        print(f"  ONNX exported to {onnx_path}")

    result = {
        "config": {
            "size": size,
            "language": language,
            "seed": seed,
            "model": asdict(model_cfg),
            "phase1": asdict(phase1_cfg),
            "phase2": asdict(phase2_cfg),
        },
        "phase1": phase1_info,
        "phase2": phase2_info,
    }

    # ─── bAbI Evaluation ────────────────────────────────────────────
    print("  Evaluating bAbI reasoning tasks...")
    babi_task_ids = eval_cfg.babi_tasks or ALL_BABI_TASKS
    babi_results, babi_predictions = eval_all_babi(
        model, tokenizer, language, device, model_cfg.ctx_len,
        task_ids=babi_task_ids
    )
    result["babi"] = babi_results

    with open(run_dir / "babi_predictions.json", "w") as f:
        json.dump(babi_predictions, f, indent=2, ensure_ascii=False)

    # ─── Narrative Evaluation ────────────────────────────────────────
    if not eval_cfg.skip_narrative_eval:
        print("  Computing test BPC...")
        test_bpc_info = compute_test_bpc(model, test_ids, tokenizer,
                                          model_cfg.ctx_len, device)
        print(f"    Test BPC: {test_bpc_info['test_bpc']}")
        result["test_bpc"] = test_bpc_info
        result["val_bpc"] = phase1_info["best_val_bpc"]

        print("  Generating narrative samples...")
        random.seed(seed)
        samples = generate_samples(
            model, tokenizer, pretrain_prompt, test_ids,
            eval_cfg.num_samples, eval_cfg.prompt_len, eval_cfg.gen_len,
            eval_cfg.temperature, eval_cfg.top_k, device
        )
        print(f"    Generated {len(samples)} samples")

        mem_info = tag_memorization(samples, pretrain_text,
                                     eval_cfg.lcs_threshold)
        result["memorization"] = mem_info

        if not eval_cfg.skip_grammar:
            print("  Evaluating grammar (novel samples only)...")
            if language == "lojban":
                grammar = eval_lojban_grammar(samples, novel_only=True)
            else:
                grammar = eval_english_grammar(samples, novel_only=True)
        else:
            grammar = {"skipped": True, "reason": "skip_grammar flag"}
        result["grammar"] = grammar

        print("  Computing structural metrics...")
        structural = compute_structural_metrics(samples, pretrain_text)
        result["structural"] = structural

        with open(run_dir / "samples.json", "w") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
    else:
        result["narrative_eval"] = "skipped"

    # ─── Save results ────────────────────────────────────────────────
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Results saved to {run_dir}/")

    # Free GPU memory
    del model
    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass

    return result


def main():
    parser = argparse.ArgumentParser(
        description="V5: Two-phase training (pretrain + bAbI SFT)")
    parser.add_argument("--size",
                        choices=["medium", "large", "all"],
                        default="all",
                        help="Model size (default: all)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Single seed (default: all 3)")
    parser.add_argument("--language",
                        choices=["english", "lojban", "all"],
                        default="all",
                        help="Language (default: all)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick sanity check: 500 steps/phase, medium, "
                             "1 seed, 2 bAbI tasks")
    parser.add_argument("--skip-grammar", action="store_true",
                        help="Skip grammar evaluation")
    parser.add_argument("--skip-calibration", action="store_true",
                        help="Skip Tatoeba calibration")
    parser.add_argument("--skip-narrative-eval", action="store_true",
                        help="Skip narrative eval (BPC, generation, grammar)")
    parser.add_argument("--babi-tasks", type=int, nargs="*", default=None,
                        help="Specific bAbI task IDs (default: all 20)")
    parser.add_argument("--phase1-steps", type=int, default=None,
                        help="Override phase 1 max steps")
    parser.add_argument("--phase2-steps", type=int, default=None,
                        help="Override phase 2 max steps")
    parser.add_argument("--export-onnx", action="store_true",
                        help="Export models to ONNX format")
    args = parser.parse_args()

    # ─── Device ──────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Device: CUDA")
    else:
        device = torch.device("cpu")
        print("Device: CPU")
    torch.set_num_threads(8)

    t_start = time.time()

    # ─── Run grid ────────────────────────────────────────────────────
    if args.quick:
        sizes = ["medium"]
        seeds = [42]
        languages = (["english", "lojban"] if args.language == "all"
                     else [args.language])
        babi_task_ids = [1, 6]
    else:
        sizes = ([args.size] if args.size != "all"
                 else ["medium", "large"])
        seeds = [args.seed] if args.seed is not None else SEEDS
        languages = ([args.language] if args.language != "all"
                     else ["english", "lojban"])
        babi_task_ids = args.babi_tasks if args.babi_tasks else ALL_BABI_TASKS

    # ─── Build configs ───────────────────────────────────────────────
    phase1_cfg = Phase1Config()
    phase2_cfg = Phase2Config()
    eval_cfg = EvalConfig(
        skip_grammar=args.skip_grammar,
        skip_narrative_eval=args.skip_narrative_eval,
        babi_tasks=babi_task_ids,
    )

    if args.phase1_steps is not None:
        phase1_cfg.max_steps = args.phase1_steps
        phase1_cfg.patience = args.phase1_steps + 1
    if args.phase2_steps is not None:
        phase2_cfg.max_steps = args.phase2_steps
        phase2_cfg.patience = args.phase2_steps + 1

    if args.quick:
        phase1_cfg.max_steps = 500
        phase1_cfg.patience = 501
        phase1_cfg.eval_interval = 50
        phase2_cfg.max_steps = 500
        phase2_cfg.patience = 501
        phase2_cfg.eval_interval = 50
        eval_cfg.num_samples = 10

    # ─── Load Pretraining Data ───────────────────────────────────────
    print("\nLoading pretraining corpus (books + FineWeb parallel)...")
    print("  English:")
    eng_pretrain = load_pretraining_corpus("english")
    print("  Lojban:")
    loj_pretrain = load_pretraining_corpus("lojban")

    min_len = min(len(eng_pretrain), len(loj_pretrain))
    eng_pretrain = eng_pretrain[:min_len]
    loj_pretrain = loj_pretrain[:min_len]
    print(f"\nPretraining text truncated to {min_len:,} chars each")

    # ─── Load bAbI Data ──────────────────────────────────────────────
    print(f"\nLoading bAbI training data ({len(babi_task_ids)} tasks)...")
    eng_babi_text = load_babi_train_text("english", babi_task_ids)
    loj_babi_text = load_babi_train_text("lojban", babi_task_ids)
    print(f"  English bAbI: {len(eng_babi_text):,} chars")
    print(f"  Lojban bAbI:  {len(loj_babi_text):,} chars")

    # Load bAbI val examples for phase 2 checkpoint selection
    print("Loading bAbI val examples for checkpoint selection...")
    eng_babi_val = []
    loj_babi_val = []
    for task_id in babi_task_ids:
        eng_babi_val.extend(load_babi_examples(task_id, "val", "english"))
        loj_babi_val.extend(load_babi_examples(task_id, "val", "lojban"))
    print(f"  English bAbI val: {len(eng_babi_val)} examples")
    print(f"  Lojban bAbI val:  {len(loj_babi_val)} examples")

    # ─── Train BPE Tokenizers ────────────────────────────────────────
    # Tokenizer trained on ALL text (pretraining + bAbI)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    eng_tok_path = RESULTS_DIR / "tokenizer_english.json"
    loj_tok_path = RESULTS_DIR / "tokenizer_lojban.json"

    eng_combined_for_tok = eng_pretrain + "\n\n" + eng_babi_text
    loj_combined_for_tok = loj_pretrain + "\n\n" + loj_babi_text

    eng_tok = BPETokenizerWrapper("English")
    loj_tok = BPETokenizerWrapper("Lojban")

    if eng_tok_path.exists():
        print(f"\nLoading saved English BPE tokenizer from {eng_tok_path}")
        eng_tok.load(eng_tok_path)
    else:
        print(f"\nTraining English BPE tokenizer (vocab={BPE_VOCAB_SIZE})...")
        eng_tok.train(eng_combined_for_tok, vocab_size=BPE_VOCAB_SIZE)
        eng_tok.save(eng_tok_path)
        print(f"  Saved to {eng_tok_path}")

    if loj_tok_path.exists():
        print(f"Loading saved Lojban BPE tokenizer from {loj_tok_path}")
        loj_tok.load(loj_tok_path)
    else:
        print(f"Training Lojban BPE tokenizer (vocab={BPE_VOCAB_SIZE})...")
        loj_tok.train(loj_combined_for_tok, vocab_size=BPE_VOCAB_SIZE)
        loj_tok.save(loj_tok_path)
        print(f"  Saved to {loj_tok_path}")

    print(f"English BPE vocab: {eng_tok.vocab_size} | "
          f"Lojban BPE vocab: {loj_tok.vocab_size}")

    # ─── Prepare Splits ──────────────────────────────────────────────
    # Phase 1: pretraining text → 90/5/5 train/val/prompt
    print("\nPreparing pretraining splits...")
    eng_p1_train, eng_p1_val, eng_p1_prompt = prepare_splits(
        eng_pretrain, eng_tok, ratios=(0.90, 0.05, 0.05))
    loj_p1_train, loj_p1_val, loj_p1_prompt = prepare_splits(
        loj_pretrain, loj_tok, ratios=(0.90, 0.05, 0.05))
    print(f"  English phase 1: {len(eng_p1_train):,} / {len(eng_p1_val):,} / "
          f"{len(eng_p1_prompt):,} tokens (train/val/prompt)")
    print(f"  Lojban phase 1:  {len(loj_p1_train):,} / {len(loj_p1_val):,} / "
          f"{len(loj_p1_prompt):,} tokens (train/val/prompt)")

    # Phase 2: bAbI text → token tensor (split done inside train_phase2)
    eng_babi_tokens = torch.tensor(eng_tok.encode(eng_babi_text), dtype=torch.long)
    loj_babi_tokens = torch.tensor(loj_tok.encode(loj_babi_text), dtype=torch.long)
    print(f"  English bAbI: {len(eng_babi_tokens):,} tokens")
    print(f"  Lojban bAbI:  {len(loj_babi_tokens):,} tokens")

    # ─── Test Data ───────────────────────────────────────────────────
    print("\nLoading test book (Metamorphosis)...")
    eng_test_text = load_book_text(TEST_BOOK, "english")
    loj_test_text = load_book_text(TEST_BOOK, "lojban")
    eng_test_ids = torch.tensor(eng_tok.encode(eng_test_text), dtype=torch.long)
    loj_test_ids = torch.tensor(loj_tok.encode(loj_test_text), dtype=torch.long)
    print(f"  English: {len(eng_test_text):,} chars -> {len(eng_test_ids):,} tokens")
    print(f"  Lojban:  {len(loj_test_text):,} chars -> {len(loj_test_ids):,} tokens")

    # ─── Tatoeba Calibration ─────────────────────────────────────────
    if not args.skip_calibration and not args.quick:
        print("\nCalibrating grammar checkers on Tatoeba...")
        random.seed(42)
        tatoeba = load_tatoeba()
        print(f"  Loaded {len(tatoeba)} parallel pairs")
        calibration = calibrate_grammar_checkers(tatoeba)
        with open(RESULTS_DIR / "calibration.json", "w") as f:
            json.dump(calibration, f, indent=2)

    # ─── Save corpus info ────────────────────────────────────────────
    corpus_info = {
        "experiment": "v5",
        "tokenizer": "bpe",
        "bpe_vocab_size": BPE_VOCAB_SIZE,
        "pretraining_chars_per_language": min_len,
        "english_babi_chars": len(eng_babi_text),
        "lojban_babi_chars": len(loj_babi_text),
        "english_bpe_vocab_size": eng_tok.vocab_size,
        "lojban_bpe_vocab_size": loj_tok.vocab_size,
        "english_phase1_train_tokens": len(eng_p1_train),
        "english_phase1_val_tokens": len(eng_p1_val),
        "lojban_phase1_train_tokens": len(loj_p1_train),
        "lojban_phase1_val_tokens": len(loj_p1_val),
        "english_babi_tokens": len(eng_babi_tokens),
        "lojban_babi_tokens": len(loj_babi_tokens),
        "english_babi_val_examples": len(eng_babi_val),
        "lojban_babi_val_examples": len(loj_babi_val),
        "english_test_chars": len(eng_test_text),
        "lojban_test_chars": len(loj_test_text),
        "babi_tasks_used": babi_task_ids,
    }
    with open(RESULTS_DIR / "corpus_info.json", "w") as f:
        json.dump(corpus_info, f, indent=2)

    # ─── Run Experiments ─────────────────────────────────────────────
    total_runs = len(sizes) * len(seeds) * len(languages)
    run_idx = 0

    for size in sizes:
        model_cfg = MODEL_CONFIGS[size]
        for seed in seeds:
            for language in languages:
                run_idx += 1

                if language == "english":
                    tok = eng_tok
                    p1_train = eng_p1_train
                    p1_val = eng_p1_val
                    p1_prompt = eng_p1_prompt
                    babi_tokens = eng_babi_tokens
                    babi_val = eng_babi_val
                    test_ids = eng_test_ids
                    test_text = eng_test_text
                    pretrain_text = eng_pretrain
                else:
                    tok = loj_tok
                    p1_train = loj_p1_train
                    p1_val = loj_p1_val
                    p1_prompt = loj_p1_prompt
                    babi_tokens = loj_babi_tokens
                    babi_val = loj_babi_val
                    test_ids = loj_test_ids
                    test_text = loj_test_text
                    pretrain_text = loj_pretrain

                print(f"\n{'#'*60}")
                print(f"# Run {run_idx}/{total_runs}: "
                      f"{size} / {language} / seed={seed}")
                print(f"#   d={model_cfg.d_model} L={model_cfg.n_layer} "
                      f"H={model_cfg.n_head} drop={model_cfg.dropout}")
                print(f"{'#'*60}")

                result = run_single(
                    size, language, seed, model_cfg, phase1_cfg, phase2_cfg,
                    eval_cfg, p1_train, p1_val, p1_prompt,
                    babi_tokens, babi_val, tok,
                    test_ids, test_text, pretrain_text,
                    device, export_onnx_flag=args.export_onnx,
                )

                # Inject data size info
                if result is not None:
                    result["phase1"]["pretrain_chars"] = min_len
                    result["phase1"]["babi_chars"] = (
                        len(eng_babi_text) if language == "english"
                        else len(loj_babi_text)
                    )
                    result_path = (RESULTS_DIR / size
                                   / f"{language}_seed{seed}"
                                   / "result.json")
                    with open(result_path, "w") as f:
                        json.dump(result, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"All {total_runs} runs complete in {elapsed/60:.1f} min")
    print(f"Results in {RESULTS_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
