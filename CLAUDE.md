# Lojban vs English Small Model Coherence Experiment

Tests whether Lojban's regular, unambiguous grammar helps small language models
produce more coherent output than English at equal parameter count.

**Hypothesis**: Lojban's regular grammar allows small models to develop reasoning
capabilities at fewer parameters than English — scaling laws for reasoning kick
in at smaller scale.

## Results Summary

See detailed writeups in:
- **V1_RESULTS.md** — V1 single-book experiment (completed, invalidated by memorization)
- **V2_RESULTS.md** — V2 multi-corpus experiment (small + medium complete, large skipped)
- **V3_RESULTS.md** — V3 bAbI reasoning experiment (completed, confounded by early stopping)
- **V3_1_RESULTS.md** — V3.1 fixed-step rerun (in progress, results in `results/v3_1/`)
- **V4_RESULTS.md** — V4 BPE tokenization experiment (completed)
- **V4_DESIGN.md** — V4 BPE tokenization experiment design
- **REASONING_EXPERIMENT.md** — Original bAbI reasoning experiment design rationale

### V1 (single book, ~10.8M params, 1 seed) — Invalidated

- 10.8M param model on ~130K chars of Alice in Wonderland
- **100% of generated samples were verbatim memorization** (both languages)
- Grammar scores (98.8% English, 100% Lojban) measured the book, not the model
- **Only valid signal**: val loss gap — Lojban 2.29 vs English 3.08 (−25%)
- Lesson: 10.8M params on 130K chars = pure memorization, need more data + less model

### V2 (4 books, 0.8M + 3.2M params, 3 seeds) — Completed

Small (~835K params) and medium (~3.2M params) runs completed. Large skipped
(data insufficient at 407K chars to benefit from 10.8M params).

**Key quantitative findings:**

| Metric | English Small | Lojban Small | English Medium | Lojban Medium |
|--------|---------------|--------------|----------------|---------------|
| Val BPC | 1.801 | **1.477** | 1.757 | **1.442** |
| Test BPC | 2.770 | **1.860** | 2.899 | **1.874** |
| Gen gap (test−val) | 0.970 | **0.384** | 1.141 | **0.432** |
| Grammar | 73.5% | **100.0%** | 79.6% | **100.0%** |
| Memorized | 0/100 | 3-6/100 | 0/100 | 2-7/100 |

- Lojban BPC ~18% lower (val) and ~33% lower (test), consistent across all seeds/sizes
- Lojban grammar 100% at all sizes; English 73→80% with 4x params
- Lojban generalizes better: test−val gap 2.5x smaller than English
- Medium barely improved over small — **data is the bottleneck** (407K chars)
- English medium suffers Book of Esther formatting contamination (~20-30% of samples)

**Qualitative finding**: Medium Lojban shows emerging semantic coherence
(multi-clause propositions, contextually appropriate dialogue) while English
remains word salad. But this is subjective and unmeasurable — motivates the
reasoning experiment.

### V3 (tiny models + bAbI reasoning, 5 sizes, 3 seeds) — Completed

Combined training (narrative + all 20 bAbI tasks). See V3_RESULTS.md for
full writeup.

**Key quantitative findings:**

| Metric | English Base | Lojban Base |
|--------|-------------|-------------|
| Val BPC | 0.960 | 0.977 |
| Test BPC | **2.827** | 2.857 |
| Grammar | 96.8% | **100.0%** |
| bAbI seen | **46.4%** | 20.5% |
| bAbI unseen | **38.0%** | 18.0% |
| Avg best_step | 4,367 | 1,100 |

**Critical insight: A language with more character-level entropy takes longer
to converge on character prediction.** Lojban's lower entropy causes early
stopping to fire 4x sooner, giving it 4x less bAbI exposure. Lojban
mode-collapses (predicting the same answer for all examples) while English
diversifies — but this is a training duration artifact, not a reasoning
difference. English's "reasoning" gains are mostly pattern matching on 3
tasks (coreference + induction); neither language learns yes/no discrimination.

### V3.1 (fixed 10K steps, no early stopping, 5 sizes, 3 seeds) — In Progress

Same architecture/data as V3, but `--fixed-steps 10000` removes early stopping.
Checkpoint still selected by best narrative val BPC. Results in `results/v3_1/`.
Run script: `run_v3_1.sh` on studio.

**Status**: nano (6/6), micro (6/6), mini (5/6), small and base not started.

**Key quantitative findings (averages across seeds):**

| Metric | EN nano | LJ nano | EN micro | LJ micro | EN mini | LJ mini |
|--------|---------|---------|----------|----------|---------|---------|
| Val BPC | 1.244 | **0.946** | 1.117 | **0.915** | 1.200 | **0.962** |
| Test BPC | 3.453 | **2.813** | 3.064 | **2.604** | 3.083 | **2.899** |
| Grammar | 98.2% | **100.0%** | 98.7% | **100.0%** | 98.3% | **100.0%** |
| bAbI seen | 19.0% | **20.8%** | **24.4%** | 20.9% | **28.6%** | 22.1% |
| bAbI unseen | 16.2% | **18.3%** | **21.0%** | 19.1% | **23.6%** | 19.0% |
| Avg best_step | 6,800 | 7,533 | 7,733 | 5,133 | 7,450 | 2,100 |
| Memorized | 0 | 0 | 0 | 0 | 0 | 0 |

**Comparison to V3**: The bAbI gap has narrowed dramatically. In V3, English base
had 46.4% vs Lojban 20.5% (26pp gap). In V3.1, the largest gap is mini at
28.6% vs 22.1% (6.5pp). At nano scale, Lojban actually slightly outperforms
English (20.8% vs 19.0%). Fixed steps successfully removed the gross training
duration confound.

**But a subtler confound remains**: Even with 10K fixed steps, the *best
checkpoint* is still selected by narrative val BPC. Lojban's best_step for mini
averages 2,100 vs English's 7,450 — the saved Lojban model is from early
training when bAbI patterns haven't been fully absorbed. The checkpoint selection
criterion (narrative loss) is orthogonal to bAbI performance.

**Qualitative finding on tasks 11/13 (coreference)**: English gets 73.5% on
task 11 vs Lojban's 17.5%. But this is a **vocabulary diversity** issue, not a
reasoning difference. Lojban mode-collapses to 2 location tokens (`lo panka`
66%, `lo purdi` 24% of all predictions), while English produces all 12 locations.
Multi-word Lojban locations (`lo jukpa kumfa` = 3 tokens) are harder for a tiny
model to represent distinctly than single-word English locations (`kitchen`).
English still fails on 3 of 12 locations (school 15%, market 9%, bedroom 7%),
suggesting partial memorization rather than true coreference resolution.
Tasks 17/18: both languages predict only yes/go'i for 100% of examples,
getting 64%/68% via majority-class exploitation — zero reasoning.

### Key Methodological Lessons

1. **Any experiment using character/token prediction loss for early stopping will
systematically undertrain lower-entropy languages on secondary objectives.**
Character-prediction convergence speed is orthogonal to reasoning ability, but
early stopping conflates them. V3→V3.1 showed that fixing training steps helps
but doesn't fully resolve this when checkpoint selection still uses char loss.

2. **Character-level tokenization penalizes multi-word expressions.** Lojban
locations are 2-3 tokens (`lo panka`, `lo jukpa kumfa`) while English locations
are single tokens (`park`, `kitchen`). At tiny model scale (67K-261K params),
this difference dominates over any grammatical advantage. A V4 experiment should
use subword/word-level tokenization to normalize this.

3. **Checkpoint selection must align with the evaluation objective.** Selecting
by narrative val loss then evaluating bAbI accuracy is a misaligned objective.
The "best" checkpoint for narrative generation may be the worst for reasoning.

### V4 (BPE tokenization, 2 sizes, 3 seeds) — Completed

Uses BPE tokenization (vocab=1024) to equalize token-level information between
languages. See **V4_DESIGN.md** for design, **V4_RESULTS.md** for full results.

**Key quantitative findings:**

| Metric | EN medium | LJ medium | EN large | LJ large |
|--------|-----------|-----------|----------|----------|
| Test BPC | 2.657 | **2.251** | 2.543 | **2.321** |
| Grammar | 99.2% | **100.0%** | 98.9% | **100.0%** |
| bAbI seen | **20.8%** | 19.5% | **20.8%** | 14.5% |
| bAbI unseen | **14.4%** | 10.1% | **17.4%** | 10.7% |
| Avg best_step | 967 | 767 | 1,433 | 533 |
| Memorized | 0 | 0 | 0 | 0 |

**BPE did not resolve the checkpoint selection confound — it made it worse.**
With 1024 vocab (vs 73-85 char vocab), models memorize narrative patterns far
faster. Val BPC bottoms out at step 300-1000 (vs 2,100-7,533 in V3.1). Lojban
converges even faster, so its saved checkpoint reflects 3-10% of total training,
with minimal bAbI exposure. Large Lojban is catastrophically unstable: 2 of 3
seeds save at step 300, collapsing bAbI accuracy to ~10.5%.

**No evidence of reasoning in any condition.** All tasks are at chance baselines.
Tasks 17/18 score 64/68% via 100% yes/go'i prediction (majority class). No task
shows signal above noise for either language. Both languages at 570K-758K params
are too small for bAbI reasoning regardless of tokenization.

**Files**: `experiment_v4.py` (main script), `bpe_tokenizer.py` (tokenizer module)

## Project Structure

```
experiment.py          # V1 script: single book, single size, single seed
experiment_v2.py       # V2 script: multi-corpus, multi-size, multi-seed suite
experiment_v3.py       # V3 script: tiny models + bAbI reasoning tasks
experiment_v4.py       # V4 script: BPE tokenization + bAbI reasoning
bpe_tokenizer.py       # BPE tokenizer wrapper (used by V4)
analyze_results.py     # Post-hoc analysis: t-tests, effect sizes, scaling trends
eval_english.py        # Standalone English grammar eval (on studio only)
V1_RESULTS.md          # Detailed V1 results and post-mortem
V3_RESULTS.md          # Detailed V3 results and analysis
V3_1_RESULTS.md        # V3.1 fixed-step results (in progress)
V2_RESULTS.md          # Detailed V2 results (small + medium)
V4_RESULTS.md          # Detailed V4 results (completed)
V4_DESIGN.md           # V4 BPE experiment design
REASONING_EXPERIMENT.md # Reasoning experiment design (bAbI tasks)
alice_english.txt      # Original training data (V1)
alice_lojban.txt       # Original training data (V1)
babi/                  # bAbI task generator module
  vocab.py               # Vocabulary pools with train/test splits
  tasks.py               # All 20 task generators (English + Lojban)
  generate.py            # Data generation script
  data/                  # Generated data (canonical source)
corpus/                # Extended corpus (V2 + V3)
  alice_in_wonderland/   # Training book 1
  wizard_of_oz/          # Training book 2
  esther/                # Training book 3
  in_a_grove/            # Training book 4
  metamorphosis/         # Held-out test book
  tatoeba/               # Parallel sentences for grammar checker calibration
  babi/                  # bAbI data copy for V3 (20 task dirs, 8 files each)
  extract_texts.py       # HTML extraction utility
ilmentufa/             # Lojban grammar parser (camxes) - used for eval
results/
  english_samples.json   # V1 generated samples (on studio)
  lojban_samples.json    # V1 generated samples (on studio)
  english_grammar.json   # V1 grammar eval (on studio)
  v2/                    # V2 results (on studio)
    calibration.json       # Tatoeba grammar checker baseline rates
    corpus_info.json       # Data split statistics
    small/                 # Small model results (complete, 6 runs)
      english_seed{42,137,2024}/
        result.json          # Metrics (BPC, grammar, structural, memorization)
        samples.json         # Generated text samples
      lojban_seed{42,137,2024}/
    medium/                # Medium model results (complete, 6 runs)
  v3/                    # V3 results (early stopping, completed)
    corpus_info.json       # Combined data statistics
    {nano,micro,mini,small,base}/
      {english,lojban}_seed{42,137,2024}/
        result.json          # All metrics (bAbI accuracy + narrative)
        babi_predictions.json # Per-example bAbI predictions
  v3_1/                  # V3.1 results (fixed 10K steps, no early stopping)
        samples.json         # Narrative generated samples
  v4/                    # V4 results (BPE tokenization)
    tokenizer_english.json # Saved English BPE tokenizer
    tokenizer_lojban.json  # Saved Lojban BPE tokenizer
    corpus_info.json       # BPE-specific data statistics
    calibration.json       # Tatoeba grammar checker baseline
    {medium,large}/
      {english,lojban}_seed{42,137,2024}/
        result.json          # All metrics (bAbI accuracy + narrative)
        babi_predictions.json # Per-example bAbI predictions
        samples.json         # Narrative generated samples
```

## How to Run

### V4 Experiment (current)

**Locally (quick sanity check):**
```bash
cd /Users/billy/repo/lojban_experiment
python3 experiment_v4.py --quick --skip-grammar     # medium, 500 steps, 2 bAbI tasks
python3 experiment_v4.py --quick --skip-narrative-eval  # bAbI only, fastest
```

**Full run options:**
```bash
python3 experiment_v4.py --size medium              # all 6 medium runs (3 seeds x 2 langs)
python3 experiment_v4.py --size large --seed 42     # single seed, large model
python3 experiment_v4.py                            # all 12 runs (2 sizes x 2 x 3)
python3 experiment_v4.py --skip-grammar             # skip grammar eval (no Java/Node)
python3 experiment_v4.py --skip-narrative-eval      # bAbI accuracy only, no narrative eval
python3 experiment_v4.py --babi-tasks 1 6 15        # specific bAbI tasks only
python3 experiment_v4.py --max-steps 5000           # override default 10K steps
```

Requires: `torch`, `tokenizers`. Optional: `language_tool_python` + Java (English grammar), `node` (Lojban grammar via camxes).

### V3 Experiment

**Full run options:**
```bash
python3 experiment_v3.py --size nano                # all 6 nano runs (3 seeds x 2 langs)
python3 experiment_v3.py --size small --seed 42     # single seed, small model
python3 experiment_v3.py                            # all 30 runs (5 sizes x 2 x 3)
python3 experiment_v3.py --skip-grammar             # skip grammar eval (no Java/Node)
python3 experiment_v3.py --skip-narrative-eval      # bAbI accuracy only, no narrative eval
python3 experiment_v3.py --babi-tasks 1 6 15        # specific bAbI tasks only
python3 experiment_v3.py --fixed-steps 10000        # V3.1: no early stopping, results in v3_1/
```

**Run order (recommended):**
```bash
python3 experiment_v3.py --size nano     # ~20 min (6 runs)
python3 experiment_v3.py --size micro    # ~40 min
python3 experiment_v3.py --size mini     # ~90 min
python3 experiment_v3.py --size small    # ~2 hrs
python3 experiment_v3.py --size base     # ~3 hrs
```

Requires: `torch`. Optional: `language_tool_python` + Java (English grammar), `node` (Lojban grammar via camxes).

### V2 Experiment

**Locally (quick sanity check):**
```bash
cd /Users/billy/repo/lojban_experiment
python3 experiment_v2.py --quick                    # 500 steps, 1 seed, small model
python3 experiment_v2.py --quick --skip-grammar     # even faster (no Java/Node needed)
```

**Full run options:**
```bash
python3 experiment_v2.py --size small               # all 6 small runs (3 seeds x 2 langs)
python3 experiment_v2.py --size medium --seed 42    # single seed, medium model
python3 experiment_v2.py                            # all 18 runs (~8-10 hrs on MPS)
python3 experiment_v2.py --skip-grammar             # skip grammar eval (no Java/Node)
python3 experiment_v2.py --skip-calibration         # skip Tatoeba calibration
```

**Analyze results:**
```bash
python3 analyze_results.py                          # summary tables, t-tests, scaling
python3 analyze_results.py --detail                 # per-run detail
python3 analyze_results.py --format markdown        # markdown tables
python3 analyze_results.py --size small             # just small model results
```

Requires: `torch`, `language_tool_python` (optional), `node` (optional, for camxes), Java (optional, for LanguageTool).

### V1 Experiment (original, single-book)

```bash
python3 experiment.py
```

Requires: `torch`, `language_tool_python`, `node` (for camxes), Java (for LanguageTool).

### On Studio (remote, Apple Silicon)

Studio is configured in `~/.ssh/config` as host `studio` (bilalba.duckdns.org:3235, user bilal).

**Copy project:**
```bash
scp -r /Users/billy/repo/lojban_experiment studio:~/
```

**Run experiment:**
The system python (`/Applications/Xcode.app/.../python3`) is Python 3.9 with torch
installed to user site-packages. Homebrew python3 is 3.13 and does NOT have torch.
Node is at `/opt/homebrew/bin/node`. Java (openjdk) is at `/opt/homebrew/opt/openjdk/bin/java`.

Create and run `~/lojban_experiment/run.sh`:
```bash
#!/bin/bash
export PATH="/opt/homebrew/opt/openjdk/bin:/opt/homebrew/bin:$PATH"
cd ~/lojban_experiment
# V2: run by size (checkpoint between sizes)
/Applications/Xcode.app/Contents/Developer/usr/bin/python3 -u experiment_v2.py --size small > experiment_v2.log 2>&1 &
echo $! > experiment.pid
echo "Started with PID: $(cat experiment.pid)"
```

IMPORTANT notes for studio:
- Always use `-u` flag with python3 (unbuffered output), otherwise experiment.log stays empty due to buffering
- Always use the full Xcode python3 path — homebrew python3 doesn't have torch
- `nohup` does NOT work over SSH on this machine (fails with "can't detach from console" and silently drops the command)
- Use `bash run.sh` to launch, not `nohup`
- PATH must include `/opt/homebrew/opt/openjdk/bin` for LanguageTool (Java) and `/opt/homebrew/bin` for node (camxes)

**Check status:**
```bash
ssh studio 'tail -20 ~/lojban_experiment/experiment_v2.log'
ssh studio 'ps -p $(cat ~/lojban_experiment/experiment.pid) -o pid,stat,etime,%cpu'
ssh studio 'ls -la ~/lojban_experiment/results/v2/small/'  # check completed runs
```

**Run English eval only** (if LanguageTool crashed but samples exist):
There's an `eval_english.py` on studio that loads saved samples and runs LanguageTool
without retraining. Note: `language_tool_python` Match objects use snake_case attributes
(`rule_issue_type`, `rule_id`, `category`) not camelCase.

## V2 Experiment Design

See experiment_v2.py for implementation, V2_RESULTS.md for full results.

### Data
- **Training**: 4 books (~407K chars per language, truncated to equal length)
  - Alice in Wonderland, Wizard of Oz, Book of Esther, In a Grove
- **Test**: Metamorphosis (~120K chars) — stylistically distinct, held out entirely
- **Split**: 90% train / 5% val / 5% prompt-source (within training books)
- **Calibration**: Tatoeba parallel sentences (200 sampled, 100% pass both checkers)

### Model Sizes

| Size   | d_model | n_layer | n_head | ~Params | Dropout | Status |
|--------|---------|---------|--------|---------|---------|--------|
| Small  | 128     | 4       | 2      | ~0.8M   | 0.15    | Done   |
| Medium | 256     | 4       | 4      | ~3.2M   | 0.20    | Done   |
| Large  | 384     | 6       | 6      | ~10.8M  | 0.30    | Skipped (data insufficient) |

### Key Improvements over V1
- **Early stopping** (patience=1500 steps, up to 15K max)
- **Higher dropout** scaled by model size
- **Best checkpoint** saved by val loss
- **BPC metric** (bits-per-character), normalizes across vocab sizes
- **Memorization detection** via longest common substring (LCS > 50 chars)
- **Grammar eval on novel text only** (non-memorized samples)
- **Tatoeba calibration** establishes baseline pass rates for grammar checkers
- **3 seeds** (42, 137, 2024) for confidence intervals
- **Held-out test book** (Metamorphosis) for generalization measurement

## V3 Experiment Design

See experiment_v3.py for implementation, REASONING_EXPERIMENT.md for motivation.

### Goals
1. **Find where Lojban grammar breaks**: 5 sizes from ~67K to ~837K params
2. **Measure reasoning**: bAbI exact-match accuracy (all 20 tasks)
3. **Combined training**: narrative corpus (~407K chars) + bAbI tasks (~2.4-2.6M chars)

### Model Sizes

| Size  | d_model | n_layer | n_head | ctx_len | dropout | ~Params |
|-------|---------|---------|--------|---------|---------|---------|
| nano  | 48      | 2       | 2      | 128     | 0.05    | ~67K    |
| micro | 64      | 3       | 2      | 128     | 0.08    | ~164K   |
| mini  | 80      | 3       | 2      | 256     | 0.10    | ~261K   |
| small | 96      | 4       | 2      | 256     | 0.12    | ~480K   |
| base  | 128     | 4       | 2      | 256     | 0.15    | ~837K   |

- nano/micro use ctx_len=128, batch_size=128 (smaller context, compensating batch)
- mini/small/base use ctx_len=256, batch_size=64 (comparable to V2)

### Training Data
- **Narrative**: Same 4 books as V2 (~407K chars/lang, truncated to equal)
- **bAbI**: All 20 tasks x 1000 train examples (~2.4M EN / ~2.6M LJ chars)
- **Combined**: ~2.8M EN / ~3.0M LJ chars total
- Tokenizer built from combined text (vocab ~85 EN / ~73 LJ)
- 90/5/5 split; early stopping on narrative val loss; max_steps=20000

### bAbI Evaluation
- **Exact match accuracy**: feed context+question, greedy decode answer, compare to expected
- Evaluated on **test_seen** (training vocab) and **test_unseen** (held-out vocab)
- Per-task and overall accuracy reported
- 20 tasks testing: fact lookup, multi-hop, yes/no, counting, lists, negation,
  indefinite knowledge, coreference, conjunction, time, deduction, induction,
  spatial reasoning, size reasoning, path finding, motivation

### Narrative Evaluation (retained from V2)
- Test BPC on held-out Metamorphosis
- 50 generated samples (80% in-domain, 20% out-of-domain)
- Memorization detection (LCS threshold=50)
- Grammar (camxes for Lojban, LanguageTool for English) on novel samples
- Structural metrics (char KL, n-gram diversity, repetition, word-length similarity)

### Results Schema
Per run: `results/v3/{size}/{language}_seed{seed}/`
- `result.json` — config, training info, bAbI accuracy (per-task + overall for
  test_seen/test_unseen), test BPC, grammar, structural, memorization
- `babi_predictions.json` — per-example expected/predicted/correct for error analysis
- `samples.json` — narrative generated samples

### Key Differences from V2
- 5 smaller sizes (67K-837K) instead of 3 larger (800K-10.8M)
- bAbI reasoning as primary new metric (exact-match accuracy)
- Combined training data (~2.8M chars vs ~407K)
- Variable ctx_len per size (128 or 256)
- 30 total runs (5 sizes x 2 langs x 3 seeds)
