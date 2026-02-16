# V3.1 Experiment Results: Fixed Training Steps (No Early Stopping)

**Status**: In progress. Nano (6/6), micro (6/6), mini (5/6) complete. Small
and base not yet started.

## Motivation

V3 found that English dramatically outperformed Lojban on bAbI reasoning
(46.4% vs 20.5% at base size), but this was confounded by early stopping:
Lojban's lower character entropy caused convergence 4x faster, giving it 4x
less bAbI exposure. V3.1 reruns the same experiment with `--fixed-steps 10000`
to remove the training duration confound.

## Changes from V3

Only one change: `--fixed-steps 10000` (patience set to 10001, effectively
disabling early stopping). Everything else is identical — same model sizes,
same data, same seeds, same evaluation. Results saved to `results/v3_1/`
instead of `results/v3/`.

**What this fixes**: Both languages now train for exactly 10,000 steps, seeing
equal amounts of bAbI data.

**What this does NOT fix**: The best checkpoint is still selected by narrative
val BPC (lowest combined val loss during training). This means the saved model
may not correspond to peak bAbI performance.

---

## Quantitative Results

### BPC (Bits Per Character)

| Size | English Val | Lojban Val | English Test | Lojban Test |
|------|------------|------------|-------------|-------------|
| nano | 1.244 | **0.946** | 3.453 | **2.813** |
| micro | 1.117 | **0.915** | 3.064 | **2.604** |
| mini | 1.200 | **0.962** | 3.083 | **2.899** |

Mean across seeds (3 for nano/micro, 2 for mini).

BPC patterns are consistent with V3: Lojban has ~20-24% lower val BPC and
~6-18% lower test BPC at all sizes. The gap narrows slightly with scale.

### Best Step (Checkpoint Selection)

| Size | English avg best_step | Lojban avg best_step | Ratio |
|------|----------------------|---------------------|-------|
| nano | 6,800 | 7,533 | 0.9x |
| micro | 7,733 | 5,133 | 1.5x |
| mini | 7,450 | 2,100 | **3.5x** |

At nano scale, the best-step gap is negligible (both models plateau around
step 7K). At mini scale, the gap reappears: Lojban's narrative val loss
bottoms out at step ~2,100 while English continues improving until step ~7,450.

This is the **subtler confound** that fixed steps does not resolve. Even though
both languages train for 10K steps, the *saved checkpoint* reflects where
narrative val loss was lowest. Lojban's checkpoint comes from step 2,100 — the
model at that point has seen only ~21% of the total bAbI training data. English's
checkpoint at step 7,450 has seen ~75%.

### Comparison: V3 vs V3.1 best_step

| Size | V3 EN | V3.1 EN | V3 LJ | V3.1 LJ |
|------|-------|---------|-------|---------|
| nano | 9,567 | 6,800 | 7,200 | 7,533 |
| micro | 8,667 | 7,733 | 3,833 | 5,133 |
| mini | 9,933 | 7,450 | 2,300 | 2,100 |

V3 English ran longer because early stopping patience allowed overshoot. V3.1
caps at 10K but the val-loss-selected checkpoint still favors English at mini+.
Lojban best_step barely changes between V3 and V3.1 — the narrative loss curve
shape is the same, just without the early stop cutoff.

### Grammar

| Size | English | Lojban |
|------|---------|--------|
| nano | 98.2% | **100.0%** |
| micro | 98.7% | **100.0%** |
| mini | 98.3% | **100.0%** |

Identical to V3. Lojban 100% at all sizes. English ~98%.

### Memorization

Zero memorization across all completed runs. Consistent with V3.

---

## bAbI Reasoning Accuracy

### Overall Accuracy

| Size | EN Seen | LJ Seen | EN Unseen | LJ Unseen |
|------|---------|---------|-----------|-----------|
| nano | 19.0% | **20.8%** | 16.2% | **18.3%** |
| micro | **24.4%** | 20.9% | **21.0%** | 19.1% |
| mini | **28.6%** | 22.1% | **23.6%** | 19.0% |

### V3 vs V3.1 bAbI Comparison

| Size | V3 EN Seen | V3.1 EN Seen | V3 LJ Seen | V3.1 LJ Seen |
|------|-----------|-------------|-----------|-------------|
| nano | 21.0% | 19.0% | 20.3% | **20.8%** |
| micro | **28.8%** | 24.4% | 20.7% | **20.9%** |
| mini | **39.6%** | 28.6% | 21.5% | **22.1%** |

The fixed-step approach had two effects:

1. **Lojban improved slightly** (20.3→20.8% at nano, 21.5→22.1% at mini). The
   additional training beyond V3's early-stop point helped marginally.

2. **English got worse** (21.0→19.0% at nano, 39.6→28.6% at mini). This is
   counterintuitive. English's best checkpoint in V3 was selected at step ~9,900
   (mini), benefiting from the full training duration. In V3.1, the best val BPC
   lands at step ~7,450 — still late, but the cap at 10K prevents overshoot that
   sometimes helped English in V3.

3. **The gap narrowed dramatically**: V3 mini had an 18.1pp gap (39.6 vs 21.5).
   V3.1 mini has a 6.5pp gap (28.6 vs 22.1). At nano, Lojban actually leads
   by 1.8pp.

### Per-Task Accuracy (mini, test_seen)

| Task | Description | V3.1 EN | V3.1 LJ | V3 EN | V3 LJ | Notes |
|------|-------------|---------|---------|-------|-------|-------|
| 1 | Single fact | 26.7% | 13.8% | 32.5% | 13.8% | |
| 2 | Two facts | 8.8% | 10.5% | 11.5% | 10.5% | LJ slightly ahead |
| 3 | Three facts | 26.8% | 9.8% | 48.0% | 9.8% | EN dropped from V3 |
| 4 | Two arg | 10.5% | 10.2% | 20.7% | 9.3% | Near parity |
| 5 | Three arg | 12.8% | 8.8% | 16.8% | 8.8% | |
| 6 | Yes/no | 52.2% | 47.2% | 50.3% | 51.8% | Both ~chance |
| 7 | Counting | 41.0% | 30.2% | 38.2% | 19.7% | LJ improved |
| 8 | Lists/sets | 0.0% | 0.8% | 0.0% | 0.7% | Neither learns |
| 9 | Negation | 48.3% | 51.5% | 50.0% | 49.0% | Both ~chance |
| 10 | Indefinite | 35.5% | 31.0% | 33.3% | 33.3% | Similar |
| 11 | Coreference | **49.2%** | 16.5% | **82.7%** | 11.2% | EN dropped 33pp |
| 12 | Conjunction | 30.8% | 15.5% | 41.3% | 12.0% | |
| 13 | Compound coref | **50.5%** | 11.8% | **80.2%** | 10.0% | EN dropped 30pp |
| 14 | Time | 10.8% | 5.5% | 23.2% | 9.7% | |
| 15 | Deduction | 10.0% | 11.2% | 11.5% | 11.5% | Parity |
| 16 | Induction | 26.2% | **35.8%** | 47.0% | 20.3% | **LJ ahead** |
| 17 | Positional | 64.0% | 64.0% | 64.0% | 64.0% | Majority class |
| 18 | Size | 68.0% | 68.3% | 67.3% | 59.3% | Majority class |
| 19 | Path finding | 0.0% | 0.0% | 0.0% | 0.0% | Neither learns |
| 20 | Motivation | 0.0% | 0.0% | 0.0% | 0.0% | Neither learns |

Key changes from V3:

- **Tasks 11/13 (coreference)**: English dropped from ~80-83% to ~49-51%.
  These were English's strongest tasks in V3, inflated by longer training and
  pattern-matching on location words. With the checkpoint at a different
  point, the advantage halves.
- **Task 16 (induction)**: Lojban now *outperforms* English (35.8% vs 26.2%).
  This is the only task where Lojban clearly leads.
- **Tasks 6, 9, 17, 18**: Unchanged at majority-class baseline for both
  languages. Zero reasoning signal.

---

## The Mode Collapse Problem (Revisited)

### Lojban still collapses on location vocabulary

V3.1's fixed training did not fix Lojban's mode collapse on location-answer
tasks. At mini scale:

**Lojban mini seed 42 — tasks 1, 11, 13 prediction distributions:**

| Task | `lo panka` | `lo purdi` | Other | Diversity |
|------|-----------|-----------|-------|-----------|
| T01 | 65 (33%) | 109 (55%) | 26 (13%) | 4 |
| T11 | 131 (66%) | 47 (24%) | 22 (11%) | 6 |
| T13 | 104 (52%) | 86 (43%) | 10 (5%) | 5 |

**English mini seed 42 — same tasks:**

| Task | Top 3 predictions | Diversity |
|------|-------------------|-----------|
| T01 | bathroom: 33, park: 26, playground: 25 | 12 |
| T11 | bathroom: 40, park: 28, playground: 28 | 12 |
| T13 | bathroom: 37, park: 29, playground: 27 | 12 |

Lojban uses 2-6 distinct predictions; English uses all 12. The pattern is
consistent across seeds.

### Why: multi-token vs single-token locations

The root cause is character-level tokenization interacting with Lojban's
multi-word location expressions:

| English | Chars | Lojban | Chars |
|---------|-------|--------|-------|
| park | 4 | lo panka | 8 |
| kitchen | 7 | lo jukpa kumfa | 14 |
| bathroom | 8 | lo lumku'a | 9 |
| playground | 10 | lo kelci stuzi | 14 |

A 261K-param character-level model must maintain coherent generation across
8-14 characters for each Lojban location. It collapses to the shortest/most
frequent tokens (`lo panka`, `lo purdi`) because it can't reliably produce
12 distinct multi-character sequences. English locations are shorter and more
diverse at the character level.

This is not a reasoning failure — it's a **representation bottleneck**.

### Where Lojban mode collapse breaks

Interestingly, Lojban does NOT collapse on non-location tasks:

- **Task 16 (induction, color answers)**: 5 distinct predictions (`xekri`,
  `blanu`, `blabi`, etc.) — single-word Lojban answers. Lojban scores **35.8%**
  vs English 26.2%.
- **Task 7 (counting)**: 3 distinct predictions (`re`, `na go'i`, `no`).
  Lojban scores 30.2% vs V3's 19.7%.

When Lojban answers are single tokens, the mode collapse disappears and
performance is competitive or better. This confirms the issue is multi-token
generation, not reasoning capacity.

---

## Structural Metrics (mini, averaged)

| Metric | English | Lojban |
|--------|---------|--------|
| Repetition r10 | 0.503 | 0.575 |
| Repetition r20 | 0.174 | 0.254 |
| Ngram div (3) | 0.104 | 0.087 |
| Ngram div (4) | 0.189 | 0.167 |
| Char KL div | 0.097 | 0.194 |
| Word len sim | 0.919 | 0.849 |

Similar to V3. Lojban has higher repetition and higher char KL divergence.
English has more n-gram diversity. Both are worse than V2 due to smaller model
sizes and bAbI-dominated training data.

---

## Per-Run Detail

```
Size   Lang     Seed   BestStep  ValBPC  TestBPC  SeenAcc  UnseenAcc  Grammar  Memo
------------------------------------------------------------------------------------
nano   english  42       7300    1.335   3.427    17.8%    15.8%      97.9%    0
nano   english  137      6400    1.190   3.475    17.5%    15.3%      98.9%    0
nano   english  2024     6700    1.208   3.458    21.6%    17.5%      97.8%    0
nano   lojban   42       7300    0.918   2.809    19.5%    17.4%     100.0%    0
nano   lojban   137      8900    0.967   2.815    21.1%    18.7%     100.0%    0
nano   lojban   2024     6400    0.953   2.813    21.9%    18.9%     100.0%    0
micro  english  42       8100    1.131   3.054    23.2%    20.1%      98.3%    0
micro  english  137      6400    1.065   3.089    29.2%    23.9%      98.3%    0
micro  english  2024     8700    1.157   3.049    20.7%    18.9%      99.5%    0
micro  lojban   42       3400    0.916   2.698    19.2%    18.4%     100.0%    0
micro  lojban   137      7000    0.919   2.530    21.4%    19.6%     100.0%    0
micro  lojban   2024     5000    0.911   2.583    22.0%    19.2%     100.0%    0
mini   english  42       7300    1.196   3.081    32.4%    26.6%      98.3%    0
mini   english  137      7600    1.204   3.086    24.8%    20.6%      98.4%    0
mini   lojban   42       2200    0.974   2.896    22.4%    19.4%     100.0%    0
mini   lojban   137      2000    0.950   2.902    21.8%    18.7%     100.0%    0
```

Mini english_seed2024 and mini lojban_seed2024 not yet complete. Small and
base sizes not yet started.

---

## Key Findings

### 1. Fixed steps dramatically narrows the bAbI gap

V3's headline result (English 46.4% vs Lojban 20.5% at base) was largely a
training duration artifact. With equal training steps, the gap at mini drops
from 18.1pp to 6.5pp. At nano, Lojban slightly outperforms English. The
gross confound from V3 is resolved.

### 2. A subtler confound remains: checkpoint selection

Even with fixed steps, the best checkpoint is selected by narrative val BPC.
Lojban's checkpoint is from step ~2,100 (mini) while English's is from step
~7,450. The saved Lojban model hasn't absorbed as many bAbI patterns as the
saved English model. A fully clean comparison would require selecting
checkpoints by bAbI validation accuracy, or evaluating at a fixed step for
both languages.

### 3. Lojban's mode collapse is a multi-token representation issue

Lojban collapses to 2-3 location predictions not because it can't reason,
but because character-level generation of multi-word Lojban locations
(`lo jukpa kumfa` = 14 chars) overwhelms the model's capacity. When answers
are single tokens (task 16 colors, task 7 numbers), Lojban performs
competitively or better than English.

### 4. English's coreference advantage is fragile

English's dominant tasks 11/13 dropped from ~80-83% (V3) to ~49-51% (V3.1)
with a different checkpoint. The V3 result reflected a specific snapshot where
pattern-matching on location tokens happened to be strong — not robust
coreference resolution.

### 5. Neither language does real reasoning

The evidence against genuine reasoning:
- Tasks 17/18: Both predict only yes/go'i (100% of predictions), scoring 64%/68%
  via majority-class exploitation
- Tasks 8, 19, 20: 0% for both — true multi-hop and complex tasks are unsolved
- Tasks 6, 9: Both at ~50% (chance) — no yes/no discrimination
- English's best tasks (11, 13) are solvable by surface string matching

### 6. Lojban grammar remains perfectly learnable at tiny scale

100% grammaticality across all 10 completed Lojban runs (67K-261K params),
consistent with V2 and V3. This is the one unambiguous finding across all
experiment versions.

---

## Implications for V4

V3.1 clarified which confounds are fundamental vs fixable:

**Fixed by V3.1**: The gross training duration artifact (Lojban training 4x
shorter due to early stopping).

**Not fixed by V3.1**:
1. **Checkpoint selection by narrative loss** — still creates unequal bAbI
   exposure at the evaluated checkpoint
2. **Character-level tokenization** — penalizes multi-word Lojban expressions,
   causing mode collapse on location answers
3. **bAbI tasks don't test grammar** — coreference and memory tasks test
   pattern matching, not the structural parsing advantage Lojban should have

A V4 experiment should address these with:
- Subword/word-level tokenization (normalizes multi-token locations)
- Checkpoint selection by bAbI accuracy (aligns objective with evaluation)
- Two-stage training (pretrain narrative, fine-tune bAbI)
- Tasks that test structural parsing and compositional generalization

---

## Files

Results are stored at `studio:~/lojban_experiment/results/v3_1/`:
- `{nano,micro,mini}/{english,lojban}_seed{42,137,2024}/`
  - `result.json` — all metrics
  - `babi_predictions.json` — per-example predictions
  - `samples.json` — narrative generated samples
