# V3 Experiment Results: Tiny Models + bAbI Reasoning

## Experiment Configuration

### Data
- **Narrative corpus**: Same 4 parallel books as V2 (Alice in Wonderland, Wizard
  of Oz, Book of Esther, In a Grove) — 407,194 chars per language
- **bAbI reasoning tasks**: All 20 tasks x 1000 training examples — 2,390,610
  chars (English), 2,627,365 chars (Lojban)
- **Combined training data**: ~2.8M chars English, ~3.0M chars Lojban
  (narrative ~14-15% of total, bAbI ~85-87%)
- **Split**: 90% train / 5% val / 5% prompt-source (of combined text)
- **Test corpus**: Metamorphosis (held out entirely) — 121K chars English,
  114K chars Lojban
- **bAbI test**: 200 examples/task, test_seen (training vocab) + test_unseen
  (held-out vocab)
- **Calibration**: Tatoeba parallel sentences — 200 sampled, 100% pass rate

### Models

| Size | d_model | n_layer | n_head | ctx_len | dropout | ~Params |
|------|---------|---------|--------|---------|---------|---------|
| nano | 48 | 2 | 2 | 128 | 0.05 | ~67K |
| micro | 64 | 3 | 2 | 128 | 0.08 | ~164K |
| mini | 80 | 3 | 2 | 256 | 0.10 | ~261K |
| small | 96 | 4 | 2 | 256 | 0.12 | ~480K |
| base | 128 | 4 | 2 | 256 | 0.15 | ~837K |

All character-level GPT models. nano/micro use batch_size=128, ctx_len=128.
mini/small/base use batch_size=64, ctx_len=256. Max 20,000 steps, cosine LR
schedule (3e-4 peak, 100-step warmup), early stopping with patience 1,500 steps
on combined val loss.

### Seeds
3 seeds per configuration: 42, 137, 2024. 30 total runs (5 sizes x 2 langs x 3
seeds).

---

## Quantitative Results

### BPC (Bits Per Character)

| Size | English Val | Lojban Val | English Test | Lojban Test |
|------|------------|------------|-------------|-------------|
| nano | 1.180 | **0.932** | 3.309 | **2.756** |
| micro | 1.061 | **0.914** | 2.933 | **2.667** |
| mini | 1.045 | **0.950** | 2.931 | **2.866** |
| small | 0.945 | 0.949 | 2.830 | **2.728** |
| base | 0.960 | 0.977 | **2.827** | 2.857 |

Mean across 3 seeds. Val BPC is on the combined (narrative + bAbI) validation
split. Test BPC is on held-out Metamorphosis.

**Scaling pattern diverges from V2.** In V2 (narrative-only, 0.8M-3.2M params),
Lojban had a consistent ~18% val BPC advantage at all sizes. Here the picture is
different:

- At nano/micro, Lojban's val BPC advantage is large (0.93 vs 1.18 at nano)
- By small/base, val BPC converges to near-parity (~0.95 for both)
- Test BPC (Metamorphosis, pure narrative) also converges, with English
  slightly better at base

The convergence is likely because the combined training data is ~85% bAbI. Both
languages' bAbI text is highly templated, reducing the structural advantage
Lojban had on narrative text. The larger models learn the templates quickly,
leaving val BPC dominated by residual narrative loss — which is similar for both
at these tiny model sizes.

### Generalization Gap (Test BPC - Val BPC)

| Size | English | Lojban |
|------|---------|--------|
| nano | +2.128 | +1.824 |
| micro | +1.872 | +1.753 |
| mini | +1.886 | +1.916 |
| small | +1.885 | +1.778 |
| base | +1.867 | +1.880 |

The generalization gap is much larger than V2 (V2: English ~1.0, Lojban ~0.4)
because the models trained on ~85% bAbI templates but are tested on literary
narrative. The gap is roughly equal between languages — training on bAbI hurt
narrative generalization equally.

### Grammar

| Size | English | Lojban |
|------|---------|--------|
| nano | 98.7% | **100.0%** |
| micro | 98.1% | **100.0%** |
| mini | 97.5% | **100.0%** |
| small | 97.1% | **100.0%** |
| base | 96.8% | **100.0%** |

Lojban remains at 100% across all sizes, consistent with V2.

English grammar is notably higher than V2 (96-99% here vs 73-80% in V2). This
is because the combined training data includes bAbI text — short, grammatically
simple sentences that are easy to imitate. The generated samples contain more
simple sentence structures, boosting grammaticality scores. English grammar
actually trends *down* slightly with scale (98.7% → 96.8%), possibly because
larger models generate more varied (and riskier) constructions.

### bAbI Reasoning Accuracy

| Size | English Seen | Lojban Seen | English Unseen | Lojban Unseen |
|------|-------------|-------------|----------------|---------------|
| nano | 21.0% | 20.3% | 18.3% | 18.5% |
| micro | 28.8% | 20.7% | 24.4% | 18.4% |
| mini | 39.6% | 21.5% | 31.4% | 18.8% |
| small | 44.3% | 24.7% | 35.8% | 21.5% |
| base | **46.4%** | 20.5% | **38.0%** | 18.0% |

Mean across 3 seeds. "Seen" = test examples using training vocabulary. "Unseen"
= test examples using held-out vocabulary.

**English scales dramatically; Lojban stays flat.** English accuracy more than
doubles from nano to base (21% → 46%). Lojban hovers around 20% regardless of
model size.

However, this headline finding is misleading. The analysis below shows that
Lojban's failure is primarily a mode-collapse artifact driven by a training
confound, not evidence that Lojban can't learn reasoning.

### Per-Task bAbI Accuracy (base, test_seen)

| Task | Description | English | Lojban | Notes |
|------|-------------|---------|--------|-------|
| 1 | Single supporting fact | 39.2% | 13.0% | |
| 2 | Two supporting facts | 29.7% | 10.7% | |
| 3 | Three supporting facts | 69.2% | 9.7% | |
| 4 | Two arg relations | 44.0% | 8.5% | |
| 5 | Three arg relations | 29.0% | 10.2% | |
| 6 | Yes/no questions | 52.8% | 52.0% | Both near chance (50%) |
| 7 | Counting | 41.2% | 19.8% | |
| 8 | Lists / sets | 0.0% | 0.8% | Neither learns |
| 9 | Simple negation | 52.0% | 52.5% | Both near chance (50%) |
| 10 | Indefinite knowledge | 34.3% | 30.5% | Closest gap |
| 11 | Basic coreference | 99.7% | 11.5% | English near-perfect |
| 12 | Conjunction | 50.7% | 11.8% | |
| 13 | Compound coreference | 97.3% | 11.7% | English near-perfect |
| 14 | Time reasoning | 41.5% | 10.3% | |
| 15 | Basic deduction | 20.0% | 11.7% | |
| 16 | Basic induction | 95.3% | 24.3% | English near-perfect |
| 17 | Positional reasoning | 64.0% | 62.3% | Both = majority class |
| 18 | Size reasoning | 68.0% | 58.7% | Both near majority class |
| 19 | Path finding | 0.0% | 0.0% | Neither learns |
| 20 | Agent motivations | 0.0% | 0.0% | Neither learns |

Three categories emerge:

1. **Neither language learns** (tasks 8, 19, 20): 0% for both. These require
   capabilities beyond what ~837K params can support.

2. **Both at chance level** (tasks 6, 9, 17, 18): Yes/no tasks where both
   models simply predict the majority class. English base predicts "yes" 100%
   of the time on task 17 (which is 64% yes → 128/200 "correct"). Neither
   model discriminates.

3. **English scales, Lojban doesn't** (tasks 1-5, 7, 11-16): This is where
   the gap opens. English improves dramatically on tasks 3, 11, 13, 16 (up to
   99.7%). Lojban stays at ~10%.

---

## The Mode Collapse Problem

### What Lojban predictions look like

Task 1 (single supporting fact), base model, seed 42:

| # | English expected | English predicted | | Lojban expected | Lojban predicted |
|---|-----------------|-------------------|---|-----------------|------------------|
| 1 | market | garden (XX) | | lo zarci | lo purdi (XX) |
| 2 | hallway | hallway (OK) | | lo vrogai | lo purdi (XX) |
| 3 | office | market (XX) | | lo briju | lo purdi (XX) |
| 4 | garden | garden (OK) | | lo purdi | lo purdi (OK) |
| 5 | hallway | hallway (OK) | | lo vrogai | lo purdi (XX) |

English produces 12 unique predictions spread roughly uniformly across the 12
possible answers. Lojban produces **1 unique prediction**: `lo purdi` ("garden")
for all 200 test examples.

This is complete mode collapse. For tasks 1-5 and 11-14, Lojban base outputs the
same answer for every input. It scores ~8-13% — matching only the examples where
the expected answer happens to be the collapsed prediction.

### Mode collapse is not just about multi-word answers

Initial hypothesis: Lojban answers are multi-word (`lo purdi` = 2 words, `lo
jukpa kumfa` = 3 words) while English answers are single-word (`garden`),
making them harder to generate correctly.

This hypothesis is **wrong**. Task 16 (basic induction) has single-word Lojban
answers (`blabi`, `xunre`, `crino`, `xekri`, `blanu`). English base scores
95.3% with 5 unique predictions spread evenly. Lojban base scores 24.3% with
only 3 unique predictions (`xunre` at 66%, `xekri` at 20%, `blanu` at 14%).

Stripping the `lo` prefix from Lojban answers (comparing only the content word)
does not change accuracy. The model isn't "knowing the right answer but
formatting it wrong" — it genuinely only generates one answer.

### English also mode-collapses at nano, then breaks out

English nano shows the same pattern: task 1 has only 4 unique predictions with
`bedroom` at 68%. Task 13 at nano has `bedroom` at 92%. But by base size,
English breaks out into 12 unique predictions with a flat distribution. Lojban
never breaks out.

---

## Root Cause: Early Stopping and Training Exposure

### The training duration gap

| Size | English avg best_step | Lojban avg best_step | Ratio |
|------|----------------------|---------------------|-------|
| nano | 9,567 | 7,200 | 1.3x |
| micro | 8,667 | 3,833 | 2.3x |
| mini | 9,933 | 2,300 | 4.3x |
| small | 6,400 | 2,133 | 3.0x |
| base | 4,367 | 1,100 | **4.0x** |

At base size, English trains **4x longer** than Lojban before early stopping.

### Estimated bAbI exposure

Training data is ~85-87% bAbI by character count, so most random training chunks
are bAbI. But the total exposure still differs dramatically:

| Size | English bAbI epochs | Lojban bAbI epochs |
|------|--------------------|--------------------|
| nano | 62.2 | 43.2 |
| micro | 56.4 | 23.0 |
| mini | 64.6 | 13.8 |
| small | 41.6 | 12.8 |
| base | **28.4** | **6.6** |

At base, Lojban sees ~7 epochs of bAbI data. English sees ~28 epochs — **4x
more exposure to reasoning examples**.

### Why Lojban converges so fast

The loss trajectory for base/seed42 tells the story:

**English base/42:**
- Step 100: val_bpc 3.475
- Step 500: val_bpc 2.200
- Step 1300: val_bpc 1.342 (still improving)
- Step 5400: val_bpc 0.942 (best)
- Step 6900: early stopped

**Lojban base/42:**
- Step 100: val_bpc 3.192
- Step 500: val_bpc 1.703
- Step 1000: val_bpc 0.975
- Step 1300: val_bpc 0.958 (best)
- Step 2800: early stopped

Lojban reaches val_bpc < 1.0 at step 1000. English doesn't reach the same level
until step ~5000. Lojban's lower character entropy means the model can predict
the next character accurately with less training. The combined val loss (which is
~85% bAbI text) plateaus because the model has learned to compress the bAbI
character sequences — but it hasn't learned the Q&A structure needed to answer
questions correctly.

After best_step, Lojban's val loss bounces between 0.96-1.06, never improving.
Train BPC continues dropping to 0.5-0.6 — the model is overfitting the training
set but not learning new generalizable patterns.

### The irony

Lojban's lower entropy — the property that gives it better BPC and perfect
grammar — is what kills its bAbI performance. The model converges on character
prediction so quickly that early stopping fires before it has enough gradient
signal to learn the bAbI task structure. The bAbI "reasoning" signal is much
weaker than the character-prediction signal and gets drowned out.

### The small/seed42 outlier confirms this

The one Lojban run that performed well on bAbI (small/seed42, 35% overall) is
the one where training lasted longer. At seed 42, the small Lojban model trained
for 3,400 best steps (vs 1,500 for seeds 137 and 2024). This run shows 10
unique predictions on task 1 instead of 1 — it broke out of mode collapse
because it had more training time.

---

## Narrative Evaluation

### Memorization

No memorization detected in any V3 run (0/50 samples exceeding LCS threshold
of 50 chars at any size or language). Average LCS is 14-17 chars. This contrasts
with V2 where Lojban had 2-7 memorized samples per run.

The improvement is likely because the combined training data is ~7x larger
(~2.8-3.0M chars vs ~407K) while the model sizes are smaller (67K-837K vs
835K-3.2M).

### Structural Metrics (base, averaged)

| Metric | English | Lojban |
|--------|---------|--------|
| Repetition r10 | 0.464 | 0.581 |
| Repetition r20 | 0.158 | 0.278 |
| Repetition r50 | 0.001 | 0.005 |
| Ngram div (3) | 0.115 | 0.087 |
| Ngram div (4) | 0.208 | 0.164 |
| Ngram div (5) | 0.282 | 0.224 |
| Char KL div | 0.080 | 0.201 |
| Word len sim | 0.893 | 0.849 |

Both languages show higher repetition than V2 (English r10: 0.46 vs 0.13;
Lojban r10: 0.58 vs 0.29). Char KL divergence is also much higher (0.08/0.20
vs 0.01/0.006). The generated text is lower quality than V2 — expected, since
these models are 1-12x smaller and trained on a different data mix (85% bAbI
templates, 15% narrative).

---

## What English Actually Learned

English's bAbI gains are concentrated on a few tasks:

**Tasks that scale dramatically (nano → base):**
- Task 11 (basic coreference): 12% → **100%** (+88pp)
- Task 13 (compound coreference): 14% → **97%** (+83pp)
- Task 16 (basic induction): 27% → **95%** (+69pp)
- Task 3 (three supporting facts): 11% → 69% (+58pp)

**Tasks that barely improve:**
- Task 6 (yes/no): 50% → 53% (chance = 50%)
- Task 7 (counting): 38% → 41% (chance = 20-41%)
- Task 17 (positional): 64% → 64% (majority = 64%)
- Task 18 (size reasoning): 66% → 68% (majority = 68%)
- Tasks 8, 19, 20: ~0% → ~0%

The tasks English "solves" (11, 13, 16) share a property: they have relatively
simple surface-level patterns the model can exploit without deep reasoning.
Coreference tasks (11, 13) follow a "X is Y. Y went to Z. Where is X?" pattern
that reduces to string matching. Task 16 (induction) has fixed color-animal
mappings.

The tasks that stay flat are either inherently at chance (yes/no where both
classes are 50%) or require capabilities these tiny models don't have (path
finding, list manipulation).

### Seen vs unseen accuracy drop

| Task | English seen → unseen | Lojban seen → unseen |
|------|----------------------|---------------------|
| 1 | 39% → 31% (−8pp) | 13% → 7% (−6pp) |
| 6 | 53% → 56% (+3pp) | 52% → 49% (−4pp) |
| 16 | 95% → 73% (**−22pp**) | 24% → 18% (−6pp) |

English task 16 drops 22 percentage points from seen to unseen vocabulary. This
suggests the model partially memorized specific entity-answer mappings rather
than learning a general rule. Lojban's drop is smaller, but its seen accuracy is
already near-chance.

---

## Confounds and Limitations

### 1. Early stopping on combined val loss is biased

The most important confound. Early stopping criterion is combined val cross-
entropy (narrative + bAbI). Lojban converges faster on character prediction,
triggering early stopping before the model can learn bAbI task structure. English
models train 2-4x longer and see 4x more bAbI examples. This alone could
explain the entire bAbI accuracy gap.

### 2. bAbI answer format disadvantages Lojban

English bAbI answers are single words (`garden`, `kitchen`, `yes`, `no`).
Lojban answers are 1-3 words (`lo purdi`, `lo jukpa kumfa`, `na go'i`).
Character-level greedy decoding must maintain coherence over a longer sequence
for Lojban. While stripping the `lo` prefix didn't change accuracy (ruling out
a simple formatting issue), the multi-word structure creates a deeper mode-
collapse basin.

### 3. Neither model is truly reasoning

English's best tasks (11, 13, 16) are solvable by surface pattern matching.
The yes/no tasks (6, 9, 17, 18) show both languages at majority-class level —
neither learns to discriminate. Tasks requiring actual multi-hop reasoning (8,
19, 20) show 0% for both. These are tiny models (~67K-837K params); genuine
reasoning may require orders of magnitude more capacity.

### 4. Combined training hurts narrative evaluation

The models are 85% bAbI-trained. Narrative BPC and structural metrics are
worse than V2 because the models have less narrative exposure. The BPC
convergence between languages at larger sizes may reflect bAbI template
compression rather than the narrative signal we care about.

---

## Key Findings

### 1. Lojban grammar remains perfect at all sizes

100% grammaticality from 67K params to 837K params, across all 15 Lojban runs.
This extends V2's finding down to 12x smaller models. Lojban grammar is learned
at the smallest scale we tested.

### 2. A language with more character-level entropy takes longer to converge on character prediction

This is the most important methodological insight from V3. Lojban's regular
grammar constrains what characters can follow what — after `l` in particle
position it's almost always `o` (for `lo`) or `i` (for `li'u`), after `.` it's
almost always `i` or a name consonant. English has more character-level
uncertainty — after `th` you could get `e`, `a`, `i`, `o`, `r`, `ough`, etc.

Lower character entropy means faster convergence on next-character prediction.
This is neither good nor bad for reasoning — it's orthogonal. But when early
stopping is based on character-prediction loss, it creates a confound: the
lower-entropy language stops training earlier, getting less exposure to any
secondary learning objective (like bAbI reasoning) that's embedded in the data.

This insight generalizes beyond Lojban vs English: **any experiment that uses
character/token prediction loss for early stopping will systematically
undertrain lower-entropy languages on secondary objectives.**

### 3. bAbI results are confounded by training dynamics

The headline result (English 46% vs Lojban 20% at base) is not a clean
comparison. Lojban trains 4x fewer steps, sees 4x fewer bAbI examples, and
mode-collapses because its fast character-prediction convergence triggers early
stopping prematurely. The experiment as designed cannot distinguish "Lojban is
worse at reasoning" from "Lojban didn't get enough training on reasoning tasks."

### 4. Lojban mode-collapses on bAbI while English diversifies

At base size, Lojban predicts a single answer for entire tasks (e.g., `lo purdi`
for all 200 task-1 examples). English produces 12 unique predictions spread
across the answer space. Both languages start mode-collapsed at nano, but English
breaks out with scale while Lojban doesn't — because Lojban's training is cut
short.

### 5. English's "reasoning" is mostly pattern matching on 3 tasks

English's overall accuracy is carried by tasks 11, 13, and 16 (coreference and
induction), which jump to 95-100%. These are solvable by surface-level string
matching without genuine multi-hop reasoning. Tasks requiring real reasoning
(multi-hop, path finding, counting) show modest or zero improvement.

### 6. Neither language learns yes/no discrimination

Tasks 6, 9, 17, and 18 (yes/no) show both languages performing at exactly
majority-class baseline. English base predicts "yes" 100% of the time on
task 17. Neither model learned to use context to determine the answer.

---

## What Would Fix This

To get a clean comparison of reasoning by language, the experiment design needs
to address the training duration confound:

1. **Fixed training steps** instead of early stopping: Train both languages for
   the same number of steps (e.g., 10K or 20K), comparing bAbI accuracy at equal
   training exposure. This removes the confound entirely.

2. **Early stop on bAbI accuracy**: Instead of combined val loss, stop when bAbI
   validation accuracy plateaus. This ensures both languages get enough reasoning
   signal before stopping.

3. **Separate training phases**: Train on narrative first (to convergence), then
   fine-tune on bAbI. This prevents the fast narrative convergence from drowning
   out the reasoning signal.

4. **bAbI-only training**: Drop narrative entirely and train pure bAbI. This
   isolates the reasoning question from the language modeling question.

5. **Equalized answer format**: Restructure Lojban bAbI to use single-word
   answers (e.g., just `purdi` instead of `lo purdi`) to remove the multi-token
   generation confound.

---

## Summary Table (base size, mean of 3 seeds)

| Metric | English | Lojban |
|--------|---------|--------|
| Params | ~837K | ~837K |
| Val BPC | 0.960 | 0.977 |
| Test BPC | **2.827** | 2.857 |
| Grammar | 96.8% | **100.0%** |
| bAbI seen (overall) | **46.4%** | 20.5% |
| bAbI unseen (overall) | **38.0%** | 18.0% |
| bAbI seen (task 11) | **99.7%** | 11.5% |
| bAbI seen (task 6, yes/no) | 52.8% | 52.0% |
| Memorized (any run) | 0 | 0 |
| Avg best_step | 4,367 | 1,100 |
| bAbI epochs at best_step | **28.4** | 6.6 |

---

## Files

Results are stored at `studio:~/lojban_experiment/results/v3/`:
- `calibration.json` — Tatoeba grammar checker baseline
- `corpus_info.json` — combined data split statistics
- `{nano,micro,mini,small,base}/{english,lojban}_seed{42,137,2024}/`
  - `result.json` — all metrics (training, bAbI, grammar, structural, memorization)
  - `babi_predictions.json` — per-example expected/predicted/correct
  - `samples.json` — narrative generated samples
