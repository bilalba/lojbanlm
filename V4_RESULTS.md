# V4 Experiment Results: BPE Tokenization + bAbI Reasoning

## Experiment Configuration

### Motivation

V3/V3.1 identified two confounds that favored English on bAbI reasoning:
1. **Character-level entropy difference**: Lojban's lower character entropy
   (~73 chars vs ~85) caused faster convergence, triggering earlier checkpoint
   selection and less bAbI exposure in the saved model.
2. **Multi-token representation bottleneck**: Lojban locations are multi-word
   (`lo jukpa kumfa` = 14 chars) while English locations are single-word
   (`kitchen` = 7 chars), causing mode collapse at tiny model scale.

V4 uses BPE tokenization (vocab=1024) to equalize token-level information
density across languages and normalize multi-word expressions into fewer tokens.

### Data
- **Narrative corpus**: Same 4 parallel books as V2/V3 (Alice in Wonderland,
  Wizard of Oz, Book of Esther, In a Grove) -- 407,194 chars per language
- **bAbI reasoning tasks**: All 20 tasks x 1000 training examples -- 2,390,610
  chars (English), 2,627,365 chars (Lojban)
- **Combined training data**: ~2.8M chars English, ~3.0M chars Lojban
- **BPE tokenization**: 640,890 train tokens (English), 767,739 train tokens
  (Lojban) -- Lojban has 0.281 tokens/char vs English's 0.255 tokens/char
- **Split**: 90% train / 5% val / 5% prompt-source (in token space)
- **Test corpus**: Metamorphosis (held out entirely) -- 44,466 tokens English,
  43,806 tokens Lojban
- **Calibration**: Tatoeba parallel sentences -- 200 sampled, 100% pass rate
  for both camxes and LanguageTool

### BPE Tokenizer
- **Byte-level BPE** via HuggingFace `tokenizers` library
- **256 byte base tokens + ~768 learned merges = 1024 total vocab** per language
- **Separate tokenizer trained per language** on combined (narrative + bAbI)
  corpus
- No special tokens (no PAD, no EOS), same as V3
- Byte-level fallback ensures zero OOV
- Tokenizers saved to `results/v4/tokenizer_{english,lojban}.json`

### Models

| Size | d_model | n_layer | n_head | head_dim | vocab | ctx_len | dropout | ~Params |
|------|---------|---------|--------|----------|-------|---------|---------|---------|
| medium | 96 | 4 | 4 | 24 | 1024 | 256 | 0.15 | **~570K** |
| large | 128 | 3 | 4 | 32 | 1024 | 256 | 0.15 | **~758K** |

Both use batch_size=64, tied embeddings (embedding table = output head), cosine
LR schedule (3e-4 peak, 100-step warmup), fixed 10,000 steps (no early stopping).
Checkpoint selected by best narrative val BPC.

### Seeds
3 seeds per configuration: 42, 137, 2024. 12 total runs (2 sizes x 2 langs x 3
seeds). Completed in ~217 minutes on studio (Apple Silicon).

---

## Quantitative Results

### BPC (Bits Per Character)

| Size | English Val | Lojban Val | English Test | Lojban Test |
|------|------------|------------|-------------|-------------|
| medium | 3.467 | **3.293** | 2.657 | **2.251** |
| large | 3.281 | **3.204** | 2.543 | **2.321** |

Mean across 3 seeds. Val BPC is on the combined (narrative + bAbI) validation
split at the best checkpoint. Test BPC is on held-out Metamorphosis, computed
by decoding BPE tokens to count actual characters (making it comparable to V3's
character-level BPC).

**Lojban BPC advantage persists.** Test BPC is 15.3% lower at medium (2.251 vs
2.657) and 8.7% lower at large (2.321 vs 2.543). This is consistent with
V2/V3/V3.1 -- Lojban text is structurally easier to predict at the character
level regardless of tokenization. BPE did not equalize this.

**Val BPC is higher than test BPC** (negative generalization gap). This seems
paradoxical but is explained by data composition: val BPC is measured on mixed
narrative + bAbI validation data, while test BPC is narrative-only
(Metamorphosis). The BPE tokenizer was trained on combined data, so
narrative-only test data has more predictable patterns.

### Best Step (Checkpoint Selection)

| Size | English avg | Lojban avg | Ratio (EN/LJ) |
|------|------------|------------|----------------|
| medium | 967 | 767 | 1.3x |
| large | 1,433 | 533 | **2.7x** |

The checkpoint selection confound from V3.1 persists and is **amplified**. With
BPE tokenization, narrative val loss bottoms out much earlier than with
character-level tokenization (V3.1 best_steps were 2,100-7,733). Lojban's
checkpoint is especially early at large scale -- averaging step 533 out of
10,000, meaning the saved model reflects only 5.3% of total training.

| Run | Seed 42 | Seed 137 | Seed 2024 |
|-----|---------|----------|-----------|
| medium english | 1,000 | 1,000 | 900 |
| medium lojban | 1,000 | 500 | 800 |
| large english | 800 | 800 | 2,700 |
| large lojban | **300** | 1,000 | **300** |

Large Lojban is catastrophically unstable: 2 of 3 seeds save at step 300 (3% of
training). The model at step 300 has seen ~2-3 passes over bAbI data, far too
few to learn any task patterns.

### Grammar

| Size | English | Lojban |
|------|---------|--------|
| medium | 99.2% | **100.0%** |
| large | 98.9% | **100.0%** |

Identical pattern to V2/V3/V3.1. Lojban 100% at all sizes. English ~99%.

### Memorization

Zero memorization across all 12 runs (LCS threshold=50, avg LCS 16-22 chars).

### Training Dynamics

All 12 runs show aggressive overfitting. Val BPC averaged across seeds:

| Step | med EN | med LJ | lar EN | lar LJ |
|------|--------|--------|--------|--------|
| 100 | 6.876 | 7.009 | 6.085 | 6.224 |
| 300 | 3.791 | **3.477** | 3.608 | **3.211** |
| 500 | 3.715 | **3.371** | 3.519 | 3.347 |
| 1,000 | **3.497** | 3.393 | **3.355** | 3.438 |
| 2,000 | 3.781 | 3.686 | 3.481 | 3.654 |
| 5,000 | 4.041 | 3.649 | 3.681 | 3.836 |
| 10,000 | 4.206 | 3.745 | 3.729 | 4.149 |

The val loss curve is U-shaped: rapid improvement from step 1 to ~500-1000,
then steady degradation. All runs overfit substantially -- the overfit ratio
(step-10K val BPC / min val BPC) ranges from 1.10x to 1.36x. 10,000 steps is
far more than needed; models converge by step ~1,000 and spend the remaining
9,000 steps memorizing training data.

Lojban converges faster initially (lower val BPC at step 300-500) but also
overfits more aggressively, especially at large scale (overfit ratio 1.30x vs
English's 1.14x). This pushes its optimal checkpoint earlier, amplifying the
checkpoint selection confound.

---

## bAbI Reasoning Accuracy

### Overall Accuracy

| Size | EN Seen | LJ Seen | EN Unseen | LJ Unseen |
|------|---------|---------|-----------|-----------|
| medium | **20.8%** +/- 0.5% | 19.5% +/- 1.1% | **14.4%** +/- 1.3% | 10.1% +/- 1.0% |
| large | **20.8%** +/- 0.2% | 14.5% +/- 6.7% | **17.4%** +/- 1.3% | 10.7% +/- 4.4% |

English leads at both sizes, but the margin varies. At medium, the seen gap is
only 1.3pp (within noise). At large, the seen gap is 6.3pp -- but this is
driven entirely by large Lojban's catastrophic variance (see below).

### Seed-Level Variance

| Config | Seed 42 | Seed 137 | Seed 2024 | Mean | Std |
|--------|---------|----------|-----------|------|-----|
| medium english | 21.4% | 20.4% | 20.8% | 20.8% | 0.4pp |
| medium lojban | 20.8% | 18.8% | 18.9% | 19.5% | 0.9pp |
| large english | 21.1% | 20.6% | 20.9% | 20.8% | 0.2pp |
| large lojban | 10.8% | **22.2%** | 10.5% | 14.5% | **5.5pp** |

Large Lojban is the critical outlier with 11.8pp range across seeds. Seed 137
(best_step=1000) scores 22.2% -- comparable to all other conditions. Seeds 42
and 2024 (best_step=300) collapse to ~10.5%. The checkpoint timing is the sole
explanation: at step 300, the model hasn't trained long enough to learn any bAbI
patterns.

### Per-Task Accuracy (test_seen, averaged across 3 seeds)

| Task | Description | MED EN | MED LJ | LRG EN | LRG LJ | Chance |
|------|-------------|--------|--------|--------|--------|--------|
| 1 | Single fact | 5.5% | 7.0% | 8.0% | 8.3% | 8% |
| 2 | Two facts | 9.7% | 8.8% | 9.5% | 8.2% | 8% |
| 3 | Three facts | 8.8% | 9.0% | 5.5% | 10.0% | 8% |
| 4 | Two arg relations | 8.7% | 8.7% | 8.3% | 5.2% | 8% |
| 5 | Three arg relations | 7.3% | 11.8% | 9.3% | 6.7% | 8% |
| 6 | Yes/No questions | 50.0% | 40.3% | 50.7% | 27.8% | 50% |
| 7 | Counting | **41.0%** | **41.0%** | **38.5%** | 13.7% | 20% |
| 8 | Lists/Sets | 1.2% | 0.0% | 1.0% | 1.0% | 1% |
| 9 | Simple negation | 48.7% | 48.7% | 50.7% | 19.7% | 50% |
| 10 | Indefinite knowledge | 33.7% | 32.3% | 32.7% | 31.5% | 33% |
| 11 | Basic coreference | 9.7% | 7.2% | 8.3% | 11.7% | 8% |
| 12 | Conjunction | 9.3% | 5.3% | 8.7% | 6.5% | 8% |
| 13 | Compound coreference | 9.3% | 7.0% | 8.2% | 12.2% | 8% |
| 14 | Time reasoning | 8.8% | 3.0% | 8.5% | 3.0% | 8% |
| 15 | Basic deduction | 10.3% | 8.0% | 10.2% | 7.5% | 10% |
| 16 | Basic induction | 23.0% | 20.0% | 27.0% | 22.5% | 25% |
| 17 | Positional reasoning | **64.0%** | **64.0%** | **64.0%** | 38.7% | 50% |
| 18 | Size reasoning | **68.0%** | **68.0%** | **68.0%** | 56.0% | 50% |
| 19 | Path finding | 0.0% | 0.0% | 0.0% | 0.0% | 5% |
| 20 | Agent motivation | 0.0% | 0.0% | 0.0% | 0.0% | 5% |

Every task in every condition is at or near its chance baseline. No task shows
evidence of genuine reasoning above statistical noise.

### Mode Collapse

Mode collapse (>50% of predictions = single answer) affects all conditions:

| Condition | Tasks collapsed (of 20) | Collapsed via yes/go'i | Collapsed via location/other |
|-----------|------------------------|------------------------|------------------------------|
| medium English | **12** | T06, T07, T09, T17, T18, T20 | T03, T04, T05, T12, T14, T15 |
| medium Lojban | 6 | T07, T09, T17, T18 | T10, T16 |
| large English | **10** | T06, T07, T09, T17, T18, T19 | T04, T05, T10, T16 |
| large Lojban | **12** | T17, T18 | T01, T09, T10, T11, T12, T13, T14, T15, T16, T20 |

Medium Lojban actually has the **fewest** collapsed tasks (6/20). Large Lojban
has the most degenerate behavior: on tasks 14 and 20, it produces repetitive
garbage (`lo nu klama lo nu klama lo nu klama...`) -- the step-300 checkpoint
hasn't learned to terminate bAbI answers.

**BPE did not eliminate mode collapse.** It shifted the pattern -- V3's Lojban
mode-collapsed to 2-3 location tokens (`lo panka`, `lo purdi`), while V4's
collapse is more varied -- but the fundamental behavior of predicting a single
dominant answer per task persists.

### Yes/No Task Analysis (tasks 6, 9, 17, 18)

| Task | Lang | Pred yes/go'i | Pred no/na go'i | Majority % | Accuracy |
|------|------|---------------|-----------------|------------|----------|
| T17 (medium) | EN | 100% | 0% | 64% (yes) | **64.0%** |
| T17 (medium) | LJ | 100% | 0% | 64% (go'i) | **64.0%** |
| T18 (medium) | EN | 100% | 0% | 68% (yes) | **68.0%** |
| T18 (medium) | LJ | 100% | 0% | 68% (go'i) | **68.0%** |
| T06 (medium) | EN | 64% | 36% | 50.5% | 50.0% |
| T09 (medium) | EN | 66% | 34% | 50.5% | 48.7% |

Tasks 17 and 18 predict 100% yes/go'i at medium scale for both languages --
accuracy equals the majority-class fraction exactly. This is pure majority-class
exploitation with zero reasoning, identical to V3/V3.1.

Large Lojban's yes/no performance degrades further: T17 drops to 38.7% and T18
to 56.0% because 218-365 predictions are degenerate garbage strings instead of
valid yes/no tokens.

---

## Cross-Version Comparison

### bAbI Accuracy Across All Experiments

| Experiment | Params | EN Seen | LJ Seen | Gap | Notes |
|------------|--------|---------|---------|-----|-------|
| V3 base | 837K | 46.4% | 20.5% | +25.9pp | Early stopping confound |
| V3.1 nano | 67K | 19.0% | 20.8% | -1.8pp | Fixed 10K steps, char-level |
| V3.1 micro | 164K | 24.4% | 20.9% | +3.5pp | Fixed 10K steps, char-level |
| V3.1 mini | 261K | 28.6% | 22.1% | +6.5pp | Fixed 10K steps, char-level |
| **V4 medium** | **570K** | **20.8%** | **19.5%** | **+1.3pp** | BPE, best_step ~870 |
| **V4 large** | **759K** | **20.8%** | **14.5%** | **+6.3pp** | BPE, best_step ~983 |

### Best Step Across Versions

| Experiment | EN avg best_step | LJ avg best_step | Ratio |
|------------|-----------------|------------------|-------|
| V3 base | 4,367 | 1,100 | 4.0x |
| V3.1 mini | 7,450 | 2,100 | 3.5x |
| V4 medium | 967 | 767 | 1.3x |
| V4 large | 1,433 | 533 | 2.7x |

BPE tokenization brought the best_step ratio closer at medium (1.3x vs V3.1's
3.5x) but the absolute steps are dramatically lower. Both languages converge in
~1,000 steps with BPE instead of ~5,000-7,000 with character-level tokenization.
The models overfit far more aggressively with the larger vocabulary.

### Test BPC Across Versions

| Experiment | EN Test BPC | LJ Test BPC | LJ advantage |
|------------|-------------|-------------|--------------|
| V2 small (835K) | 2.770 | 1.860 | 33% |
| V3 base (837K) | 2.827 | 2.857 | -1% |
| V3.1 mini (261K) | 3.083 | 2.899 | 6% |
| **V4 medium (570K)** | **2.657** | **2.251** | **15%** |
| **V4 large (759K)** | **2.543** | **2.321** | **9%** |

V4 test BPC is better than V3/V3.1 (despite measuring BPC the same way over
actual characters). BPE enables the model to represent narrative text more
efficiently. Lojban's BPC advantage (~9-15%) is between V2's large advantage
(33%, narrative-only data) and V3's near-parity (bAbI-dominated data).

---

## Qualitative Results: Narrative Generation

### In-Domain Samples (bAbI prompts)

Both languages produce fluent bAbI-format text. This is expected -- ~85% of
training data is bAbI tasks.

**English medium (seed 42)**:
```
Bob went to the school.
Kate went to the bedroom.
Is John in the office? yes

Lucy traveled to the garden.
Sandra is no longer in the bathroom.
David went to the market.
Is Tom in the bedroom? yes
```
Fluent, well-formed bAbI text with proper question-answer structure, diverse
names and locations. The model has thoroughly learned the bAbI format.

**Lojban medium (seed 42)**:
```
la .alis. pu klama lo ckule
la .pet. pu klama lo zarci
la .sam. pu zvati lo kelci stuzi
la .sandr. pu lo sipna kumfa
la .ninas. pu lo gusta
la .ninas. cu zvati ma? lo briju
```
Correct Lojban bAbI format: `la .X. pu klama lo Y` (X went to Y), `la .X. cu
zvati ma?` (where is X?). Diverse locations including multi-word expressions
(`kelci stuzi`, `sipna kumfa`). Minor errors: some lines drop the verb `klama`
(e.g., `la .sandr. pu lo sipna kumfa` -- missing `klama`).

**Lojban large (seed 137)** shows better task diversity:
```
la .pet. pu klama lo panka
la .frank. pu klama lo jukpa kumfa
xu la .deiv. cu zvati lo jukpa kumfa? na go'i

la .djan. pu klama lo jukpa kumfa
la .frank. ca na zvati lo larcu kumfa
la .alis. pu klama lo briju
xu la .keit. cu zvati lo zarci? na go'i
```
This includes yes/no questions with negation (`xu ... na go'i`), negative
state tracking (`ca na zvati`), and successful generation of multi-word
locations (`jukpa kumfa`, `larcu kumfa`, `kelci stuzi`). BPE tokenization does
help with multi-word Lojban locations in generation, even though it did not
help with bAbI accuracy.

### Out-of-Domain Samples (Metamorphosis prompts)

Out-of-domain narrative quality is poor for both languages -- the models are
fundamentally bAbI pattern generators.

**English medium (seed 42)**:
```
'I am you, so he will be his haled, "Will the poor the king.
'I am to stangained a met.
'RHhy," said the gans, but Dorothy.
"You'vely. And he shouldde the Woodman.
```
Heavily contaminated by training book vocabulary (Scarecrow, Dorothy, Woodman,
king). Fragments of dialogue structure but surrounding text is word salad with
nonsense tokens ("stangained", "RHhy", "gans").

**English large (seed 2024)** is slightly more fluent:
```
"I get me, that is to go with a little time?" asked Dorothy.
"And I did not be sur little part of the Wicked Witch of the
Wicked Witch, the Scarecrow.
"What is a lad!" cried Dorothy.
"Why, and we are not have been killed me," remarked Dorothy.
```
Recognizable dialogue structure with proper verb framing ("asked Dorothy",
"cried Dorothy", "remarked Dorothy"). Content is still nonsensical but the
syntactic scaffolding is more intact.

**Lojban medium (seed 42)**:
```
gi'e lo se ckra la .i ko lo cicti lo so'a lo mi na ka'i je la
cifma .i je'u gi'u ma kau lo nu la .i .i gi'e lo ka'u lo nu lo
xarbi'o lo du'i je mi na'u lo nu ca lo ka'e lo nu ca lo nixli
cu cusku lo nu se ka'a te zu'o'a lo nu mi ba'e ku'e da cu cusku
```
Maintains Lojban grammatical scaffolding -- real function words (`gi'e`, `lo nu`,
`cu cusku`, `ka'u`, `ba'e`), discourse connectives, and subordination structures.
Semantically incoherent but structurally valid. Note `lo nu ... lo nu ... lo nu`
nesting -- a pathological over-production of event abstractions.

**Lojban large (seed 137)** produces the most structured output:
```
ni'o lu mi ca bo la cpitepygau cu cusku .i mi na se ki'u bo mi'u
bo do klama ba'u bo mi nau bo mi ma kau .i mi za'u da lo nu .i
mi'i ri'u .obu se djica lo nu ri .i lu ta'a ja'a mi li'u
ni'o sei la .oz. cu cusku li'o sei la cpitepygau cu cusku
```
Paragraph markers (`ni'o`), quotation frames (`lu ... li'u`), attributed speech
(`sei la .oz. cu cusku`), negation (`na`), and logical connectives. Uses
character names from training data (`.oz.` = Oz, `cpitepygau`). The grammar
"skeleton" is remarkably consistent despite zero semantic coherence.

### Sample Quality Summary

Neither language produces coherent narrative prose. Both are dominated by bAbI
pattern generation (~85% of training data is bAbI). English produces more
"readable" nonsense (recognizable words, dialogue fragments). Lojban produces
more "grammatically valid" nonsense (100% grammar pass rate, correct particle
usage, proper discourse structure). This replicates V2/V3 findings and is not
meaningfully changed by BPE tokenization.

---

## Structural Metrics (averaged across seeds)

| Metric | MED EN | MED LJ | LRG EN | LRG LJ |
|--------|--------|--------|--------|--------|
| Char KL divergence | 0.094 | 0.209 | 0.091 | 0.229 |
| N-gram diversity (3) | 0.060 | 0.035 | 0.061 | 0.035 |
| N-gram diversity (4) | 0.107 | 0.064 | 0.111 | 0.065 |
| N-gram diversity (5) | 0.150 | 0.093 | 0.156 | 0.094 |
| Repetition r10 | 0.662 | 0.730 | 0.645 | 0.736 |
| Repetition r20 | 0.350 | 0.364 | 0.295 | 0.395 |
| Repetition r50 | 0.018 | 0.004 | 0.002 | 0.004 |
| Word length sim | 0.890 | 0.826 | 0.874 | 0.826 |

Same patterns as V3/V3.1. Lojban has higher character KL divergence (generated
text deviates more from reference distribution), lower n-gram diversity (more
repetitive at the n-gram level), higher short-range repetition (r10, r20), and
lower word-length similarity to reference text. English has higher n-gram
diversity -- its generated text is more varied at the surface level despite
being semantically incoherent.

---

## Key Findings

### 1. BPE tokenization did not help Lojban on bAbI

V4 medium Lojban (570K params, 19.5% seen) performs comparably to V3.1 nano
Lojban (67K params, 20.8% seen) -- an 8.5x parameter increase produced no
improvement. The hypothesis that BPE would normalize multi-token locations and
equalize convergence speed is not supported. Both languages remain at or near
chance baselines on all tasks.

### 2. The checkpoint selection confound is catastrophically worse with BPE

BPE models overfit narrative text far more aggressively than character-level
models. Val BPC bottoms out at step 300-1000 (vs 2,100-7,533 in V3.1), meaning
the "best" checkpoint comes from very early training when bAbI patterns are
barely absorbed. This affects Lojban more severely (best_step avg 533 at large
vs English's 1,433), creating an even larger confound than V3.1.

The root cause: with 1024 vocab (vs 73-85 char vocab), the model has far more
expressiveness per parameter and memorizes narrative patterns faster. The
U-shaped val loss curve is more extreme -- sharp descent to step ~500, then
steady climb through step 10,000.

### 3. BPE equalized convergence speed at medium but not large scale

At medium, the best_step ratio is 1.3x (EN 967 vs LJ 767) -- close to the
20% target in V4_DESIGN.md. At large, it widens to 2.7x (EN 1,433 vs LJ 533).
The equalization hypothesis partially holds: BPE brings convergence closer at
smaller scale, but larger models re-amplify the difference because Lojban's
structurally lower entropy causes faster memorization regardless of
tokenization.

### 4. No evidence of reasoning in any condition

Across all 4 conditions and 20 tasks, no task exceeds its chance baseline by
more than noise:
- Tasks 17/18: 64%/68% via 100% yes/go'i prediction (= majority class exactly)
- Task 7: 41% via 100% "two"/"re" prediction (= most frequent answer)
- Tasks 6/9: ~50% (coin flip)
- Tasks 1-5, 11-16: at chance (8-25%)
- Tasks 8, 19, 20: 0% (too complex)

This is consistent across all experiment versions. 570K-758K parameter models
with ~640K-768K training tokens do not learn multi-step reasoning, regardless
of language or tokenization.

### 5. Lojban grammar remains 100% at all scales

The one robust, replicable finding across V1-V4: Lojban grammar is perfectly
learnable at tiny model scale. Every Lojban run across all versions (67K-758K
params) scores 100% grammaticality. English ranges from 96-99.6%.

### 6. Lojban BPC advantage is structural, not tokenization-dependent

Lojban's test BPC is 9-15% lower than English's across V4's BPE tokenization,
compared to 6-33% across V2/V3's character-level tokenization. The advantage
persists regardless of tokenization scheme, confirming it reflects Lojban's
inherently lower character-level entropy and more regular orthography.

---

## What V4 Rules Out

V4 was designed to test whether BPE tokenization would:
1. **Equalize convergence speed** -- Partially (at medium), but not at large.
2. **Fix multi-word mode collapse** -- No. Mode collapse shifted patterns but
   persists. Large Lojban produces degenerate repetitive garbage at early
   checkpoints.
3. **Improve Lojban bAbI accuracy** -- No. Lojban medium (570K) matches V3.1
   nano (67K) despite 8.5x more parameters.

The fundamental problem is not tokenization -- it's that **checkpoint selection
by narrative val loss is orthogonal to bAbI performance**, and this confound is
amplified by any factor that makes one language converge on narrative prediction
faster (whether character entropy or BPE memorization).

---

## Implications for Future Work

If continuing this line of research, the following changes would be necessary:

1. **Separate checkpoint selection from evaluation metric.** Save checkpoints at
   fixed intervals and evaluate bAbI at each. Report bAbI accuracy at the
   checkpoint with best bAbI validation accuracy, not best narrative loss.

2. **Two-stage training.** Pretrain on narrative (to learn language structure),
   then fine-tune on bAbI (to learn reasoning). This decouples the two
   objectives and prevents narrative convergence from cutting short bAbI
   training.

3. **Larger models and more data.** At 570K-758K params, neither language shows
   any reasoning signal. The minimum scale for bAbI reasoning may be well above
   1M parameters, at which point the structural advantage of Lojban (if any)
   might manifest.

4. **Tasks that test grammar, not memory.** bAbI tasks primarily test entity
   tracking and pattern matching -- skills that depend on vocabulary memorization,
   not grammatical structure. Tasks testing compositional generalization,
   structural parsing, or systematic argument binding would more directly
   evaluate Lojban's hypothesized advantage.

---

## Per-Run Detail

```
Size    Lang     Seed   BestStep  ValBPC  TestBPC  SeenAcc  UnseenAcc  Grammar  Memo  AvgLCS  Time(s)
------------------------------------------------------------------------------------------------------------------------
medium  english  42       1000    3.448   2.646    21.4%    14.1%      99.6%    0     16.7    859
medium  english  137      1000    3.566   2.654    20.4%    15.8%      98.8%    0     16.5    856
medium  english  2024      900    3.388   2.672    20.8%    13.3%      99.2%    0     16.0    848
medium  lojban   42       1000    3.298   2.191    20.8%    10.8%     100.0%    0     19.7    870
medium  lojban   137       500    3.161   2.317    18.8%     8.9%     100.0%    0     19.1    852
medium  lojban   2024      800    3.420   2.246    18.9%    10.5%     100.0%    0     20.5    851
large   english  42        800    3.364   2.605    21.1%    16.9%      99.0%    0     16.6    619
large   english  137       800    3.254   2.613    20.6%    16.4%      99.1%    0     15.9    647
large   english  2024     2700    3.225   2.412    20.9%    18.9%      98.6%    0     17.1    636
large   lojban   42        300    3.143   2.414    10.8%     8.1%     100.0%    0     18.3    624
large   lojban   137      1000    3.304   2.140    22.2%    15.8%     100.0%    0     21.6    636
large   lojban   2024      300    3.167   2.409    10.5%     8.1%     100.0%    0     19.2    638
```

---

## Files

Results stored at `results/v4/`:
- `tokenizer_english.json`, `tokenizer_lojban.json` -- saved BPE tokenizers
- `corpus_info.json` -- BPE-specific data statistics
- `calibration.json` -- Tatoeba grammar checker baseline
- `{medium,large}/{english,lojban}_seed{42,137,2024}/`
  - `result.json` -- all metrics
  - `babi_predictions.json` -- per-example predictions
  - `samples.json` -- narrative generated samples
