# V2 Experiment Results: Small and Medium Models

## Experiment Configuration

### Data
- **Training corpus**: 4 parallel books (Alice in Wonderland, Wizard of Oz,
  Book of Esther, In a Grove) — 407,194 chars per language
- **Split**: 90% train (366,474 tokens) / 5% val (20,360) / 5% prompt-source (20,360)
- **Test corpus**: Metamorphosis (held out entirely) — 121K chars English, 114K chars Lojban
- **Calibration**: Tatoeba parallel sentences — 200 sampled, 100% pass rate for
  both camxes (Lojban) and LanguageTool (English)

### Models

| Size | d_model | n_layer | n_head | Params (Eng) | Params (Loj) | Dropout |
|------|---------|---------|--------|--------------|--------------|---------|
| Small | 128 | 4 | 2 | 836,992 | 835,456 | 0.15 |
| Medium | 256 | 4 | 4 | 3,246,848 | 3,243,776 | 0.20 |

Parameter difference between languages is due to vocab size: English 85 chars,
Lojban 73 chars. Both are character-level GPT models with context length 256,
batch size 64, max 15,000 steps, cosine LR schedule (3e-4 peak, 100 step warmup),
early stopping with patience 1,500 steps.

### Generation
- 100 samples per run, 256 chars each
- Prompts drawn from held-out prompt split (not training data)
- Temperature 0.8, top-k 40
- Memorization detected via longest common substring (LCS > 50 chars)
- Grammar evaluated on novel (non-memorized) samples only

### Seeds
3 seeds per configuration: 42, 137, 2024.

---

## Quantitative Results

### BPC (Bits Per Character)

| | English Small | Lojban Small | English Medium | Lojban Medium |
|---|---|---|---|---|
| **Val BPC** | | | | |
| seed 42 | 1.7958 | 1.4932 | 1.7618 | 1.4489 |
| seed 137 | 1.7876 | 1.4716 | 1.7598 | 1.4336 |
| seed 2024 | 1.8199 | 1.4656 | 1.7502 | 1.4425 |
| **mean** | **1.801** | **1.477** | **1.757** | **1.442** |
| **Test BPC** | | | | |
| seed 42 | 2.8081 | 1.8765 | 2.7522 | 1.8834 |
| seed 137 | 2.6961 | 1.8531 | 3.1278 | 1.8874 |
| seed 2024 | 2.8079 | 1.8512 | 2.8157 | 1.8507 |
| **mean** | **2.770** | **1.860** | **2.899** | **1.874** |

Lojban val BPC is ~18% lower at both sizes. Lojban test BPC is ~33-35% lower.
The gap is remarkably stable across seeds and model sizes.

### Scaling: Small → Medium

| Metric | English | Lojban |
|--------|---------|--------|
| Val BPC | 1.801 → 1.757 (−2.4%) | 1.477 → 1.442 (−2.4%) |
| Test BPC | 2.770 → 2.899 (+4.7%) | 1.860 → 1.874 (+0.7%) |
| Best step | ~10.7K → ~3.8K | ~13.1K → ~3.2K |
| Train time | ~13.3 min → ~11.7 min | ~15.7 min → ~10.5 min |

4x parameters bought almost nothing for either language. Val BPC improved by
only 2.4% for both. Test BPC actually got slightly worse — the models fit the
training distribution faster (early-stopping 3-4x earlier) but don't generalize
better to the held-out Metamorphosis text.

### Generalization Gap (Test BPC − Val BPC)

| | English Small | Lojban Small | English Medium | Lojban Medium |
|---|---|---|---|---|
| seed 42 | +1.012 | +0.383 | +0.990 | +0.435 |
| seed 137 | +0.909 | +0.382 | +1.368 | +0.454 |
| seed 2024 | +0.988 | +0.386 | +1.066 | +0.408 |
| **mean** | **+0.970** | **+0.384** | **+1.141** | **+0.432** |

English's generalization gap is ~2.5x larger than Lojban's, and it got **worse**
at medium (0.97 → 1.14). Lojban's gap also grew slightly (0.38 → 0.43) but
remained much more contained.

This means: when asked to model text from a stylistically different book
(Metamorphosis vs the training books), English models fall apart much more than
Lojban models. Lojban's regular grammar transfers better across domains.

### Overfitting (Train BPC vs Val BPC at Best Checkpoint)

| | English Small | Lojban Small | English Medium | Lojban Medium |
|---|---|---|---|---|
| Train BPC at best step | ~1.76 | ~1.27 | ~1.53 | ~1.22 |
| Val BPC at best step | ~1.80 | ~1.48 | ~1.76 | ~1.44 |
| Overfit gap | ~0.04 | ~0.21 | ~0.23 | ~0.22 |

Small English barely overfits (train ≈ val), while small Lojban has a 0.21 gap.
At medium, English overfitting jumps to 0.23 (matching Lojban). This suggests
the small English model hasn't really "learned" the language deeply — it's near
ceiling on both train and val because it can't model English well at 800K params.
The medium English model starts actually fitting the training data (train BPC
drops from 1.76 to 1.53) but this doesn't help generalization.

### Grammar

| | English Small | Lojban Small | English Medium | Lojban Medium |
|---|---|---|---|---|
| seed 42 | 75.6% | 100.0% | 77.8% | 100.0% |
| seed 137 | 72.3% | 100.0% | 81.4% | 100.0% |
| seed 2024 | 72.5% | 100.0% | 79.6% | 100.0% |
| **mean** | **73.5%** | **100.0%** | **79.6%** | **100.0%** |

English grammar improved +6 percentage points at medium. Lojban was already
saturated at 100% at small and stayed there.

English grammar was evaluated with LanguageTool (category=GRAMMAR filter),
Lojban with camxes (full parseability). Calibration on Tatoeba showed 100%
pass rate for both tools, so baseline bias is not an issue.

Grammar is evaluated on novel samples only (non-memorized).

### Memorization

| | English Small | Lojban Small | English Medium | Lojban Medium |
|---|---|---|---|---|
| Memorized / 100 | 0, 0, 0 | 3, 5, 6 | 0, 0, 0 | 2, 7, 7 |
| Avg LCS (chars) | 22, 22, 22 | 32, 32, 33 | 23, 23, 24 | 30, 32, 33 |

English never crosses the 50-char LCS threshold (avg LCS ~22-24). Lojban has
a small number of memorized samples (2-7 per run) with avg LCS ~30-33.

Lojban's higher memorization may reflect its smaller vocabulary (73 vs 85 chars)
and more repetitive structure — common particles like `lo`, `cu`, `gi'e`, `ni'o`
recur frequently, making longer substring matches more likely even in novel text.

### Structural Metrics

| Metric | English Small | Lojban Small | English Medium | Lojban Medium |
|--------|---------------|--------------|----------------|---------------|
| Repetition r10 | 0.131 | 0.292 | 0.152 | 0.292 |
| Repetition r20 | 0.007 | 0.045 | 0.010 | 0.044 |
| Repetition r50 | 0.000 | 0.000 | 0.000 | 0.000 |
| Ngram div (3) | 0.118 | 0.068 | 0.118 | 0.072 |
| Ngram div (4) | 0.260 | 0.150 | 0.258 | 0.157 |
| Ngram div (5) | 0.407 | 0.258 | 0.402 | 0.264 |
| Char KL div | 0.010 | 0.006 | 0.012 | 0.010 |
| Word len sim | 0.915 | 0.947 | 0.937 | 0.955 |

Lojban has ~2x the repetition rate at r10 and ~1.7x lower n-gram diversity.
This reflects Lojban's genuinely more repetitive surface form — function
particles (`lo`, `cu`, `.i`, `gi'e`, `be`, `pe`) make up a large fraction of
all characters. This is intrinsic to the language, not a model deficiency.

Both languages show r50 = 0.000, meaning no long degenerate loops.

Char KL divergence (how well the model matches the training character
distribution) is lower for Lojban — the generated text has a more accurate
character frequency distribution.

Word length similarity is higher for Lojban (~0.95 vs ~0.93) — generated
Lojban words match the training word length distribution more closely.

### Training Dynamics

| | English Small | Lojban Small | English Medium | Lojban Medium |
|---|---|---|---|---|
| Early stopped? | Yes (all 3) | 2 of 3 | Yes (all 3) | Yes (all 3) |
| Total steps | 11.9-13.4K | 13-15K | 5.3-5.4K | 4.5-4.9K |
| Best step | 10.4-11.9K | 11.5-14K | 3.8-3.9K | 3.0-3.4K |
| Time | 12.5-14.0 min | 14.3-16.4 min | 11.6-11.8 min | 10.2-10.9 min |

Small Lojban trained longer before early stopping (or hit the 15K ceiling).
This suggests the Lojban loss landscape was still yielding improvements at
15K steps — more training could have helped.

Medium models early-stopped much sooner (3-4K vs 10-14K steps). The 4x
parameter increase lets them memorize the training data faster, but the
validation loss stopped improving quickly.

### Learning Speed (BPC thresholds, seed 42)

How quickly each model reached various BPC levels:

| BPC threshold | English Small | Lojban Small | English Medium | Lojban Medium |
|---------------|---------------|--------------|----------------|---------------|
| 3.0 | step 700 | step 400 | step 400 | step 200 |
| 2.5 | step 1,200 | step 700 | step 600 | step 400 |
| 2.0 | step 3,100 | step 1,300 | step 1,300 | step 700 |
| 1.8 | step 6,600 | step 2,400 | step 2,400 | step 1,200 |
| 1.5 | never | step 5,000 | never | step 2,000 |

Lojban consistently reaches each BPC threshold 2-3x faster. English never
reaches BPC 1.5 at either scale.

---

## Qualitative Analysis: Generated Samples

### English Small (~835K params)

Typical output (novel, non-memorized):

> **Prompt**: "ng the rest of my bonds, I liste"
> **Generated**: ned it dead him to feet.' 'But why had to get him variently.
> It was a grass careful cuniousity,' said Alice. The answered of the
> Caterpillar, who was foot the month wicked by and the strangers, pleased
> to her, and the legs of the silver shoes so brights

**Characteristics**:
- Recognizable English words and attempted sentence structure
- Frequent grammatical failures: "why had to get him variently", "The answered
  of the Caterpillar", "who was foot the month wicked"
- Identifiable fragments of meaning within broken sentences: "I was hope you
  cannot my back to go at the great company"
- Characters from training data appear (Alice, Caterpillar, Scarecrow, Tin
  Woodman, Dorothy, Lion) but are mixed across books randomly
- Dialogue structure attempted with quotes and "said X" — sometimes works,
  often breaks mid-sentence
- No coherent narrative flow between sentences

### Lojban Small (~835K params)

> **Prompt**: "to'e curmi lo nu ky cliva lo kum"
> **Generated**: fa fau lo nu no'a lo pa temci cu troci .i je la .dorotis. cu
> sanli fa lo nu lo solri cu se cusku «lu mi se nitcu lo ka ckire kei gi'e na
> djica lo nu do bi'o troci lo nu mi di'a co'e li'u» ni'o «lu ti ki'u ma
> tolcri --sei la cpitepygau cu cusku-- ...

**Characteristics**:
- 100% grammatically correct — every sentence parses with camxes
- Discourse structure is flawless: paragraph markers (`ni'o`), quotation
  delimiters (`lu`/`li'u`), dialogue attribution (`sei la X cu cusku`)
- Character names used consistently from appropriate source texts
- But when translated, propositions are often semantically random:
  - `mi se nitcu lo ka ckire` = "I need the property of being grateful" — coherent
  - `lo solri cu se cusku` = "the sun was said" — nonsensical
  - `ti ki'u ma tolcri` = "this, because of what, is hidden" — grammatical but arbitrary
- Heavy reliance on common structural particles (`lo`, `cu`, `gi'e`, `.i je`,
  `sei`, `ni'o`, `lu`/`li'u`) which are very frequent in Lojban text
- Lower vocabulary diversity than English (n-gram div 0.068 vs 0.118)

### English Medium (~3.2M params)

> **Prompt**: "more blood flowed into my mouth."
> **Generated**: But the Lion were all did not know where you have been a wood
> grandful to round you can help me." "How do you do?" asked the Scarecrow.
> "Of course it they had been twelfth silver had gone how of this room," he
> said, "for I have myself I shall be very sh...

**Characteristics**:
- Grammar improved noticeably over small (73% → 80%)
- Dialogue structure is more consistently correct: proper quote placement,
  "asked the Scarecrow" / "he said" attributions
- Some near-coherent exchanges: "How do you do?" asked the Scarecrow is a
  reasonable dialogue fragment
- **Major problem: Book of Esther formatting contamination.** ~20-30% of
  samples collapse into indented verse-style text:
  > provinces of the king's commandment, for his shrill brought yourself
  > provinces of the king's proces of Media, the king's commandment...
  Esther's highly structured formatting (verse numbers, indentation) is a
  strong attractor in the training data that the model falls into.
- Semantic coherence at the clause level is marginally better than small, but
  still largely word salad between clauses

### Lojban Medium (~3.2M params)

> **Prompt**: "termafyfe'i pu mrobi'o .i mi nel"
> **Generated**: ci lo nu darxi li'u» .i ca lo nu lo mi prenu cu retsku «lu
> mi pu jinvi lo du'u ma kau do ba ba'o sidju lo nu mi ka'e se xagmau li'u»
> ni'o «lu mi'o na crori'a lo termafyfe'i pe la stuna --sei lo bolci cu
> cusku-- ...

**Characteristics**:
- Still 100% grammatically correct
- **Notably more coherent at the proposition level than small**:
  - `mi pu jinvi lo du'u do ba ba'o sidju` = "I thought that you would have
    finished helping" — multi-clause temporal reasoning
  - `mi'o na crori'a lo termafyfe'i` = "we don't wage war on the Scarecrow" —
    coherent proposition with named entity
  - `mi se nitcu lo ka ckire` = "I need to be grateful" — coherent intent
  - `ko mi virnu gi'e ta'irva'u do` = "be brave for me and instead of you" —
    coherent request
  - `mi djica lo nu xruti lo mi se klacpe` = "I want to return to where I was
    called from" — desire + spatial reasoning
- Dialogue responses sometimes contextually appropriate:
  - Character asks question → response addresses it
  - `xu do xruti?` ("did you return?") → `na kakne` ("unable to") — reasonable
- Still no narrative coherence across paragraphs
- Repetition rate and n-gram diversity barely changed from small

---

## Key Findings

### 1. Lojban is fundamentally easier to model (Val/Test BPC)

The BPC gap is consistent, large, and stable:
- Val BPC: Lojban ~18% lower at both sizes
- Test BPC: Lojban ~33-35% lower at both sizes
- The gap doesn't shrink at medium — if anything it's slightly larger

This is the cleanest result. BPC is normalized across vocab sizes (unlike raw
cross-entropy), so this reflects genuine predictability differences. Lojban's
regular grammar makes the next character more predictable.

### 2. Grammar is free for Lojban

100% grammaticality at 835K params, unchanged at 3.2M. The model learned
Lojban grammar completely at the smallest size tested. English grammar improved
from 73% to 80% with 4x parameters and still has substantial room to grow.

This means: at small scale, Lojban models have already solved grammar and can
allocate all capacity to higher-order patterns. English models are still
spending capacity on grammar at medium scale.

### 3. Scaling from small to medium added almost nothing

For both languages, the val BPC improved by only ~2.4% and test BPC actually
got slightly worse. The medium models early-stopped at 3-4K steps vs 10-14K
for small — they memorize the training data faster but don't generalize better.

**Diagnosis**: The models are data-starved. At 407K chars (~80K words), there
isn't enough text for 3.2M parameters to learn from. The training data is the
bottleneck, not model capacity.

### 4. Lojban generalizes better across domains

The test BPC gap (test − val) is ~2.5x larger for English (~1.0) than Lojban
(~0.4). When the model encounters Metamorphosis (stylistically different from
the training books), English falls apart much more.

This suggests Lojban's regular grammar provides transferable structure. English
models may be overfitting to style-specific patterns (word choice, sentence
rhythm) that don't transfer.

### 5. The Esther contamination problem

English medium samples suffer from severe format contamination — the Book of
Esther's verse numbering and indentation pattern is a strong attractor that
pulls ~20-30% of samples into repetitive "the king's commandment" / "provinces"
text. This wasn't as severe at small scale.

This is a data issue, not a fundamental language issue. But it illustrates how
English models struggle with heterogeneous formatting. Lojban's corpus doesn't
have this problem because Lojban text is more uniformly formatted.

### 6. Lojban medium shows hints of emerging semantic coherence

This is the most speculative finding. Medium Lojban samples contain more
multi-clause propositions that translate to coherent meaning:
- Temporal reasoning: "I thought you would have finished helping"
- Spatial reasoning: "I want to return to where I was called from"
- Emotional state: "be brave for me"
- Contextual dialogue: questions followed by relevant responses

English medium does not show a comparable leap — it improved grammatically but
the semantic content is still largely word salad.

**However**, this assessment is subjective. We identified coherent fragments by
manual inspection and translation. We cannot quantify this or determine whether
the apparent coherence is genuine or an artifact of selection bias (cherry-picking
good fragments) or Lojban's formulaic structure (fewer possible constructions
making random combinations look more meaningful).

---

## Open Questions

### Can we measure semantic coherence objectively?

The biggest gap in V2 is the lack of a rigorous semantic metric. Grammar is
measurable (camxes/LanguageTool). BPC is measurable. But "does this make sense?"
is not. We have three options:

1. **bAbI-style reasoning tasks** with verifiable correct answers — the cleanest
   approach but requires a new experiment with template-generated data
   (see REASONING_EXPERIMENT.md)
2. **Human evaluation** — hand-label samples for coherence, but expensive and
   subjective
3. **LLM-as-judge** — use GPT-4/Claude to rate coherence, as in TinyStories.
   Scalable but introduces evaluator bias

Option 1 is the strongest for our hypothesis about reasoning scaling.

### Is the Lojban advantage about grammar or about data?

Lojban has a smaller character vocabulary (73 vs 85), more regular structure,
and more predictable character sequences. The BPC advantage could come from:

- **Grammar regularity** (our hypothesis) — fewer possible sentence structures
  to learn, leaving capacity for semantics
- **Lower entropy language** — Lojban text is inherently more predictable
  character-by-character because of frequent short function words
- **Smaller effective vocabulary** — fewer unique character patterns to model

We can't fully disentangle these with BPC alone. The reasoning experiment
(bAbI tasks) would help because it holds semantic content constant and only
varies the grammar wrapping.

### Does the advantage grow or shrink at larger scale?

We only have small and medium. At both, the gap was stable (~18% val, ~33%
test). Three possibilities:

- **Gap grows**: Lojban keeps pulling ahead — strongest result for the hypothesis
- **Gap stable**: Lojban has a constant offset advantage — grammar regularity
  helps but doesn't compound
- **Gap shrinks**: English eventually catches up as it learns grammar at scale —
  suggests the advantage is just a head start, not a fundamental difference

The large model runs (~10M params) will test this. But given that medium was
data-starved, large may be even more so. We may not see meaningful differences
without more training data.

### Is the training data sufficient?

407K chars (~80K words) is very small. For reference:
- Chinchilla-optimal for 3.2M params would be ~64M tokens (800x more)
- TinyStories used ~500M tokens for models of similar size
- Even small language model papers typically use 10-100M tokens

Both languages are severely undertrained. Medium showed barely any improvement
over small because it ran out of learnable signal in the data. The large model
(10.8M params) will be even more starved.

For the literary text experiment to be more informative, we would need at least
10-50x more data per language. Natural Lojban text of this quantity does not
exist (~2M chars total in the world), so either:
- We generate synthetic Lojban via LLM translation (separate project)
- We pivot to the template-based reasoning experiment where data is unlimited

### Why does English test BPC get worse at medium?

English test BPC went from 2.770 (small) to 2.899 (medium) — a 4.7% regression.
Seed 137 is the worst outlier (3.128 at medium vs 2.696 at small). Lojban test
BPC was essentially flat (1.860 → 1.874).

Possible explanations:
- Medium English overfits to the training books' style more aggressively (it
  early-stops at 3.8K steps with train BPC 1.53, meaning it's memorizing fast)
- Metamorphosis (the test book) is stylistically distant from the training mix,
  and a bigger model is more sensitive to style mismatch
- The Esther formatting attractor is stronger in the medium model, and
  Metamorphosis has no similar formatting, increasing the BPC penalty

### Is Lojban's repetition a problem or a feature?

Lojban's 2x higher repetition rate (r10: 0.29 vs 0.13) and lower n-gram
diversity (0.068 vs 0.118) could mean:

- **Problem**: The model is producing formulaic output, recycling the same
  particle sequences without semantic variation. The "coherence" we see is
  just a small template pool making random combinations look structured.
- **Feature**: Lojban genuinely has more repeated substrings in natural text
  because function words like `lo`, `cu`, `.i`, `gi'e` are very frequent.
  The model is accurately reproducing the language's statistical properties.

Likely both. The char KL divergence is lower for Lojban (0.006 vs 0.010),
meaning the character distribution of generated text matches training data
better. This supports the "feature" interpretation. But the semantic analysis
question remains.

### What would the large model tell us?

Predictions:
- Large English (~10.8M params) will probably early-stop even faster (~2K steps)
  and continue to be data-starved
- Grammar might improve to ~85%+
- Test BPC may continue to worsen (more overfitting, worse generalization)
- Lojban large will remain at 100% grammar, val BPC ~1.40-1.43
- The result may not be very informative given data constraints

The most useful next step is probably the reasoning experiment (bAbI tasks)
rather than waiting for large model results on the same insufficient data.

---

## Summary Table

| Metric | English Small | Lojban Small | English Medium | Lojban Medium |
|--------|---------------|--------------|----------------|---------------|
| Params | 836,992 | 835,456 | 3,246,848 | 3,243,776 |
| Val BPC (mean) | 1.801 | **1.477** | 1.757 | **1.442** |
| Test BPC (mean) | 2.770 | **1.860** | 2.899 | **1.874** |
| Gen gap (test−val) | 0.970 | **0.384** | 1.141 | **0.432** |
| Grammar (mean) | 73.5% | **100.0%** | 79.6% | **100.0%** |
| Memorized (range) | 0 | 3-6 | 0 | 2-7 |
| Repetition r10 | **0.131** | 0.292 | **0.152** | 0.292 |
| Ngram div 3 | **0.118** | 0.068 | **0.118** | 0.072 |
| Char KL div | 0.010 | **0.006** | 0.012 | **0.010** |
| Best step (mean) | 10.7K | 13.1K | 3.8K | 3.2K |
| Semantic coherence | Low | Low | Low | **Moderate** (subjective) |

Bold indicates the better value per row.

---

## Files

Results are stored at `studio:~/lojban_experiment/results/v2/`:
- `calibration.json` — Tatoeba grammar checker baseline
- `corpus_info.json` — data split statistics
- `small/{english,lojban}_seed{42,137,2024}/result.json` — per-run metrics
- `small/{english,lojban}_seed{42,137,2024}/samples.json` — generated samples
- `medium/...` — same structure
