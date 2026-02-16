# V1 Experiment Results: Single Book, Single Seed

## Overview

V1 was the initial proof-of-concept run: one book (Alice in Wonderland), one
model size (~10.8M params), one seed, no overfitting controls. It established
that there is a measurable difference between Lojban and English learnability,
but the results were severely compromised by memorization.

Run date: 2025-02-15, on studio (Apple Silicon, MPS backend).

## Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | Character-level GPT |
| d_model | 384 |
| n_layer | 6 |
| n_head | 6 |
| Dropout | 0.1 |
| Context length | 256 |
| Batch size | 64 |
| Training steps | 5,000 |
| Learning rate | 3e-4 (cosine decay) |
| Warmup steps | 100 |
| Seed | 42 |
| Training data | Alice in Wonderland (single book, both languages) |
| Generation | 100 samples, 256 chars, temperature=0.8, top-k=40 |

### Data

| | English | Lojban |
|---|---|---|
| Source | alice_english.txt | alice_lojban.txt |
| Total chars | 144,518 | 153,812 |
| Train chars | 130,066 | 138,430 |
| Val chars | 14,452 | 15,382 |
| Vocab size | 73 | 57 |

No held-out test set. No prompt-source split. Prompts were drawn from
validation data.

## Training Results

| Metric | English | Lojban |
|--------|---------|--------|
| Parameters | 10,773,888 | 10,767,744 |
| Final train loss | 0.0722 | 0.0609 |
| Final val loss | 3.0782 | 2.2904 |
| Train BPC | 0.104 | 0.088 |
| Val BPC | 4.440 | 3.304 |
| Training time | 1,573s (26 min) | 1,583s (26 min) |

Both models reached extremely low training loss (~0.07), indicating near-perfect
memorization of the training data. The val loss was 40-50x higher than train
loss — catastrophic overfitting.

Even so, Lojban's val loss (2.29) was 25% lower than English's (3.08). This
gap, measured on text the model hadn't memorized, was the most meaningful
signal from V1.

## Memorization Analysis

Post-hoc analysis using longest common substring (LCS) matching against the
training text:

| | English | Lojban |
|---|---|---|
| Samples with LCS > 50 | **100/100** (100%) | **100/100** (100%) |
| Min LCS | 146 chars | 59 chars |
| Max LCS | 150 chars | 150 chars |
| Mean LCS | 149.9 chars | 128.5 chars |

**Every single generated sample from both models was near-verbatim
reproduction of the training data.** English samples were essentially perfect
copies (LCS ~150 out of 256 generated chars). Lojban samples had slightly
more variation but were still overwhelmingly memorized.

This means the grammaticality evaluation below is measuring the grammar of
*Alice in Wonderland*, not the grammar of *model-generated text*.

## Grammar Evaluation

| | English | Lojban |
|---|---|---|
| Total sentences evaluated | 254 | 318 |
| Error-free sentences | 251 | 318 |
| Grammaticality rate | 98.8% | 100.0% |
| Avg grammar errors/sent | 0.012 | 0.0 |

English evaluated with LanguageTool (GRAMMAR category filter).
Lojban evaluated with camxes (full parseability).

**These numbers are meaningless for the research question.** Since all output
is memorized, we're measuring the grammaticality of the original book
translations, not the model's ability to generate grammatical text. The 98.8%
English rate reflects 3 sentences where LanguageTool flagged the original
Alice in Wonderland text — a limitation of the grammar checker, not the model.

## Sample Quality

### English V1

All 100 samples are verbatim Alice in Wonderland text:

> **Prompt**: "for making her escape; so she se"
> **Generated**: t off at once, and ran till she was quite tired and out of
> breath, and till the puppy's bark sounded quite faint in the distance.
> 'And yet what a dear little puppy it was!' said Alice, as she leant
> against a buttercup to rest herself...

This is copied character-for-character from the training text.

### Lojban V1

Also overwhelmingly memorized, but with slightly more variation. The
least-memorized sample (LCS=59 out of 256 chars):

> **Prompt**: "zgike li'u» ni'o «lu xu go'i fi"
> **Generated**: lo nu lumci --sei la cakyrespa cu cusku-- li'u» ni'o «lu
> li'a na go'i --sei la .alis. cu fengu cusku-- li'u» ni'o «lu ua lo me
> do moi na mutce lo ka xamgu ckule --sei la jitfa cakyrespa cu surbi'o
> mutce tonga cusku-- ...

Even this "least memorized" sample is recognizably from the training text,
just possibly spliced from different sections.

## What V1 Established

### Meaningful findings

1. **Val loss gap**: Lojban 2.29 vs English 3.08 (−25%). When forced to
   generalize beyond memorized text, the model does substantially better on
   Lojban. This signal survived despite the memorization problem.

2. **Feasibility**: The experimental setup works — character-level GPT on
   MPS, grammar evaluation with LanguageTool and camxes, generation and
   evaluation pipeline all functional.

### Not meaningful

1. **Grammar comparison**: Invalidated by 100% memorization. Both languages
   scored >98% because the output was copied from professionally translated
   books.

2. **Sample quality comparison**: Impossible — all samples are training data
   reproductions, not model-generated text.

3. **Any claim about "coherence" or "reasoning"**: The model is a lookup
   table for Alice in Wonderland, not a language model.

## Why V1 Failed

### Root cause: Massive model on tiny data

10.8M parameters trained on ~130K characters is roughly 80x over-parameterized.
By Chinchilla scaling, 10.8M params wants ~200M tokens. We gave it ~130K. The
model had more than enough capacity to memorize every character of the training
data, and that's exactly what it did.

### Contributing factors

- **No dropout** (only 0.1, and the model memorized anyway)
- **No early stopping** (fixed 5,000 steps, well past the point of overfitting)
- **No held-out test set** (couldn't measure generalization)
- **No memorization detection** (discovered post-hoc)
- **Single seed** (no confidence intervals)
- **Single book** (results could be corpus-specific)
- **Training loss 0.07 was a red flag** — this is near-zero cross-entropy,
  meaning the model predicts the next character almost perfectly, which is
  only possible through memorization

## Lessons for V2

V1's failures directly motivated V2's design:

| V1 Problem | V2 Fix |
|------------|--------|
| Massive overfitting | Higher dropout (0.15-0.30), early stopping (patience=1500) |
| Over-parameterized | Three model sizes (0.8M, 3.2M, 10.8M) |
| Tiny training data | 4 books instead of 1 (~407K chars vs ~130K) |
| No test set | Held-out Metamorphosis for test BPC |
| No memorization check | LCS detection, grammar eval on novel samples only |
| Single seed | 3 seeds (42, 137, 2024) |
| Single book | 4 training books across different genres |
| No calibration | Tatoeba baseline for grammar checker pass rates |
| Training loss not monitored for pathology | BPC thresholds and overfitting gap tracked |

## Historical Value

V1 is a cautionary tale about training neural language models on insufficient
data. The val loss gap (Lojban 2.29 vs English 3.08) was the only valid signal,
and it was strong enough to justify V2. Everything else — grammaticality scores,
sample quality, coherence claims — was an artifact of memorization.

The V1 result is included in the project history for completeness, but no
conclusions should be drawn from it beyond "Lojban may be more learnable, but
we need a much better experiment to confirm this."
