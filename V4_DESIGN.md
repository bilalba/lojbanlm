# V4 Experiment: BPE Tokenization

## Motivation

V3/V3.1 revealed that character-level entropy differences between English
(~85 chars, irregular spelling) and Lojban (~73 chars, regular spelling)
confound training dynamics. Lojban's lower character entropy causes faster
convergence on character prediction, which triggers early stopping sooner
and gives Lojban less gradient signal for bAbI reasoning tasks.

V4 uses BPE tokenization so that tokens carry roughly equal information in
both languages, equalizing convergence speed and removing this confound.

## Key Changes from V3

| Aspect | V3 | V4 |
|--------|----|----|
| Tokenizer | Character-level (~73-85 vocab) | BPE (vocab=1024) |
| Model sizes | 5 (nano-base, 67K-837K params) | 2 (medium ~570K, large ~758K) |
| Default training | Early stopping (patience=1500) | Fixed 10K steps |
| BPC computation | Loss / char count (tokens=chars) | Loss / actual decoded char count |
| bAbI generation | Check newline per char | Decode tokens, check for newline in text |
| Sample generation | 256 chars | 128 tokens (~256+ chars) |

## Architecture

With vocab=1024 and tied embeddings, the embedding table (1024 * d_model) is a
significant fraction of the parameter budget. Model dimensions adjusted to hit
~500K and ~800K targets:

| Size | d_model | n_layer | n_head | head_dim | vocab | ~Params |
|------|---------|---------|--------|----------|-------|---------|
| medium | 96 | 4 | 4 | 24 | 1024 | **~570K** |
| large | 128 | 3 | 4 | 32 | 1024 | **~758K** |

Both sizes use ctx_len=256 tokens and batch_size=64.

## BPE Tokenizer Design

Implemented in `bpe_tokenizer.py` using HuggingFace `tokenizers` library:

- **Byte-level BPE**: 256 byte base tokens + ~768 learned merges = 1024 vocab
- **Separate tokenizer per language**: trained on each language's combined corpus
- **Saved to disk**: `results/v4/tokenizer_{english,lojban}.json` for reproducibility
- **No special tokens**: no PAD, no EOS (same as V3's CharTokenizer)
- **Byte-level fallback**: ensures zero OOV tokens

### Equalization Hypothesis

With character-level tokenization, English needs ~85 symbols to represent text
while Lojban uses ~73. This means each English character carries less information
(higher entropy per char), slowing convergence on character prediction.

BPE with equal vocab size (1024) should produce tokens carrying roughly equal
information in both languages. The key metric to verify this is `tokens_per_char`
ratio in corpus_info.json and `best_step` convergence comparison across languages.

## BPC Computation

V3 assumed tokens = characters, so `total_chars += ctx_len`. With BPE, each
token may represent multiple characters. V4 decodes target token windows to
count actual characters:

```python
y_ids = test_ids[start + 1:start + ctx_len + 1].tolist()
chars_in_window = len(tokenizer.decode(y_ids))
total_chars += chars_in_window
```

This ensures BPC is comparable across tokenization schemes and languages.

## bAbI Answer Generation

V3 checks `char == "\n"` per generated character. With BPE, a single token may
contain a newline character. V4 decodes all generated tokens so far and checks
for newline in the decoded text:

```python
generated_ids.append(next_id)
text_so_far = tokenizer.decode(generated_ids)
if "\n" in text_so_far:
    text_so_far = text_so_far[:text_so_far.index("\n")]
    break
```

`max_answer_tokens=20` (was 50 chars in V3; 20 BPE tokens covers bAbI answers).

## Results Directory

```
results/v4/
  tokenizer_english.json      # Saved BPE tokenizer
  tokenizer_lojban.json
  corpus_info.json             # Includes BPE-specific stats
  calibration.json             # Tatoeba grammar checker baseline
  medium/
    english_seed{42,137,2024}/
      result.json
      babi_predictions.json
      samples.json
    lojban_seed{42,137,2024}/
  large/
    ...
```

## CLI Usage

```bash
# Quick sanity check (~2 min)
python3 experiment_v4.py --quick --skip-grammar

# Single size
python3 experiment_v4.py --size medium
python3 experiment_v4.py --size large

# All runs (12 total: 2 sizes x 2 langs x 3 seeds)
python3 experiment_v4.py

# Override training steps
python3 experiment_v4.py --max-steps 5000

# Other options
python3 experiment_v4.py --skip-grammar           # no Java/Node needed
python3 experiment_v4.py --skip-narrative-eval     # bAbI only
python3 experiment_v4.py --babi-tasks 1 6 15       # specific tasks
python3 experiment_v4.py --language lojban          # single language
```

## Verification Checklist

1. `pip install tokenizers` (if not already)
2. `python3 experiment_v4.py --quick --skip-grammar` -- sanity check
   - Verify both languages produce reasonable output
   - Check BPE vocab sizes are ~1024
   - Check tokens_per_char ratios in output
3. Key equalization check: compare `best_step` for English vs Lojban
   - If within ~20% of each other, BPE successfully equalized convergence
4. Full run: `python3 experiment_v4.py --size medium` then `--size large`

## Dependencies

- `torch` (required)
- `tokenizers` (required, `pip install tokenizers`)
- `language_tool_python` + Java (optional, English grammar)
- `node` (optional, Lojban grammar via camxes)

## Files

- `experiment_v4.py` -- main experiment script (forked from experiment_v3.py)
- `bpe_tokenizer.py` -- BPE tokenizer wrapper module
- `V4_DESIGN.md` -- this file
