"""
V4 Training Dynamics Analysis
Computes derived statistics for English and Lojban medium seed42 runs.
"""

def main():
    # === Constants ===
    BATCH_SIZE = 64
    CTX_LEN = 256
    PARAMS = 570_000  # approximate
    CHINCHILLA_RATIO = 20  # tokens per param

    # === Per-language data ===
    langs = {
        "English": {
            "train_tokens": 640_890,
            "vocab_size": 1024,
            "best_step": 1000,
            "best_val_loss": 2.3898,
            "train_loss_at_best": 1.8881,
            "final_step": 10_000,
            "final_val_loss": 2.9677,
            "final_train_loss": 1.4148,
            "narrative_chars": 407_194,
            "babi_chars": 2_390_610,
        },
        "Lojban": {
            "train_tokens": 767_739,
            "vocab_size": 1024,
            "best_step": 1000,
            "best_val_loss": 2.2858,
            "train_loss_at_best": 1.1743,
            "final_step": 10_000,
            "final_val_loss": 2.5422,
            "final_train_loss": 0.9299,
            "narrative_chars": 407_194,
            "babi_chars": 2_627_365,
        },
    }

    tokens_per_step = BATCH_SIZE * CTX_LEN
    chinchilla_optimal = CHINCHILLA_RATIO * PARAMS

    print("=" * 72)
    print("V4 TRAINING DYNAMICS ANALYSIS — Medium, seed42")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Section 1: Throughput and epoch counts
    # ------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("1. THROUGHPUT AND EPOCH COUNTS")
    print("-" * 72)
    print(f"  Batch size:        {BATCH_SIZE}")
    print(f"  Context length:    {CTX_LEN}")
    print(f"  Tokens per step:   {tokens_per_step:,}")
    print()

    for name, d in langs.items():
        total_at_best = tokens_per_step * d["best_step"]
        total_at_final = tokens_per_step * d["final_step"]
        epochs_at_best = total_at_best / d["train_tokens"]
        epochs_at_final = total_at_final / d["train_tokens"]

        print(f"  {name}:")
        print(f"    Train tokens in corpus:          {d['train_tokens']:>12,}")
        print(f"    Best step:                       {d['best_step']:>12,}")
        print(f"    Tokens processed by best step:   {total_at_best:>12,}")
        print(f"    Epochs by best step:             {epochs_at_best:>12.2f}")
        print(f"    Tokens processed by step 10000:  {total_at_final:>12,}")
        print(f"    Epochs by step 10000:            {epochs_at_final:>12.2f}")
        print()

    # ------------------------------------------------------------------
    # Section 2: Chinchilla scaling analysis
    # ------------------------------------------------------------------
    print("-" * 72)
    print("2. CHINCHILLA SCALING ANALYSIS")
    print("-" * 72)
    print(f"  Model params (approx):    {PARAMS:>12,}")
    print(f"  Chinchilla ratio:         {CHINCHILLA_RATIO} tokens/param")
    print(f"  Chinchilla-optimal tokens:{chinchilla_optimal:>12,}")
    print()

    for name, d in langs.items():
        deficit = chinchilla_optimal / d["train_tokens"]
        surplus_at_final = (tokens_per_step * d["final_step"]) / chinchilla_optimal
        print(f"  {name}:")
        print(f"    Actual train tokens:             {d['train_tokens']:>12,}")
        print(f"    Chinchilla-optimal tokens:       {chinchilla_optimal:>12,}")
        print(f"    Data deficit ratio:              {deficit:>12.2f}x")
        print(f"      (need {deficit:.1f}x more unique tokens for Chinchilla-optimal)")
        print(f"    Tokens seen at step 10000:       {tokens_per_step * d['final_step']:>12,}")
        print(f"    Ratio to Chinchilla-optimal:     {surplus_at_final:>12.2f}x")
        print(f"      (seeing each token ~{(tokens_per_step * d['final_step']) / d['train_tokens']:.1f}x "
              f"but only have {d['train_tokens']/chinchilla_optimal:.2%} of needed unique data)")
        print()

    # ------------------------------------------------------------------
    # Section 3: Data composition
    # ------------------------------------------------------------------
    print("-" * 72)
    print("3. DATA COMPOSITION (characters, pre-BPE)")
    print("-" * 72)

    for name, d in langs.items():
        total_chars = d["narrative_chars"] + d["babi_chars"]
        narr_pct = d["narrative_chars"] / total_chars * 100
        babi_pct = d["babi_chars"] / total_chars * 100
        chars_per_token = total_chars / d["train_tokens"]
        print(f"  {name}:")
        print(f"    Narrative chars:   {d['narrative_chars']:>12,}  ({narr_pct:.1f}%)")
        print(f"    bAbI chars:        {d['babi_chars']:>12,}  ({babi_pct:.1f}%)")
        print(f"    Total chars:       {total_chars:>12,}")
        print(f"    Train tokens:      {d['train_tokens']:>12,}")
        print(f"    Chars per token:   {chars_per_token:>12.2f}")
        print()

    # ------------------------------------------------------------------
    # Section 4: bAbI capacity analysis
    # ------------------------------------------------------------------
    print("-" * 72)
    print("4. bAbI CAPACITY ANALYSIS")
    print("-" * 72)
    n_tasks = 20
    examples_per_task = 1000
    total_babi_examples = n_tasks * examples_per_task
    print(f"  Number of bAbI tasks:              {n_tasks}")
    print(f"  Training examples per task:        {examples_per_task:,}")
    print(f"  Total bAbI training examples:      {total_babi_examples:,}")
    print(f"  Model parameters:                  {PARAMS:,}")
    print(f"  Params per bAbI example:           {PARAMS / total_babi_examples:.1f}")
    print(f"  Params per task:                   {PARAMS / n_tasks:,.0f}")
    print()
    print("  For context:")
    print(f"    - 570K params must learn 20 distinct task formats")
    print(f"    - That is {PARAMS // n_tasks:,} params per task")
    print(f"    - Each task has only 1,000 training examples")
    print(f"    - The model must ALSO learn narrative language modeling")
    print(f"    - bAbI is ~85% of the data by chars, so dominates training")
    print(f"    - But 20 different formats compete for limited capacity")
    print()

    # How many bAbI tokens per epoch?
    for name, d in langs.items():
        babi_token_share = d["babi_chars"] / (d["narrative_chars"] + d["babi_chars"])
        babi_tokens_approx = d["train_tokens"] * babi_token_share
        babi_tokens_per_task = babi_tokens_approx / n_tasks
        print(f"  {name} bAbI token budget:")
        print(f"    bAbI share of tokens (approx):   {babi_token_share:.1%}")
        print(f"    bAbI tokens (approx):            {babi_tokens_approx:,.0f}")
        print(f"    Tokens per task (approx):        {babi_tokens_per_task:,.0f}")
        epochs_at_final = (tokens_per_step * d["final_step"]) / d["train_tokens"]
        babi_exposures_per_task = epochs_at_final * examples_per_task
        print(f"    At {epochs_at_final:.1f} epochs, each task example seen ~{epochs_at_final:.1f}x")
        print(f"    Total task-example exposures:     {babi_exposures_per_task:,.0f} per task")
        print()

    # ------------------------------------------------------------------
    # Section 5: Overfitting trajectory
    # ------------------------------------------------------------------
    print("-" * 72)
    print("5. OVERFITTING TRAJECTORY (train-val loss gap)")
    print("-" * 72)
    print()
    print(f"  {'':20s} {'English':>12s}  {'Lojban':>12s}")
    print(f"  {'':20s} {'-------':>12s}  {'------':>12s}")

    en = langs["English"]
    lj = langs["Lojban"]

    # At best step
    en_gap_best = en["best_val_loss"] - en["train_loss_at_best"]
    lj_gap_best = lj["best_val_loss"] - lj["train_loss_at_best"]

    print(f"  At best step (1000):")
    print(f"    Train loss:        {en['train_loss_at_best']:>12.4f}  {lj['train_loss_at_best']:>12.4f}")
    print(f"    Val loss:          {en['best_val_loss']:>12.4f}  {lj['best_val_loss']:>12.4f}")
    print(f"    Gap (val-train):   {en_gap_best:>12.4f}  {lj_gap_best:>12.4f}")
    print()

    # At final step
    en_gap_final = en["final_val_loss"] - en["final_train_loss"]
    lj_gap_final = lj["final_val_loss"] - lj["final_train_loss"]

    print(f"  At step 10000:")
    print(f"    Train loss:        {en['final_train_loss']:>12.4f}  {lj['final_train_loss']:>12.4f}")
    print(f"    Val loss:          {en['final_val_loss']:>12.4f}  {lj['final_val_loss']:>12.4f}")
    print(f"    Gap (val-train):   {en_gap_final:>12.4f}  {lj_gap_final:>12.4f}")
    print()

    # Change
    en_gap_change = en_gap_final - en_gap_best
    lj_gap_change = lj_gap_final - lj_gap_best
    en_val_change = en["final_val_loss"] - en["best_val_loss"]
    lj_val_change = lj["final_val_loss"] - lj["best_val_loss"]
    en_train_change = en["final_train_loss"] - en["train_loss_at_best"]
    lj_train_change = lj["final_train_loss"] - lj["train_loss_at_best"]

    print(f"  Change (step 1000 -> 10000):")
    print(f"    Train loss delta:  {en_train_change:>+12.4f}  {lj_train_change:>+12.4f}")
    print(f"    Val loss delta:    {en_val_change:>+12.4f}  {lj_val_change:>+12.4f}")
    print(f"    Gap growth:        {en_gap_change:>+12.4f}  {lj_gap_change:>+12.4f}")
    print()

    print(f"  Interpretation:")
    print(f"    English: train loss drops {abs(en_train_change):.3f} but val rises {en_val_change:.3f}")
    print(f"      -> overfitting gap grows from {en_gap_best:.3f} to {en_gap_final:.3f} ({en_gap_final/en_gap_best:.1f}x)")
    print(f"    Lojban:  train loss drops {abs(lj_train_change):.3f} but val rises {lj_val_change:.3f}")
    print(f"      -> overfitting gap grows from {lj_gap_best:.3f} to {lj_gap_final:.3f} ({lj_gap_final/lj_gap_best:.1f}x)")
    print()

    # Already overfitting at step 1000?
    print(f"  Already overfitting at best step (1000)?")
    print(f"    English gap = {en_gap_best:.4f} — {'yes, moderate gap' if en_gap_best > 0.3 else 'mild gap' if en_gap_best > 0.1 else 'no, well-fit'}")
    print(f"    Lojban gap  = {lj_gap_best:.4f} — {'yes, LARGE gap' if lj_gap_best > 0.5 else 'yes, moderate gap' if lj_gap_best > 0.3 else 'mild gap' if lj_gap_best > 0.1 else 'no, well-fit'}")
    print()

    lj_gap_ratio = lj_gap_best / en_gap_best if en_gap_best > 0 else float('inf')
    print(f"  NOTE: Lojban already has {lj_gap_ratio:.1f}x the overfitting gap of English")
    print(f"  at step 1000. This is surprising — Lojban has MORE training tokens")
    print(f"  ({langs['Lojban']['train_tokens']:,} vs {langs['English']['train_tokens']:,}).")
    print(f"  Possible explanation: BPE tokenizer with vocab=1024 may compress Lojban")
    print(f"  more aggressively (fewer unique subwords needed for regular morphology),")
    print(f"  leading to lower effective diversity despite more raw tokens.")
    print()

    # ------------------------------------------------------------------
    # Section 6: BPE compression analysis
    # ------------------------------------------------------------------
    print("-" * 72)
    print("6. BPE COMPRESSION ANALYSIS")
    print("-" * 72)
    print()
    for name, d in langs.items():
        total_chars = d["narrative_chars"] + d["babi_chars"]
        chars_per_token = total_chars / d["train_tokens"]
        # 90% of total goes to train split
        total_chars_90 = total_chars * 0.90
        train_tokens = d["train_tokens"]
        print(f"  {name}:")
        print(f"    Total chars (all data):  {total_chars:>12,}")
        print(f"    Train tokens (BPE):      {train_tokens:>12,}")
        print(f"    Approx chars/token:      {chars_per_token:>12.2f}")
        # Higher chars/token = more compression
        print()

    en_cpt = (langs["English"]["narrative_chars"] + langs["English"]["babi_chars"]) / langs["English"]["train_tokens"]
    lj_cpt = (langs["Lojban"]["narrative_chars"] + langs["Lojban"]["babi_chars"]) / langs["Lojban"]["train_tokens"]
    print(f"  English: {en_cpt:.2f} chars/token")
    print(f"  Lojban:  {lj_cpt:.2f} chars/token")
    if lj_cpt > en_cpt:
        print(f"  Lojban compresses {lj_cpt/en_cpt:.2f}x more chars per BPE token.")
        print(f"  This means each Lojban token carries more character-level information,")
        print(f"  but also that 767K Lojban tokens represent {767739 * lj_cpt / 1e6:.1f}M chars")
        print(f"  while 640K English tokens represent {640890 * en_cpt / 1e6:.1f}M chars.")
    else:
        print(f"  English compresses {en_cpt/lj_cpt:.2f}x more chars per BPE token.")
    print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print()
    print("  1. Both languages hit best val loss at step 1000 out of 10000,")
    print("     meaning 90% of training is wasted (overfitting).")
    print()
    print(f"  2. By step 1000, the model has already cycled through the data")
    en_epochs_1k = (tokens_per_step * 1000) / langs["English"]["train_tokens"]
    lj_epochs_1k = (tokens_per_step * 1000) / langs["Lojban"]["train_tokens"]
    print(f"     ~{en_epochs_1k:.1f}x (EN) / ~{lj_epochs_1k:.1f}x (LJ) — already heavily overfit!")
    print()
    print(f"  3. Chinchilla says 570K params needs {chinchilla_optimal:,} tokens.")
    print(f"     English has {langs['English']['train_tokens']:,} ({langs['English']['train_tokens']/chinchilla_optimal:.1%}).")
    print(f"     Lojban has {langs['Lojban']['train_tokens']:,} ({langs['Lojban']['train_tokens']/chinchilla_optimal:.1%}).")
    print(f"     Both are severely data-starved ({chinchilla_optimal / langs['English']['train_tokens']:.0f}-{chinchilla_optimal / langs['Lojban']['train_tokens']:.0f}x deficit).")
    print()
    print(f"  4. 20 bAbI tasks with 1000 examples each compete for 570K params.")
    print(f"     That is {PARAMS // n_tasks:,} params per task — enough for pattern matching")
    print(f"     but not for generalizable reasoning.")
    print()
    print(f"  5. Lojban overfits MORE than English at best step ({lj_gap_best:.3f} vs {en_gap_best:.3f}")
    print(f"     val-train gap) despite having 20% more training tokens. The BPE")
    print(f"     tokenizer likely compresses Lojban's regular morphology more")
    print(f"     aggressively, reducing effective vocabulary diversity.")
    print()


if __name__ == "__main__":
    main()
