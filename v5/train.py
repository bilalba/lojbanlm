"""Training loops for V5: phase 1 (pretraining) and phase 2 (bAbI SFT)."""

import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import GPT
from .data import get_batch, make_fixed_val_batches
from .evaluate import eval_babi_accuracy


# ─── Shared Helpers ─────────────────────────────────────────────────────────

def eval_val_loss(model, val_batches, vocab_size):
    """Average cross-entropy loss over fixed validation batches."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in val_batches:
            logits = model(x)
            total += F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)).item()
    return total / len(val_batches)


def _make_lr_schedule(warmup_steps, max_steps):
    """Cosine annealing with linear warmup."""
    def lr_fn(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return lr_fn


def _seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Phase 1: Language Pretraining ──────────────────────────────────────────

def train_phase1(model, train_data, val_data, config, vocab_size, device):
    """Pretrain on general text. Checkpoint by val loss (BPC).

    Args:
        model: GPT model (already on device)
        train_data: 1D tensor of token IDs (training split)
        val_data: 1D tensor of token IDs (validation split)
        config: Phase1Config
        vocab_size: int
        device: torch device

    Returns:
        (model with best checkpoint loaded, info dict)
    """
    print(f"\n  Phase 1: Pretraining ({config.max_steps} steps, lr={config.lr})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                   weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _make_lr_schedule(config.warmup_steps, config.max_steps))

    ctx_len = model.ctx_len
    val_batches = make_fixed_val_batches(val_data, config.batch_size,
                                          ctx_len, device, n_batches=10)

    model.train()
    t0 = time.time()
    log = []
    best_val_loss = float("inf")
    best_state = None
    best_step = 0
    no_improve = 0
    final_step = config.max_steps

    for step in range(1, config.max_steps + 1):
        x, y = get_batch(train_data, config.batch_size, ctx_len, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()

        if step % config.eval_interval == 0 or step == 1:
            val_loss = eval_val_loss(model, val_batches, vocab_size)
            val_bpc = val_loss / math.log(2)
            train_bpc = loss.item() / math.log(2)
            model.train()

            elapsed = time.time() - t0
            print(f"    step {step:5d} | train_bpc {train_bpc:.3f} | "
                  f"val_bpc {val_bpc:.3f} | "
                  f"lr {scheduler.get_last_lr()[0]:.6f} | {elapsed:.0f}s")

            log.append({
                "step": step,
                "train_loss": round(loss.item(), 4),
                "val_loss": round(val_loss, 4),
                "train_bpc": round(train_bpc, 4),
                "val_bpc": round(val_bpc, 4),
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
                best_step = step
                no_improve = 0
            else:
                no_improve += config.eval_interval

            if no_improve >= config.patience:
                print(f"    Early stopping at step {step} "
                      f"(best at step {best_step})")
                final_step = step
                break
    else:
        final_step = config.max_steps

    total_time = time.time() - t0
    best_val_bpc = best_val_loss / math.log(2)
    print(f"    Phase 1 done in {total_time:.0f}s | "
          f"best val_bpc {best_val_bpc:.3f} at step {best_step}")

    if best_state is not None:
        model.load_state_dict({k: v.to(device)
                               for k, v in best_state.items()})

    info = {
        "phase": 1,
        "n_params": model.count_params(),
        "vocab_size": vocab_size,
        "total_time_s": round(total_time, 1),
        "total_steps": final_step,
        "best_step": best_step,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_bpc": round(best_val_bpc, 4),
        "final_train_loss": round(loss.item(), 4),
        "early_stopped": no_improve >= config.patience,
        "log": log,
    }
    return model, info


# ─── Phase 2: bAbI SFT ─────────────────────────────────────────────────────

def train_phase2(model, babi_train_data, babi_val_examples, tokenizer,
                 config, vocab_size, device):
    """Fine-tune on bAbI text. Checkpoint by bAbI val accuracy.

    Args:
        model: GPT model (pretrained, on device)
        babi_train_data: 1D tensor of token IDs (bAbI training text)
        babi_val_examples: list of bAbI val example dicts for accuracy eval
        tokenizer: BPETokenizerWrapper
        config: Phase2Config
        vocab_size: int
        device: torch device

    Returns:
        (model with best checkpoint loaded, info dict)
    """
    print(f"\n  Phase 2: bAbI SFT ({config.max_steps} steps, lr={config.lr})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                   weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _make_lr_schedule(config.warmup_steps, config.max_steps))

    ctx_len = model.ctx_len

    # Split bAbI train data 90/10 for train loss / val loss monitoring
    n = len(babi_train_data)
    split_idx = int(n * 0.9)
    babi_train_split = babi_train_data[:split_idx]
    babi_val_split = babi_train_data[split_idx:]

    val_batches = make_fixed_val_batches(babi_val_split, config.batch_size,
                                          ctx_len, device, n_batches=10)

    model.train()
    t0 = time.time()
    log = []
    best_babi_accuracy = -1.0
    best_state = None
    best_step = 0
    no_improve = 0
    final_step = config.max_steps

    for step in range(1, config.max_steps + 1):
        x, y = get_batch(babi_train_split, config.batch_size, ctx_len, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()

        if step % config.eval_interval == 0 or step == 1:
            val_loss = eval_val_loss(model, val_batches, vocab_size)
            val_bpc = val_loss / math.log(2)
            train_bpc = loss.item() / math.log(2)

            # Evaluate bAbI accuracy on val examples
            babi_result = eval_babi_accuracy(
                model, tokenizer, babi_val_examples, device, ctx_len
            )
            babi_acc = babi_result["accuracy"]
            model.train()

            elapsed = time.time() - t0
            print(f"    step {step:5d} | train_bpc {train_bpc:.3f} | "
                  f"val_bpc {val_bpc:.3f} | "
                  f"babi_val_acc {babi_acc:.3f} | {elapsed:.0f}s")

            log.append({
                "step": step,
                "train_loss": round(loss.item(), 4),
                "val_loss": round(val_loss, 4),
                "train_bpc": round(train_bpc, 4),
                "val_bpc": round(val_bpc, 4),
                "babi_val_accuracy": round(babi_acc, 4),
            })

            # Checkpoint by bAbI accuracy (the key V5 innovation)
            if babi_acc > best_babi_accuracy:
                best_babi_accuracy = babi_acc
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
                best_step = step
                no_improve = 0
            else:
                no_improve += config.eval_interval

            if no_improve >= config.patience:
                print(f"    Early stopping at step {step} "
                      f"(best at step {best_step})")
                final_step = step
                break
    else:
        final_step = config.max_steps

    total_time = time.time() - t0
    print(f"    Phase 2 done in {total_time:.0f}s | "
          f"best bAbI val acc {best_babi_accuracy:.3f} at step {best_step}")

    if best_state is not None:
        model.load_state_dict({k: v.to(device)
                               for k, v in best_state.items()})

    info = {
        "phase": 2,
        "total_time_s": round(total_time, 1),
        "total_steps": final_step,
        "best_step": best_step,
        "best_babi_val_accuracy": round(best_babi_accuracy, 4),
        "best_val_bpc": round(
            min((e["val_bpc"] for e in log), default=0), 4),
        "final_train_loss": round(loss.item(), 4),
        "early_stopped": no_improve >= config.patience,
        "log": log,
    }
    return model, info
