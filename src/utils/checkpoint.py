"""Resumable checkpointing utilities for long training runs on HPC.

SCIENTIFIC / OPERATIONAL PURPOSE:
    Isaac (and most HPC clusters) enforce a 24hr wall-time limit per job.
    Our models take ~40hr to train 100 epochs. Without proper resume logic,
    a killed job throws away all progress. This module provides a consistent
    save/load protocol across all training scripts.

    WHAT MUST BE CHECKPOINTED for true bit-for-bit resume:
    - model weights
    - optimizer state (Adam has per-parameter moment estimates)
    - scheduler state (cosine LR position)
    - GradScaler state (AMP loss scale factor)
    - Python/numpy/torch RNG states (for exact augmentation reproducibility)
    - current epoch number
    - best_iou so far (otherwise resumed run may overwrite the best model)
    - training history list (for complete loss curves)

    WITHOUT all of these, the resumed run is a DIFFERENT experiment. Reviewers
    will notice if the LR schedule jumps or if the "best" checkpoint regresses
    on resume — so this module is the foundation for clean paper-ready results.

Usage in a training script:
    from src.utils.checkpoint import save_checkpoint, load_checkpoint, resolve_resume_path

    start_epoch, best_iou, history = 1, 0.0, []
    resume_path = resolve_resume_path(args.resume, args.auto_resume, ckpt_dir, args.run_name)
    if resume_path:
        start_epoch, best_iou, history = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, device
        )

    for epoch in range(start_epoch, args.epochs + 1):
        ...
        save_checkpoint(
            ckpt_dir / f"{args.run_name}_latest.pt",
            epoch, model, optimizer, scheduler, scaler, best_iou, history,
        )
"""

import json
import random
from pathlib import Path

import numpy as np
import torch


def _get_rng_state():
    """Snapshot all RNG states for exact reproducibility on resume."""
    state = {
        "python_random": random.getstate(),
        "numpy_random":  np.random.get_state(),
        "torch_random":  torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_random"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state):
    """Restore RNG states. Silently skip any missing keys (old checkpoints)."""
    if "python_random" in state:
        random.setstate(state["python_random"])
    if "numpy_random" in state:
        np.random.set_state(state["numpy_random"])
    if "torch_random" in state:
        torch.set_rng_state(state["torch_random"])
    if "torch_cuda_random" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda_random"])


def save_checkpoint(path, epoch, model, optimizer, scheduler, scaler,
                    best_iou, history, extra=None):
    """Save a complete training state snapshot.

    The saved file can resume training EXACTLY where it left off, including
    the LR schedule position and the random seed state.

    Args:
        path:      Output .pt file path.
        epoch:     Last completed epoch number.
        model:     nn.Module to save.
        optimizer: torch Optimizer (saves momentum/variance estimates).
        scheduler: LR scheduler.
        scaler:    GradScaler (or None if AMP disabled).
        best_iou:  Best validation IoU observed so far.
        history:   List of per-epoch metric dicts.
        extra:     Dict of extra fields to store (e.g. val metrics of this epoch).
    """
    state = {
        "epoch":                 epoch,
        "model_state_dict":      model.state_dict(),
        "optimizer_state_dict":  optimizer.state_dict(),
        "scheduler_state_dict":  scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict":     scaler.state_dict() if scaler is not None else None,
        "best_iou":              best_iou,
        "history":               history,
        "rng_state":             _get_rng_state(),
    }
    if extra is not None:
        state.update(extra)

    # Atomic write: save to .tmp then rename. Prevents corruption if job is
    # killed mid-save (SLURM SIGTERM).
    tmp_path = str(path) + ".tmp"
    torch.save(state, tmp_path)
    Path(tmp_path).replace(path)


def load_checkpoint(path, model, optimizer=None, scheduler=None,
                    scaler=None, device="cpu", strict=True):
    """Restore training state from checkpoint.

    Args:
        path:      Checkpoint .pt path.
        model:     Model to load weights into.
        optimizer: Optimizer to restore (optional — pass None for eval-only).
        scheduler: Scheduler to restore.
        scaler:    GradScaler to restore.
        device:    Map location.
        strict:    Strict key matching for model state dict.

    Returns:
        start_epoch: int — epoch to resume from (last_epoch + 1).
        best_iou:    float — best IoU so far.
        history:     list — previous per-epoch records.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Model weights
    model.load_state_dict(ckpt["model_state_dict"], strict=strict)

    # Optional: optimizer, scheduler, scaler
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    # RNG — only restore if present (backward compat with older checkpoints)
    if "rng_state" in ckpt:
        _set_rng_state(ckpt["rng_state"])

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_iou    = float(ckpt.get("best_iou", 0.0))
    history     = ckpt.get("history", [])

    print(f"[Resume] Loaded checkpoint from {path}")
    print(f"[Resume] Epoch: {ckpt.get('epoch')} (resuming at {start_epoch})")
    print(f"[Resume] Best IoU so far: {best_iou:.4f}")
    print(f"[Resume] History entries: {len(history)}")

    return start_epoch, best_iou, history


def resolve_resume_path(resume_arg, auto_resume, ckpt_dir, run_name):
    """Pick the checkpoint to resume from based on user args.

    Priority:
    1. If --resume <path> explicitly given → use that.
    2. If --auto_resume → look for {run_name}_latest.pt in ckpt_dir.
    3. Otherwise → None (fresh training).

    Returns:
        Path or None.
    """
    if resume_arg:
        p = Path(resume_arg)
        if not p.exists():
            raise FileNotFoundError(f"--resume path does not exist: {p}")
        return p

    if auto_resume:
        latest = Path(ckpt_dir) / f"{run_name}_latest.pt"
        if latest.exists():
            print(f"[Auto-resume] Found latest checkpoint: {latest}")
            return latest
        print(f"[Auto-resume] No checkpoint found at {latest} — starting fresh")

    return None


def save_history(history, path):
    """Write training history to JSON. Atomic write for safety."""
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(history, f, indent=2)
    Path(tmp).replace(path)
