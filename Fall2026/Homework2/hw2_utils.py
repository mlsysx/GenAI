"""Provided utilities for CAI 5607 Homework 2.

Students implement the model and response-only masking in the notebook.
This module supplies data loading, evaluation, training-loop boilerplate, and
logging. No external services, model downloads, or API keys are used.
"""
from __future__ import annotations

import json
import math
import itertools
import random
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

LABELS = ("leak", "noise", "heat", "unknown")
IGNORE_INDEX = -100


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CharCodec:
    """The supplied corpus inventory plus distinct BOS, EOS, and PAD token IDs."""

    def __init__(self, text: str):
        if not isinstance(text, str) or not text:
            raise ValueError("The corpus must be a nonempty string.")
        self.chars = sorted(set(text))
        self.stoi = {char: i for i, char in enumerate(self.chars)}
        self.itos = {i: char for char, i in self.stoi.items()}
        n = len(self.chars)
        self.bos_id, self.eos_id, self.pad_id = n, n + 1, n + 2
        self.special_names = {n: "<BOS>", n + 1: "<EOS>", n + 2: "<PAD>"}
        self.vocab_size = n + 3

    def encode(self, text: str) -> list[int]:
        missing = set(text) - self.stoi.keys()
        if missing:
            raise ValueError(f"Characters outside the course inventory: {sorted(missing)!r}")
        return [self.stoi[c] for c in text]

    def decode(self, ids) -> str:
        # Preserve special tokens visibly instead of silently hiding errors.
        indices = [int(i) for i in ids]
        if any(i < 0 or i >= self.vocab_size for i in indices):
            raise ValueError("Cannot decode an invalid token ID (including the loss-only -100 label).")
        return "".join(self.itos[i] if i in self.itos else self.special_names[i] for i in indices)


def load_records(path: str | Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        required = {"id", "prompt", "response", "template_id", "asset_id"}
        if not required <= row.keys() or row["response"] not in LABELS:
            raise ValueError(f"Invalid task record on line {line_no} of {path}")
        if any(not isinstance(row[k], str) or not row[k] for k in required):
            raise ValueError(f"Empty or non-string task field on line {line_no} of {path}")
        if row["asset_id"] not in row["prompt"]:
            raise ValueError(f"Asset identifier is missing from the prompt on line {line_no}")
        rows.append(row)
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Task data must be nonempty and have unique record IDs.")
    return rows


def assert_split_disjoint(*partitions: list[dict]) -> None:
    """Check IDs, templates, assets, full prompts, and instantiated wording across splits.

    Only metadata validation is performed here; no model is trained or evaluated.
    Shared vocabulary and general issue terms are intentional, not identical templates.
    """
    for left, right in itertools.combinations(partitions, 2):
        for key in ("id", "template_id", "asset_id", "prompt"):
            overlap = {r[key] for r in left} & {r[key] for r in right}
            if overlap:
                raise ValueError(f"Task partitions overlap in {key}: {sorted(overlap)[:3]}")
        def wording(record):
            return record["prompt"].replace(record["asset_id"], "{asset}")
        if {wording(r) for r in left} & {wording(r) for r in right}:
            raise ValueError("Wording is repeated across task partitions under different IDs.")


def text_batch(sequence: torch.Tensor, length: int, batch_size: int, device):
    """Random windows: inputs and targets are already shifted by one character."""
    if length < 1 or batch_size < 1 or sequence.ndim != 1 or len(sequence) <= length:
        raise ValueError("Expected a one-dimensional sequence longer than the context.")
    starts = torch.randint(len(sequence) - length, (batch_size,))
    offsets = torch.arange(length)
    x = sequence[starts[:, None] + offsets]
    y = sequence[starts[:, None] + offsets + 1]
    return x.to(device), y.to(device)


def sequence_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean over supervised targets. Targets have already been shifted once."""
    if logits.ndim != 3 or targets.ndim != 2 or logits.shape[:2] != targets.shape:
        raise ValueError("Expected logits [B,L,V] and targets [B,L].")
    if not torch.any(targets != IGNORE_INDEX):
        raise ValueError("This batch has no supervised target tokens.")
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                           ignore_index=IGNORE_INDEX)


@torch.no_grad()
def evaluate_text(model: nn.Module, sequence: torch.Tensor, length: int, device,
                  batch_size: int = 128, limit: int | None = None) -> float:
    """NLL in nats: predict each target after L characters exactly once.

    Scores positions L ... len(sequence)-1, always from a full L-character
    context. Evaluation windows never cross a partition boundary. A limit
    selects the FIRST target positions, for reproducible learning curves.
    """
    if sequence.ndim != 1 or length < 1 or batch_size < 1:
        raise ValueError("Use a one-dimensional sequence and positive context/batch sizes.")
    total = len(sequence) - length
    if total <= 0:
        raise ValueError("Evaluation partition is shorter than the context.")
    if limit is not None:
        total = min(total, limit)
    if total <= 0:
        raise ValueError("Evaluation limit must be positive.")
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    offsets = torch.arange(length)
    try:
        for start in range(0, total, batch_size):
            starts = torch.arange(start, min(start + batch_size, total))
            x = sequence[starts[:, None] + offsets].to(device)
            y = sequence[starts + length].to(device)
            logits = model(x)[:, -1, :]
            loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
    finally:
        model.train(was_training)
    return loss_sum / total


def fit_bigram(sequence: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """HW1 baseline supplied so this assignment focuses on the Transformer."""
    if sequence.ndim != 1 or len(sequence) < 2 or vocab_size < 1:
        raise ValueError("Bigram fitting needs at least two token IDs and a positive vocabulary size.")
    if sequence.min() < 0 or sequence.max() >= vocab_size:
        raise ValueError("Invalid bigram token ID.")
    indices = sequence[:-1] * vocab_size + sequence[1:]
    counts = torch.bincount(indices, minlength=vocab_size**2).reshape(vocab_size, vocab_size)
    smoothed = counts.to(torch.float32) + 1.0
    return smoothed / smoothed.sum(dim=1, keepdim=True)


def evaluate_bigram(probabilities: torch.Tensor, sequence: torch.Tensor,
                    length: int) -> float:
    """Uses exactly the same target positions as evaluate_text, but one-character context."""
    if length < 1 or len(sequence) <= length:
        raise ValueError("Partition is shorter than the context.")
    p = probabilities[sequence[length - 1:-1], sequence[length:]]
    return (-p.log()).mean().item()


def synchronize(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def train_steps(model: nn.Module, get_batch: Callable, steps: int, lr: float,
                device, evaluate: Callable[[], dict], interval: int = 100):
    """Provided loop; a NEW optimizer is created on each call, including SFT.

    Checkpoints are not selected from the test set. The returned model is the
    final-step model. Timing includes periodic validation. Gradient clipping
    uses norm 1.0. AdamW weight decay is 0.01.
    """
    if steps <= 0 or interval <= 0 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("steps, interval, and learning rate must be positive")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    history = []
    synchronize(device)
    start_time = time.perf_counter()
    model.train()
    for step in range(1, steps + 1):
        inputs, targets = get_batch()
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = sequence_cross_entropy(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if step == 1 or step % interval == 0 or step == steps:
            metrics = evaluate()
            record = {"step": step, "batch_loss": float(loss.item()), **metrics}
            history.append(record)
            print(" | ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                             for k, v in record.items()))
            model.train()
    synchronize(device)
    return history, time.perf_counter() - start_time


def cpu_state(model: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def generate_text(model: nn.Module, codec: CharCodec, prefix: str,
                  length: int, temperature: float, device) -> str:
    if not prefix or length < 0 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Use a nonempty prefix, nonnegative length, and positive temperature.")
    was_training = model.training
    model.eval()
    ids = codec.encode(prefix)
    try:
        for _ in range(length):
            x = torch.tensor([ids[-model.context_length:]], dtype=torch.long, device=device)
            logits = model(x)[0, -1].clone() / temperature
            # Plain-text display samples are conditioned on not generating special tokens.
            # This restriction is NOT applied in the SFT task evaluator.
            logits[[codec.bos_id, codec.eos_id, codec.pad_id]] = float("-inf")
            next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
            ids.append(next_id)
    finally:
        model.train(was_training)
    return codec.decode(ids)


@torch.no_grad()
def greedy_task_response(model: nn.Module, codec: CharCodec, prompt: str, device,
                         max_new_tokens: int = 12) -> tuple[str, list[int]]:
    if not prompt or max_new_tokens < 1:
        raise ValueError("Use a nonempty prompt and positive generation budget.")
    ids = [codec.bos_id] + codec.encode(prompt)
    if len(ids) + max_new_tokens > model.context_length:
        raise ValueError("Prompt plus generation budget exceeds the context window.")
    generated = []
    for _ in range(max_new_tokens):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        nxt = model(x)[0, -1].argmax().item()
        if nxt == codec.eos_id:
            break
        ids.append(nxt)
        generated.append(nxt)
    return codec.decode(generated), generated


@torch.no_grad()
def evaluate_task(model: nn.Module, codec: CharCodec, records: list[dict], device):
    """Unrestricted greedy decoding. Unexpected extra text or special tokens count as errors."""
    if not records:
        raise ValueError("Cannot evaluate an empty task dataset.")
    was_training = model.training
    model.eval()
    saved = []
    synchronize(device)
    start = time.perf_counter()
    try:
        for r in records:
            raw, ids = greedy_task_response(model, codec, r["prompt"], device)
            prediction = raw.strip().lower()
            saved.append({"id": r["id"], "prompt": r["prompt"], "gold": r["response"], "asset_id": r["asset_id"], "template_id": r["template_id"],
                          "raw_response": raw, "generated_ids": ids,
                          "prediction": prediction, "correct": prediction == r["response"]})
    finally:
        model.train(was_training)
    synchronize(device)
    elapsed = time.perf_counter() - start
    accuracy = sum(r["correct"] for r in saved) / len(saved)
    return {"accuracy": accuracy, "correct": sum(r["correct"] for r in saved),
            "total": len(saved), "evaluation_seconds": elapsed}, saved


@torch.no_grad()
def evaluate_sft_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device,
                      batch_size: int = 64) -> float:
    was_training = model.training
    model.eval()
    total_loss, total_targets = 0.0, 0
    try:
        for start in range(0, len(x), batch_size):
            bx, by = x[start:start + batch_size].to(device), y[start:start + batch_size].to(device)
            logits = model(bx)
            total_loss += F.cross_entropy(logits.reshape(-1, logits.size(-1)), by.reshape(-1),
                                         ignore_index=IGNORE_INDEX, reduction="sum").item()
            total_targets += (by != IGNORE_INDEX).sum().item()
    finally:
        model.train(was_training)
    if total_targets == 0:
        raise ValueError("No response/EOS targets were supervised.")
    return total_loss / total_targets


def save_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_jsonl(path: str | Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
