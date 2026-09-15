from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def neural_v1_ar_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Autoregressive intra-group cross entropy.

    logits:
        [B, T, G, V]

    targets:
        [B, T, G]

    target_mask:
        [B, T, G]
    """

    batch_size, sequence_length, group_size, vocab_size = logits.shape

    loss = F.cross_entropy(
        logits.reshape(
            batch_size * sequence_length * group_size,
            vocab_size,
        ),
        targets.reshape(
            batch_size * sequence_length * group_size,
        ),
        reduction="none",
    )

    mask = target_mask.reshape(-1).float()

    loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)

    return loss


@torch.no_grad()
def evaluate_neural_v1_ar(
    model,
    loader,
    device: torch.device,
    max_batches: int | None = None,
):
    model.eval()

    total_loss = 0.0
    total_batches = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x, targets, target_mask = batch

        x = x.to(device)
        targets = targets.to(device)
        target_mask = target_mask.to(device)

        # ---------------------------------------------------------
        # Construct teacher-forced previous symbols
        #
        # target:
        #
        #   [y1, y2, y3, y4]
        #
        # previous:
        #
        #   [BOS, y1, y2, y3]
        # ---------------------------------------------------------

        bos_id = 258

        previous_symbols = torch.empty_like(targets)

        previous_symbols[..., 0] = bos_id
        previous_symbols[..., 1:] = targets[..., :-1]

        logits = model(
            x,
            previous_symbols,
        )

        loss = neural_v1_ar_loss(
            logits,
            targets,
            target_mask,
        )

        total_loss += loss.item()
        total_batches += 1

    if total_batches == 0:
        return float("nan")

    return total_loss / total_batches


def train_neural_v1_ar(
    model,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer,
    device: torch.device,
    steps: int = 5000,
    eval_every: int = 500,
    max_eval_batches: int | None = 100,
):
    model.train()

    history = []

    train_iter = iter(train_loader)

    total_positions = 0
    total_training_time = 0.0

    progress = tqdm(
        range(steps),
        desc="Train",
    )

    for step in progress:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        x, targets, target_mask = batch

        x = x.to(device)
        targets = targets.to(device)
        target_mask = target_mask.to(device)

        # Teacher forcing.
        bos_id = 258

        previous_symbols = torch.empty_like(targets)

        previous_symbols[..., 0] = bos_id
        previous_symbols[..., 1:] = targets[..., :-1]

        optimizer.zero_grad(set_to_none=True)

        start_time = time.perf_counter()

        logits = model(
            x,
            previous_symbols,
        )

        loss = neural_v1_ar_loss(
            logits,
            targets,
            target_mask,
        )

        loss.backward()

        optimizer.step()

        step_time = time.perf_counter() - start_time

        batch_positions = target_mask.sum().item()

        total_positions += int(batch_positions)

        total_training_time += step_time

        # ---------------------------------------------------------
        # Validation
        # ---------------------------------------------------------

        validation_loss = None

        if step == 0 or (step + 1) % eval_every == 0 or step == steps - 1:
            validation_loss = evaluate_neural_v1_ar(
                model,
                valid_loader,
                device,
                max_batches=max_eval_batches,
            )

            model.train()

        record = {
            "step": step,
            "train_loss": float(loss.item()),
            "val_loss": (None if validation_loss is None else float(validation_loss)),
            "time": float(step_time),
            "wall_clock_seconds": float(total_training_time),
            "positions": int(batch_positions),
            "total_positions": int(total_positions),
            "positions_per_second": float(
                batch_positions / step_time if step_time > 0 else 0.0
            ),
        }

        history.append(record)

        if validation_loss is not None:
            print(
                f"step={step:05d} "
                f"train loss={loss.item():.4f} "
                f"validation loss={validation_loss:.4f} "
                f"time={step_time:.3f}s"
            )

    return history
