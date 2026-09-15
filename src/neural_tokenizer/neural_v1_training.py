from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from tqdm import tqdm


def neural_v1_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate masked byte/symbol cross-entropy.

    logits:
        [B, T, 4, V]

    targets:
        [B, T, 4]

    target_mask:
        [B, T, 4]
    """

    vocab_size = logits.shape[-1]

    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
        reduction="none",
    )

    mask = target_mask.reshape(-1).float()

    loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)

    return loss


@torch.no_grad()
def evaluate_neural_v1(
    model,
    loader,
    device,
    max_batches=None,
):
    """
    Evaluate Neural V1.

    Returns:
        Mean NLL per predicted symbol.
    """

    model.eval()

    total_nll = 0.0
    total_symbols = 0

    if max_batches is not None:
        total = min(
            max_batches,
            len(loader),
        )
    else:
        total = len(loader)

    progress = tqdm(
        loader,
        desc="val",
        leave=False,
        total=total,
    )

    for batch_idx, (
        x,
        y,
        target_mask,
    ) in enumerate(progress):

        if max_batches is not None and batch_idx >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)
        target_mask = target_mask.to(device)

        logits = model(x)

        vocab_size = logits.shape[-1]

        losses = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            y.reshape(-1),
            reduction="none",
        )

        mask = target_mask.reshape(-1).float()

        total_nll += (losses * mask).sum().item()

        total_symbols += target_mask.sum().item()

    model.train()

    if total_symbols == 0:
        raise RuntimeError("Validation contains zero valid symbols.")

    return total_nll / total_symbols


def train_neural_v1(
    model,
    loader,
    optimizer,
    device,
    steps,
    val_loader=None,
    val_every=500,
    val_batches=100,
):
    """
    Train Neural V1 for a fixed number of optimizer steps.
    """

    model.train()

    iterator = iter(loader)

    history = []

    total_positions = 0
    total_training_time = 0.0

    progress = tqdm(
        range(steps),
        desc="Train",
    )

    for step in progress:

        try:
            x, y, target_mask = next(iterator)

        except StopIteration:
            iterator = iter(loader)
            x, y, target_mask = next(iterator)

        x = x.to(device)
        y = y.to(device)
        target_mask = target_mask.to(device)

        start = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)

        loss = neural_v1_loss(
            logits,
            y,
            target_mask,
        )

        loss.backward()

        optimizer.step()

        elapsed = time.perf_counter() - start

        batch_positions = target_mask.sum().item()

        total_positions += int(batch_positions)

        total_training_time += elapsed

        val_loss = None

        if val_loader is not None and (step % val_every == 0 or step == steps - 1):
            val_loss = evaluate_neural_v1(
                model,
                val_loader,
                device,
                max_batches=val_batches,
            )

            progress.write(
                f"step={step:05d} "
                f"train loss={loss.item():.4f} "
                f"validation loss={val_loss:.4f} "
                f"time={elapsed:.3f}s"
            )

        history.append(
            {
                "step": int(step),
                "train_loss": float(loss.item()),
                "": (float(val_loss) if val_loss is not None else None),
                "time": float(elapsed),
                "wall_clock_seconds": float(total_training_time),
                "positions": int(batch_positions),
                "total_positions": int(total_positions),
                "positions_per_second": float(
                    batch_positions / elapsed if elapsed > 0 else 0.0
                ),
            }
        )

    if len(history) != steps:
        raise RuntimeError(f"Expected {steps} history records, " f"got {len(history)}.")

    return history
