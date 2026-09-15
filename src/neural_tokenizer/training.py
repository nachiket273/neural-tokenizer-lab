from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    max_batches=None,
):
    """
    Evaluate mean cross-entropy loss per target token.

    Returns:
        Mean NLL in nats/token.
    """

    model.eval()

    total_nll = 0.0
    total_tokens = 0

    if max_batches is not None:
        total = min(max_batches, len(loader))
    else:
        total = len(loader)

    progress = tqdm(
        loader,
        desc="val",
        leave=False,
        total=total,
    )

    for batch_idx, (x, y) in enumerate(progress):

        if max_batches is not None and batch_idx >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="sum",
        )

        total_nll += loss.item()
        total_tokens += y.numel()

    model.train()

    if total_tokens == 0:
        raise RuntimeError("Validation loader produced zero target tokens.")

    return total_nll / total_tokens


def train(
    model,
    loader,
    optimizer,
    device,
    steps,
    val_loader=None,
    val_every=100,
    val_batches=None,
):
    """
    Train a language model for a fixed number of optimizer steps.

    History contains one record per optimization step.

    Each record contains:

        step
        train_loss
        val_loss
        time
        wall_clock_seconds
        positions
        total_positions
        positions_per_second
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

        # ----------------------------------------------------
        # Get next batch
        # ----------------------------------------------------

        try:
            x, y = next(iterator)

        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)

        x = x.to(device)
        y = y.to(device)

        # ----------------------------------------------------
        # Optimization step
        # ----------------------------------------------------

        start = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )

        loss.backward()

        optimizer.step()

        elapsed = time.perf_counter() - start

        # ----------------------------------------------------
        # Counters
        # ----------------------------------------------------

        batch_positions = y.numel()

        total_positions += batch_positions

        total_training_time += elapsed

        positions_per_second = batch_positions / elapsed if elapsed > 0 else 0.0

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_loss = None

        if val_loader is not None and (step % val_every == 0 or step == steps - 1):
            val_loss = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                max_batches=val_batches,
            )

            progress.write(
                f"step={step:05d} "
                f"train loss={loss.item():.4f} "
                f"validation loss={val_loss:.4f} "
                f"time={elapsed:.3f}s"
            )

        # ----------------------------------------------------
        # Record history
        # ----------------------------------------------------

        record = {
            "step": step,
            "train_loss": float(loss.item()),
            "val_loss": (float(val_loss) if val_loss is not None else None),
            "time": float(elapsed),
            "wall_clock_seconds": float(total_training_time),
            "positions": int(batch_positions),
            "total_positions": int(total_positions),
            "positions_per_second": float(positions_per_second),
        }

        history.append(record)

    # --------------------------------------------------------
    # Sanity check
    # --------------------------------------------------------

    if len(history) != steps:
        raise RuntimeError(
            f"Training history has {len(history)} " f"records, expected {steps}."
        )

    return history
