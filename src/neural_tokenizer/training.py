import time

import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    max_batches: int | None = None,
) -> float:
    """
    Calculate mean validation NLL per token.

    This is useful for monitoring a single tokenizer/model
    but should NOT be directly compared between BPE and byte
    models because their token vocabularies differ.
    """

    model.eval()

    total_nll = 0.0
    total_tokens = 0

    progress = tqdm(
        loader,
        desc="val",
        leave=False,
        total=max_batches,
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

    return total_nll / total_tokens


def train(
    model,
    loader,
    optimizer,
    device,
    steps: int,
    val_loader=None,
    val_every: int = 100,
    val_batches: int | None = None,
):
    """
    Train a language model and return training history.

    Returned losses are NLL per representation token.
    They are useful for tracking learning but are not
    directly comparable across different tokenizers.
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
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)

        x = x.to(device)
        y = y.to(device)

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

        batch_positions = y.numel()

        total_positions += batch_positions
        total_training_time += elapsed

        record = {
            "step": step,
            "train_loss": loss.item(),
            "time": elapsed,
            "positions": batch_positions,
            "total_positions": total_positions,
            "positions_per_second": (batch_positions / elapsed),
        }

        if val_loader is not None and (step % val_every == 0 or step == steps - 1):
            val_loss = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                max_batches=val_batches,
            )

            record["val_loss"] = val_loss

            progress.write(
                f"step={step:05d} "
                f"train loss={loss.item():.4f} "
                f"validation loss={val_loss:.4f} "
                f"time={elapsed:.3f}s"
            )
        else:
            record["val_loss"] = None

    return history
