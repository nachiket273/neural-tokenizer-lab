from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate_neural_v1_bpb(
    model,
    texts,
    tokenizer,
    device,
    context_length=256,
):
    """
    Evaluate Neural V1 in bits per original UTF-8 byte.

    The model predicts groups of four symbols.

    EOS is included using the same convention as the existing
    byte baseline.
    """

    model.eval()

    total_nll = 0.0
    total_bytes = 0

    group_size = 4

    for text in texts:

        ids = tokenizer.encode(text)

        if len(ids) < 2:
            continue

        tokens = torch.tensor(
            ids,
            dtype=torch.long,
        )

        # ----------------------------------------------------
        # Pad to a multiple of four.
        # ----------------------------------------------------

        remainder = len(tokens) % group_size

        if remainder != 0:
            padding = torch.full(
                (group_size - remainder,),
                tokenizer.pad_id,
                dtype=torch.long,
            )

            tokens = torch.cat([tokens, padding])

        groups = tokens.reshape(
            -1,
            group_size,
        )

        # ----------------------------------------------------
        # Predict next groups.
        # ----------------------------------------------------

        for start in range(
            0,
            len(groups) - 1,
            context_length,
        ):

            chunk = groups[start : start + context_length + 1]

            if len(chunk) < 2:
                continue

            x = chunk[:-1].unsqueeze(0).to(device)

            y = chunk[1:].unsqueeze(0).to(device)

            logits = model(x)

            mask = y != tokenizer.pad_id

            losses = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.shape[-1],
                ),
                y.reshape(-1),
                reduction="none",
            )

            mask_flat = mask.reshape(-1).float()

            total_nll += (losses * mask_flat).sum().item()

        total_bytes += len(text.encode("utf-8"))

    model.train()

    if total_bytes == 0:
        raise RuntimeError("No validation bytes.")

    return total_nll / total_bytes / math.log(2)
