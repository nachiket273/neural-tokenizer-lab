from __future__ import annotations

import math

import torch


@torch.no_grad()
def evaluate_neural_v1_ar_bpb(
    model,
    texts,
    tokenizer,
    device: torch.device,
    group_size: int = 4,
    bos_id: int = 258,
    max_groups: int = 256,
):
    """
    Evaluate Neural V1.1 in bits per original UTF-8 byte.

    Uses teacher forcing, as appropriate for NLL/BPB evaluation.
    """

    model.eval()

    total_nll = 0.0
    total_bytes = 0

    for text in texts:
        symbols = tokenizer.encode(text)

        total_bytes += len(text.encode("utf-8"))

        if len(symbols) == 0:
            continue

        # Pad final group.
        remainder = len(symbols) % group_size

        if remainder != 0:
            padding_needed = group_size - remainder

            symbols = symbols + [tokenizer.pad_id] * padding_needed

        symbols_tensor = torch.tensor(
            symbols,
            dtype=torch.long,
        )

        groups = symbols_tensor.reshape(
            -1,
            group_size,
        )

        # Need T+1 groups for next-group prediction.
        if len(groups) < 2:
            continue

        for start in range(
            0,
            len(groups) - 1,
            max_groups,
        ):
            chunk = groups[start : start + max_groups + 1]

            if len(chunk) < 2:
                continue

            x = chunk[:-1].unsqueeze(0).to(device)
            targets = chunk[1:].unsqueeze(0).to(device)

            target_mask = targets != tokenizer.pad_id

            previous_symbols = torch.empty_like(targets)

            previous_symbols[..., 0] = bos_id
            previous_symbols[..., 1:] = targets[..., :-1]

            logits = model(
                x,
                previous_symbols,
            )

            log_probs = torch.log_softmax(
                logits,
                dim=-1,
            )

            token_log_probs = log_probs.gather(
                dim=-1,
                index=targets.unsqueeze(-1),
            ).squeeze(-1)

            total_nll += -token_log_probs[target_mask].sum().item()

    if total_bytes == 0:
        return float("nan")

    return total_nll / (math.log(2.0) * total_bytes)
