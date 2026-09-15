from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.data import Dataset


class NeuralV2Dataset(Dataset):
    """
    Dataset for Neural Tokenizer V2.

    Each sample contains:

        input:
            1024 source symbols

        target:
            the same source stream represented as
            256 groups of 4 symbols.

    The tokenizer sees all 1024 symbols.

    The language model predicts the next 4-symbol
    group from each adaptive latent token.
    """

    def __init__(
        self,
        texts: Sequence[str],
        tokenizer,
        source_length: int = 1024,
        group_size: int = 4,
        stride: int | None = None,
    ) -> None:

        self.source_length = source_length
        self.group_size = group_size
        self.stride = stride or source_length

        if source_length % group_size != 0:
            raise ValueError("source_length must be divisible by group_size.")

        self.num_tokens = source_length // group_size

        # --------------------------------------------------------
        # Build continuous symbol stream
        # --------------------------------------------------------

        symbols: list[int] = []

        for text in texts:
            symbols.extend(tokenizer.encode(text))

        self.symbols = torch.tensor(
            symbols,
            dtype=torch.long,
        )

        # --------------------------------------------------------
        # Number of complete source windows
        # --------------------------------------------------------
        window_length = 2 * self.source_length

        if len(self.symbols) <= window_length:
            self.num_samples = 0
        else:
            self.num_samples = (
                (len(self.symbols) - 1 - window_length) // self.stride
            ) + 1

        self.pad_id = tokenizer.pad_id

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):

        start = index * self.stride

        window_length = 2 * self.source_length

        chunk = self.symbols[start : start + window_length]

        if len(chunk) != window_length:
            raise RuntimeError(
                f"Incomplete sample at index={index}: "
                f"expected {window_length} symbols, "
                f"got {len(chunk)}"
            )

        x = chunk[: self.source_length]

        target_stream = chunk[self.source_length : 2 * self.source_length]

        y = target_stream.reshape(
            self.num_tokens,
            self.group_size,
        )

        target_mask = y != self.pad_id

        return x, y, target_mask
