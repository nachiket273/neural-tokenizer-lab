from collections.abc import Sequence

import torch
from torch.utils.data import Dataset


class LanguageModelDataset(Dataset):
    """
    Fixed-length language-model dataset.

    Each sample is a non-overlapping sequence of
    `context_length` input tokens and one shifted
    target sequence.

    Example with context_length=4:

        tokens:
        [0 1 2 3 4 5 6 7 8 ...]

        sample 0:
        x = [0 1 2 3]
        y = [1 2 3 4]

        sample 1:
        x = [4 5 6 7]
        y = [5 6 7 8]
    """

    def __init__(
        self,
        texts: Sequence[str],
        tokenizer,
        context_length: int = 256,
        stride: int | None = None,
    ):
        self.context_length = context_length
        self.stride = stride or context_length

        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []

        for text in texts:
            ids = tokenizer.encode(text)

            if len(ids) < context_length + 1:
                continue

            for start in range(
                0,
                len(ids) - context_length,
                self.stride,
            ):
                chunk = ids[start : start + context_length + 1]

                if len(chunk) < context_length + 1:
                    break

                x = torch.tensor(
                    chunk[:-1],
                    dtype=torch.long,
                )

                y = torch.tensor(
                    chunk[1:],
                    dtype=torch.long,
                )

                self.samples.append((x, y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[index]
