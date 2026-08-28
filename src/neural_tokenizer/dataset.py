from collections.abc import Sequence

import torch
from torch.utils.data import Dataset


class LanguageModelDataset(Dataset):
    def __init__(
        self, texts: Sequence[str], tokenizer, context_length: int = 256
    ) -> None:
        self.context_length = context_length
        encoded: list[int] = []

        for text in texts:
            encoded.extend(tokenizer.encode(text))

        self.data = torch.tensor(encoded, dtype=torch.long)

    def __len__(self) -> int:
        return max(0, len(self.data) - self.context_length)

    def __getitem__(self, index: int):
        chunk = self.data[index : index + self.context_length + 1]
        return chunk[:-1], chunk[1:]
