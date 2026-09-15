from collections.abc import Sequence

import torch
from torch.utils.data import Dataset


class LanguageModelDataset(Dataset):
    """
    Language-model dataset using non-overlapping chunks.

    All tokenized texts are concatenated into a single stream.
    EOS tokens separate individual texts.

    Example:

        tokens:
        [0 1 2 3 4 5 6 7 8 ...]

        context_length = 4

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
    ) -> None:
        self.context_length = context_length
        self.stride = stride or context_length

        encoded: list[int] = []

        for text in texts:
            encoded.extend(tokenizer.encode(text))

        self.data = torch.tensor(
            encoded,
            dtype=torch.long,
        )

        # Number of complete input/target chunks.
        if len(self.data) <= context_length:
            self.num_samples = 0
        else:
            self.num_samples = (
                (len(self.data) - 1 - context_length) // self.stride
            ) + 1

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        start = index * self.stride

        chunk = self.data[start : start + self.context_length + 1]

        x = chunk[:-1]
        y = chunk[1:]

        return x, y


class NeuralV1Dataset(Dataset):
    """
    Dataset for Neural Tokenizer V1.

    The corpus is represented as:

        UTF-8 bytes + EOS

    Then symbols are grouped into groups of 4.

    Each training sample contains:

        input:
            256 groups × 4 symbols

        target:
            256 groups × 4 symbols

        target_mask:
            mask identifying valid target symbols

    Thus one Transformer position corresponds to approximately
    four original UTF-8 bytes.
    """

    def __init__(
        self,
        texts: Sequence[str],
        tokenizer,
        context_length: int = 256,
        group_size: int = 4,
        stride: int | None = None,
    ) -> None:

        self.context_length = context_length
        self.group_size = group_size
        self.stride = stride or context_length
        self.pad_id = tokenizer.pad_id

        symbols: list[int] = []

        for text in texts:
            symbols.extend(tokenizer.encode(text))

        self.symbols = torch.tensor(
            symbols,
            dtype=torch.long,
        )

        # Number of complete groups.
        num_groups = len(self.symbols) // self.group_size

        remainder = len(self.symbols) % self.group_size

        if remainder != 0:
            padding_needed = self.group_size - remainder

            padding = torch.full(
                (padding_needed,),
                self.pad_id,
                dtype=torch.long,
            )

            self.symbols = torch.cat(
                [
                    self.symbols,
                    padding,
                ]
            )

            num_groups += 1

        self.num_groups = num_groups

        self.groups = self.symbols.reshape(
            self.num_groups,
            self.group_size,
        )

        # We need context_length + 1 groups:
        #
        # x = groups[0:256]
        # y = groups[1:257]
        #
        if self.num_groups <= context_length:
            self.num_samples = 0
        else:
            self.num_samples = (
                (self.num_groups - 1 - context_length) // self.stride
            ) + 1

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(
        self,
        index: int,
    ):
        start = index * self.stride

        chunk = self.groups[start : start + self.context_length + 1]

        x = chunk[:-1]
        y = chunk[1:]

        # Valid target positions.
        #
        # PAD symbols are not part of the original corpus and
        # therefore should not contribute to the loss.
        target_mask = y != self.pad_id

        return x, y, target_mask
