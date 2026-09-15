from __future__ import annotations

import torch
from torch import nn


class NeuralTokenizerV1(nn.Module):
    """
    Neural Tokenizer V1.

    Converts groups of 4 byte/special symbols into one continuous
    latent token.

    Input:
        [batch, sequence_length, 4]

    Output:
        [batch, sequence_length, latent_dim]

    Architecture:

        4 symbols
            ↓
        embedding
            ↓
        4 × embedding_dim
            ↓
        FFN
            ↓
        latent token
    """

    def __init__(
        self,
        vocab_size: int = 259,
        embedding_dim: int = 64,
        group_size: int = 4,
        hidden_dim: int = 256,
        latent_dim: int = 128,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.group_size = group_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
        )

        input_dim = group_size * embedding_dim

        self.encoder = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                latent_dim,
            ),
        )

        # Keep embedding initialization controlled.
        nn.init.normal_(
            self.embedding.weight,
            mean=0.0,
            std=0.02,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Integer tensor with shape:

                [batch, sequence_length, group_size]

        Returns
        -------
        torch.Tensor
            Continuous neural tokens:

                [batch, sequence_length, latent_dim]
        """

        if x.ndim != 3:
            raise ValueError(
                "Expected input with shape "
                "[batch, sequence_length, group_size], "
                f"got {tuple(x.shape)}"
            )

        if x.shape[-1] != self.group_size:
            raise ValueError(
                f"Expected group size {self.group_size}, " f"got {x.shape[-1]}"
            )

        # [B, T, 4] → [B, T, 4, 64]
        embedded = self.embedding(x)

        # [B, T, 4, 64] → [B, T, 256]
        flattened = embedded.reshape(
            embedded.shape[0],
            embedded.shape[1],
            self.group_size * self.embedding_dim,
        )

        # [B, T, 256] → [B, T, 128]
        latent = self.encoder(flattened)

        return latent


if __name__ == "__main__":
    tokenizer = NeuralTokenizerV1()

    x = torch.randint(
        low=0,
        high=259,
        size=(2, 8, 4),
    )

    z = tokenizer(x)

    print("Input shape: ", x.shape)
    print("Output shape:", z.shape)
