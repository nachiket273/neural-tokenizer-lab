from __future__ import annotations

import torch
from torch import nn

from neural_tokenizer.neural_v1 import NeuralTokenizerV1


class NeuralV1LanguageModel(nn.Module):
    """
    Neural Tokenizer V1 + causal Transformer language model.

    Input:
        [B, T, 4]

    where each Transformer position represents four
    consecutive byte/special-token symbols.

    Neural tokenizer:

        [B, T, 4]
            ↓
        [B, T, d_model]

    Transformer:

        [B, T, d_model]
            ↓
        [B, T, d_model]

    Output:

        [B, T, 4, vocab_size]

    Each Transformer position predicts the four symbols
    belonging to the next group.
    """

    def __init__(
        self,
        input_vocab_size: int = 259,
        output_vocab_size: int = 259,
        group_size: int = 4,
        tokenizer_embedding_dim: int = 64,
        tokenizer_hidden_dim: int = 256,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        max_seq_len: int = 256,
    ) -> None:
        super().__init__()

        self.group_size = group_size
        self.output_vocab_size = output_vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # ----------------------------------------------------
        # Neural tokenizer
        # ----------------------------------------------------

        self.neural_tokenizer = NeuralTokenizerV1(
            vocab_size=input_vocab_size,
            embedding_dim=tokenizer_embedding_dim,
            group_size=group_size,
            hidden_dim=tokenizer_hidden_dim,
            latent_dim=d_model,
        )

        # ----------------------------------------------------
        # Positional embedding
        # ----------------------------------------------------

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                max_seq_len,
                d_model,
            )
        )

        nn.init.normal_(
            self.position_embedding,
            mean=0.0,
            std=0.02,
        )

        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # ----------------------------------------------------
        # Final normalization
        # ----------------------------------------------------

        self.norm = nn.LayerNorm(d_model)

        # ----------------------------------------------------
        # Output heads
        # ----------------------------------------------------

        self.output_heads = nn.ModuleList(
            [
                nn.Linear(
                    d_model,
                    output_vocab_size,
                )
                for _ in range(group_size)
            ]
        )

    def _causal_mask(
        self,
        sequence_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Construct a causal attention mask.

        Position i may attend only to positions <= i.
        """

        return torch.triu(
            torch.full(
                (
                    sequence_length,
                    sequence_length,
                ),
                float("-inf"),
                device=device,
            ),
            diagonal=1,
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

                [B, T, group_size]

        Returns
        -------
        logits:
            Tensor with shape:

                [B, T, group_size, vocab_size]
        """

        if x.ndim != 3:
            raise ValueError(
                "Expected input shape " "[B, T, group_size], " f"got {tuple(x.shape)}"
            )

        batch_size, sequence_length, group_size = x.shape

        if group_size != self.group_size:
            raise ValueError(
                f"Expected group size " f"{self.group_size}, " f"got {group_size}"
            )

        if sequence_length > self.max_seq_len:
            raise ValueError(
                f"Sequence length {sequence_length} "
                f"exceeds max_seq_len "
                f"{self.max_seq_len}"
            )

        # ----------------------------------------------------
        # Neural tokenizer
        # ----------------------------------------------------
        #
        # [B, T, 4]
        #     ↓
        # [B, T, 128]
        #

        x = self.neural_tokenizer(x)

        # ----------------------------------------------------
        # Positional information
        # ----------------------------------------------------

        x = (
            x
            + self.position_embedding[
                :,
                :sequence_length,
                :,
            ]
        )

        # ----------------------------------------------------
        # Causal self-attention
        # ----------------------------------------------------

        causal_mask = self._causal_mask(
            sequence_length,
            x.device,
        )

        x = self.transformer(
            x,
            mask=causal_mask,
        )

        x = self.norm(x)

        # ----------------------------------------------------
        # Predict next group
        # ----------------------------------------------------
        #
        # Each head predicts one of the four symbols.
        #
        # [B, T, 128]
        #       ↓
        # [B, T, 4, 259]
        #

        logits = torch.stack(
            [head(x) for head in self.output_heads],
            dim=2,
        )

        return logits


if __name__ == "__main__":

    model = NeuralV1LanguageModel()

    x = torch.randint(
        low=0,
        high=259,
        size=(2, 256, 4),
    )

    logits = model(x)

    print(
        "Input shape: ",
        x.shape,
    )

    print(
        "Output shape:",
        logits.shape,
    )

    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Parameters:   " f"{total_parameters:,}")
