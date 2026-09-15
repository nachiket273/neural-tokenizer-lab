from __future__ import annotations

import torch
from torch import nn

from neural_tokenizer.neural_v1 import NeuralTokenizerV1


class NeuralV1ARLanguageModel(nn.Module):
    """
    Neural Tokenizer V1.1.

    Architecture:

        4 input symbols
            ↓
        NeuralTokenizerV1
            ↓
        continuous neural token [128]
            ↓
        main Transformer
            ↓
        4-position causal intra-group decoder
            ↓
        4 autoregressive symbol predictions

    The main Transformer is intentionally kept identical to V1.
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
        ar_decoder_layers: int = 1,
        ar_decoder_d_ff: int = 256,
    ) -> None:
        super().__init__()

        self.group_size = group_size
        self.output_vocab_size = output_vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # ---------------------------------------------------------
        # Neural tokenizer
        # ---------------------------------------------------------

        self.neural_tokenizer = NeuralTokenizerV1(
            vocab_size=input_vocab_size,
            embedding_dim=tokenizer_embedding_dim,
            group_size=group_size,
            hidden_dim=tokenizer_hidden_dim,
            latent_dim=d_model,
        )

        # ---------------------------------------------------------
        # Main Transformer
        # ---------------------------------------------------------

        self.position_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        nn.init.normal_(
            self.position_embedding,
            mean=0.0,
            std=0.02,
        )

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

        self.norm = nn.LayerNorm(d_model)

        # ---------------------------------------------------------
        # Autoregressive intra-group decoder
        # ---------------------------------------------------------

        # Embedding of the previously generated symbol.
        #
        # Input at position j contains:
        #
        #   h_t
        #   +
        #   embedding(previous symbol)
        #   +
        #   intra-group position embedding
        #
        self.previous_symbol_embedding = nn.Embedding(
            output_vocab_size,
            d_model,
        )

        self.intra_position_embedding = nn.Parameter(
            torch.zeros(1, group_size, d_model)
        )

        nn.init.normal_(
            self.intra_position_embedding,
            mean=0.0,
            std=0.02,
        )

        ar_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ar_decoder_d_ff,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.ar_decoder = nn.TransformerEncoder(
            ar_layer,
            num_layers=ar_decoder_layers,
        )

        self.ar_norm = nn.LayerNorm(d_model)

        # One shared output projection.
        #
        # Unlike V1, we do NOT have four independent heads.
        self.output_projection = nn.Linear(
            d_model,
            output_vocab_size,
        )

    def _causal_mask(
        self,
        sequence_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Prevent each intra-group position from seeing future positions.
        """
        return torch.triu(
            torch.full(
                (sequence_length, sequence_length),
                float("-inf"),
                device=device,
            ),
            diagonal=1,
        )

    def forward(
        self,
        x: torch.Tensor,
        previous_symbols: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Input symbol groups.

            Shape:
                [B, T, group_size]

        previous_symbols:
            Teacher-forced previous symbols.

            Shape:
                [B, T, group_size]

            For a target group:

                target = [y1, y2, y3, y4]

            previous_symbols should be:

                [BOS, y1, y2, y3]

        Returns
        -------
        logits:
            Shape:

                [B, T, group_size, vocab_size]
        """

        if x.ndim != 3:
            raise ValueError(
                "Expected x shape [B, T, group_size], " f"got {tuple(x.shape)}"
            )

        if previous_symbols.ndim != 3:
            raise ValueError(
                "Expected previous_symbols shape "
                "[B, T, group_size], "
                f"got {tuple(previous_symbols.shape)}"
            )

        batch_size, sequence_length, group_size = x.shape

        if group_size != self.group_size:
            raise ValueError(
                f"Expected group_size={self.group_size}, " f"got {group_size}"
            )

        if previous_symbols.shape != x.shape:
            raise ValueError(
                "previous_symbols must have the same shape as x. "
                f"Got x={tuple(x.shape)}, "
                f"previous_symbols={tuple(previous_symbols.shape)}"
            )

        if sequence_length > self.max_seq_len:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"max_seq_len={self.max_seq_len}"
            )

        # ---------------------------------------------------------
        # Neural tokenizer
        # ---------------------------------------------------------

        z = self.neural_tokenizer(x)

        # z:
        # [B, T, d_model]

        # ---------------------------------------------------------
        # Main Transformer
        # ---------------------------------------------------------

        z = z + self.position_embedding[:, :sequence_length, :]

        causal_mask = self._causal_mask(
            sequence_length,
            z.device,
        )

        h = self.transformer(
            z,
            mask=causal_mask,
        )

        h = self.norm(h)

        # h:
        # [B, T, d_model]

        # ---------------------------------------------------------
        # Intra-group autoregressive decoder
        # ---------------------------------------------------------

        # Previous symbol embeddings:
        #
        # [B, T, 4] → [B, T, 4, d_model]
        prev_emb = self.previous_symbol_embedding(previous_symbols)

        # Expand main Transformer representation across
        # the four intra-group positions.
        #
        # [B, T, d_model]
        #       ↓
        # [B, T, 4, d_model]
        h_expanded = h.unsqueeze(2).expand(
            batch_size,
            sequence_length,
            self.group_size,
            self.d_model,
        )

        # Intra-group positional information.
        pos_emb = self.intra_position_embedding.unsqueeze(0).unsqueeze(0)

        # Combined decoder input:
        #
        # h_t
        # + previous symbol
        # + intra-group position
        #
        decoder_input = h_expanded + prev_emb + pos_emb

        # Flatten B and T so each group can be processed
        # independently by the small 4-position decoder.
        #
        # [B, T, 4, d]
        #       ↓
        # [B*T, 4, d]
        decoder_input = decoder_input.reshape(
            batch_size * sequence_length,
            self.group_size,
            self.d_model,
        )

        ar_mask = self._causal_mask(
            self.group_size,
            decoder_input.device,
        )

        decoder_output = self.ar_decoder(
            decoder_input,
            mask=ar_mask,
        )

        decoder_output = self.ar_norm(decoder_output)

        # Shared vocabulary projection.
        #
        # [B*T, 4, d]
        #       ↓
        # [B*T, 4, V]
        logits = self.output_projection(decoder_output)

        # Restore batch/group structure.
        #
        # [B*T, 4, V]
        #       ↓
        # [B, T, 4, V]
        logits = logits.reshape(
            batch_size,
            sequence_length,
            self.group_size,
            self.output_vocab_size,
        )

        return logits


if __name__ == "__main__":
    model = NeuralV1ARLanguageModel()

    batch_size = 2
    sequence_length = 8
    group_size = 4
    vocab_size = 259

    x = torch.randint(
        low=0,
        high=vocab_size,
        size=(
            batch_size,
            sequence_length,
            group_size,
        ),
    )

    previous_symbols = torch.randint(
        low=0,
        high=vocab_size,
        size=(
            batch_size,
            sequence_length,
            group_size,
        ),
    )

    logits = model(
        x,
        previous_symbols,
    )

    print("Input shape:           ", x.shape)
    print("Previous symbols shape:", previous_symbols.shape)
    print("Output shape:          ", logits.shape)

    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    tokenizer_parameters = sum(
        p.numel() for p in model.neural_tokenizer.parameters() if p.requires_grad
    )

    print(f"Total parameters:      {total_parameters:,}")
    print(f"Tokenizer parameters:  {tokenizer_parameters:,}")
