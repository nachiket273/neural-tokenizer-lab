from __future__ import annotations

import torch
from torch import nn

from neural_tokenizer.neural_v1 import NeuralTokenizerV1


class NeuralV1GRULanguageModel(nn.Module):
    """
    Neural Tokenizer V1.2.

    Controlled ablation of V1.1.

    Main architecture:

        4 input symbols
            ↓
        NeuralTokenizerV1
            ↓
        continuous neural token [d_model]
            ↓
        causal main Transformer
            ↓
        h_t
            ↓
        4-position autoregressive GRU
            ↓
        shared vocabulary projection

    The only intended architectural difference from V1.1
    is the replacement of the 1-layer Transformer-based
    intra-group decoder with a GRU.
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
        gru_hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        self.group_size = group_size
        self.output_vocab_size = output_vocab_size
        self.d_model = d_model
        self.gru_hidden_dim = gru_hidden_dim
        self.max_seq_len = max_seq_len

        # ========================================================
        # Neural tokenizer
        # ========================================================

        self.neural_tokenizer = NeuralTokenizerV1(
            vocab_size=input_vocab_size,
            embedding_dim=tokenizer_embedding_dim,
            group_size=group_size,
            hidden_dim=tokenizer_hidden_dim,
            latent_dim=d_model,
        )

        # ========================================================
        # Main Transformer
        # ========================================================

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

        # ========================================================
        # Intra-group autoregressive decoder
        # ========================================================

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

        # The GRU processes exactly four positions per group.
        #
        # Input:
        #   h_t + previous-symbol embedding + position embedding
        #
        # Hidden state:
        #   autoregressive state across the four symbols.
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.gru_norm = nn.LayerNorm(gru_hidden_dim)

        # Shared output projection.
        self.output_projection = nn.Linear(
            gru_hidden_dim,
            output_vocab_size,
        )

    def _causal_mask(
        self,
        sequence_length: int,
        device: torch.device,
    ) -> torch.Tensor:
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
            Input groups.

            Shape:
                [B, T, G]

        previous_symbols:
            Teacher-forced previous symbols.

            Shape:
                [B, T, G]

            For target:

                [y1, y2, y3, y4]

            input is:

                [BOS, y1, y2, y3]

        Returns
        -------
        logits:
            [B, T, G, V]
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

        # ========================================================
        # Neural tokenizer
        # ========================================================

        z = self.neural_tokenizer(x)

        # [B, T, d_model]

        # ========================================================
        # Main Transformer
        # ========================================================

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

        # [B, T, d_model]

        # ========================================================
        # Build intra-group decoder input
        # ========================================================

        prev_emb = self.previous_symbol_embedding(previous_symbols)

        # [B, T, G, d_model]

        h_expanded = h.unsqueeze(2).expand(
            batch_size,
            sequence_length,
            self.group_size,
            self.d_model,
        )

        pos_emb = self.intra_position_embedding.unsqueeze(0).unsqueeze(0)

        decoder_input = h_expanded + prev_emb + pos_emb

        # [B, T, G, d_model]

        # ========================================================
        # Flatten groups
        # ========================================================

        decoder_input = decoder_input.reshape(
            batch_size * sequence_length,
            self.group_size,
            self.d_model,
        )

        # ========================================================
        # GRU
        # ========================================================

        gru_output, _ = self.gru(decoder_input)

        # [B*T, G, gru_hidden_dim]

        gru_output = self.gru_norm(gru_output)

        # ========================================================
        # Vocabulary prediction
        # ========================================================

        logits = self.output_projection(gru_output)

        # [B*T, G, V]

        logits = logits.reshape(
            batch_size,
            sequence_length,
            self.group_size,
            self.output_vocab_size,
        )

        return logits


if __name__ == "__main__":
    model = NeuralV1GRULanguageModel()

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
