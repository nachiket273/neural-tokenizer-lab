from __future__ import annotations

import torch
from torch import nn

from neural_tokenizer.neural_v2_tokenizer import (
    NeuralV2Tokenizer,
)


class NeuralV2LanguageModel(nn.Module):
    """
    Neural Tokenizer V2 language model.

    Architecture:

        1024 source symbols
              ↓
        Neural V2 tokenizer
              ↓
        256 adaptive latent tokens
              ↓
        4-layer causal Transformer
              ↓
        1-layer GRU autoregressive decoder
              ↓
        next-symbol prediction
    """

    def __init__(
        self,
        input_vocab_size: int = 259,
        output_vocab_size: int = 259,
        source_length: int = 1024,
        num_tokens: int = 256,
        tokenizer_embedding_dim: int = 64,
        boundary_hidden_dim: int = 128,
        boundary_kernel_size: int = 5,
        pooling_temperature: float = 0.75,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        gru_hidden_dim: int = 128,
        group_size: int = 4,
        pad_id: int = 258,
    ) -> None:
        super().__init__()

        self.source_length = source_length
        self.num_tokens = num_tokens
        self.d_model = d_model
        self.group_size = group_size
        self.output_vocab_size = output_vocab_size

        # ========================================================
        # V2 tokenizer
        # ========================================================

        self.neural_tokenizer = NeuralV2Tokenizer(
            vocab_size=input_vocab_size,
            embedding_dim=tokenizer_embedding_dim,
            boundary_hidden_dim=boundary_hidden_dim,
            boundary_kernel_size=boundary_kernel_size,
            num_tokens=num_tokens,
            d_model=d_model,
            pad_id=pad_id,
            pooling_temperature=pooling_temperature,
        )

        # ========================================================
        # Main Transformer
        # ========================================================

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                num_tokens,
                d_model,
            )
        )

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
        # Intra-token autoregressive GRU
        # ========================================================

        self.previous_symbol_embedding = nn.Embedding(
            output_vocab_size,
            d_model,
        )

        self.intra_position_embedding = nn.Parameter(
            torch.zeros(
                1,
                group_size,
                d_model,
            )
        )

        nn.init.normal_(
            self.intra_position_embedding,
            mean=0.0,
            std=0.02,
        )

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.gru_norm = nn.LayerNorm(gru_hidden_dim)

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
        previous_symbols: torch.Tensor,
        boundary_probs_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x:
            Source symbols.

            Shape:
                [B, 1024]

        previous_symbols:
            Teacher-forced previous symbols.

            Shape:
                [B, 256, 4]

        Returns
        -------
        logits:
            [B, 256, 4, vocab_size]

        boundary_probs:
            [B, 1024]
        """

        if x.ndim != 2:
            raise ValueError(
                "Expected x shape [B, source_length], " f"got {tuple(x.shape)}"
            )

        if x.shape[1] != self.source_length:
            raise ValueError(
                f"Expected source_length={self.source_length}, " f"got {x.shape[1]}"
            )

        if previous_symbols.ndim != 3:
            raise ValueError(
                "Expected previous_symbols shape [B, T, G], "
                f"got {tuple(previous_symbols.shape)}"
            )

        batch_size = x.shape[0]

        if previous_symbols.shape[1] != self.num_tokens:
            raise ValueError(
                f"Expected {self.num_tokens} latent tokens, "
                f"got {previous_symbols.shape[1]}"
            )

        if previous_symbols.shape[2] != self.group_size:
            raise ValueError(
                f"Expected group_size={self.group_size}, "
                f"got {previous_symbols.shape[2]}"
            )

        # ========================================================
        # V2 tokenizer
        # ========================================================

        latent_tokens, boundary_probs = self.neural_tokenizer(
            x, boundary_probs_override
        )

        # [B, 256, 128]

        # ========================================================
        # Main Transformer
        # ========================================================

        latent_tokens = latent_tokens + self.position_embedding

        causal_mask = self._causal_mask(
            self.num_tokens,
            latent_tokens.device,
        )

        h = self.transformer(
            latent_tokens,
            mask=causal_mask,
        )

        h = self.norm(h)

        # [B, 256, 128]

        # ========================================================
        # GRU decoder
        # ========================================================

        prev_emb = self.previous_symbol_embedding(previous_symbols)

        # [B, 256, 4, 128]

        h_expanded = h.unsqueeze(2).expand(
            batch_size,
            self.num_tokens,
            self.group_size,
            self.d_model,
        )

        pos_emb = self.intra_position_embedding.unsqueeze(0).unsqueeze(0)

        decoder_input = h_expanded + prev_emb + pos_emb

        decoder_input = decoder_input.reshape(
            batch_size * self.num_tokens,
            self.group_size,
            self.d_model,
        )

        gru_output, _ = self.gru(decoder_input)

        gru_output = self.gru_norm(gru_output)

        logits = self.output_projection(gru_output)

        logits = logits.reshape(
            batch_size,
            self.num_tokens,
            self.group_size,
            self.output_vocab_size,
        )

        return logits, boundary_probs


if __name__ == "__main__":
    model = NeuralV2LanguageModel()

    batch_size = 2

    source_length = 1024
    num_tokens = 256
    group_size = 4
    vocab_size = 259

    x = torch.randint(
        low=0,
        high=vocab_size,
        size=(
            batch_size,
            source_length,
        ),
        dtype=torch.long,
    )

    previous_symbols = torch.randint(
        low=0,
        high=vocab_size,
        size=(
            batch_size,
            num_tokens,
            group_size,
        ),
        dtype=torch.long,
    )

    logits, boundary_probs = model(
        x,
        previous_symbols,
    )

    print("=" * 70)
    print("NEURAL TOKENIZER V2 — INTEGRATED MODEL")
    print("=" * 70)

    print()
    print("Input shape:")
    print("  ", x.shape)

    print()
    print("Previous-symbol shape:")
    print("  ", previous_symbols.shape)

    print()
    print("Logits shape:")
    print("  ", logits.shape)

    print()
    print("Boundary probabilities:")
    print("  ", boundary_probs.shape)

    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    tokenizer_parameters = sum(
        p.numel() for p in model.neural_tokenizer.parameters() if p.requires_grad
    )

    print()
    print(f"Total parameters:     {parameters:,}")
    print(f"Tokenizer parameters: {tokenizer_parameters:,}")

    # ============================================================
    # Gradient test
    # ============================================================

    loss = logits.pow(2).mean()

    loss.backward()

    boundary_grad = model.neural_tokenizer.boundary_predictor.output.weight.grad

    print()
    print(
        "Boundary predictor receives gradient:",
        boundary_grad is not None,
    )

    if boundary_grad is not None:
        print(
            "Mean boundary gradient:",
            f"{boundary_grad.abs().mean().item():.10e}",
        )

    print()
    print("=" * 70)
