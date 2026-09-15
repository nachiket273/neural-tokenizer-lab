from __future__ import annotations

import torch
from torch import nn

from neural_tokenizer.neural_v2_boundary import (
    NeuralV2BoundaryPredictor,
)
from neural_tokenizer.neural_v2_pooling import (
    NeuralV2AdaptivePooling,
)


class NeuralV2Tokenizer(nn.Module):
    """
    Neural Tokenizer V2.

    Pipeline:

        symbols
            ↓
        symbol embedding
            ↓
        boundary predictor
            ↓
        soft boundary probabilities
            ↓
        differentiable adaptive pooling
            ↓
        adaptive continuous tokens
            ↓
        projection to d_model

    Input:
        [B, N]

    Output:
        latent_tokens:
            [B, K, d_model]

        boundary_probs:
            [B, N]
    """

    def __init__(
        self,
        vocab_size: int = 259,
        embedding_dim: int = 64,
        boundary_hidden_dim: int = 128,
        boundary_kernel_size: int = 5,
        num_tokens: int = 256,
        d_model: int = 128,
        pad_id: int = 257,
        pooling_temperature: float = 0.75,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_tokens = num_tokens
        self.d_model = d_model
        self.pad_id = pad_id

        # --------------------------------------------------------
        # Shared source embedding
        # --------------------------------------------------------

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_id,
        )

        nn.init.normal_(
            self.embedding.weight,
            mean=0.0,
            std=0.02,
        )

        # --------------------------------------------------------
        # Boundary predictor
        # --------------------------------------------------------

        self.boundary_predictor = NeuralV2BoundaryPredictor(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=boundary_hidden_dim,
            kernel_size=boundary_kernel_size,
            pad_id=pad_id,
        )

        # --------------------------------------------------------
        # Adaptive pooling
        # --------------------------------------------------------

        self.pooling = NeuralV2AdaptivePooling(
            num_tokens=num_tokens,
            temperature=pooling_temperature,
        )

        # --------------------------------------------------------
        # Project pooled representation to Transformer dimension
        # --------------------------------------------------------

        self.output_projection = nn.Linear(
            embedding_dim,
            d_model,
        )

    def forward(
        self,
        x: torch.Tensor,
        boundary_probs_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x:
            Symbol IDs.

            Shape:
                [B, N]

        Returns
        -------
        latent_tokens:
            [B, K, d_model]

        boundary_probs:
            [B, N]
        """

        if x.ndim != 2:
            raise ValueError("Expected input shape [B, N], " f"got {tuple(x.shape)}")

        # --------------------------------------------------------
        # Source representations
        # --------------------------------------------------------

        features = self.embedding(x)

        # [B, N, embedding_dim]

        # --------------------------------------------------------
        # Boundary probabilities
        # --------------------------------------------------------

        _, boundary_probs = self.boundary_predictor(x)

        # [B, N]

        if boundary_probs_override is not None:
            boundary_probs = boundary_probs_override

            if boundary_probs.shape != x.shape:
                raise ValueError(
                    "Expected boundary_probs_override shape "
                    f"[B, N], got {tuple(boundary_probs.shape)}"
                )

            boundary_probs = boundary_probs.to(
                device=x.device,
                dtype=features.dtype,
            )

        # --------------------------------------------------------
        # Differentiable adaptive pooling
        # --------------------------------------------------------

        pooled = self.pooling(
            features,
            boundary_probs,
        )

        # [B, K, embedding_dim]

        # --------------------------------------------------------
        # Project to LM dimension
        # --------------------------------------------------------

        latent_tokens = self.output_projection(pooled)

        # [B, K, d_model]

        return latent_tokens, boundary_probs


if __name__ == "__main__":
    model = NeuralV2Tokenizer(
        vocab_size=259,
        embedding_dim=64,
        boundary_hidden_dim=128,
        boundary_kernel_size=5,
        num_tokens=256,
        d_model=128,
        pad_id=257,
        pooling_temperature=0.75,
    )

    x = torch.randint(
        low=0,
        high=259,
        size=(2, 1024),
        dtype=torch.long,
    )

    latent_tokens, boundary_probs = model(x)

    print("=" * 70)
    print("NEURAL TOKENIZER V2")
    print("=" * 70)

    print()
    print("Input shape:")
    print("  ", x.shape)

    print()
    print("Boundary probability shape:")
    print("  ", boundary_probs.shape)

    print()
    print("Latent token shape:")
    print("  ", latent_tokens.shape)

    print()
    print("Mean boundary rate:")
    print(f"  {boundary_probs.mean().item():.6f}")

    print()
    print("Expected:")
    print("  Source symbols: 1024")
    print("  Neural tokens:  256")
    print("  Compression:    4.0 symbols/token")

    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print()
    print(f"Tokenizer parameters: {parameters:,}")

    # ------------------------------------------------------------
    # Gradient test
    # ------------------------------------------------------------

    loss = latent_tokens.pow(2).mean()

    loss.backward()

    print()
    print(
        "Boundary predictor gradient:",
        model.boundary_predictor.output.weight.grad is not None,
    )

    print("=" * 70)
