from __future__ import annotations

import torch
from torch import nn


def boundary_binary_loss(
    boundary_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Encourage boundary probabilities toward 0 or 1.

    L_binary = mean(p * (1 - p))

    This is:
        0       when p = 0 or 1
        0.1875  when p = 0.25
        0.25    when p = 0.5

    Minimizing this term therefore encourages confident
    boundary decisions without directly controlling the
    overall boundary rate.
    """

    return torch.mean(boundary_probs * (1.0 - boundary_probs))


def boundary_rate_loss(
    boundary_probs: torch.Tensor,
    target_symbols_per_token: float = 4.0,
) -> torch.Tensor:
    """
    Penalize deviation from the desired average
    symbols-per-token rate.

    If target_symbols_per_token = 4:

        target boundary rate = 1 / 4 = 0.25

    Parameters
    ----------
    boundary_probs:
        [B, N]

    target_symbols_per_token:
        Desired average number of source symbols per token.

    Returns
    -------
    Scalar rate loss.
    """

    if boundary_probs.ndim != 2:
        raise ValueError(
            "Expected boundary_probs shape [B, N], "
            f"got {tuple(boundary_probs.shape)}"
        )

    target_boundary_rate = 1.0 / target_symbols_per_token

    actual_boundary_rate = boundary_probs.mean()

    return (actual_boundary_rate - target_boundary_rate).pow(2)


class NeuralV2BoundaryPredictor(nn.Module):
    """
    Neural Tokenizer V2 - Step A.

    Predicts a soft boundary probability for every input symbol.

    Input:
        x: [B, N]
            Integer symbol IDs.

    Output:
        boundary_logits: [B, N]
        boundary_probs:  [B, N]

    A boundary probability p_i represents the model's
    confidence that a token boundary should occur after
    source position i.

    Architecture:

        symbol IDs
             |
             v
        Symbol Embedding
             |
             v
        causal Conv1D
             |
             v
        GELU
             |
             v
        Linear
             |
             v
        boundary logits
             |
             v
        sigmoid
             |
             v
        boundary probabilities
    """

    def __init__(
        self,
        vocab_size: int = 259,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        kernel_size: int = 5,
        pad_id: int = 257,
    ) -> None:
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be odd for the causal " "padding calculation."
            )

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.pad_id = pad_id

        # --------------------------------------------------------
        # Symbol embedding
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
        # Causal convolution
        # --------------------------------------------------------
        #
        # We want the boundary after position i to depend only
        # on positions <= i.
        #
        # For kernel size 5:
        #
        # position i sees:
        #
        #   i-4, i-3, i-2, i-1, i
        #
        # and never i+1, i+2, ...
        #
        # PyTorch Conv1d padding is symmetric, so we explicitly
        # remove the future-padding positions after convolution.
        # --------------------------------------------------------

        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
        )

        # --------------------------------------------------------
        # Boundary prediction head
        # --------------------------------------------------------

        self.activation = nn.GELU()

        self.output = nn.Linear(
            hidden_dim,
            1,
        )

        # Initialize the boundary head so that the initial
        # boundary probability is close to 0.25.
        #
        # Why 0.25?
        #
        # Our initial target is approximately:
        #
        #   4 symbols / token
        #
        # Therefore we want roughly:
        #
        #   1 boundary / 4 symbols
        #
        # sigmoid(logit) = 0.25
        #
        # logit(0.25) = log(0.25 / 0.75)
        #              ≈ -1.0986
        #
        nn.init.zeros_(self.output.weight)

        initial_boundary_prob = 0.25

        initial_logit = torch.log(
            torch.tensor(initial_boundary_prob) / (1.0 - initial_boundary_prob)
        )

        nn.init.constant_(
            self.output.bias,
            initial_logit.item(),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict boundary probabilities.

        Parameters
        ----------
        x:
            Integer symbol IDs.

            Shape:
                [B, N]

        Returns
        -------
        boundary_logits:
            Shape:
                [B, N]

        boundary_probs:
            Shape:
                [B, N]

            Values lie in [0, 1].
        """

        if x.ndim != 2:
            raise ValueError("Expected input shape [B, N], " f"got {tuple(x.shape)}")

        if x.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise ValueError(
                "Input must contain integer symbol IDs, " f"got dtype={x.dtype}"
            )

        # --------------------------------------------------------
        # Embedding
        # --------------------------------------------------------

        embedded = self.embedding(x)

        # [B, N, embedding_dim]

        # --------------------------------------------------------
        # Conv1D expects:
        #
        # [B, channels, sequence]
        #
        # so transpose.
        # --------------------------------------------------------

        conv_input = embedded.transpose(1, 2)

        # [B, embedding_dim, N]

        features = self.conv(conv_input)

        # Because we used padding=kernel_size-1, the convolution
        # produces extra positions at the right.
        #
        # Keep only the original N positions.
        features = features[:, :, : x.shape[1]]

        # [B, hidden_dim, N]

        features = features.transpose(1, 2)

        # [B, N, hidden_dim]

        features = self.activation(features)

        # --------------------------------------------------------
        # Boundary logits
        # --------------------------------------------------------

        boundary_logits = self.output(features).squeeze(-1)

        # [B, N]

        boundary_probs = torch.sigmoid(boundary_logits)

        # [B, N]

        return boundary_logits, boundary_probs


if __name__ == "__main__":
    # ============================================================
    # Basic shape test
    # ============================================================

    batch_size = 2
    sequence_length = 32
    vocab_size = 259

    model = NeuralV2BoundaryPredictor(
        vocab_size=vocab_size,
        embedding_dim=64,
        hidden_dim=128,
        kernel_size=5,
        pad_id=257,
    )

    x = torch.randint(
        low=0,
        high=vocab_size,
        size=(
            batch_size,
            sequence_length,
        ),
        dtype=torch.long,
    )

    logits, probs = model(x)

    print("=" * 70)
    print("NEURAL TOKENIZER V2 — STEP A")
    print("BOUNDARY PREDICTOR")
    print("=" * 70)

    print()
    print("Input shape:          ", x.shape)
    print("Boundary logits shape:", logits.shape)
    print("Boundary probs shape: ", probs.shape)

    print()
    print("Probability range:")
    print(f"  min = {probs.min().item():.6f}")
    print(f"  max = {probs.max().item():.6f}")
    print(f"  mean = {probs.mean().item():.6f}")

    # ------------------------------------------------------------
    # Parameter count
    # ------------------------------------------------------------

    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print()
    print(f"Parameters: {parameters:,}")

    # ------------------------------------------------------------
    # Gradient test
    # ------------------------------------------------------------

    loss = probs.mean()

    loss.backward()

    gradients_present = all(
        p.grad is not None for p in model.parameters() if p.requires_grad
    )

    print()
    print(
        "All trainable parameters have gradients:",
        gradients_present,
    )

    print()
    print("=" * 70)

    # ============================================================
    # Causality test
    # ============================================================

    model.zero_grad(set_to_none=True)

    x1 = torch.randint(
        low=0,
        high=vocab_size,
        size=(1, 32),
        dtype=torch.long,
    )

    x2 = x1.clone()

    # Change only a future symbol.
    x2[0, 20] = (x2[0, 20] + 1) % vocab_size

    _, probs1 = model(x1)
    _, probs2 = model(x2)

    # Predictions before position 20 must remain unchanged.
    max_difference = (probs1[0, :20] - probs2[0, :20]).abs().max().item()

    print()
    print("Causality test")
    print("-" * 40)

    print(
        "Max difference before changed position:",
        f"{max_difference:.10f}",
    )

    print(
        "Causal:",
        max_difference < 1e-7,
    )

    # ============================================================
    # Rate-loss test
    # ============================================================

    rate_loss = boundary_rate_loss(
        probs,
        target_symbols_per_token=4.0,
    )

    print()
    print("Rate loss test")
    print("-" * 40)

    print(
        "Initial boundary rate:",
        f"{probs.mean().item():.6f}",
    )

    print(
        "Target boundary rate:",
        f"{1.0 / 4.0:.6f}",
    )

    print(
        "Rate loss:",
        f"{rate_loss.item():.10f}",
    )
