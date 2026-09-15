from __future__ import annotations

import torch
from torch import nn


class NeuralV2AdaptivePooling(nn.Module):
    """
    Neural Tokenizer V2 - Step B.

    Differentiable adaptive pooling from source-symbol
    representations to a fixed number of continuous
    neural-token slots.

    Input
    -----
    features:
        [B, N, D]

        B = batch size
        N = number of source symbols
        D = feature dimension

    boundary_probs:
        [B, N]

        Soft probability that a boundary occurs after
        each source position.

    Output
    ------
    pooled:
        [B, K, D]

        K = number of neural-token slots.

    Core idea
    ---------
    1. Convert boundary probabilities into cumulative
       soft token coordinates.

           c_i = sum_{j <= i} p_j

    2. Define K target token centers.

    3. Assign every source position softly to nearby
       token centers using a Gaussian kernel.

    4. Normalize the assignments.

    5. Compute weighted averages of source features.

    Everything is differentiable.
    """

    def __init__(
        self,
        num_tokens: int = 256,
        temperature: float = 0.75,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive.")

        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        self.num_tokens = num_tokens
        self.temperature = temperature
        self.eps = eps

    def forward(
        self,
        features: torch.Tensor,
        boundary_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        features:
            Source representations.

            Shape:
                [B, N, D]

        boundary_probs:
            Boundary probabilities.

            Shape:
                [B, N]

        Returns
        -------
        pooled:
            Adaptive continuous token representations.

            Shape:
                [B, K, D]
        """

        if features.ndim != 3:
            raise ValueError(
                "Expected features shape [B, N, D], " f"got {tuple(features.shape)}"
            )

        if boundary_probs.ndim != 2:
            raise ValueError(
                "Expected boundary_probs shape [B, N], "
                f"got {tuple(boundary_probs.shape)}"
            )

        batch_size, sequence_length, feature_dim = features.shape

        if boundary_probs.shape != (
            batch_size,
            sequence_length,
        ):
            raise ValueError(
                "boundary_probs must have shape [B, N] " "matching features."
            )

        # --------------------------------------------------------
        # 1. Cumulative soft boundary coordinate
        # --------------------------------------------------------
        #
        # c_i = sum_{j <= i} p_j
        #
        # Shape:
        #   [B, N]
        #
        cumulative_positions = torch.cumsum(
            boundary_probs,
            dim=1,
        )

        # --------------------------------------------------------
        # 2. Create target token centers
        # --------------------------------------------------------
        #
        # If there are K token slots:
        #
        #   0.5, 1.5, 2.5, ..., K-0.5
        #
        # These represent the centers of the K adaptive
        # token regions.
        #
        token_centers = (
            torch.arange(
                self.num_tokens,
                device=features.device,
                dtype=features.dtype,
            )
            + 0.5
        )

        # [K]

        # --------------------------------------------------------
        # 3. Convert cumulative positions into soft
        #    source-to-token distances.
        # --------------------------------------------------------
        #
        # cumulative_positions:
        #   [B, N]
        #
        # token_centers:
        #   [K]
        #
        # Difference:
        #   [B, N, K]
        #
        distances = cumulative_positions.unsqueeze(-1) - token_centers.view(1, 1, -1)

        # [B, N, K]

        # --------------------------------------------------------
        # 4. Gaussian soft assignment
        # --------------------------------------------------------
        #
        # Larger temperature:
        #   broader assignments
        #
        # Smaller temperature:
        #   sharper assignments
        #
        weights = torch.exp(-0.5 * (distances / self.temperature).pow(2))

        # [B, N, K]

        # --------------------------------------------------------
        # 5. Normalize over source positions
        # --------------------------------------------------------
        #
        # Every token slot becomes a weighted average
        # of the source representations.
        #
        normalization = (
            weights.sum(
                dim=1,
                keepdim=True,
            )
            + self.eps
        )

        normalized_weights = weights / normalization

        # [B, N, K]

        # --------------------------------------------------------
        # 6. Weighted pooling
        # --------------------------------------------------------
        #
        # features:
        #   [B, N, D]
        #
        # normalized_weights:
        #   [B, N, K]
        #
        # Result:
        #   [B, K, D]
        #
        pooled = torch.einsum(
            "bnk,bnd->bkd",
            normalized_weights,
            features,
        )

        return pooled


def adaptive_pooling_rate(
    boundary_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Return the average boundary rate.

    Parameters
    ----------
    boundary_probs:
        [B, N]

    Returns
    -------
    Scalar tensor.

    Example
    -------
    If average boundary probability is 0.25:

        rate = 0.25

    corresponding to approximately:

        4 source symbols / token
    """

    if boundary_probs.ndim != 2:
        raise ValueError(
            "Expected boundary_probs shape [B, N], "
            f"got {tuple(boundary_probs.shape)}"
        )

    return boundary_probs.mean()


def adaptive_pooling_rate_loss(
    boundary_probs: torch.Tensor,
    target_symbols_per_token: float = 4.0,
) -> torch.Tensor:
    """
    Penalize deviation from the desired average token rate.

    For four symbols per token:

        target boundary rate = 1 / 4 = 0.25
    """

    if target_symbols_per_token <= 0:
        raise ValueError("target_symbols_per_token must be positive.")

    target_rate = 1.0 / target_symbols_per_token

    actual_rate = adaptive_pooling_rate(boundary_probs)

    return (actual_rate - target_rate).pow(2)


if __name__ == "__main__":
    # ============================================================
    # Configuration
    # ============================================================

    batch_size = 2
    num_source_symbols = 1024
    num_tokens = 256
    feature_dim = 64

    print("=" * 70)
    print("NEURAL TOKENIZER V2 — STEP B")
    print("DIFFERENTIABLE ADAPTIVE POOLING")
    print("=" * 70)

    # ============================================================
    # Create pooling layer
    # ============================================================

    pooling = NeuralV2AdaptivePooling(
        num_tokens=num_tokens,
        temperature=0.75,
    )

    # ============================================================
    # Random source features
    # ============================================================

    features = torch.randn(
        batch_size,
        num_source_symbols,
        feature_dim,
    )

    # ============================================================
    # Test 1: uniform boundaries
    # ============================================================

    uniform_probs = torch.full(
        (
            batch_size,
            num_source_symbols,
        ),
        0.25,
    )

    pooled = pooling(
        features,
        uniform_probs,
    )

    print()
    print("Test 1 — Uniform boundaries")
    print("-" * 40)

    print(
        "Input features shape:",
        features.shape,
    )

    print(
        "Boundary probabilities:",
        uniform_probs.shape,
    )

    print(
        "Pooled output shape:",
        pooled.shape,
    )

    print(
        "Expected output shape:",
        (
            batch_size,
            num_tokens,
            feature_dim,
        ),
    )

    # ============================================================
    # Test 2: rate
    # ============================================================

    rate = adaptive_pooling_rate(uniform_probs)

    rate_loss = adaptive_pooling_rate_loss(
        uniform_probs,
        target_symbols_per_token=4.0,
    )

    print()
    print("Test 2 — Rate")
    print("-" * 40)

    print(
        "Boundary rate:",
        f"{rate.item():.6f}",
    )

    print(
        "Target rate:",
        f"{1.0 / 4.0:.6f}",
    )

    print(
        "Rate loss:",
        f"{rate_loss.item():.10f}",
    )

    # ============================================================
    # Test 3: non-uniform boundaries
    # ============================================================

    nonuniform_probs = torch.full(
        (
            batch_size,
            num_source_symbols,
        ),
        0.25,
    )

    # Create a region where boundaries become more likely.
    nonuniform_probs[:, 200:300] = 0.10
    nonuniform_probs[:, 300:350] = 0.50

    pooled_nonuniform = pooling(
        features,
        nonuniform_probs,
    )

    print()
    print("Test 3 — Non-uniform boundaries")
    print("-" * 40)

    print(
        "Uniform mean boundary rate:",
        f"{uniform_probs.mean().item():.6f}",
    )

    print(
        "Non-uniform mean boundary rate:",
        f"{nonuniform_probs.mean().item():.6f}",
    )

    print(
        "Non-uniform output shape:",
        pooled_nonuniform.shape,
    )

    # ============================================================
    # Test 4: outputs should change when boundaries change
    # ============================================================

    difference = (pooled - pooled_nonuniform).abs().mean().item()

    print()
    print("Test 4 — Adaptive response")
    print("-" * 40)

    print(
        "Mean pooled-feature difference:",
        f"{difference:.8f}",
    )

    print(
        "Pooling responds to boundaries:",
        difference > 1e-8,
    )

    # ============================================================
    # Test 5: gradient flow
    # ============================================================

    features_grad = torch.randn(
        batch_size,
        num_source_symbols,
        feature_dim,
        requires_grad=True,
    )

    boundary_logits = torch.randn(
        batch_size,
        num_source_symbols,
        requires_grad=True,
    )

    boundary_probs_grad = torch.sigmoid(boundary_logits)

    pooled_grad = pooling(
        features_grad,
        boundary_probs_grad,
    )

    loss = pooled_grad.pow(2).mean()

    loss.backward()

    print()
    print("Test 5 — Gradient flow")
    print("-" * 40)

    print(
        "Feature gradients present:",
        features_grad.grad is not None,
    )

    print(
        "Boundary-logit gradients present:",
        boundary_logits.grad is not None,
    )

    if boundary_logits.grad is not None:
        print(
            "Mean boundary gradient:",
            f"{boundary_logits.grad.abs().mean().item():.10e}",
        )

    print()
    print("=" * 70)
    print("STEP B TESTS COMPLETE")
    print("=" * 70)
