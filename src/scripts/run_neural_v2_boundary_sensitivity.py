from __future__ import annotations

import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader

from neural_tokenizer.neural_v2_boundary import (
    boundary_binary_loss,
    boundary_rate_loss,
)
from neural_tokenizer.neural_v2_dataset import NeuralV2Dataset
from neural_tokenizer.neural_v2_model import NeuralV2LanguageModel
from neural_tokenizer.tokenizer import ByteTokenizer

# ============================================================
# Configuration
# ============================================================

TRAIN_SIZE = 50_000
VALID_SIZE = 5_000

SOURCE_LENGTH = 1024
GROUP_SIZE = 4
NUM_TOKENS = SOURCE_LENGTH // GROUP_SIZE

BATCH_SIZE = 4

TOKENIZER_EMBEDDING_DIM = 64
BOUNDARY_HIDDEN_DIM = 128
BOUNDARY_KERNEL_SIZE = 5
POOLING_TEMPERATURE = 0.75

D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
D_FF = 512
GRU_HIDDEN_DIM = 128

TARGET_SYMBOLS_PER_TOKEN = 4.0

LAMBDA_RATE = 1.0
LAMBDA_BINARY = 0.01

SEED = 42

# Use the same best checkpoint produced by the training runner.
CHECKPOINT_PATH = Path("experiments/neural_v2/best.pt")

OUTPUT_DIR = Path("experiments/neural_v2/boundary_sensitivity")

# Keep this identical to the earlier sensitivity experiment.
MAX_VALIDATION_BATCHES = 250

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Reproducibility
# ============================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Data
# ============================================================


def load_tinystories():
    print("\nLoading TinyStories...")

    dataset = load_dataset("roneneldan/TinyStories")

    train_texts = [row["text"] for row in dataset["train"].select(range(TRAIN_SIZE))]

    valid_texts = [
        row["text"] for row in dataset["validation"].select(range(VALID_SIZE))
    ]

    print(f"Training stories:   {len(train_texts):,}")

    print(f"Validation stories: {len(valid_texts):,}")

    return train_texts, valid_texts


# ============================================================
# Corpus statistics
# ============================================================


def get_statistics(
    texts,
    tokenizer,
):
    total_bytes = 0
    total_symbols = 0

    for text in texts:
        total_bytes += len(text.encode("utf-8"))

        total_symbols += len(tokenizer.encode(text))

    return {
        "bytes": total_bytes,
        "symbols": total_symbols,
        "bytes_per_symbol": (total_bytes / total_symbols),
    }


# ============================================================
# Parameter counting
# ============================================================


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)

    tokenizer = sum(
        p.numel() for p in model.neural_tokenizer.parameters() if p.requires_grad
    )

    return {
        "total": total,
        "neural_tokenizer": tokenizer,
        "transformer_and_gru": (total - tokenizer),
    }


# ============================================================
# Boundary metrics
# ============================================================


def calculate_boundary_entropy(
    boundary_probs: torch.Tensor,
) -> float:
    eps = 1e-7

    p = boundary_probs.clamp(
        min=eps,
        max=1.0 - eps,
    )

    entropy = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))

    return entropy.mean().item()


def calculate_boundary_metrics(
    boundary_probs: torch.Tensor,
):
    mean = boundary_probs.mean().item()

    return {
        "boundary_mean": mean,
        "boundary_std": (boundary_probs.std().item()),
        "boundary_min": (boundary_probs.min().item()),
        "boundary_max": (boundary_probs.max().item()),
        "boundary_entropy": (calculate_boundary_entropy(boundary_probs)),
        "low_confident": ((boundary_probs < 0.1).float().mean().item()),
        "high_confident": ((boundary_probs > 0.9).float().mean().item()),
        "symbols_per_token": (1.0 / max(mean, 1e-8)),
    }


# ============================================================
# Boundary interventions
# ============================================================


def make_uniform_boundaries(
    batch_size: int,
    device: torch.device,
):
    return torch.full(
        (
            batch_size,
            SOURCE_LENGTH,
        ),
        1.0 / GROUP_SIZE,
        device=device,
        dtype=torch.float32,
    )


def make_regular_boundaries(
    batch_size: int,
    device: torch.device,
):
    boundaries = torch.zeros(
        (
            batch_size,
            SOURCE_LENGTH,
        ),
        device=device,
        dtype=torch.float32,
    )

    positions = torch.arange(
        GROUP_SIZE - 1,
        SOURCE_LENGTH,
        GROUP_SIZE,
        device=device,
    )

    boundaries[:, positions] = 1.0

    return boundaries


def make_random_boundaries(
    batch_size: int,
    device: torch.device,
):
    boundaries = torch.zeros(
        (
            batch_size,
            SOURCE_LENGTH,
        ),
        device=device,
        dtype=torch.float32,
    )

    num_boundaries = round(SOURCE_LENGTH / GROUP_SIZE)

    for batch_idx in range(batch_size):

        positions = torch.randperm(
            SOURCE_LENGTH,
            device=device,
        )[:num_boundaries]

        boundaries[batch_idx, positions] = 1.0

    return boundaries


def make_shuffled_boundaries(
    boundary_probs: torch.Tensor,
):
    """
    Shuffle boundary probabilities independently
    within each sample.

    Preserves:
        - mean boundary rate
        - probability distribution
        - entropy

    Destroys:
        - positional structure
    """

    shuffled = boundary_probs.clone()

    for batch_idx in range(boundary_probs.shape[0]):
        permutation = torch.randperm(
            boundary_probs.shape[1],
            device=boundary_probs.device,
        )

        shuffled[batch_idx] = boundary_probs[
            batch_idx,
            permutation,
        ]

    return shuffled


# ============================================================
# Model
# ============================================================


def load_model(tokenizer):

    model = NeuralV2LanguageModel(
        input_vocab_size=tokenizer.vocab_size,
        output_vocab_size=tokenizer.vocab_size,
        source_length=SOURCE_LENGTH,
        num_tokens=NUM_TOKENS,
        tokenizer_embedding_dim=(TOKENIZER_EMBEDDING_DIM),
        boundary_hidden_dim=(BOUNDARY_HIDDEN_DIM),
        boundary_kernel_size=(BOUNDARY_KERNEL_SIZE),
        pooling_temperature=(POOLING_TEMPERATURE),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        gru_hidden_dim=GRU_HIDDEN_DIM,
        group_size=GROUP_SIZE,
    ).to(DEVICE)

    print()
    print("Loading checkpoint")
    print("-" * 60)

    print(f"Checkpoint: {CHECKPOINT_PATH}")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.eval()

    params = count_parameters(model)

    print(f"Parameters: {params['total']:,}")

    print(f"Tokenizer:  " f"{params['neural_tokenizer']:,}")

    print(f"Transformer + GRU: " f"{params['transformer_and_gru']:,}")

    return model


# ============================================================
# Latent representation diagnostics
# ============================================================


def latent_difference(
    reference: torch.Tensor,
    candidate: torch.Tensor,
):
    """
    Compare [B,T,D] latent representations.

    Returns:
        mean absolute difference
        mean L2 difference
        mean cosine similarity
    """

    difference = candidate - reference

    mean_absolute_difference = difference.abs().mean().item()

    l2_difference = difference.pow(2).sum(dim=-1).sqrt().mean().item()

    reference_flat = reference.reshape(
        -1,
        reference.shape[-1],
    )

    candidate_flat = candidate.reshape(
        -1,
        candidate.shape[-1],
    )

    cosine_similarity = (
        F.cosine_similarity(
            reference_flat,
            candidate_flat,
            dim=-1,
        )
        .mean()
        .item()
    )

    return {
        "latent_mean_abs_difference": (mean_absolute_difference),
        "latent_mean_l2_difference": (l2_difference),
        "latent_cosine_similarity": (cosine_similarity),
    }


# ============================================================
# Single configuration evaluation
# ============================================================


@torch.no_grad()
def evaluate_configuration(
    model,
    loader,
    bytes_per_symbol,
    configuration,
):
    model.eval()

    total_nll = 0.0
    total_symbols = 0

    total_rate_loss = 0.0
    total_binary_loss = 0.0

    boundary_values = []

    latent_differences = []
    latent_l2_differences = []
    latent_cosine_similarities = []

    batches_seen = 0

    start_time = time.perf_counter()

    for batch_idx, batch in enumerate(loader):

        if batch_idx >= MAX_VALIDATION_BATCHES:
            break

        x, y, target_mask = batch

        x = x.to(
            DEVICE,
            non_blocking=True,
        )

        y = y.to(
            DEVICE,
            non_blocking=True,
        )

        target_mask = target_mask.to(
            DEVICE,
            non_blocking=True,
        )

        batch_size = x.shape[0]

        # ----------------------------------------------------
        # EXACT SAME TEACHER FORCING AS TRAINING RUNNER
        # ----------------------------------------------------

        previous_symbols = torch.empty_like(y)

        previous_symbols[:, :, 0] = model.neural_tokenizer.pad_id

        previous_symbols[:, :, 1:] = y[:, :, :-1]

        # ----------------------------------------------------
        # Boundary configuration
        # ----------------------------------------------------

        if configuration == "learned":

            boundary_override = None

        elif configuration == "shuffled":

            # First obtain learned probabilities.
            _, learned_boundary_probs = model.neural_tokenizer(x)

            boundary_override = make_shuffled_boundaries(learned_boundary_probs)

        elif configuration == "uniform":
            boundary_override = make_uniform_boundaries(
                batch_size,
                DEVICE,
            )

        elif configuration == "regular":
            boundary_override = make_regular_boundaries(
                batch_size,
                DEVICE,
            )

        elif configuration == "random":
            boundary_override = make_random_boundaries(
                batch_size,
                DEVICE,
            )

        else:
            raise ValueError(f"Unknown configuration: " f"{configuration}")

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        logits, boundary_probs = model(
            x,
            previous_symbols,
            boundary_probs_override=(boundary_override),
        )

        # ----------------------------------------------------
        # Latent representation
        # ----------------------------------------------------

        if configuration == "learned":

            learned_latents = model.neural_tokenizer(x)[0]

        else:

            candidate_latents = model.neural_tokenizer(
                x,
                boundary_probs_override=(boundary_override),
            )[0]

            # We need the learned representation
            # for the same input as reference.
            learned_latents = model.neural_tokenizer(x)[0]

            latent_metrics = latent_difference(
                learned_latents,
                candidate_latents,
            )

            latent_differences.append(latent_metrics["latent_mean_abs_difference"])

            latent_l2_differences.append(latent_metrics["latent_mean_l2_difference"])

            latent_cosine_similarities.append(
                latent_metrics["latent_cosine_similarity"]
            )

        # ----------------------------------------------------
        # LM loss
        # ----------------------------------------------------

        vocab_size = logits.shape[-1]

        losses = F.cross_entropy(
            logits.reshape(
                -1,
                vocab_size,
            ),
            y.reshape(-1),
            reduction="none",
        )

        losses = losses.reshape_as(y)

        valid_losses = losses[target_mask]

        total_nll += valid_losses.sum().item()

        total_symbols += valid_losses.numel()

        # ----------------------------------------------------
        # Boundary losses
        # ----------------------------------------------------

        rate_loss = boundary_rate_loss(
            boundary_probs,
            target_symbols_per_token=(TARGET_SYMBOLS_PER_TOKEN),
        )

        binary_loss = boundary_binary_loss(boundary_probs)

        total_rate_loss += rate_loss.item()

        total_binary_loss += binary_loss.item()

        boundary_values.append(boundary_probs.detach().float().cpu())

        batches_seen += 1

    elapsed = time.perf_counter() - start_time

    if total_symbols == 0:
        raise RuntimeError("No valid symbols evaluated.")

    avg_nll = total_nll / total_symbols

    # --------------------------------------------------------
    # EXACT SAME BPB CONVERSION AS TRAINING RUNNER
    # --------------------------------------------------------

    bpb = avg_nll / np.log(2.0) / bytes_per_symbol

    boundary_tensor = torch.cat(
        boundary_values,
        dim=0,
    )

    boundary_metrics = calculate_boundary_metrics(boundary_tensor)

    result = {
        "configuration": configuration,
        "lm_loss": avg_nll,
        "bpb": bpb,
        "evaluation_time_seconds": elapsed,
        "batches_evaluated": batches_seen,
        "valid_symbols_evaluated": (total_symbols),
        "rate_loss": (total_rate_loss / batches_seen),
        "binary_loss": (total_binary_loss / batches_seen),
    }

    result.update(boundary_metrics)

    # --------------------------------------------------------
    # Latent intervention metrics
    # --------------------------------------------------------

    if configuration == "learned":

        result["latent_mean_abs_difference"] = 0.0

        result["latent_mean_l2_difference"] = 0.0

        result["latent_cosine_similarity"] = 1.0

    else:

        result["latent_mean_abs_difference"] = float(np.mean(latent_differences))

        result["latent_mean_l2_difference"] = float(np.mean(latent_l2_differences))

        result["latent_cosine_similarity"] = float(np.mean(latent_cosine_similarities))

    return result


# ============================================================
# Plotting
# ============================================================


def save_bar_plot(
    names,
    values,
    ylabel,
    title,
    path,
    horizontal_line=None,
):

    plt.figure(figsize=(9, 6))

    plt.bar(
        names,
        values,
    )

    if horizontal_line is not None:

        plt.axhline(
            horizontal_line,
            linestyle="--",
            label=f"Target = {horizontal_line}",
        )

        plt.legend()

    plt.ylabel(ylabel)
    plt.xlabel("Boundary configuration")

    plt.title(title)

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()


def save_plots(
    results,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    names = [r["configuration"] for r in results]

    # --------------------------------------------------------
    # BPB
    # --------------------------------------------------------

    save_bar_plot(
        names,
        [r["bpb"] for r in results],
        "Validation BPB",
        "Neural V2: BPB under boundary interventions",
        OUTPUT_DIR / "boundary_sensitivity_bpb.png",
    )

    # --------------------------------------------------------
    # Boundary rate
    # --------------------------------------------------------

    save_bar_plot(
        names,
        [r["boundary_mean"] for r in results],
        "Mean boundary probability",
        "Neural V2: boundary rate under interventions",
        OUTPUT_DIR / "boundary_sensitivity_rate.png",
        horizontal_line=0.25,
    )

    # --------------------------------------------------------
    # Boundary entropy
    # --------------------------------------------------------

    save_bar_plot(
        names,
        [r["boundary_entropy"] for r in results],
        "Boundary entropy (nats)",
        "Neural V2: boundary entropy under interventions",
        OUTPUT_DIR / "boundary_sensitivity_entropy.png",
    )

    # --------------------------------------------------------
    # Latent difference
    # --------------------------------------------------------

    save_bar_plot(
        names,
        [r["latent_mean_abs_difference"] for r in results],
        "Mean absolute latent difference",
        "Neural V2: latent representation sensitivity",
        OUTPUT_DIR / "boundary_sensitivity_latent_difference.png",
    )

    # --------------------------------------------------------
    # Latent cosine similarity
    # --------------------------------------------------------

    save_bar_plot(
        names,
        [r["latent_cosine_similarity"] for r in results],
        "Cosine similarity to learned representation",
        "Neural V2: latent representation cosine similarity",
        OUTPUT_DIR / "boundary_sensitivity_latent_cosine.png",
    )


# ============================================================
# Main
# ============================================================


def main():

    set_seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("NEURAL TOKENIZER V2 — " "BOUNDARY SENSITIVITY")
    print("=" * 80)

    print(f"Device: {DEVICE}")

    print(f"Seed:   {SEED}")

    if torch.cuda.is_available():

        print("GPU:    " f"{torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    train_texts, valid_texts = load_tinystories()

    tokenizer = ByteTokenizer()

    train_stats = get_statistics(
        train_texts,
        tokenizer,
    )

    valid_stats = get_statistics(
        valid_texts,
        tokenizer,
    )

    print()
    print("Corpus")
    print("-" * 60)

    print(f"Train bytes:       " f"{train_stats['bytes']:,}")

    print(f"Train symbols:     " f"{train_stats['symbols']:,}")

    print(f"Bytes/symbol:      " f"{train_stats['bytes_per_symbol']:.4f}")

    print(f"Valid bytes:       " f"{valid_stats['bytes']:,}")

    print(f"Valid symbols:     " f"{valid_stats['symbols']:,}")

    # --------------------------------------------------------
    # Validation dataset
    # --------------------------------------------------------

    valid_dataset = NeuralV2Dataset(
        valid_texts,
        tokenizer,
        source_length=SOURCE_LENGTH,
        group_size=GROUP_SIZE,
        stride=SOURCE_LENGTH,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    print()
    print("Dataset")
    print("-" * 60)

    print(f"Validation samples: " f"{len(valid_dataset):,}")

    print(f"Source length:      " f"{SOURCE_LENGTH}")

    print(f"Latent tokens:      " f"{NUM_TOKENS}")

    print(f"Group size:         " f"{GROUP_SIZE}")

    print(f"Validation batches: " f"{MAX_VALIDATION_BATCHES}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = load_model(tokenizer)

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    configurations = [
        "learned",
        "shuffled",
        "uniform",
        "regular",
        "random",
    ]

    results = []

    print()
    print("=" * 80)
    print("BOUNDARY INTERVENTION EVALUATION")
    print("=" * 80)

    for configuration in configurations:

        print()
        print("-" * 80)

        print(f"Evaluating: " f"{configuration.upper()}")

        print("-" * 80)

        metrics = evaluate_configuration(
            model=model,
            loader=valid_loader,
            bytes_per_symbol=(train_stats["bytes_per_symbol"]),
            configuration=configuration,
        )

        results.append(metrics)

        print(f"BPB:              " f"{metrics['bpb']:.4f}")

        print(f"LM loss:           " f"{metrics['lm_loss']:.4f}")

        print(f"Boundary mean:     " f"{metrics['boundary_mean']:.6f}")

        print(f"Boundary std:      " f"{metrics['boundary_std']:.6f}")

        print(f"Boundary min:      " f"{metrics['boundary_min']:.6f}")

        print(f"Boundary max:      " f"{metrics['boundary_max']:.6f}")

        print(f"Entropy:           " f"{metrics['boundary_entropy']:.6f}")

        print(f"Binary loss:       " f"{metrics['binary_loss']:.6f}")

        print(f"Symbols/token:     " f"{metrics['symbols_per_token']:.3f}")

        print(f"Low confident:     " f"{metrics['low_confident']:.4f}")

        print(f"High confident:    " f"{metrics['high_confident']:.4f}")

        print(f"Latent |Δ|:        " f"{metrics['latent_mean_abs_difference']:.6f}")

        print(f"Latent L2 Δ:       " f"{metrics['latent_mean_l2_difference']:.6f}")

        print(f"Latent cosine:     " f"{metrics['latent_cosine_similarity']:.6f}")

        print(f"Evaluation time:   " f"{metrics['evaluation_time_seconds']:.2f}s")

    # --------------------------------------------------------
    # Sanity check
    # --------------------------------------------------------

    learned_bpb = results[0]["bpb"]

    print()
    print("=" * 80)
    print("SENSITIVITY SANITY CHECK")
    print("=" * 80)

    print(f"Learned BPB: " f"{learned_bpb:.4f}")

    print(
        "\nThis should approximately match "
        "the training runner's validation BPB "
        "when evaluated over the same number "
        "of batches."
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    output = {
        "experiment": ("neural_v2_boundary_sensitivity"),
        "seed": SEED,
        "device": str(DEVICE),
        "checkpoint": str(CHECKPOINT_PATH),
        "source_length": SOURCE_LENGTH,
        "num_tokens": NUM_TOKENS,
        "group_size": GROUP_SIZE,
        "batch_size": BATCH_SIZE,
        "max_validation_batches": (MAX_VALIDATION_BATCHES),
        "train_bytes_per_symbol": (train_stats["bytes_per_symbol"]),
        "results": results,
    }

    results_path = OUTPUT_DIR / "boundary_sensitivity_results.json"

    with results_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    save_plots(results)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("BOUNDARY SENSITIVITY COMPLETE")
    print("=" * 80)

    print()

    print(
        f"{'Configuration':<15}"
        f"{'BPB':>10}"
        f"{'Mean p':>10}"
        f"{'Entropy':>12}"
        f"{'|Δz|':>12}"
        f"{'Cos(z)':>12}"
    )

    print("-" * 71)

    for result in results:

        print(
            f"{result['configuration']:<15}"
            f"{result['bpb']:>10.4f}"
            f"{result['boundary_mean']:>10.4f}"
            f"{result['boundary_entropy']:>12.4f}"
            f"{result['latent_mean_abs_difference']:>12.6f}"
            f"{result['latent_cosine_similarity']:>12.6f}"
        )

    print()
    print(f"JSON: {results_path}")

    print(f"Output directory: " f"{OUTPUT_DIR.resolve()}")

    print("=" * 80)


if __name__ == "__main__":
    main()
