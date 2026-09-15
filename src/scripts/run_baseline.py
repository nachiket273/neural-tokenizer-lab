from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from neural_tokenizer.dataset import LanguageModelDataset
from neural_tokenizer.model import TinyTransformer
from neural_tokenizer.tokenizer import BPETokenizer, ByteTokenizer
from neural_tokenizer.training import train

# ============================================================
# Configuration
# ============================================================

TRAIN_SIZE = 50_000
VALID_SIZE = 5_000

CONTEXT_LENGTH = 256
STRIDE = 256
BATCH_SIZE = 64

D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
D_FF = 512

LEARNING_RATE = 3e-4

# Baseline 1
BASELINE_1_STEPS = 5_000

# Baseline 2
BASELINE_2_BPE_STEPS = 5_000

VAL_EVERY = 500
VAL_BATCHES = 100

BPE_VOCAB_SIZE = 8192
BPE_TOKENIZER_PATH = Path("notebooks/artifacts/bpe/tokenizer.json")

SEED = 42

EXPERIMENT_ROOT = Path("experiments")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Reproducibility
# ============================================================


def set_seed(seed: int) -> None:
    """Set all relevant random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic behavior where practical.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Dataset loading
# ============================================================


def load_tinystories() -> tuple[list[str], list[str]]:
    """Load the fixed TinyStories train/validation subsets."""

    print("\nLoading TinyStories...")

    dataset = load_dataset("roneneldan/TinyStories")

    train_texts = [
        example["text"] for example in dataset["train"].select(range(TRAIN_SIZE))
    ]

    valid_texts = [
        example["text"] for example in dataset["validation"].select(range(VALID_SIZE))
    ]

    print(f"Training stories:   {len(train_texts):,}")
    print(f"Validation stories: {len(valid_texts):,}")

    return train_texts, valid_texts


# ============================================================
# Corpus statistics
# ============================================================


def get_corpus_statistics(
    texts: list[str],
    tokenizer,
) -> dict[str, float | int]:
    """
    Calculate original UTF-8 bytes and tokenizer tokens.

    Important:
    - bytes are measured from the original text
    - token count includes the tokenizer EOS token
    """

    total_bytes = 0
    total_tokens = 0

    for text in texts:
        total_bytes += len(text.encode("utf-8"))
        total_tokens += len(tokenizer.encode(text))

    bytes_per_token = total_bytes / total_tokens

    return {
        "bytes": total_bytes,
        "tokens": total_tokens,
        "bytes_per_token": bytes_per_token,
        "tokens_per_byte": total_tokens / total_bytes,
    }


def print_corpus_statistics(
    name: str,
    train_stats: dict,
    valid_stats: dict,
) -> None:
    """Print corpus statistics."""

    print(f"\n{name} corpus statistics")
    print("-" * 60)

    print(f"Train bytes:          " f"{train_stats['bytes']:,}")
    print(f"Train tokens:         " f"{train_stats['tokens']:,}")
    print(f"Train bytes/token:    " f"{train_stats['bytes_per_token']:.4f}")

    print(f"Valid bytes:          " f"{valid_stats['bytes']:,}")
    print(f"Valid tokens:         " f"{valid_stats['tokens']:,}")
    print(f"Valid bytes/token:    " f"{valid_stats['bytes_per_token']:.4f}")


# ============================================================
# Equal-byte calculation
# ============================================================


def calculate_equal_byte_steps(
    reference_steps: int,
    reference_bytes_per_token: float,
    target_bytes_per_token: float,
) -> int:
    """
    Calculate target steps required to process approximately
    the same number of original UTF-8 bytes.

    Each optimization step processes:

        batch_size * context_length

    token positions.

    Therefore:

        target_steps =
            reference_steps
            * reference_bytes_per_token
            / target_bytes_per_token
    """

    steps = reference_steps * reference_bytes_per_token / target_bytes_per_token

    return int(round(steps))


# ============================================================
# DataLoaders
# ============================================================


def create_dataloaders(
    train_texts: list[str],
    valid_texts: list[str],
    tokenizer,
    seed: int,
) -> tuple[DataLoader, DataLoader, LanguageModelDataset, LanguageModelDataset]:
    """
    Create datasets and deterministic DataLoaders.

    The LanguageModelDataset concatenates the tokenized stories
    into one stream and then takes non-overlapping windows.

    This avoids the previous problem where short stories caused
    the final partial chunk to be discarded separately for each
    story.
    """

    train_dataset = LanguageModelDataset(
        train_texts,
        tokenizer,
        context_length=CONTEXT_LENGTH,
        stride=STRIDE,
    )

    valid_dataset = LanguageModelDataset(
        valid_texts,
        tokenizer,
        context_length=CONTEXT_LENGTH,
        stride=STRIDE,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    return (
        train_loader,
        valid_loader,
        train_dataset,
        valid_dataset,
    )


# ============================================================
# Model creation
# ============================================================


def create_model(vocab_size: int) -> TinyTransformer:
    """Create a fresh TinyTransformer."""

    model = TinyTransformer(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_seq_len=CONTEXT_LENGTH,
    )

    model = model.to(DEVICE)

    return model


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    """Count total, embedding, and transformer parameters."""

    total = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    embedding = 0

    if hasattr(model, "token_embedding"):
        embedding = sum(
            parameter.numel()
            for parameter in model.token_embedding.parameters()
            if parameter.requires_grad
        )

    transformer = total - embedding

    return {
        "total": total,
        "embedding": embedding,
        "transformer": transformer,
    }


# ============================================================
# BPB evaluation
# ============================================================


@torch.no_grad()
def evaluate_bits_per_byte(
    model: torch.nn.Module,
    texts: list[str],
    tokenizer,
    device: torch.device,
) -> float:
    """
    Calculate normalized negative log-likelihood in bits/byte.

    The denominator is the exact number of original UTF-8 bytes.

    NOTE:
    This baseline uses the current tokenizer convention where
    EOS is appended to each encoded example. EOS contributes to
    the model NLL but has no corresponding original byte. Therefore
    this metric should currently be interpreted as a normalized
    BPB diagnostic rather than a fully formal byte-level likelihood.

    The same convention is applied to both BPE and byte models.
    """

    model.eval()

    total_nll = 0.0
    total_bytes = 0

    for text in texts:
        ids = tokenizer.encode(text)

        if len(ids) < 2:
            continue

        tokens = torch.tensor(
            ids,
            dtype=torch.long,
        )

        # Evaluate the sequence in chunks.
        #
        # Unlike the training dataset, this evaluator retains
        # the final partial sequence.
        for start in range(
            0,
            len(tokens) - 1,
            CONTEXT_LENGTH,
        ):
            chunk = tokens[start : start + CONTEXT_LENGTH + 1]

            if len(chunk) < 2:
                continue

            x = chunk[:-1].unsqueeze(0).to(device)
            y = chunk[1:].unsqueeze(0).to(device)

            logits = model(x)

            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                reduction="sum",
            )

            total_nll += loss.item()

        total_bytes += len(text.encode("utf-8"))

    if total_bytes == 0:
        raise ValueError("No validation bytes found.")

    # NLL is in nats.
    # Convert to bits using log2(e).
    bpb = total_nll / total_bytes / math.log(2)

    model.train()

    return bpb


# ============================================================
# History post-processing
# ============================================================


def add_derived_metrics(
    history: list[dict],
    bytes_per_token: float,
) -> list[dict]:
    """
    Add cumulative original-byte estimates and BPB.

    Training NLL is in nats/token.

    Since each training step processes a fixed number of token
    positions, we can estimate original bytes processed using
    the corpus-level average bytes/token.
    """

    for record in history:
        total_tokens = record["total_positions"]

        record["estimated_bytes_processed"] = total_tokens * bytes_per_token

        record["estimated_bits_processed"] = record["estimated_bytes_processed"] * 8

        # Normalized training BPB diagnostic.
        record["train_bpb"] = record["train_loss"] / math.log(2) / bytes_per_token

        if record.get("val_loss") is not None:
            record["val_bpb"] = record["val_loss"] / math.log(2) / bytes_per_token
        else:
            record["val_bpb"] = None

    return history


# ============================================================
# Plotting helpers
# ============================================================


def extract_validation_points(
    history: list[dict],
) -> tuple[list[int], list[float], list[float]]:
    """Extract validation steps, BPB and wall-clock times."""

    steps = []
    bpb = []
    wall_clock = []

    cumulative_time = 0.0

    for record in history:
        cumulative_time += record["time"]

        if record.get("val_bpb") is not None:
            steps.append(record["step"])
            bpb.append(record["val_bpb"])
            wall_clock.append(cumulative_time)

    return steps, bpb, wall_clock


def plot_bpb_vs_steps(
    history: list[dict],
    output_path: Path,
    title: str,
) -> None:
    """Plot validation BPB against optimization steps."""

    steps, bpb, _ = extract_validation_points(history)

    plt.figure(figsize=(8, 5))

    plt.plot(
        steps,
        bpb,
        marker="o",
        markersize=3,
    )

    plt.xlabel("Optimization steps")
    plt.ylabel("Validation BPB")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_bpb_vs_bytes(
    history: list[dict],
    output_path: Path,
    title: str,
) -> None:
    """Plot validation BPB against estimated original bytes processed."""

    _, bpb, _ = extract_validation_points(history)

    bytes_processed = []

    for record in history:
        if record.get("val_bpb") is not None:
            bytes_processed.append(record["estimated_bytes_processed"])

    plt.figure(figsize=(8, 5))

    plt.plot(
        bytes_processed,
        bpb,
        marker="o",
        markersize=3,
    )

    plt.xlabel("Estimated original UTF-8 bytes processed")
    plt.ylabel("Validation BPB")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_bpb_vs_time(
    history: list[dict],
    output_path: Path,
    title: str,
) -> None:
    """Plot validation BPB against wall-clock training time."""

    _, bpb, wall_clock = extract_validation_points(history)

    plt.figure(figsize=(8, 5))

    plt.plot(
        wall_clock,
        bpb,
        marker="o",
        markersize=3,
    )

    plt.xlabel("Training time (seconds)")
    plt.ylabel("Validation BPB")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close()


def save_experiment_plots(
    history: list[dict],
    output_dir: Path,
    experiment_name: str,
) -> None:
    """Generate all three plots for an experiment."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_bpb_vs_steps(
        history,
        output_dir / "bpb_vs_steps.png",
        f"{experiment_name}: BPB vs optimization steps",
    )

    plot_bpb_vs_bytes(
        history,
        output_dir / "bpb_vs_bytes.png",
        f"{experiment_name}: BPB vs original bytes processed",
    )

    plot_bpb_vs_time(
        history,
        output_dir / "bpb_vs_time.png",
        f"{experiment_name}: BPB vs wall-clock time",
    )


# ============================================================
# JSON serialization
# ============================================================


def save_json(
    data,
    output_path: Path,
) -> None:
    """Save JSON with readable formatting."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
        )


# ============================================================
# Single experiment
# ============================================================


def run_experiment(
    *,
    experiment_name: str,
    tokenizer_name: str,
    tokenizer,
    train_texts: list[str],
    valid_texts: list[str],
    steps: int,
    experiment_dir: Path,
) -> dict:
    """Run one complete experiment."""

    print("\n")
    print("=" * 80)
    print(f"EXPERIMENT: {experiment_name}")
    print("=" * 80)

    print(f"Tokenizer:       {tokenizer_name}")
    print(f"Vocabulary size: {tokenizer.vocab_size:,}")
    print(f"Training steps:  {steps:,}")
    print(f"Device:          {DEVICE}")

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    train_stats = get_corpus_statistics(
        train_texts,
        tokenizer,
    )

    valid_stats = get_corpus_statistics(
        valid_texts,
        tokenizer,
    )

    print_corpus_statistics(
        tokenizer_name,
        train_stats,
        valid_stats,
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    (
        train_loader,
        valid_loader,
        train_dataset,
        valid_dataset,
    ) = create_dataloaders(
        train_texts,
        valid_texts,
        tokenizer,
        seed=SEED,
    )

    print("\nDataset")
    print("-" * 60)
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Valid samples: {len(valid_dataset):,}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    set_seed(SEED)

    model = create_model(
        tokenizer.vocab_size,
    )

    params = count_parameters(model)

    print("\nModel parameters")
    print("-" * 60)
    print(f"Total:       {params['total']:,}")
    print(f"Embedding:   {params['embedding']:,}")
    print(f"Transformer: {params['transformer']:,}")

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\nStarting training...\n")

    start_time = time.perf_counter()

    history = train(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        device=DEVICE,
        steps=steps,
        val_loader=valid_loader,
        val_every=VAL_EVERY,
        val_batches=VAL_BATCHES,
    )

    training_wall_clock = time.perf_counter() - start_time

    # --------------------------------------------------------
    # Derived metrics
    # --------------------------------------------------------

    history = add_derived_metrics(
        history,
        bytes_per_token=train_stats["bytes_per_token"],
    )

    # Add actual cumulative wall-clock time.
    cumulative_time = 0.0

    for record in history:
        cumulative_time += record["time"]
        record["wall_clock_seconds"] = cumulative_time

    # --------------------------------------------------------
    # Exact validation BPB diagnostic
    # --------------------------------------------------------

    print("\nCalculating validation BPB...")

    validation_bpb = evaluate_bits_per_byte(
        model=model,
        texts=valid_texts,
        tokenizer=tokenizer,
        device=DEVICE,
    )

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    final_record = history[-1]

    result = {
        "experiment_name": experiment_name,
        "tokenizer": tokenizer_name,
        "vocab_size": tokenizer.vocab_size,
        "steps": steps,
        "context_length": CONTEXT_LENGTH,
        "stride": STRIDE,
        "batch_size": BATCH_SIZE,
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS,
        "d_ff": D_FF,
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
        "device": str(DEVICE),
        "parameters": params,
        "train_statistics": train_stats,
        "valid_statistics": valid_stats,
        "train_dataset_samples": len(train_dataset),
        "valid_dataset_samples": len(valid_dataset),
        "final_train_loss": final_record["train_loss"],
        "final_validation_loss": final_record.get("val_loss"),
        "final_validation_bpb_from_history": final_record.get("val_bpb"),
        "validation_bpb": validation_bpb,
        "training_wall_clock_seconds": training_wall_clock,
        "estimated_original_bytes_processed": final_record["estimated_bytes_processed"],
        "estimated_tokens_processed": final_record["total_positions"],
    }

    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_json(
        result,
        experiment_dir / "results.json",
    )

    save_json(
        history,
        experiment_dir / "history.json",
    )

    config = {
        "experiment_name": experiment_name,
        "tokenizer": tokenizer_name,
        "steps": steps,
        "train_size": TRAIN_SIZE,
        "valid_size": VALID_SIZE,
        "context_length": CONTEXT_LENGTH,
        "stride": STRIDE,
        "batch_size": BATCH_SIZE,
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS,
        "d_ff": D_FF,
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
        "bpe_vocab_size": BPE_VOCAB_SIZE,
        "device": str(DEVICE),
    }

    save_json(
        config,
        experiment_dir / "config.json",
    )

    save_experiment_plots(
        history,
        experiment_dir / "plots",
        experiment_name,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\nExperiment complete")
    print("-" * 60)

    print(f"Final train loss:       " f"{result['final_train_loss']:.4f}")

    print(f"Final validation loss:  " f"{result['final_validation_loss']:.4f}")

    print(f"Validation BPB:         " f"{result['validation_bpb']:.4f}")

    print(f"Training time:          " f"{training_wall_clock / 60:.2f} min")

    print(
        f"Estimated bytes:        "
        f"{result['estimated_original_bytes_processed']:,.0f}"
    )

    print(f"Results saved to:       " f"{experiment_dir}")

    return {
        "result": result,
        "history": history,
    }


# ============================================================
# Comparison plots
# ============================================================


def plot_comparison(
    histories: dict[str, list[dict]],
    output_dir: Path,
) -> None:
    """
    Generate combined BPE-vs-byte plots.

    These plots are useful for directly comparing the two
    tokenization strategies.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # BPB vs steps
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    for label, history in histories.items():
        steps, bpb, _ = extract_validation_points(history)

        plt.plot(
            steps,
            bpb,
            marker="o",
            markersize=3,
            label=label,
        )

    plt.xlabel("Optimization steps")
    plt.ylabel("Validation BPB")
    plt.title("BPE vs byte: BPB vs optimization steps")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "comparison_bpb_vs_steps.png",
        dpi=150,
    )

    plt.close()

    # --------------------------------------------------------
    # BPB vs original bytes
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    for label, history in histories.items():

        _, bpb, _ = extract_validation_points(history)

        bytes_processed = [
            record["estimated_bytes_processed"]
            for record in history
            if record.get("val_bpb") is not None
        ]

        plt.plot(
            bytes_processed,
            bpb,
            marker="o",
            markersize=3,
            label=label,
        )

    plt.xlabel("Estimated original UTF-8 bytes processed")
    plt.ylabel("Validation BPB")
    plt.title("BPE vs byte: BPB vs original bytes")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "comparison_bpb_vs_bytes.png",
        dpi=150,
    )

    plt.close()

    # --------------------------------------------------------
    # BPB vs wall-clock time
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    for label, history in histories.items():

        _, bpb, wall_clock = extract_validation_points(history)

        plt.plot(
            wall_clock,
            bpb,
            marker="o",
            markersize=3,
            label=label,
        )

    plt.xlabel("Training time (seconds)")
    plt.ylabel("Validation BPB")
    plt.title("BPE vs byte: BPB vs wall-clock time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "comparison_bpb_vs_time.png",
        dpi=150,
    )

    plt.close()


# ============================================================
# Main
# ============================================================


def main() -> None:
    """Run Baseline 1 and Baseline 2."""

    print("=" * 80)
    print("NEURAL TOKENIZER — BASELINE EXPERIMENTS")
    print("=" * 80)

    print(f"Device: {DEVICE}")
    print(f"Seed:   {SEED}")

    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    train_texts, valid_texts = load_tinystories()

    # --------------------------------------------------------
    # Load tokenizers
    # --------------------------------------------------------

    if not BPE_TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"BPE tokenizer not found at " f"{BPE_TOKENIZER_PATH}")

    bpe_tokenizer = BPETokenizer(BPE_TOKENIZER_PATH)

    byte_tokenizer = ByteTokenizer()

    # --------------------------------------------------------
    # Corpus statistics
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("TOKENIZATION STATISTICS")
    print("=" * 80)

    bpe_train_stats = get_corpus_statistics(
        train_texts,
        bpe_tokenizer,
    )

    byte_train_stats = get_corpus_statistics(
        train_texts,
        byte_tokenizer,
    )

    bpe_valid_stats = get_corpus_statistics(
        valid_texts,
        bpe_tokenizer,
    )

    byte_valid_stats = get_corpus_statistics(
        valid_texts,
        byte_tokenizer,
    )

    print("\nBPE")
    print("-" * 60)
    print(f"Train tokens:      " f"{bpe_train_stats['tokens']:,}")
    print(f"Train bytes:       " f"{bpe_train_stats['bytes']:,}")
    print(f"Bytes/token:       " f"{bpe_train_stats['bytes_per_token']:.4f}")

    print("\nBYTE")
    print("-" * 60)
    print(f"Train tokens:      " f"{byte_train_stats['tokens']:,}")
    print(f"Train bytes:       " f"{byte_train_stats['bytes']:,}")
    print(f"Bytes/token:       " f"{byte_train_stats['bytes_per_token']:.4f}")

    compression_ratio = bpe_train_stats["tokens"] / byte_train_stats["tokens"]

    byte_per_bpe_token = (
        bpe_train_stats["bytes_per_token"] / byte_train_stats["bytes_per_token"]
    )

    print("\nTokenization ratio")
    print("-" * 60)

    print(f"Byte/BPE token ratio: " f"{byte_per_bpe_token:.4f}")

    print(f"BPE/byte token ratio: " f"{compression_ratio:.4f}")

    # --------------------------------------------------------
    # Equal-byte calculation
    # --------------------------------------------------------

    baseline_2_byte_steps = calculate_equal_byte_steps(
        reference_steps=BASELINE_2_BPE_STEPS,
        reference_bytes_per_token=(bpe_train_stats["bytes_per_token"]),
        target_bytes_per_token=(byte_train_stats["bytes_per_token"]),
    )

    print("\n")
    print("=" * 80)
    print("BASELINE 2 — EQUAL ORIGINAL BYTES")
    print("=" * 80)

    print(f"BPE steps:  {BASELINE_2_BPE_STEPS:,}")

    print(f"Byte steps: {baseline_2_byte_steps:,}")

    print("\nApproximate step ratio:")

    print(f"Byte/BPE = " f"{baseline_2_byte_steps / BASELINE_2_BPE_STEPS:.2f}x")

    # --------------------------------------------------------
    # Experiment directories
    # --------------------------------------------------------

    baseline_1_dir = EXPERIMENT_ROOT / "baseline_001_equal_steps"

    baseline_2_dir = EXPERIMENT_ROOT / "baseline_002_equal_bytes"

    # --------------------------------------------------------
    # Baseline 1
    # --------------------------------------------------------

    baseline_1_bpe = run_experiment(
        experiment_name="Baseline 1 — Equal Optimization Steps — BPE",
        tokenizer_name="BPE",
        tokenizer=bpe_tokenizer,
        train_texts=train_texts,
        valid_texts=valid_texts,
        steps=BASELINE_1_STEPS,
        experiment_dir=(baseline_1_dir / "bpe"),
    )

    baseline_1_byte = run_experiment(
        experiment_name="Baseline 1 — Equal Optimization Steps — BYTE",
        tokenizer_name="BYTE",
        tokenizer=byte_tokenizer,
        train_texts=train_texts,
        valid_texts=valid_texts,
        steps=BASELINE_1_STEPS,
        experiment_dir=(baseline_1_dir / "byte"),
    )

    # --------------------------------------------------------
    # Baseline 2
    # --------------------------------------------------------

    baseline_2_bpe = run_experiment(
        experiment_name="Baseline 2 — Equal Original Bytes — BPE",
        tokenizer_name="BPE",
        tokenizer=bpe_tokenizer,
        train_texts=train_texts,
        valid_texts=valid_texts,
        steps=BASELINE_2_BPE_STEPS,
        experiment_dir=(baseline_2_dir / "bpe"),
    )

    baseline_2_byte = run_experiment(
        experiment_name="Baseline 2 — Equal Original Bytes — BYTE",
        tokenizer_name="BYTE",
        tokenizer=byte_tokenizer,
        train_texts=train_texts,
        valid_texts=valid_texts,
        steps=baseline_2_byte_steps,
        experiment_dir=(baseline_2_dir / "byte"),
    )

    # --------------------------------------------------------
    # Combined plots
    # --------------------------------------------------------

    plot_comparison(
        {
            "BPE": baseline_1_bpe["history"],
            "BYTE": baseline_1_byte["history"],
        },
        baseline_1_dir / "comparison",
    )

    plot_comparison(
        {
            "BPE": baseline_2_bpe["history"],
            "BYTE": baseline_2_byte["history"],
        },
        baseline_2_dir / "comparison",
    )

    # --------------------------------------------------------
    # Combined summary
    # --------------------------------------------------------

    summary = {
        "baseline_1_equal_steps": {
            "bpe": baseline_1_bpe["result"],
            "byte": baseline_1_byte["result"],
        },
        "baseline_2_equal_bytes": {
            "bpe": baseline_2_bpe["result"],
            "byte": baseline_2_byte["result"],
        },
        "tokenization_statistics": {
            "bpe_train": bpe_train_stats,
            "byte_train": byte_train_stats,
            "bpe_valid": bpe_valid_stats,
            "byte_valid": byte_valid_stats,
            "byte_to_bpe_bytes_per_token_ratio": (byte_per_bpe_token),
            "baseline_2_byte_steps": (baseline_2_byte_steps),
        },
    }

    save_json(
        summary,
        EXPERIMENT_ROOT / "baseline_summary.json",
    )

    # --------------------------------------------------------
    # Final console summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("FINAL BASELINE SUMMARY")
    print("=" * 80)

    print("\nBaseline 1 — Equal Optimization Steps")
    print("-" * 80)

    print(f"BPE  validation BPB: " f"{baseline_1_bpe['result']['validation_bpb']:.4f}")

    print(f"BYTE validation BPB: " f"{baseline_1_byte['result']['validation_bpb']:.4f}")

    print(
        f"BPE  training time: "
        f"{baseline_1_bpe['result']['training_wall_clock_seconds'] / 60:.2f} min"
    )

    print(
        f"BYTE training time: "
        f"{baseline_1_byte['result']['training_wall_clock_seconds'] / 60:.2f} min"
    )

    print("\nBaseline 2 — Equal Original Bytes")
    print("-" * 80)

    print(f"BPE  steps: " f"{BASELINE_2_BPE_STEPS:,}")

    print(f"BYTE steps: " f"{baseline_2_byte_steps:,}")

    print(f"BPE  validation BPB: " f"{baseline_2_bpe['result']['validation_bpb']:.4f}")

    print(f"BYTE validation BPB: " f"{baseline_2_byte['result']['validation_bpb']:.4f}")

    print("\n")
    print("=" * 80)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 80)

    print(f"\nResults directory: " f"{EXPERIMENT_ROOT.resolve()}")


if __name__ == "__main__":
    main()
