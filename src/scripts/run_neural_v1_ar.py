from __future__ import annotations

import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from neural_tokenizer.dataset import NeuralV1Dataset
from neural_tokenizer.neural_v1_ar_eval import (
    evaluate_neural_v1_ar_bpb,
)
from neural_tokenizer.neural_v1_ar_model import (
    NeuralV1ARLanguageModel,
)
from neural_tokenizer.neural_v1_ar_training import (
    train_neural_v1_ar,
)
from neural_tokenizer.tokenizer import ByteTokenizer

# ============================================================
# Configuration
# ============================================================

TRAIN_SIZE = 50_000
VALID_SIZE = 5_000

CONTEXT_LENGTH = 256
GROUP_SIZE = 4
STRIDE = 256

BATCH_SIZE = 64

TOKENIZER_EMBEDDING_DIM = 64
TOKENIZER_HIDDEN_DIM = 256

D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
D_FF = 512

LEARNING_RATE = 3e-4

# Approximately equal-byte budget to BPE 5,000-step baseline.
STEPS = 5000

VAL_EVERY = 500
VAL_BATCHES = 100

SEED = 42

OUTPUT_DIR = Path("experiments/neural_v1_ar")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AR_DECODER_LAYERS = 1
AR_DECODER_D_FF = 256

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

    print(f"Training stories:   " f"{len(train_texts):,}")

    print(f"Validation stories: " f"{len(valid_texts):,}")

    return train_texts, valid_texts


# ============================================================
# Statistics
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

    transformer = total - tokenizer

    return {
        "total": total,
        "neural_tokenizer": tokenizer,
        "transformer": transformer,
    }


# ============================================================
# Save JSON
# ============================================================


def save_json(
    data,
    path,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
        )


# ============================================================
# Plots
# ============================================================


def save_plots(
    history,
    output_dir,
    bytes_per_symbol,
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    steps = []
    bpb = []
    bytes_processed = []
    wall_clock = []

    total_tokens = 0

    for record in history:

        total_tokens = record["total_positions"]

        if record["val_loss"] is None:
            continue

        steps.append(record["step"])

        # Validation loss is nats/symbol.
        #
        # Each neural token represents ~4 symbols,
        # so:
        #
        # nats/symbol
        # ----------------
        # bytes/symbol
        #
        # gives nats/byte.
        #
        # Convert nats → bits.

        val_bpb = record["val_loss"] / bytes_per_symbol / np.log(2)

        bpb.append(val_bpb)

        bytes_processed.append(total_tokens * bytes_per_symbol)

        wall_clock.append(record["wall_clock_seconds"])

    # --------------------------------------------------------
    # BPB vs steps
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        steps,
        bpb,
        marker="o",
        markersize=3,
    )

    plt.xlabel("Optimization steps")

    plt.ylabel("Validation BPB")

    plt.title("Neural V1: BPB vs optimization steps")

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / "bpb_vs_steps.png",
        dpi=150,
    )

    plt.close()

    # --------------------------------------------------------
    # BPB vs bytes
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        bytes_processed,
        bpb,
        marker="o",
        markersize=3,
    )

    plt.xlabel("Estimated original UTF-8 bytes processed")

    plt.ylabel("Validation BPB")

    plt.title("Neural V1: BPB vs original bytes")

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / "bpb_vs_bytes.png",
        dpi=150,
    )

    plt.close()

    # --------------------------------------------------------
    # BPB vs time
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        wall_clock,
        bpb,
        marker="o",
        markersize=3,
    )

    plt.xlabel("Training compute time (seconds)")

    plt.ylabel("Validation BPB")

    plt.title("Neural V1: BPB vs training compute time")

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / "bpb_vs_time.png",
        dpi=150,
    )

    plt.close()


# ============================================================
# Main
# ============================================================


def main():

    print("=" * 80)
    print("NEURAL TOKENIZER V1.1 AUTOREGRESSIVE")
    print("=" * 80)

    print(f"Device: {DEVICE}")

    print(f"Seed:   {SEED}")

    if torch.cuda.is_available():

        print("GPU:    " f"{torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed(SEED)

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    train_texts, valid_texts = load_tinystories()

    # --------------------------------------------------------
    # Byte tokenizer
    # --------------------------------------------------------

    tokenizer = ByteTokenizer()

    train_stats = get_statistics(
        train_texts,
        tokenizer,
    )

    valid_stats = get_statistics(
        valid_texts,
        tokenizer,
    )

    print("\nCorpus")
    print("-" * 60)

    print(f"Train bytes:       " f"{train_stats['bytes']:,}")

    print(f"Train symbols:     " f"{train_stats['symbols']:,}")

    print(f"Bytes/symbol:      " f"{train_stats['bytes_per_symbol']:.4f}")

    print(f"Valid bytes:       " f"{valid_stats['bytes']:,}")

    print(f"Valid symbols:     " f"{valid_stats['symbols']:,}")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = NeuralV1Dataset(
        train_texts,
        tokenizer,
        context_length=CONTEXT_LENGTH,
        group_size=GROUP_SIZE,
        stride=STRIDE,
    )

    valid_dataset = NeuralV1Dataset(
        valid_texts,
        tokenizer,
        context_length=CONTEXT_LENGTH,
        group_size=GROUP_SIZE,
        stride=STRIDE,
    )

    print("\nDataset")
    print("-" * 60)

    print(f"Train groups:      " f"{train_dataset.num_groups:,}")

    print(f"Train samples:     " f"{len(train_dataset):,}")

    print(f"Valid groups:      " f"{valid_dataset.num_groups:,}")

    print(f"Valid samples:     " f"{len(valid_dataset):,}")

    print(f"Group size:        " f"{GROUP_SIZE} symbols")

    print(f"Effective context: " f"{CONTEXT_LENGTH * GROUP_SIZE} symbols")

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    generator = torch.Generator()

    generator.manual_seed(SEED)

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

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    set_seed(SEED)

    model = NeuralV1ARLanguageModel(
        input_vocab_size=tokenizer.vocab_size,
        output_vocab_size=tokenizer.vocab_size,
        group_size=GROUP_SIZE,
        tokenizer_embedding_dim=TOKENIZER_EMBEDDING_DIM,
        tokenizer_hidden_dim=TOKENIZER_HIDDEN_DIM,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_seq_len=CONTEXT_LENGTH,
        ar_decoder_layers=AR_DECODER_LAYERS,
        ar_decoder_d_ff=AR_DECODER_D_FF,
    )

    model = model.to(DEVICE)

    params = count_parameters(model)

    print("\nModel parameters")
    print("-" * 60)

    print(f"Total:             " f"{params['total']:,}")

    print(f"Neural tokenizer:  " f"{params['neural_tokenizer']:,}")

    print(f"Transformer:       " f"{params['transformer']:,}")

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

    history = train_neural_v1_ar(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        device=DEVICE,
        steps=STEPS,
        eval_every=VAL_EVERY,
        max_eval_batches=VAL_BATCHES,
    )

    total_training_time = time.perf_counter() - start_time

    # --------------------------------------------------------
    # Exact BPB
    # --------------------------------------------------------

    print("\nCalculating validation BPB...")

    validation_bpb = evaluate_neural_v1_ar_bpb(
        model=model,
        texts=valid_texts,
        tokenizer=tokenizer,
        device=DEVICE,
        group_size=GROUP_SIZE,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = {
        "experiment": "neural_v1_ar",
        "train_size": TRAIN_SIZE,
        "valid_size": VALID_SIZE,
        "context_length": CONTEXT_LENGTH,
        "group_size": GROUP_SIZE,
        "stride": STRIDE,
        "batch_size": BATCH_SIZE,
        "tokenizer_embedding_dim": (TOKENIZER_EMBEDDING_DIM),
        "tokenizer_hidden_dim": (TOKENIZER_HIDDEN_DIM),
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS,
        "d_ff": D_FF,
        "learning_rate": LEARNING_RATE,
        "steps": STEPS,
        "seed": SEED,
        "device": str(DEVICE),
    }

    results = {
        "experiment": "neural_v1",
        "validation_bpb": validation_bpb,
        "training_wall_clock_seconds": (total_training_time),
        "train_statistics": train_stats,
        "valid_statistics": valid_stats,
        "train_groups": (train_dataset.num_groups),
        "valid_groups": (valid_dataset.num_groups),
        "train_samples": len(train_dataset),
        "valid_samples": len(valid_dataset),
        "parameters": params,
    }

    save_json(
        config,
        OUTPUT_DIR / "config.json",
    )

    save_json(
        results,
        OUTPUT_DIR / "results.json",
    )

    save_json(
        history,
        OUTPUT_DIR / "history.json",
    )

    save_plots(
        history,
        OUTPUT_DIR / "plots",
        bytes_per_symbol=(train_stats["bytes_per_symbol"]),
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("NEURAL V1 COMPLETE")
    print("=" * 80)

    print(f"Validation BPB: " f"{validation_bpb:.4f}")

    print(f"Training time:  " f"{total_training_time / 60:.2f} min")

    print(f"Parameters:     " f"{params['total']:,}")

    print(f"Tokenizer:      " f"{params['neural_tokenizer']:,}")

    print(f"Transformer:    " f"{params['transformer']:,}")

    print("\nResults saved to:")

    print(OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
