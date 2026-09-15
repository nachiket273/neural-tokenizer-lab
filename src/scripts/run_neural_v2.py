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

from neural_tokenizer.neural_v2_boundary import boundary_binary_loss, boundary_rate_loss
from neural_tokenizer.neural_v2_dataset import (
    NeuralV2Dataset,
)
from neural_tokenizer.neural_v2_model import (
    NeuralV2LanguageModel,
)
from neural_tokenizer.tokenizer import ByteTokenizer

# ============================================================
# Configuration
# ============================================================

TRAIN_SIZE = 50_000
VALID_SIZE = 5_000

SOURCE_LENGTH = 1024
GROUP_SIZE = 4
NUM_TOKENS = SOURCE_LENGTH // GROUP_SIZE
STRIDE = SOURCE_LENGTH

BATCH_SIZE = 64

TOKENIZER_EMBEDDING_DIM = 64
BOUNDARY_HIDDEN_DIM = 128
BOUNDARY_KERNEL_SIZE = 5
POOLING_TEMPERATURE = 0.75

D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
D_FF = 512
GRU_HIDDEN_DIM = 128

LEARNING_RATE = 3e-4

# Start with a short controlled experiment.
STEPS = 5000

VAL_EVERY = 500
VAL_BATCHES = 100

LAMBDA_RATE = 1.0
LAMBDA_BINARY = 0.01

TARGET_SYMBOLS_PER_TOKEN = 4.0

SEED = 42

OUTPUT_DIR = Path("experiments/neural_v2")

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
        "bytes_per_symbol": total_bytes / total_symbols,
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
        "transformer_and_gru": transformer,
    }


# ============================================================
# JSON
# ============================================================


def save_json(data, path):
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
# Metrics
# ============================================================


def calculate_boundary_entropy(
    boundary_probs: torch.Tensor,
) -> float:
    """
    Mean Bernoulli entropy of boundary probabilities.

        H(p) = -p log p - (1-p) log(1-p)

    Returned in nats.
    """

    eps = 1e-7

    p = boundary_probs.clamp(
        min=eps,
        max=1.0 - eps,
    )

    entropy = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))

    return entropy.mean().item()


# ============================================================
# Validation
# ============================================================


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    eval_batches,
):
    model.eval()

    total_nll = 0.0
    total_symbols = 0

    total_rate_loss = 0.0
    total_boundary_rate = 0.0
    total_boundary_entropy = 0.0
    total_binary_loss = 0.0

    total_low_confident = 0.0
    total_high_confident = 0.0

    batches_seen = 0

    for batch_idx, batch in enumerate(loader):

        if batch_idx >= eval_batches:
            break

        x, y, target_mask = batch

        x = x.to(
            device,
            non_blocking=True,
        )

        y = y.to(
            device,
            non_blocking=True,
        )

        target_mask = target_mask.to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Teacher forcing
        #
        # For each target group:
        #
        #   y = [y0, y1, y2, y3]
        #
        #   previous = [PAD, y0, y1, y2]
        #
        # ----------------------------------------------------

        previous_symbols = torch.empty_like(y)

        previous_symbols[:, :, 0] = model.neural_tokenizer.pad_id

        previous_symbols[:, :, 1:] = y[:, :, :-1]

        logits, boundary_probs = model(
            x,
            previous_symbols,
        )

        vocab_size = logits.shape[-1]

        losses = torch.nn.functional.cross_entropy(
            logits.reshape(-1, vocab_size),
            y.reshape(-1),
            reduction="none",
        )

        losses = losses.reshape_as(y)

        valid_losses = losses[target_mask]

        total_nll += valid_losses.sum().item()
        total_symbols += valid_losses.numel()

        rate_loss = boundary_rate_loss(
            boundary_probs,
            target_symbols_per_token=TARGET_SYMBOLS_PER_TOKEN,
        )

        binary_loss = boundary_binary_loss(boundary_probs)

        total_rate_loss += rate_loss.item()
        total_binary_loss += binary_loss.item()

        total_boundary_rate += boundary_probs.mean().item()

        total_boundary_entropy += calculate_boundary_entropy(boundary_probs)
        total_low_confident += (boundary_probs < 0.1).float().mean().item()
        total_high_confident += (boundary_probs > 0.9).float().mean().item()

        batches_seen += 1

    model.train()

    if total_symbols == 0:
        raise RuntimeError("Validation produced zero valid target symbols.")

    avg_nll = total_nll / total_symbols

    return {
        "loss": avg_nll,
        "bpb": avg_nll / np.log(2.0),
        "rate_loss": (total_rate_loss / batches_seen),
        "boundary_rate": (total_boundary_rate / batches_seen),
        "boundary_entropy": (total_boundary_entropy / batches_seen),
        "binary_loss": (total_binary_loss / batches_seen),
        "boundary_mean": boundary_probs.mean().item(),
        "boundary_std": boundary_probs.std().item(),
        "boundary_min": boundary_probs.min().item(),
        "boundary_max": boundary_probs.max().item(),
        "low_confident": (total_low_confident / batches_seen),
        "high_confident": (total_high_confident / batches_seen),
    }


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

    validation_records = [
        record for record in history if record["val_loss"] is not None
    ]

    if not validation_records:
        return

    steps = [record["step"] for record in validation_records]

    bpb = [record["val_bpb"] for record in validation_records]

    bytes_processed = [
        record["estimated_original_bytes"] for record in validation_records
    ]

    wall_clock = [record["wall_clock_seconds"] for record in validation_records]

    boundary_rate = [record["val_boundary_rate"] for record in validation_records]

    symbols_per_token = [
        record["val_symbols_per_token"] for record in validation_records
    ]

    boundary_entropy = [record["val_boundary_entropy"] for record in validation_records]

    binary_loss = [record["val_binary_loss"] for record in validation_records]

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
    plt.title("Neural V2: BPB vs optimization steps")

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
    # BPB vs original bytes
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
    plt.title("Neural V2: BPB vs original bytes")

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
    # BPB vs training time
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
    plt.title("Neural V2: BPB vs training compute time")

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

    # --------------------------------------------------------
    # Boundary rate vs steps
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        steps,
        boundary_rate,
        marker="o",
        markersize=3,
        label="Observed boundary rate",
    )

    plt.axhline(
        TARGET_SYMBOLS_PER_TOKEN**-1,
        linestyle="--",
        label="Target boundary rate",
    )

    plt.xlabel("Optimization steps")
    plt.ylabel("Boundary probability")
    plt.title("Neural V2: boundary rate vs optimization steps")

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir / "boundary_rate_vs_steps.png",
        dpi=150,
    )

    plt.close()

    # --------------------------------------------------------
    # Symbols per token vs steps
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        steps,
        symbols_per_token,
        marker="o",
        markersize=3,
    )

    plt.axhline(
        TARGET_SYMBOLS_PER_TOKEN,
        linestyle="--",
        label="Target",
    )

    plt.xlabel("Optimization steps")
    plt.ylabel("Estimated source symbols per token")
    plt.title("Neural V2: symbols per token vs optimization steps")

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir / "symbols_per_token_vs_steps.png",
        dpi=150,
    )

    plt.close()

    # --------------------------------------------------------
    # Boundary entropy vs steps
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        steps,
        boundary_entropy,
        marker="o",
        markersize=3,
    )

    plt.xlabel("Optimization steps")
    plt.ylabel("Mean boundary entropy (nats)")
    plt.title("Neural V2: boundary entropy vs optimization steps")

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / "boundary_entropy_vs_steps.png",
        dpi=150,
    )

    plt.close()

    # --------------------------------------------------------
    # Boundary binary loss vs steps
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        steps,
        binary_loss,
        marker="o",
        markersize=3,
    )

    plt.xlabel("Optimization steps")
    plt.ylabel("Boundary binary loss")
    plt.title("Neural V2: boundary binary loss vs optimization steps")

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / "boundary_binary_loss_vs_steps.png",
        dpi=150,
    )

    plt.close()


# ============================================================
# Main
# ============================================================


def main():

    print("=" * 80)
    print("NEURAL TOKENIZER V2")
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

    train_dataset = NeuralV2Dataset(
        train_texts,
        tokenizer,
        source_length=SOURCE_LENGTH,
        group_size=GROUP_SIZE,
        stride=STRIDE,
    )

    valid_dataset = NeuralV2Dataset(
        valid_texts,
        tokenizer,
        source_length=SOURCE_LENGTH,
        group_size=GROUP_SIZE,
        stride=STRIDE,
    )

    print("\nDataset")
    print("-" * 60)

    print(f"Train samples:     " f"{len(train_dataset):,}")

    print(f"Valid samples:     " f"{len(valid_dataset):,}")

    print(f"Source length:     " f"{SOURCE_LENGTH} symbols")

    print(f"Latent tokens:     " f"{NUM_TOKENS}")

    print(f"Group size:        " f"{GROUP_SIZE} symbols")

    print(f"Target compression:" f" {GROUP_SIZE:.1f} symbols/token")

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

    model = NeuralV2LanguageModel(
        input_vocab_size=tokenizer.vocab_size,
        output_vocab_size=tokenizer.vocab_size,
        source_length=SOURCE_LENGTH,
        num_tokens=NUM_TOKENS,
        tokenizer_embedding_dim=TOKENIZER_EMBEDDING_DIM,
        boundary_hidden_dim=BOUNDARY_HIDDEN_DIM,
        boundary_kernel_size=BOUNDARY_KERNEL_SIZE,
        pooling_temperature=POOLING_TEMPERATURE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        gru_hidden_dim=GRU_HIDDEN_DIM,
        group_size=GROUP_SIZE,
    )

    model = model.to(DEVICE)

    params = count_parameters(model)

    print("\nModel parameters")
    print("-" * 60)

    print(f"Total:              " f"{params['total']:,}")

    print(f"Neural tokenizer:   " f"{params['neural_tokenizer']:,}")

    print(f"Transformer + GRU:  " f"{params['transformer_and_gru']:,}")

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\nStarting training...\n")

    start_time = time.perf_counter()

    history = []

    train_iterator = iter(train_loader)

    best_bpb = float("inf")

    for step in range(1, STEPS + 1):

        try:
            x, y, target_mask = next(train_iterator)

        except StopIteration:
            train_iterator = iter(train_loader)

            x, y, target_mask = next(train_iterator)

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

        # ----------------------------------------------------
        # Teacher forcing
        # ----------------------------------------------------

        previous_symbols = torch.empty_like(y)

        previous_symbols[:, :, 0] = tokenizer.pad_id

        previous_symbols[:, :, 1:] = y[:, :, :-1]

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        logits, boundary_probs = model(
            x,
            previous_symbols,
        )

        low_confident = (boundary_probs < 0.1).float().mean()

        high_confident = (boundary_probs > 0.9).float().mean()

        # ----------------------------------------------------
        # LM loss
        # ----------------------------------------------------

        vocab_size = logits.shape[-1]

        losses = torch.nn.functional.cross_entropy(
            logits.reshape(-1, vocab_size),
            y.reshape(-1),
            reduction="none",
        )

        losses = losses.reshape_as(y)

        valid_losses = losses[target_mask]

        lm_loss = valid_losses.mean()

        # ----------------------------------------------------
        # Boundary-rate loss
        # ----------------------------------------------------

        rate_loss = boundary_rate_loss(
            boundary_probs,
            target_symbols_per_token=TARGET_SYMBOLS_PER_TOKEN,
        )

        binary_loss = boundary_binary_loss(boundary_probs)

        # ----------------------------------------------------
        # Total loss
        # ----------------------------------------------------

        loss = lm_loss + LAMBDA_RATE * rate_loss + LAMBDA_BINARY * binary_loss

        # ----------------------------------------------------
        # Backprop
        # ----------------------------------------------------

        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        # ----------------------------------------------------
        # Training metrics
        # ----------------------------------------------------

        elapsed = time.perf_counter() - start_time

        train_bpb = lm_loss.item() / np.log(2.0) / train_stats["bytes_per_symbol"]

        train_boundary_rate = boundary_probs.detach().mean().item()

        train_symbols_per_token = 1.0 / max(
            train_boundary_rate,
            1e-8,
        )

        train_boundary_entropy = calculate_boundary_entropy(boundary_probs.detach())

        # Each sample consumes SOURCE_LENGTH
        # source symbols.
        #
        # Convert symbols to estimated original
        # UTF-8 bytes using corpus-level bytes/symbol.

        estimated_original_bytes = (
            step * BATCH_SIZE * SOURCE_LENGTH * train_stats["bytes_per_symbol"]
        )

        record = {
            "step": step,
            "train_loss": lm_loss.item(),
            "train_total_loss": loss.item(),
            "train_rate_loss": rate_loss.item(),
            "train_binary_loss": binary_loss.item(),
            "train_bpb": train_bpb,
            "train_boundary_rate": train_boundary_rate,
            "train_symbols_per_token": (train_symbols_per_token),
            "train_boundary_entropy": (train_boundary_entropy),
            "estimated_original_bytes": (estimated_original_bytes),
            "wall_clock_seconds": elapsed,
            "val_loss": None,
            "val_bpb": None,
            "val_rate_loss": None,
            "val_binary_loss": None,
            "val_boundary_rate": None,
            "val_symbols_per_token": None,
            "val_boundary_entropy": None,
        }

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        if step == 1 or step % VAL_BATCHES == 0 or step == STEPS:
            print(
                f"step {step:5d} | "
                f"loss {loss.item():.4f} | "
                f"LM {lm_loss.item():.4f} | "
                f"rate {rate_loss.item():.6f} | "
                f"binary {binary_loss.item():.6f} | "
                f"train BPB {train_bpb:.4f} | "
                f"boundary {train_boundary_rate:.4f} | "
                f"sym/token "
                f"{train_symbols_per_token:.2f} | "
                f"low confident "
                f"{low_confident:.6f} | "
                f"high confident "
                f"{high_confident:.6f} | "
                f"time {elapsed / 60:.2f} min"
            )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if step == 1 or step % VAL_EVERY == 0 or step == STEPS:
            val_metrics = evaluate(
                model=model,
                loader=valid_loader,
                device=DEVICE,
                eval_batches=VAL_BATCHES,
            )

            record["val_loss"] = val_metrics["loss"]

            record["val_bpb"] = val_metrics["bpb"] / train_stats["bytes_per_symbol"]

            record["val_rate_loss"] = val_metrics["rate_loss"]

            record["val_boundary_rate"] = val_metrics["boundary_rate"]

            record["val_symbols_per_token"] = 1.0 / max(
                val_metrics["boundary_rate"],
                1e-8,
            )

            record["val_boundary_entropy"] = val_metrics["boundary_entropy"]

            record["boundary_mean"] = val_metrics["boundary_mean"]
            record["boundary_std"] = val_metrics["boundary_std"]
            record["boundary_min"] = val_metrics["boundary_min"]
            record["boundary_max"] = val_metrics["boundary_max"]

            record["val_binary_loss"] = val_metrics["binary_loss"]
            record["low_confident"] = val_metrics["low_confident"]
            record["high_confident"] = val_metrics["high_confident"]

            print(
                f"  validation | "
                f"loss {record['val_loss']:.4f} | "
                f"BPB {record['val_bpb']:.4f} | "
                f"rate "
                f"{record['val_rate_loss']:.6f} | "
                f"binary "
                f"{record['val_binary_loss']:.6f} | "
                f"boundary "
                f"{record['val_boundary_rate']:.4f} | "
                f"sym/token "
                f"{record['val_symbols_per_token']:.2f} | "
                f"entropy "
                f"{record['val_boundary_entropy']:.4f} | "
                f"boundary mean "
                f"{record['boundary_mean']:.4f} | "
                f"boundary std "
                f"{record['boundary_std']:.4f} | "
                f"boundary min "
                f"{record['boundary_min']:.4f} | "
                f"boundary max "
                f"{record['boundary_max']:.4f} | "
                f"low confident "
                f"{record['low_confident']:.6f} | "
                f"high confident "
                f"{record['high_confident']:.6f} "
            )

            if record["val_bpb"] < best_bpb:
                best_bpb = record["val_bpb"]

                torch.save(
                    {
                        "step": step,
                        "model_state_dict": (model.state_dict()),
                        "optimizer_state_dict": (optimizer.state_dict()),
                        "val_bpb": best_bpb,
                        "args": {
                            "source_length": SOURCE_LENGTH,
                            "group_size": GROUP_SIZE,
                            "num_tokens": NUM_TOKENS,
                            "batch_size": BATCH_SIZE,
                            "learning_rate": LEARNING_RATE,
                            "lambda_rate": LAMBDA_RATE,
                            "pooling_temperature": (POOLING_TEMPERATURE),
                            "seed": SEED,
                        },
                    },
                    OUTPUT_DIR / "best.pt",
                )

                print(f"  saved best checkpoint " f"(BPB={best_bpb:.4f})")

        history.append(record)

    total_training_time = time.perf_counter() - start_time

    # --------------------------------------------------------
    # Final checkpoint
    # --------------------------------------------------------

    torch.save(
        {
            "step": STEPS,
            "model_state_dict": (model.state_dict()),
            "optimizer_state_dict": (optimizer.state_dict()),
            "best_val_bpb": best_bpb,
        },
        OUTPUT_DIR / "final.pt",
    )

    # --------------------------------------------------------
    # Final validation metrics
    # --------------------------------------------------------

    final_metrics = evaluate(
        model=model,
        loader=valid_loader,
        device=DEVICE,
        eval_batches=VAL_BATCHES,
    )

    final_validation_bpb = final_metrics["bpb"] / train_stats["bytes_per_symbol"]

    # --------------------------------------------------------
    # Configuration JSON
    # --------------------------------------------------------

    config = {
        "experiment": "neural_v2",
        "train_size": TRAIN_SIZE,
        "valid_size": VALID_SIZE,
        "source_length": SOURCE_LENGTH,
        "group_size": GROUP_SIZE,
        "num_tokens": NUM_TOKENS,
        "stride": STRIDE,
        "batch_size": BATCH_SIZE,
        "tokenizer_embedding_dim": (TOKENIZER_EMBEDDING_DIM),
        "boundary_hidden_dim": (BOUNDARY_HIDDEN_DIM),
        "boundary_kernel_size": (BOUNDARY_KERNEL_SIZE),
        "pooling_temperature": (POOLING_TEMPERATURE),
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS,
        "d_ff": D_FF,
        "gru_hidden_dim": GRU_HIDDEN_DIM,
        "learning_rate": LEARNING_RATE,
        "steps": STEPS,
        "val_every": VAL_EVERY,
        "val_batches": VAL_BATCHES,
        "lambda_rate": LAMBDA_RATE,
        "lambda_binary": LAMBDA_BINARY,
        "target_symbols_per_token": (TARGET_SYMBOLS_PER_TOKEN),
        "seed": SEED,
        "device": str(DEVICE),
        "vocab_size": tokenizer.vocab_size,
        "pad_id": tokenizer.pad_id,
    }

    # --------------------------------------------------------
    # Results JSON
    # --------------------------------------------------------

    final_boundary_rate = final_metrics["boundary_rate"]

    final_symbols_per_token = 1.0 / max(
        final_boundary_rate,
        1e-8,
    )

    results = {
        "experiment": "neural_v2",
        "validation_bpb": final_validation_bpb,
        "best_validation_bpb": best_bpb,
        "training_wall_clock_seconds": (total_training_time),
        "train_statistics": train_stats,
        "valid_statistics": valid_stats,
        "train_samples": len(train_dataset),
        "valid_samples": len(valid_dataset),
        "source_length": SOURCE_LENGTH,
        "num_tokens": NUM_TOKENS,
        "group_size": GROUP_SIZE,
        "target_symbols_per_token": (TARGET_SYMBOLS_PER_TOKEN),
        "parameters": params,
        "final_boundary_rate": (final_boundary_rate),
        "final_symbols_per_token": (final_symbols_per_token),
        "final_boundary_entropy": (final_metrics["boundary_entropy"]),
        "final_rate_loss": (final_metrics["rate_loss"]),
        "final_binary_loss": (final_metrics["binary_loss"]),
    }

    # --------------------------------------------------------
    # Save artifacts
    # --------------------------------------------------------

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
    print("NEURAL V2 COMPLETE")
    print("=" * 80)

    print(f"Validation BPB: " f"{final_validation_bpb:.4f}")

    print(f"Best validation BPB: " f"{best_bpb:.4f}")

    print(f"Training time: " f"{total_training_time / 60:.2f} min")

    print(f"Parameters: " f"{params['total']:,}")

    print(f"Tokenizer: " f"{params['neural_tokenizer']:,}")

    print(f"Transformer + GRU: " f"{params['transformer_and_gru']:,}")

    print(f"Final boundary rate: " f"{final_boundary_rate:.6f}")

    print(f"Final symbols/token: " f"{final_symbols_per_token:.3f}")

    print(f"Final boundary entropy: " f"{final_metrics['boundary_entropy']:.6f} nats")

    print(f"Final binary loss: " f"{final_metrics['binary_loss']:.6f}")

    print("\nResults saved to:")
    print(OUTPUT_DIR.resolve())

    print("=" * 80)


if __name__ == "__main__":
    main()
