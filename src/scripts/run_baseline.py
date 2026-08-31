import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from neural_tokenizer.dataset import LanguageModelDataset
from neural_tokenizer.model import TinyTransformer
from neural_tokenizer.tokenizer import (
    BPETokenizer,
    ByteTokenizer,
)
from neural_tokenizer.training import train

# ============================================================
# Configuration
# ============================================================

TRAIN_SIZE = 50000
VALID_SIZE = 5000

CONTEXT_LENGTH = 256
STRIDE = 256
BATCH_SIZE = 64

D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
D_FF = 512

LEARNING_RATE = 3e-4
STEPS = 5_000

VAL_EVERY = 500
VAL_BATCHES = 100

SEED = 42

BPE_TOKENIZER_PATH = "notebooks/artifacts/bpe/tokenizer.json"


# ============================================================
# Reproducibility
# ============================================================


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Utilities
# ============================================================


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def count_embedding_parameters(
    model: torch.nn.Module,
) -> int:
    return model.token_embedding.weight.numel()


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Exact BPB evaluation
# ============================================================


@torch.no_grad()
def evaluate_bits_per_byte(
    model: torch.nn.Module,
    texts,
    tokenizer,
    context_length: int,
    device: torch.device,
    max_texts: int | None = None,
):
    """
    Evaluate language-model NLL against the exact UTF-8
    byte length of the original validation text.

    This deliberately calculates the denominator from the
    original text rather than trying to assign byte counts
    to individual ByteLevel BPE tokens.
    """

    model.eval()

    total_nll = 0.0
    total_bytes = 0
    total_targets = 0

    if max_texts is not None:
        texts = texts[:max_texts]

    progress = tqdm(
        texts,
        desc="BPB",
        leave=False,
    )

    for text in progress:
        ids = tokenizer.encode(text)

        # Exact denominator.
        text_bytes = len(text.encode("utf-8"))

        total_bytes += text_bytes

        # Process the token sequence in non-overlapping
        # chunks, but don't discard the final partial chunk.
        for start in range(
            0,
            len(ids) - 1,
            context_length,
        ):
            chunk = ids[start : start + context_length + 1]

            if len(chunk) < 2:
                continue

            x = torch.tensor(
                chunk[:-1],
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)

            y = torch.tensor(
                chunk[1:],
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)

            logits = model(x)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                reduction="sum",
            )

            total_nll += loss.item()
            total_targets += y.numel()

    model.train()

    bits_per_byte = total_nll / total_bytes / math.log(2)

    return {
        "nll": total_nll,
        "bytes": total_bytes,
        "targets": total_targets,
        "bits_per_byte": bits_per_byte,
    }


# ============================================================
# Experiment
# ============================================================


def run_experiment(
    name: str,
    tokenizer,
    train_texts,
    valid_texts,
    device: torch.device,
):
    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {name}")
    print("=" * 70)

    print(f"Device: {device}")
    print(f"Vocab size: " f"{tokenizer.vocab_size}")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = LanguageModelDataset(
        texts=train_texts,
        tokenizer=tokenizer,
        context_length=CONTEXT_LENGTH,
        stride=STRIDE,
    )

    valid_dataset = LanguageModelDataset(
        texts=valid_texts,
        tokenizer=tokenizer,
        context_length=CONTEXT_LENGTH,
        stride=STRIDE,
    )

    print(f"Train Dataset size: " f"{len(train_dataset):,}")

    print(f"Valid Dataset size: " f"{len(valid_dataset):,}")

    # --------------------------------------------------------
    # Deterministic DataLoaders
    # --------------------------------------------------------

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        generator=generator,
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

    model = TinyTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_seq_len=CONTEXT_LENGTH,
    ).to(device)

    total_parameters = count_parameters(model)

    embedding_parameters = count_embedding_parameters(model)

    print(f"Vocabulary size: " f"{tokenizer.vocab_size:,}")

    print(f"Parameters: " f"{total_parameters:,}")

    print(f"Embedding parameters: " f"{embedding_parameters:,}")

    print(f"Transformer parameters: " f"{total_parameters - embedding_parameters:,}")

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

    history = train(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        device=device,
        steps=STEPS,
        val_loader=valid_loader,
        val_every=VAL_EVERY,
        val_batches=VAL_BATCHES,
    )

    # --------------------------------------------------------
    # Exact BPB evaluation
    # --------------------------------------------------------

    print("\nRunning exact validation BPB...")

    bpb_result = evaluate_bits_per_byte(
        model=model,
        texts=valid_texts,
        tokenizer=tokenizer,
        context_length=CONTEXT_LENGTH,
        device=device,
    )

    print(f"Validation bytes: " f"{bpb_result['bytes']:,}")

    print(f"Validation targets: " f"{bpb_result['targets']:,}")

    print(f"Validation BPB: " f"{bpb_result['bits_per_byte']:.4f}")

    return {
        "name": name,
        "vocab_size": tokenizer.vocab_size,
        "parameters": total_parameters,
        "embedding_parameters": embedding_parameters,
        "transformer_parameters": (total_parameters - embedding_parameters),
        "train_dataset_size": len(train_dataset),
        "valid_dataset_size": len(valid_dataset),
        "history": history,
        "bpb": bpb_result,
    }


# ============================================================
# Main
# ============================================================


def main():
    set_seed(SEED)

    device = get_device()

    print("Loading TinyStories...")

    dataset = load_dataset("roneneldan/TinyStories")

    train_split = dataset["train"].select(range(TRAIN_SIZE))

    valid_split = dataset["validation"].select(range(VALID_SIZE))

    train_texts = train_split["text"]
    valid_texts = valid_split["text"]

    results = []

    # ========================================================
    # BPE
    # ========================================================

    set_seed(SEED)

    bpe_tokenizer = BPETokenizer(BPE_TOKENIZER_PATH)

    bpe_result = run_experiment(
        name="BPE",
        tokenizer=bpe_tokenizer,
        train_texts=train_texts,
        valid_texts=valid_texts,
        device=device,
    )

    results.append(bpe_result)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ========================================================
    # BYTE
    # ========================================================

    set_seed(SEED)

    byte_tokenizer = ByteTokenizer()

    byte_result = run_experiment(
        name="BYTE",
        tokenizer=byte_tokenizer,
        train_texts=train_texts,
        valid_texts=valid_texts,
        device=device,
    )

    results.append(byte_result)

    # ========================================================
    # Summary
    # ========================================================

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for result in results:
        history = result["history"]

        train_losses = [item["train_loss"] for item in history]

        val_losses = [
            item["val_loss"] for item in history if item["val_loss"] is not None
        ]

        print(f"\n{result['name']}")
        print("-" * 40)

        print(f"Vocabulary: " f"{result['vocab_size']:,}")

        print(f"Parameters: " f"{result['parameters']:,}")

        print(f"Embedding parameters: " f"{result['embedding_parameters']:,}")

        print(f"Transformer parameters: " f"{result['transformer_parameters']:,}")

        print(f"Train samples: " f"{result['train_dataset_size']:,}")

        print(f"Valid samples: " f"{result['valid_dataset_size']:,}")

        if train_losses:
            print(f"Initial train NLL/token: " f"{train_losses[0]:.4f}")

            print(f"Final train NLL/token: " f"{train_losses[-1]:.4f}")

        if val_losses:
            print(f"Initial validation NLL/token: " f"{val_losses[0]:.4f}")

            print(f"Final validation NLL/token: " f"{val_losses[-1]:.4f}")

        print(f"Validation BPB: " f"{result['bpb']['bits_per_byte']:.4f}")

        print(f"Validation bytes: " f"{result['bpb']['bytes']:,}")


if __name__ == "__main__":
    main()
