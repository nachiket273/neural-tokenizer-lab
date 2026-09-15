from __future__ import annotations

import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from neural_tokenizer.neural_v2_model import NeuralV2LanguageModel
from neural_tokenizer.tokenizer import BPETokenizer, ByteTokenizer

# ============================================================
# Configuration
# ============================================================

TRAIN_SIZE = 50_000
VALID_SIZE = 5_000

SOURCE_LENGTH = 1024
GROUP_SIZE = 4
NUM_TOKENS = SOURCE_LENGTH // GROUP_SIZE

# ------------------------------------------------------------
# Inference
# ------------------------------------------------------------

# Boundary-only inference is very cheap.
# We can therefore use a much larger batch than the
# full Neural V2 language-model evaluation.
BATCH_SIZE = 64

# Number of validation samples to inspect.
#
# None:
#     use all possible validation windows.
#
# 500:
#     useful for quick debugging.
#
# Start with 500 and increase after confirming correctness.
MAX_ALIGNMENT_SAMPLES = 500

# ------------------------------------------------------------
# Boundary extraction
# ------------------------------------------------------------

BOUNDARY_THRESHOLD = 0.5

# Position tolerance when comparing a neural boundary
# with a BPE boundary.
#
# Example:
#
#     BPE boundary at byte 120
#     neural boundary at byte 119
#
# counts as a match when tolerance >= 1.
BOUNDARY_TOLERANCE = 1

# ------------------------------------------------------------
# Checkpoint
# ------------------------------------------------------------

CHECKPOINT = Path("experiments/neural_v2/best.pt")

BPE_TOKENIZER_PATH = Path("notebooks/artifacts/bpe/tokenizer.json")

OUTPUT_DIR = Path("experiments/neural_v2/boundary_alignment")

# ------------------------------------------------------------
# Model configuration
# ------------------------------------------------------------

TOKENIZER_EMBEDDING_DIM = 64
BOUNDARY_HIDDEN_DIM = 128
BOUNDARY_KERNEL_SIZE = 5
POOLING_TEMPERATURE = 0.75

D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
D_FF = 512
GRU_HIDDEN_DIM = 128

PAD_ID = 257

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

SEED = 42

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
# Dataset
# ============================================================


class BoundaryAlignmentDataset(Dataset):
    """
    Validation dataset for comparing Neural V2 boundaries
    against BPE boundaries.

    We construct the same source representation used by
    Neural V2:

        UTF-8 bytes
        +
        EOS symbol

    Stories are concatenated into a single stream.

    Each sample contains:

        x:
            1024 source symbols

        bpe_boundaries:
            binary vector [1024]

            bpe_boundaries[i] = 1 means that a BPE token
            boundary occurs after source position i.

    The final source position is not used as a prediction
    target unless it corresponds to an actual boundary.
    """

    def __init__(
        self,
        texts: list[str],
        source_length: int = 1024,
        max_samples: int | None = None,
    ) -> None:

        self.source_length = source_length

        byte_tokenizer = ByteTokenizer()

        # ----------------------------------------------------
        # Construct byte stream
        # ----------------------------------------------------

        byte_stream: list[int] = []

        # ----------------------------------------------------
        # Construct BPE boundary stream
        # ----------------------------------------------------

        boundary_stream: list[int] = []

        print(Path.cwd())

        bpe_tokenizer = BPETokenizer(str(BPE_TOKENIZER_PATH))

        for text in texts:

            # ==================================================
            # Raw UTF-8 bytes
            # ==================================================

            raw_bytes = text.encode("utf-8")

            start_offset = len(byte_stream)

            byte_stream.extend(raw_bytes)

            # ==================================================
            # BPE boundaries
            # ==================================================
            #
            # We need one boundary for every BPE token.
            #
            # BPE offsets are character offsets.
            #
            # Neural V2 operates on UTF-8 bytes.
            #
            # Therefore:
            #
            # character offset
            #       ↓
            # UTF-8 byte offset
            #
            # ==================================================

            encoding = bpe_tokenizer.tokenizer.encode(text)

            for offset_start, offset_end in encoding.offsets:

                # Empty offsets can occur for special tokens.
                if offset_end <= offset_start:
                    continue

                # Convert character offsets into UTF-8 byte offsets.
                byte_end = len(text[:offset_end].encode("utf-8"))

                absolute_boundary = start_offset + byte_end - 1

                boundary_stream.append(absolute_boundary)

            # ==================================================
            # EOS
            # ==================================================
            #
            # ByteTokenizer adds EOS after every story.
            #
            # We represent EOS as symbol 256.
            #
            # The boundary after EOS is therefore also a
            # meaningful segmentation boundary.
            # ==================================================

            byte_stream.append(256)

            boundary_stream.append(len(byte_stream) - 1)

        # ----------------------------------------------------
        # Convert boundary positions to binary stream
        # ----------------------------------------------------

        total_length = len(byte_stream)

        boundary_mask = np.zeros(
            total_length,
            dtype=np.uint8,
        )

        for position in boundary_stream:

            if 0 <= position < total_length:

                boundary_mask[position] = 1

        # ----------------------------------------------------
        # Build samples
        # ----------------------------------------------------

        self.samples: list[tuple[np.ndarray, np.ndarray]] = []

        max_start = total_length - source_length

        if max_start < 0:

            raise ValueError(
                "Validation corpus is shorter than " f"source_length={source_length}."
            )

        # ----------------------------------------------------
        # Non-overlapping windows
        # ----------------------------------------------------
        #
        # This matches the V2 dataset stride.
        #
        # ----------------------------------------------------

        starts = range(
            0,
            max_start + 1,
            source_length,
        )

        for start in starts:

            end = start + source_length

            x = np.asarray(
                byte_stream[start:end],
                dtype=np.int64,
            )

            boundaries = boundary_mask[start:end].copy()

            self.samples.append(
                (
                    x,
                    boundaries,
                )
            )

            if max_samples is not None and len(self.samples) >= max_samples:
                break

    def __len__(self) -> int:

        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ):

        x, boundaries = self.samples[index]

        return (
            torch.from_numpy(x),
            torch.from_numpy(boundaries),
        )


# ============================================================
# Model
# ============================================================


def load_model(
    vocab_size: int,
) -> NeuralV2LanguageModel:

    print("\nLoading checkpoint")
    print("-" * 60)

    print(f"Checkpoint: {CHECKPOINT}")

    if not CHECKPOINT.exists():

        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    model = NeuralV2LanguageModel(
        input_vocab_size=vocab_size,
        output_vocab_size=vocab_size,
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
    )

    # --------------------------------------------------------
    # PyTorch >= 2.6
    #
    # Our checkpoints contain optimizer state and numpy
    # scalars, so weights_only=True can reject the file.
    #
    # This is a locally generated checkpoint that we trust.
    # --------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:

        raise KeyError("Checkpoint does not contain " "'model_state_dict'.")

    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(DEVICE)

    model.eval()

    total_parameters = sum(p.numel() for p in model.parameters())

    tokenizer_parameters = sum(p.numel() for p in (model.neural_tokenizer.parameters()))

    print(f"Parameters: {total_parameters:,}")

    print(f"Tokenizer:  {tokenizer_parameters:,}")

    print(f"Device:     {DEVICE}")

    return model


# ============================================================
# Boundary prediction
# ============================================================


@torch.no_grad()
def predict_boundaries(
    model: NeuralV2LanguageModel,
    loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run ONLY the Neural V2 boundary predictor.

    IMPORTANT:
    We intentionally do NOT call:

        model(x, previous_symbols)

    because that would execute the full Transformer +
    GRU language model.

    Instead we directly call:

        model.neural_tokenizer.boundary_predictor

    This makes the alignment experiment substantially
    cheaper and allows a large batch size.
    """

    all_probs: list[np.ndarray] = []
    all_bpe: list[np.ndarray] = []

    boundary_predictor = model.neural_tokenizer.boundary_predictor

    boundary_predictor.eval()

    start_time = time.perf_counter()

    total_samples = 0

    for batch_idx, (
        x,
        bpe_boundaries,
    ) in enumerate(loader):

        x = x.to(
            DEVICE,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # GPU boundary-only inference
        # ----------------------------------------------------

        _, boundary_probs = boundary_predictor(x)

        # ----------------------------------------------------
        # Transfer one batch to CPU.
        #
        # We do NOT synchronize per individual sample.
        # ----------------------------------------------------

        all_probs.append(boundary_probs.cpu().numpy())

        all_bpe.append(bpe_boundaries.numpy())

        total_samples += x.shape[0]

    elapsed = time.perf_counter() - start_time

    probs = np.concatenate(
        all_probs,
        axis=0,
    )

    bpe = np.concatenate(
        all_bpe,
        axis=0,
    )

    print()
    print(f"Boundary inference samples: " f"{total_samples:,}")

    print(f"Boundary inference time: " f"{elapsed:.3f} s")

    if elapsed > 0:

        print(f"Samples/sec: " f"{total_samples / elapsed:.1f}")

    return probs, bpe


# ============================================================
# Boundary extraction
# ============================================================


def extract_neural_boundaries(
    probabilities: np.ndarray,
    threshold: float,
) -> list[np.ndarray]:
    """
    Convert soft Neural V2 probabilities into hard
    boundary positions.

    A boundary is predicted after source position i when:

        p_i >= threshold
    """

    results = []

    for row in probabilities:

        positions = np.flatnonzero(row >= threshold)

        results.append(positions)

    return results


def extract_bpe_boundaries(
    boundary_masks: np.ndarray,
) -> list[np.ndarray]:

    results = []

    for row in boundary_masks:

        positions = np.flatnonzero(row > 0)

        results.append(positions)

    return results


# ============================================================
# Boundary matching
# ============================================================


def match_boundaries(
    neural_positions: np.ndarray,
    reference_positions: np.ndarray,
    tolerance: int = 1,
) -> tuple[int, int, int]:
    """
    Greedy one-to-one boundary matching.

    Returns:

        true_positives
        false_positives
        false_negatives

    A neural boundary matches a reference boundary if
    their source positions differ by <= tolerance.

    Each reference boundary can only be matched once.
    """

    if len(neural_positions) == 0:

        return (
            0,
            0,
            len(reference_positions),
        )

    if len(reference_positions) == 0:

        return (
            0,
            len(neural_positions),
            0,
        )

    matched_reference = set()

    true_positives = 0

    for neural_position in neural_positions:

        distances = np.abs(reference_positions - neural_position)

        candidates = np.flatnonzero(distances <= tolerance)

        # ----------------------------------------------------
        # Select closest unmatched reference boundary.
        # ----------------------------------------------------

        best_reference = None
        best_distance = None

        for candidate in candidates:

            candidate = int(candidate)

            if candidate in matched_reference:
                continue

            distance = int(distances[candidate])

            if best_distance is None or distance < best_distance:

                best_distance = distance
                best_reference = candidate

        if best_reference is not None:

            matched_reference.add(best_reference)

            true_positives += 1

    false_positives = len(neural_positions) - true_positives

    false_negatives = len(reference_positions) - true_positives

    return (
        true_positives,
        false_positives,
        false_negatives,
    )


# ============================================================
# Alignment metrics
# ============================================================


def calculate_alignment_metrics(
    neural_boundaries: list[np.ndarray],
    bpe_boundaries: list[np.ndarray],
    tolerance: int,
) -> dict:

    total_tp = 0
    total_fp = 0
    total_fn = 0

    total_neural_boundaries = 0
    total_bpe_boundaries = 0

    distances: list[int] = []

    per_sample = []

    for neural, bpe in zip(
        neural_boundaries,
        bpe_boundaries,
    ):

        (
            tp,
            fp,
            fn,
        ) = match_boundaries(
            neural,
            bpe,
            tolerance=tolerance,
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        total_neural_boundaries += len(neural)

        total_bpe_boundaries += len(bpe)

        # ----------------------------------------------------
        # Boundary distance statistics
        # ----------------------------------------------------

        matched = []

        used_bpe = set()

        for neural_position in neural:

            if len(bpe) == 0:
                continue

            distances_to_bpe = np.abs(bpe - neural_position)

            candidates = np.flatnonzero(distances_to_bpe <= tolerance)

            best = None
            best_distance = None

            for candidate in candidates:

                candidate = int(candidate)

                if candidate in used_bpe:
                    continue

                distance = int(distances_to_bpe[candidate])

                if best_distance is None or distance < best_distance:

                    best = candidate
                    best_distance = distance

            if best is not None:

                used_bpe.add(best)

                matched.append(best_distance)

                distances.append(best_distance)

        precision = (
            total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        )

        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per_sample.append(
            {
                "neural_boundaries": len(neural),
                "bpe_boundaries": len(bpe),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0

    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    result = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "neural_boundaries": total_neural_boundaries,
        "bpe_boundaries": total_bpe_boundaries,
        "mean_boundary_distance": (float(np.mean(distances)) if distances else None),
        "median_boundary_distance": (
            float(np.median(distances)) if distances else None
        ),
        "per_sample": per_sample,
    }

    return result


# ============================================================
# Basic statistics
# ============================================================


def calculate_boundary_statistics(
    probabilities: np.ndarray,
) -> dict:

    hard = probabilities >= BOUNDARY_THRESHOLD

    boundary_rate = hard.mean()

    symbols_per_token = 1.0 / max(
        boundary_rate,
        1e-8,
    )

    eps = 1e-7

    p = np.clip(
        probabilities,
        eps,
        1.0 - eps,
    )

    entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))

    return {
        "soft_mean": float(probabilities.mean()),
        "soft_std": float(probabilities.std()),
        "soft_min": float(probabilities.min()),
        "soft_max": float(probabilities.max()),
        "hard_boundary_rate": float(boundary_rate),
        "hard_symbols_per_token": float(symbols_per_token),
        "entropy": float(entropy.mean()),
        "low_confident_fraction": float((probabilities < 0.1).mean()),
        "high_confident_fraction": float((probabilities > 0.9).mean()),
    }


# ============================================================
# Intervention baseline
# ============================================================


def calculate_bpe_boundary_statistics(
    bpe_boundaries: list[np.ndarray],
) -> dict:

    counts = np.asarray(
        [len(x) for x in bpe_boundaries],
        dtype=np.float64,
    )

    if len(counts) == 0:

        return {
            "mean_boundaries_per_window": 0.0,
            "mean_symbols_per_token": 0.0,
        }

    mean_boundaries = counts.mean()

    mean_symbols_per_token = SOURCE_LENGTH / max(
        mean_boundaries,
        1e-8,
    )

    return {
        "mean_boundaries_per_window": float(mean_boundaries),
        "mean_symbols_per_token": float(mean_symbols_per_token),
    }


# ============================================================
# Visualization
# ============================================================


def save_probability_histogram(
    probabilities: np.ndarray,
    output_dir: Path,
) -> None:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(8, 5))

    plt.hist(
        probabilities.reshape(-1),
        bins=100,
    )

    plt.axvline(
        BOUNDARY_THRESHOLD,
        linestyle="--",
        label=(f"Threshold = " f"{BOUNDARY_THRESHOLD}"),
    )

    plt.xlabel("Neural boundary probability")

    plt.ylabel("Count")

    plt.title("Neural V2 boundary probability distribution")

    plt.grid(alpha=0.3)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir / "boundary_probability_histogram.png",
        dpi=150,
    )

    plt.close()


def save_boundary_alignment_examples(
    probabilities: np.ndarray,
    bpe_boundaries: np.ndarray,
    output_dir: Path,
    num_examples: int = 5,
) -> None:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    n = min(
        num_examples,
        len(probabilities),
    )

    for index in range(n):

        plt.figure(figsize=(12, 4))

        positions = np.arange(SOURCE_LENGTH)

        plt.plot(
            positions,
            probabilities[index],
            linewidth=1,
        )

        bpe_positions = np.flatnonzero(bpe_boundaries[index] > 0)

        for position in bpe_positions:

            plt.axvline(
                position,
                linestyle="--",
                alpha=0.35,
            )

        plt.axhline(
            BOUNDARY_THRESHOLD,
            linestyle=":",
            label="Neural threshold",
        )

        plt.xlabel("Source byte position")

        plt.ylabel("Neural boundary probability")

        plt.title("Neural V2 vs BPE boundary positions " f"(window {index})")

        plt.grid(alpha=0.3)

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            output_dir / f"alignment_window_{index:03d}.png",
            dpi=150,
        )

        plt.close()


def save_metrics_plot(
    metrics: dict,
    output_dir: Path,
) -> None:

    labels = [
        "Precision",
        "Recall",
        "F1",
    ]

    values = [
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    ]

    plt.figure(figsize=(7, 5))

    plt.bar(
        labels,
        values,
    )

    plt.ylim(
        0.0,
        1.0,
    )

    plt.ylabel("Score")

    plt.title("Neural V2 / BPE boundary alignment")

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / "boundary_alignment_metrics.png",
        dpi=150,
    )

    plt.close()


# ============================================================
# Main
# ============================================================


def main():

    print("=" * 80)
    print("NEURAL TOKENIZER V2 — BOUNDARY ALIGNMENT")
    print("=" * 80)

    print(f"Device:              {DEVICE}")

    print(f"Seed:                {SEED}")

    print(f"Checkpoint:          {CHECKPOINT}")

    print(f"Batch size:          {BATCH_SIZE}")

    print(f"Alignment threshold: {BOUNDARY_THRESHOLD}")

    print(f"Position tolerance:  {BOUNDARY_TOLERANCE} byte(s)")

    if torch.cuda.is_available():

        print(f"GPU:                 " f"{torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed(SEED)

    # --------------------------------------------------------
    # Load TinyStories
    # --------------------------------------------------------

    print("\nLoading TinyStories...")

    dataset = load_dataset("roneneldan/TinyStories")

    valid_texts = [
        row["text"] for row in dataset["validation"].select(range(VALID_SIZE))
    ]

    print(f"Validation stories: " f"{len(valid_texts):,}")

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    byte_tokenizer = ByteTokenizer()

    vocab_size = 259

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("\nBuilding alignment dataset...")
    print("-" * 60)

    alignment_dataset = BoundaryAlignmentDataset(
        texts=valid_texts,
        source_length=SOURCE_LENGTH,
        max_samples=MAX_ALIGNMENT_SAMPLES,
    )

    print(f"Alignment samples: " f"{len(alignment_dataset):,}")

    print(f"Source length: " f"{SOURCE_LENGTH}")

    print(f"Maximum source bytes: " f"{len(alignment_dataset) * SOURCE_LENGTH:,}")

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    loader = DataLoader(
        alignment_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = load_model(vocab_size=vocab_size)

    # --------------------------------------------------------
    # GPU diagnostic
    # --------------------------------------------------------

    if torch.cuda.is_available():

        torch.cuda.synchronize()

        allocated_before = torch.cuda.memory_allocated() / 1024**2

        reserved_before = torch.cuda.memory_reserved() / 1024**2

        print()
        print("GPU memory before inference")
        print("-" * 60)

        print(f"Allocated: " f"{allocated_before:.1f} MB")

        print(f"Reserved:  " f"{reserved_before:.1f} MB")

    # --------------------------------------------------------
    # Boundary prediction
    # --------------------------------------------------------

    print()
    print("=" * 80)

    print("BOUNDARY-ONLY GPU INFERENCE")

    print("=" * 80)

    probabilities, bpe_mask = predict_boundaries(
        model=model,
        loader=loader,
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    neural_stats = calculate_boundary_statistics(probabilities)

    bpe_boundaries = extract_bpe_boundaries(bpe_mask)

    bpe_stats = calculate_bpe_boundary_statistics(bpe_boundaries)

    neural_boundaries = extract_neural_boundaries(
        probabilities,
        threshold=BOUNDARY_THRESHOLD,
    )

    # --------------------------------------------------------
    # Alignment
    # --------------------------------------------------------

    metrics = calculate_alignment_metrics(
        neural_boundaries=neural_boundaries,
        bpe_boundaries=bpe_boundaries,
        tolerance=BOUNDARY_TOLERANCE,
    )

    # --------------------------------------------------------
    # GPU diagnostic after inference
    # --------------------------------------------------------

    if torch.cuda.is_available():

        torch.cuda.synchronize()

        allocated_after = torch.cuda.memory_allocated() / 1024**2

        reserved_after = torch.cuda.memory_reserved() / 1024**2

        print()
        print("GPU memory after inference")
        print("-" * 60)

        print(f"Allocated: " f"{allocated_after:.1f} MB")

        print(f"Reserved:  " f"{reserved_after:.1f} MB")

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("=" * 80)

    print("BOUNDARY ALIGNMENT RESULTS")

    print("=" * 80)

    print()
    print("Neural V2")

    print("-" * 60)

    print(f"Soft mean probability: " f"{neural_stats['soft_mean']:.6f}")

    print(f"Soft std:              " f"{neural_stats['soft_std']:.6f}")

    print(f"Soft min:              " f"{neural_stats['soft_min']:.6f}")

    print(f"Soft max:              " f"{neural_stats['soft_max']:.6f}")

    print(f"Hard boundary rate:    " f"{neural_stats['hard_boundary_rate']:.6f}")

    print(f"Hard symbols/token:    " f"{neural_stats['hard_symbols_per_token']:.3f}")

    print(f"Entropy:               " f"{neural_stats['entropy']:.6f} nats")

    print()
    print("BPE")

    print("-" * 60)

    print(f"Mean boundaries/window: " f"{bpe_stats['mean_boundaries_per_window']:.3f}")

    print(f"Mean symbols/token:     " f"{bpe_stats['mean_symbols_per_token']:.3f}")

    print()
    print("Boundary alignment")

    print("-" * 60)

    print(f"Precision:             " f"{metrics['precision']:.4f}")

    print(f"Recall:                " f"{metrics['recall']:.4f}")

    print(f"F1:                    " f"{metrics['f1']:.4f}")

    print(f"True positives:        " f"{metrics['true_positives']:,}")

    print(f"False positives:       " f"{metrics['false_positives']:,}")

    print(f"False negatives:       " f"{metrics['false_negatives']:,}")

    if metrics["mean_boundary_distance"] is not None:

        print(
            f"Mean boundary distance: " f"{metrics['mean_boundary_distance']:.3f} bytes"
        )

        print(
            f"Median boundary distance:"
            f" {metrics['median_boundary_distance']:.3f} bytes"
        )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_probability_histogram(
        probabilities,
        OUTPUT_DIR,
    )

    save_boundary_alignment_examples(
        probabilities,
        bpe_mask,
        OUTPUT_DIR,
        num_examples=5,
    )

    save_metrics_plot(
        metrics,
        OUTPUT_DIR,
    )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    results = {
        "experiment": ("neural_v2_boundary_alignment"),
        "seed": SEED,
        "device": str(DEVICE),
        "checkpoint": str(CHECKPOINT),
        "source_length": SOURCE_LENGTH,
        "group_size": GROUP_SIZE,
        "num_tokens": NUM_TOKENS,
        "batch_size": BATCH_SIZE,
        "max_alignment_samples": (MAX_ALIGNMENT_SAMPLES),
        "boundary_threshold": (BOUNDARY_THRESHOLD),
        "boundary_tolerance_bytes": (BOUNDARY_TOLERANCE),
        "neural_v2": neural_stats,
        "bpe": bpe_stats,
        "alignment": metrics,
    }

    with (OUTPUT_DIR / "boundary_alignment_results.json").open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 80)

    print("BOUNDARY ALIGNMENT COMPLETE")

    print("=" * 80)

    print()
    print("Configuration        Value")

    print("-" * 60)

    print(f"{'Neural threshold':<24}" f"{BOUNDARY_THRESHOLD}")

    print(f"{'Tolerance (bytes)':<24}" f"{BOUNDARY_TOLERANCE}")

    print(
        f"{'Neural symbols/token':<24}" f"{neural_stats['hard_symbols_per_token']:.3f}"
    )

    print(f"{'BPE symbols/token':<24}" f"{bpe_stats['mean_symbols_per_token']:.3f}")

    print(f"{'Boundary precision':<24}" f"{metrics['precision']:.4f}")

    print(f"{'Boundary recall':<24}" f"{metrics['recall']:.4f}")

    print(f"{'Boundary F1':<24}" f"{metrics['f1']:.4f}")

    print()
    print(f"Results saved to: " f"{OUTPUT_DIR}")


if __name__ == "__main__":

    main()
