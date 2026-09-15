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

BATCH_SIZE = 64
MAX_ALIGNMENT_SAMPLES = 500

BOUNDARY_THRESHOLD = 0.5

# Tolerance sweep, in source-byte positions.
TOLERANCES = [0, 1, 2, 3, 4, 5]

# Number of Monte-Carlo random boundary configurations.
# Increase to 500/1000 for final paper-quality statistics.
N_RANDOM_NULLS = 200

CHECKPOINT = Path("experiments/neural_v2/best.pt")
BPE_TOKENIZER_PATH = Path("notebooks/artifacts/bpe/tokenizer.json")

OUTPUT_DIR = Path(
    "experiments/neural_v2/boundary_null_analysis"
)

TOKENIZER_EMBEDDING_DIM = 64
BOUNDARY_HIDDEN_DIM = 128
BOUNDARY_KERNEL_SIZE = 5
POOLING_TEMPERATURE = 0.75

D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
D_FF = 512
GRU_HIDDEN_DIM = 128

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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
    Same validation-window construction as the uploaded
    Neural V2 boundary-alignment experiment.

    Each sample contains:
        x:
            1024 byte/EOS source symbols

        bpe_boundaries:
            binary vector [1024]
    """

    def __init__(
        self,
        texts: list[str],
        source_length: int = 1024,
        max_samples: int | None = None,
    ) -> None:
        self.source_length = source_length

        ByteTokenizer()
        bpe_tokenizer = BPETokenizer(
            str(BPE_TOKENIZER_PATH)
        )

        byte_stream: list[int] = []
        boundary_stream: list[int] = []

        for text in texts:
            raw_bytes = text.encode("utf-8")
            start_offset = len(byte_stream)

            byte_stream.extend(raw_bytes)

            encoding = bpe_tokenizer.tokenizer.encode(text)

            for offset_start, offset_end in encoding.offsets:
                if offset_end <= offset_start:
                    continue

                byte_end = len(
                    text[:offset_end].encode("utf-8")
                )

                absolute_boundary = (
                    start_offset + byte_end - 1
                )

                boundary_stream.append(
                    absolute_boundary
                )

            # ByteTokenizer EOS.
            byte_stream.append(256)
            boundary_stream.append(
                len(byte_stream) - 1
            )

        total_length = len(byte_stream)

        boundary_mask = np.zeros(
            total_length,
            dtype=np.uint8,
        )

        for position in boundary_stream:
            if 0 <= position < total_length:
                boundary_mask[position] = 1

        self.samples: list[
            tuple[np.ndarray, np.ndarray]
        ] = []

        max_start = total_length - source_length

        if max_start < 0:
            raise ValueError(
                "Validation corpus is shorter than "
                f"source_length={source_length}."
            )

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

            boundaries = boundary_mask[
                start:end
            ].copy()

            self.samples.append(
                (x, boundaries)
            )

            if (
                max_samples is not None
                and len(self.samples) >= max_samples
            ):
                break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
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
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    model = NeuralV2LanguageModel(
        input_vocab_size=vocab_size,
        output_vocab_size=vocab_size,
        source_length=SOURCE_LENGTH,
        num_tokens=NUM_TOKENS,
        tokenizer_embedding_dim=(
            TOKENIZER_EMBEDDING_DIM
        ),
        boundary_hidden_dim=(
            BOUNDARY_HIDDEN_DIM
        ),
        boundary_kernel_size=(
            BOUNDARY_KERNEL_SIZE
        ),
        pooling_temperature=(
            POOLING_TEMPERATURE
        ),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        gru_hidden_dim=GRU_HIDDEN_DIM,
        group_size=GROUP_SIZE,
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    total_parameters = sum(
        p.numel() for p in model.parameters()
    )

    tokenizer_parameters = sum(
        p.numel()
        for p in model.neural_tokenizer.parameters()
    )

    print(f"Parameters: {total_parameters:,}")
    print(f"Tokenizer:  {tokenizer_parameters:,}")
    print(f"Device:     {DEVICE}")

    return model


# ============================================================
# Boundary inference
# ============================================================

@torch.no_grad()
def predict_boundaries(
    model: NeuralV2LanguageModel,
    loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Boundary-only GPU inference.

    We call only the boundary predictor rather than the
    complete Neural V2 language model.
    """

    all_probs: list[np.ndarray] = []
    all_bpe: list[np.ndarray] = []

    boundary_predictor = (
        model.neural_tokenizer.boundary_predictor
    )
    boundary_predictor.eval()

    start_time = time.perf_counter()
    total_samples = 0

    for x, bpe_boundaries in loader:
        x = x.to(
            DEVICE,
            non_blocking=True,
        )

        _, boundary_probs = boundary_predictor(x)

        all_probs.append(
            boundary_probs.cpu().numpy()
        )
        all_bpe.append(
            bpe_boundaries.numpy()
        )

        total_samples += x.shape[0]

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time

    probabilities = np.concatenate(
        all_probs,
        axis=0,
    )

    bpe = np.concatenate(
        all_bpe,
        axis=0,
    )

    print()
    print(
        f"Boundary inference samples: "
        f"{total_samples:,}"
    )
    print(
        f"Boundary inference time: "
        f"{elapsed:.3f} s"
    )
    print(
        f"Samples/sec: "
        f"{total_samples / max(elapsed, 1e-12):.1f}"
    )

    return probabilities, bpe


# ============================================================
# Boundary extraction
# ============================================================

def extract_neural_boundaries(
    probabilities: np.ndarray,
    threshold: float,
) -> list[np.ndarray]:
    return [
        np.flatnonzero(row >= threshold)
        for row in probabilities
    ]


def extract_bpe_boundaries(
    boundary_masks: np.ndarray,
) -> list[np.ndarray]:
    return [
        np.flatnonzero(row > 0)
        for row in boundary_masks
    ]


# ============================================================
# One-to-one boundary matching
# ============================================================

def match_boundaries(
    predicted: np.ndarray,
    reference: np.ndarray,
    tolerance: int,
) -> tuple[int, int, int, list[int]]:
    """
    Greedy one-to-one matching.

    Both arrays are sorted boundary positions.

    A prediction can match at most one reference boundary,
    and a reference boundary can match at most one prediction.

    Returns:
        TP, FP, FN, matched distances
    """

    if len(predicted) == 0:
        return (
            0,
            0,
            len(reference),
            [],
        )

    if len(reference) == 0:
        return (
            0,
            len(predicted),
            0,
            [],
        )

    matched_reference = np.zeros(
        len(reference),
        dtype=bool,
    )

    tp = 0
    distances: list[int] = []

    # The number of boundaries is small (~250/window), so
    # this direct nearest-candidate implementation is fast
    # enough and keeps the matching definition transparent.
    for p in predicted:
        left = max(
            0,
            int(
                np.searchsorted(
                    reference,
                    p - tolerance,
                    side="left",
                )
            ),
        )

        right = min(
            len(reference),
            int(
                np.searchsorted(
                    reference,
                    p + tolerance,
                    side="right",
                )
            ),
        )

        best_idx = None
        best_distance = None

        for idx in range(left, right):
            if matched_reference[idx]:
                continue

            distance = abs(
                int(reference[idx]) - int(p)
            )

            if (
                best_distance is None
                or distance < best_distance
            ):
                best_distance = distance
                best_idx = idx

        if best_idx is not None:
            matched_reference[best_idx] = True
            tp += 1
            distances.append(
                int(best_distance)
            )

    fp = len(predicted) - tp
    fn = len(reference) - tp

    return (
        tp,
        fp,
        fn,
        distances,
    )


# ============================================================
# Aggregate metrics
# ============================================================

def calculate_metrics(
    predicted: list[np.ndarray],
    reference: list[np.ndarray],
    tolerance: int,
) -> dict:
    total_tp = 0
    total_fp = 0
    total_fn = 0

    distances: list[int] = []

    for pred, ref in zip(
        predicted,
        reference,
    ):
        tp, fp, fn, matched_distances = (
            match_boundaries(
                pred,
                ref,
                tolerance,
            )
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn
        distances.extend(
            matched_distances
        )

    precision = (
        total_tp
        / (total_tp + total_fp)
        if total_tp + total_fp > 0
        else 0.0
    )

    recall = (
        total_tp
        / (total_tp + total_fn)
        if total_tp + total_fn > 0
        else 0.0
    )

    f1 = (
        2.0 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positives": int(total_tp),
        "false_positives": int(total_fp),
        "false_negatives": int(total_fn),
        "mean_boundary_distance": (
            float(np.mean(distances))
            if distances
            else None
        ),
        "median_boundary_distance": (
            float(np.median(distances))
            if distances
            else None
        ),
    }


# ============================================================
# Null models
# ============================================================

def random_count_preserving_boundaries(
    observed: list[np.ndarray],
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """
    Null model A:

    Preserve the number of neural boundaries in every
    individual window, but destroy their positional structure.

    This is the primary null model.

    It answers:

        "Could the observed Neural/BPE alignment arise
         simply because Neural V2 predicts ~the right number
         of boundaries?"
    """

    null_boundaries: list[np.ndarray] = []

    positions = np.arange(SOURCE_LENGTH)

    for row in observed:
        count = len(row)

        if count == 0:
            null_boundaries.append(
                np.empty(0, dtype=np.int64)
            )
            continue

        count = min(count, SOURCE_LENGTH)

        sampled = rng.choice(
            positions,
            size=count,
            replace=False,
        )

        sampled.sort()

        null_boundaries.append(
            sampled.astype(np.int64)
        )

    return null_boundaries


def cyclic_shift_boundaries(
    observed: list[np.ndarray],
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """
    Null model B:

    Preserve the exact learned spacing pattern within each
    window, but randomly translate the complete boundary
    pattern around the 1024-byte window.

    This is stricter than the random-position null because
    local spacing statistics are preserved.
    """

    shifted: list[np.ndarray] = []

    for row in observed:
        if len(row) == 0:
            shifted.append(
                np.empty(0, dtype=np.int64)
            )
            continue

        shift = int(
            rng.integers(
                0,
                SOURCE_LENGTH,
            )
        )

        shifted_positions = (
            (row + shift) % SOURCE_LENGTH
        )

        shifted_positions.sort()

        shifted.append(
            shifted_positions.astype(
                np.int64
            )
        )

    return shifted


def regular_four_byte_boundaries(
    num_samples: int,
) -> list[np.ndarray]:
    """
    Deterministic baseline:

        boundary after every 4 source symbols.

    Positions:
        3, 7, 11, ..., 1023
    """

    regular = np.arange(
        GROUP_SIZE - 1,
        SOURCE_LENGTH,
        GROUP_SIZE,
        dtype=np.int64,
    )

    return [
        regular.copy()
        for _ in range(num_samples)
    ]


# ============================================================
# Tolerance sweep
# ============================================================

def evaluate_tolerance_sweep(
    observed: list[np.ndarray],
    bpe: list[np.ndarray],
    tolerances: list[int],
) -> dict:
    results = {}

    for tolerance in tolerances:
        results[str(tolerance)] = (
            calculate_metrics(
                observed,
                bpe,
                tolerance,
            )
        )

    return results


def evaluate_null_model(
    observed: list[np.ndarray],
    bpe: list[np.ndarray],
    tolerances: list[int],
    n_trials: int,
    seed: int,
) -> dict:
    """
    Monte-Carlo evaluation of the count-preserving random null.

    For every trial, boundary counts are preserved independently
    for every window.
    """

    rng = np.random.default_rng(seed)

    trial_values = {
        tolerance: []
        for tolerance in tolerances
    }

    start_time = time.perf_counter()

    for trial in range(n_trials):
        null_boundaries = (
            random_count_preserving_boundaries(
                observed,
                rng,
            )
        )

        for tolerance in tolerances:
            metrics = calculate_metrics(
                null_boundaries,
                bpe,
                tolerance,
            )

            trial_values[tolerance].append(
                metrics
            )

        if (
            (trial + 1) % 25 == 0
            or trial == 0
            or trial + 1 == n_trials
        ):
            elapsed = (
                time.perf_counter()
                - start_time
            )
            print(
                f"  Random null trial "
                f"{trial + 1}/{n_trials} "
                f"({elapsed:.1f}s)"
            )

    summary = {}

    for tolerance in tolerances:
        f1_values = np.asarray(
            [
                item["f1"]
                for item in trial_values[
                    tolerance
                ]
            ]
        )

        precision_values = np.asarray(
            [
                item["precision"]
                for item in trial_values[
                    tolerance
                ]
            ]
        )

        recall_values = np.asarray(
            [
                item["recall"]
                for item in trial_values[
                    tolerance
                ]
            ]
        )

        summary[str(tolerance)] = {
            "n_trials": int(n_trials),
            "f1_mean": float(
                f1_values.mean()
            ),
            "f1_std": float(
                f1_values.std(ddof=1)
            ) if n_trials > 1 else 0.0,
            "f1_p05": float(
                np.percentile(
                    f1_values,
                    5,
                )
            ),
            "f1_p50": float(
                np.percentile(
                    f1_values,
                    50,
                )
            ),
            "f1_p95": float(
                np.percentile(
                    f1_values,
                    95,
                )
            ),
            "precision_mean": float(
                precision_values.mean()
            ),
            "recall_mean": float(
                recall_values.mean()
            ),
        }

    return summary


# ============================================================
# Boundary statistics
# ============================================================

def calculate_boundary_statistics(
    probabilities: np.ndarray,
) -> dict:
    hard = (
        probabilities
        >= BOUNDARY_THRESHOLD
    )

    rate = hard.mean()

    eps = 1e-7
    p = np.clip(
        probabilities,
        eps,
        1.0 - eps,
    )

    entropy = -(
        p * np.log(p)
        + (1.0 - p) * np.log(1.0 - p)
    )

    return {
        "soft_mean": float(
            probabilities.mean()
        ),
        "soft_std": float(
            probabilities.std()
        ),
        "hard_boundary_rate": float(
            rate
        ),
        "hard_symbols_per_token": float(
            1.0 / max(rate, 1e-8)
        ),
        "entropy": float(
            entropy.mean()
        ),
    }


# ============================================================
# Plots
# ============================================================

def save_tolerance_plot(
    observed_results: dict,
    random_results: dict,
    cyclic_results: dict,
    regular_results: dict,
    output_dir: Path,
) -> None:
    x = np.asarray(
        TOLERANCES,
        dtype=float,
    )

    observed_f1 = np.asarray(
        [
            observed_results[str(t)]["f1"]
            for t in TOLERANCES
        ]
    )

    random_mean = np.asarray(
        [
            random_results[str(t)]["f1_mean"]
            for t in TOLERANCES
        ]
    )

    random_p05 = np.asarray(
        [
            random_results[str(t)]["f1_p05"]
            for t in TOLERANCES
        ]
    )

    random_p95 = np.asarray(
        [
            random_results[str(t)]["f1_p95"]
            for t in TOLERANCES
        ]
    )

    cyclic_f1 = np.asarray(
        [
            cyclic_results[str(t)]["f1"]
            for t in TOLERANCES
        ]
    )

    regular_f1 = np.asarray(
        [
            regular_results[str(t)]["f1"]
            for t in TOLERANCES
        ]
    )

    plt.figure(figsize=(9, 6))

    plt.plot(
        x,
        observed_f1,
        marker="o",
        label="Learned Neural V2",
    )

    plt.plot(
        x,
        random_mean,
        marker="o",
        label="Random count-preserving null",
    )

    plt.fill_between(
        x,
        random_p05,
        random_p95,
        alpha=0.2,
        label="Random null 5–95%",
    )

    plt.plot(
        x,
        cyclic_f1,
        marker="o",
        label="Cyclic-shift null",
    )

    plt.plot(
        x,
        regular_f1,
        marker="o",
        label="Regular 4-byte grid",
    )

    plt.xlabel("Position tolerance (bytes)")
    plt.ylabel("Boundary F1")
    plt.title(
        "Neural V2 boundary alignment vs null models"
    )
    plt.xticks(TOLERANCES)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir
        / "boundary_alignment_tolerance.png",
        dpi=150,
    )
    plt.close()


def save_null_distribution_plot(
    observed_results: dict,
    observed: list[np.ndarray],
    bpe: list[np.ndarray],
    output_dir: Path,
) -> None:
    """
    Distribution of F1 under the count-preserving null at
    tolerance = 1 byte.

    This is the main statistical sanity-check figure.
    """

    tolerance = 1
    rng = np.random.default_rng(
        SEED + 1000
    )

    values = []

    # Use a separate, moderately large distribution for
    # visualization. This is independent of N_RANDOM_NULLS.
    n_plot_trials = max(
        N_RANDOM_NULLS,
        500,
    )

    for _ in range(n_plot_trials):
        null = (
            random_count_preserving_boundaries(
                observed,
                rng,
            )
        )

        metrics = calculate_metrics(
            null,
            bpe,
            tolerance,
        )

        values.append(metrics["f1"])

    observed_f1 = observed_results[
        str(tolerance)
    ]["f1"]

    plt.figure(figsize=(9, 5))

    plt.hist(
        values,
        bins=30,
        alpha=0.8,
    )

    plt.axvline(
        observed_f1,
        linestyle="--",
        linewidth=2,
        label=(
            f"Observed Neural V2 "
            f"F1 = {observed_f1:.4f}"
        ),
    )

    plt.xlabel(
        "F1 under count-preserving random null"
    )
    plt.ylabel("Count")
    plt.title(
        "Neural V2 boundary alignment vs "
        "random null (tolerance = 1 byte)"
    )
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir
        / "boundary_null_distribution_tolerance_1.png",
        dpi=150,
    )
    plt.close()


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 80)
    print(
        "NEURAL TOKENIZER V2 — "
        "BOUNDARY NULL + TOLERANCE ANALYSIS"
    )
    print("=" * 80)

    print(f"Device:              {DEVICE}")
    print(f"Seed:                {SEED}")
    print(f"Checkpoint:          {CHECKPOINT}")
    print(f"Batch size:          {BATCH_SIZE}")
    print(
        f"Alignment threshold: "
        f"{BOUNDARY_THRESHOLD}"
    )
    print(
        f"Tolerances:          "
        f"{TOLERANCES}"
    )
    print(
        f"Random null trials:   "
        f"{N_RANDOM_NULLS}"
    )

    if torch.cuda.is_available():
        print(
            f"GPU:                 "
            f"{torch.cuda.get_device_name(0)}"
        )

    set_seed(SEED)

    # --------------------------------------------------------
    # Load TinyStories
    # --------------------------------------------------------

    print("\nLoading TinyStories...")

    dataset = load_dataset(
        "roneneldan/TinyStories"
    )

    valid_texts = [
        row["text"]
        for row in dataset["validation"].select(
            range(VALID_SIZE)
        )
    ]

    print(
        f"Validation stories: "
        f"{len(valid_texts):,}"
    )

    # --------------------------------------------------------
    # Build exactly the same alignment dataset
    # --------------------------------------------------------

    print("\nBuilding alignment dataset...")
    print("-" * 60)

    alignment_dataset = BoundaryAlignmentDataset(
        texts=valid_texts,
        source_length=SOURCE_LENGTH,
        max_samples=MAX_ALIGNMENT_SAMPLES,
    )

    print(
        f"Alignment samples: "
        f"{len(alignment_dataset):,}"
    )
    print(
        f"Source length:     "
        f"{SOURCE_LENGTH}"
    )
    print(
        f"Maximum source bytes: "
        f"{len(alignment_dataset) * SOURCE_LENGTH:,}"
    )

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

    model = load_model(
        vocab_size=259
    )

    # --------------------------------------------------------
    # Boundary inference
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("BOUNDARY-ONLY GPU INFERENCE")
    print("=" * 80)

    probabilities, bpe_mask = (
        predict_boundaries(
            model,
            loader,
        )
    )

    neural_boundaries = (
        extract_neural_boundaries(
            probabilities,
            BOUNDARY_THRESHOLD,
        )
    )

    bpe_boundaries = (
        extract_bpe_boundaries(
            bpe_mask
        )
    )

    neural_stats = (
        calculate_boundary_statistics(
            probabilities
        )
    )

    # --------------------------------------------------------
    # Baseline sanity check
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("BASELINE ALIGNMENT SANITY CHECK")
    print("=" * 80)

    baseline = calculate_metrics(
        neural_boundaries,
        bpe_boundaries,
        tolerance=1,
    )

    print(
        f"Observed F1 @ tolerance=1: "
        f"{baseline['f1']:.6f}"
    )
    print(
        f"Observed precision:         "
        f"{baseline['precision']:.6f}"
    )
    print(
        f"Observed recall:            "
        f"{baseline['recall']:.6f}"
    )

    # This should reproduce the uploaded script's result
    # approximately:
    #
    #   precision ~ 0.4793
    #   recall    ~ 0.4717
    #   F1        ~ 0.4755
    #
    # Small differences would indicate a construction or
    # matching change and should be investigated.

    # --------------------------------------------------------
    # Observed tolerance sweep
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("OBSERVED TOLERANCE SWEEP")
    print("=" * 80)

    observed_results = (
        evaluate_tolerance_sweep(
            neural_boundaries,
            bpe_boundaries,
            TOLERANCES,
        )
    )

    for tolerance in TOLERANCES:
        result = observed_results[
            str(tolerance)
        ]

        print(
            f"Tolerance ±{tolerance} byte(s): "
            f"Precision={result['precision']:.4f} "
            f"Recall={result['recall']:.4f} "
            f"F1={result['f1']:.4f}"
        )

    # --------------------------------------------------------
    # Cyclic-shift null
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("CYCLIC-SHIFT NULL")
    print("=" * 80)

    shift_rng = np.random.default_rng(
        SEED + 2000
    )

    shifted_boundaries = (
        cyclic_shift_boundaries(
            neural_boundaries,
            shift_rng,
        )
    )

    cyclic_results = (
        evaluate_tolerance_sweep(
            shifted_boundaries,
            bpe_boundaries,
            TOLERANCES,
        )
    )

    for tolerance in TOLERANCES:
        result = cyclic_results[
            str(tolerance)
        ]

        print(
            f"Tolerance ±{tolerance} byte(s): "
            f"Precision={result['precision']:.4f} "
            f"Recall={result['recall']:.4f} "
            f"F1={result['f1']:.4f}"
        )

    # --------------------------------------------------------
    # Regular grid baseline
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("REGULAR 4-BYTE GRID BASELINE")
    print("=" * 80)

    regular_boundaries = (
        regular_four_byte_boundaries(
            len(neural_boundaries)
        )
    )

    regular_results = (
        evaluate_tolerance_sweep(
            regular_boundaries,
            bpe_boundaries,
            TOLERANCES,
        )
    )

    for tolerance in TOLERANCES:
        result = regular_results[
            str(tolerance)
        ]

        print(
            f"Tolerance ±{tolerance} byte(s): "
            f"Precision={result['precision']:.4f} "
            f"Recall={result['recall']:.4f} "
            f"F1={result['f1']:.4f}"
        )

    # --------------------------------------------------------
    # Count-preserving random null
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "COUNT-PRESERVING RANDOM NULL "
        f"({N_RANDOM_NULLS} TRIALS)"
    )
    print("=" * 80)

    random_results = evaluate_null_model(
        neural_boundaries,
        bpe_boundaries,
        TOLERANCES,
        n_trials=N_RANDOM_NULLS,
        seed=SEED + 3000,
    )

    for tolerance in TOLERANCES:
        result = random_results[
            str(tolerance)
        ]

        print(
            f"Tolerance ±{tolerance} byte(s): "
            f"F1={result['f1_mean']:.4f} "
            f"± {result['f1_std']:.4f} "
            f"[{result['f1_p05']:.4f}, "
            f"{result['f1_p95']:.4f}]"
        )

    # --------------------------------------------------------
    # Empirical p-values
    # --------------------------------------------------------
    #
    # We compute:
    #
    #   p = P(F1_null >= F1_observed)
    #
    # using the Monte-Carlo null samples.
    #
    # The +1 correction avoids reporting p=0.
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("EMPIRICAL NULL SIGNIFICANCE")
    print("=" * 80)

    # Re-run trials once so we retain the actual samples for
    # empirical p-values. This avoids putting every individual
    # trial into the JSON result above.
    rng = np.random.default_rng(
        SEED + 4000
    )

    null_f1_samples = {
        tolerance: []
        for tolerance in TOLERANCES
    }

    for _ in range(N_RANDOM_NULLS):
        null_boundaries = (
            random_count_preserving_boundaries(
                neural_boundaries,
                rng,
            )
        )

        for tolerance in TOLERANCES:
            null_metrics = calculate_metrics(
                null_boundaries,
                bpe_boundaries,
                tolerance,
            )

            null_f1_samples[
                tolerance
            ].append(
                null_metrics["f1"]
            )

    significance = {}

    for tolerance in TOLERANCES:
        observed_f1 = observed_results[
            str(tolerance)
        ]["f1"]

        null_values = np.asarray(
            null_f1_samples[tolerance]
        )

        exceedances = int(
            np.sum(
                null_values >= observed_f1
            )
        )

        p_value = (
            exceedances + 1
        ) / (
            len(null_values) + 1
        )

        # A simple standardized effect size:
        # how many null standard deviations above the
        # null mean the observed result lies.
        null_mean = float(
            null_values.mean()
        )
        null_std = float(
            null_values.std(ddof=1)
        )

        z_score = (
            (observed_f1 - null_mean)
            / null_std
            if null_std > 0
            else None
        )

        significance[str(tolerance)] = {
            "observed_f1": float(
                observed_f1
            ),
            "null_mean": null_mean,
            "null_std": null_std,
            "empirical_p_value": float(
                p_value
            ),
            "exceedances": exceedances,
            "z_score": (
                float(z_score)
                if z_score is not None
                else None
            ),
        }

        print(
            f"Tolerance ±{tolerance}: "
            f"observed={observed_f1:.4f}, "
            f"null={null_mean:.4f}±{null_std:.4f}, "
            f"p={p_value:.5f}, "
            f"z={z_score:.2f}"
            if z_score is not None
            else
            f"Tolerance ±{tolerance}: "
            f"observed={observed_f1:.4f}, "
            f"null={null_mean:.4f}±{null_std:.4f}, "
            f"p={p_value:.5f}"
        )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_tolerance_plot(
        observed_results,
        random_results,
        cyclic_results,
        regular_results,
        OUTPUT_DIR,
    )

    save_null_distribution_plot(
        observed_results,
        neural_boundaries,
        bpe_boundaries,
        OUTPUT_DIR,
    )

    # --------------------------------------------------------
    # Save raw boundary positions
    # --------------------------------------------------------
    #
    # This is deliberate: future analyses should not need
    # to rerun GPU inference just to inspect the boundaries.
    # --------------------------------------------------------

    np.savez_compressed(
        OUTPUT_DIR
        / "boundary_positions.npz",
        neural_boundaries=np.asarray(
            [
                row.tolist()
                for row in neural_boundaries
            ],
            dtype=object,
        ),
        bpe_boundaries=np.asarray(
            [
                row.tolist()
                for row in bpe_boundaries
            ],
            dtype=object,
        ),
    )

    # --------------------------------------------------------
    # Summary table
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL TOLERANCE TABLE")
    print("=" * 80)

    print()
    print(
        "Tolerance | "
        "Observed F1 | "
        "Random F1 | "
        "Cyclic F1 | "
        "Regular F1 | "
        "Null p"
    )
    print("-" * 80)

    for tolerance in TOLERANCES:
        observed = observed_results[
            str(tolerance)
        ]["f1"]

        random_mean = random_results[
            str(tolerance)
        ]["f1_mean"]

        cyclic = cyclic_results[
            str(tolerance)
        ]["f1"]

        regular = regular_results[
            str(tolerance)
        ]["f1"]

        p_value = significance[
            str(tolerance)
        ]["empirical_p_value"]

        print(
            f"±{tolerance:^8d} | "
            f"{observed:^12.4f} | "
            f"{random_mean:^10.4f} | "
            f"{cyclic:^10.4f} | "
            f"{regular:^11.4f} | "
            f"{p_value:.5f}"
        )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    results = {
        "experiment": (
            "neural_v2_boundary_null_analysis"
        ),
        "seed": SEED,
        "device": str(DEVICE),
        "checkpoint": str(CHECKPOINT),
        "source_length": SOURCE_LENGTH,
        "group_size": GROUP_SIZE,
        "num_tokens": NUM_TOKENS,
        "batch_size": BATCH_SIZE,
        "max_alignment_samples": (
            MAX_ALIGNMENT_SAMPLES
        ),
        "boundary_threshold": (
            BOUNDARY_THRESHOLD
        ),
        "tolerances_bytes": TOLERANCES,
        "random_null_trials": N_RANDOM_NULLS,
        "neural_v2_statistics": neural_stats,
        "observed": observed_results,
        "count_preserving_random_null": (
            random_results
        ),
        "cyclic_shift_null": cyclic_results,
        "regular_four_byte_grid": (
            regular_results
        ),
        "empirical_significance": (
            significance
        ),
        "sanity_check_tolerance_1": {
            "precision": baseline[
                "precision"
            ],
            "recall": baseline[
                "recall"
            ],
            "f1": baseline["f1"],
        },
    }

    with (
        OUTPUT_DIR
        / "boundary_null_analysis_results.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print()
    print("=" * 80)
    print(
        "BOUNDARY NULL + TOLERANCE ANALYSIS COMPLETE"
    )
    print("=" * 80)

    print()
    print(
        f"Results saved to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
