import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from src.neural_tokenizer.dataset import LanguageModelDataset
from src.neural_tokenizer.model import TinyTransformer
from src.neural_tokenizer.tokenizer import BPETokenizer, ByteTokenizer
from src.neural_tokenizer.training import train

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_SIZE = 50000
BATCH_SIZE = 32
CONTEXT_LENGTH = 256
STEPS = 100
LR = 3e-4


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def run_experiments(name, tokenizer, texts):
    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {name}")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print(f"Vocab size: {tokenizer.vocab_size}")

    # -------------------------------------------------
    # Dataset
    # -------------------------------------------------

    dataset = LanguageModelDataset(
        texts=texts,
        tokenizer=tokenizer,
        context_length=CONTEXT_LENGTH,
    )

    print(f"Dataset size: {len(dataset):,}")

    loader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    # -------------------------------------------------
    # Model
    # -------------------------------------------------

    model = TinyTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_seq_len=CONTEXT_LENGTH,
    )

    print(f"Vocabulary size: {tokenizer.vocab_size:,}")
    print(f"Parameters: {count_parameters(model):,}")

    # -------------------------------------------------
    # Training
    # -------------------------------------------------

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    history = train(
        model=model,
        loader=loader,
        optimizer=optimizer,
        device=DEVICE,
        steps=STEPS,
    )

    return {
        "name": name,
        "vocab_size": tokenizer.vocab_size,
        "parameters": count_parameters(model),
        "dataset_size": len(dataset),
        "history": history,
    }


def main():
    print("Loading TinyStories...")

    dataset = load_dataset("roneneldan/TinyStories")

    train_dataset = dataset["train"].select(range(TRAIN_SIZE))

    texts = train_dataset["text"]

    # -------------------------------------------------
    # Tokenizers
    # -------------------------------------------------\
    bpe_tokenizer = BPETokenizer(Tokenizer.from_file("artifacts/bpe/tokenizer.json"))

    byte_tokenizer = ByteTokenizer()

    # -------------------------------------------------
    # Experiments
    # -------------------------------------------------
    results = []

    results.append(
        run_experiments(
            name="BPE",
            tokenizer=bpe_tokenizer,
            texts=texts,
        )
    )

    results.append(
        run_experiments(
            name="BYTE",
            tokenizer=byte_tokenizer,
            texts=texts,
        )
    )

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for result in results:
        history = result["history"]

        print(f"\n{result['name']}")
        print("-" * 40)
        print(f"Vocabulary: " f"{result['vocab_size']:,}")
        print(f"Parameters: " f"{result['parameters']:,}")
        print(f"Dataset samples: " f"{result['dataset_size']:,}")

        if history:
            print(f"Initial loss: " f"{history[0]['loss']:.4f}")
            print(f"Final loss: " f"{history[-1]['loss']:.4f}")


if __name__ == "__main__":
    main()
