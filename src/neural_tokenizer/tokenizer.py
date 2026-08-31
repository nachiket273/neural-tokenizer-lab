from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = [
    "<pad>",
    "<bos>",
    "<eos>",
]


class BPETokenizer:
    """Lossless ByteLevel BPE tokenizer."""

    def __init__(self, tokenizer_path: str | Path):
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))

        self.pad_id = self.tokenizer.token_to_id("<pad>")
        self.bos_id = self.tokenizer.token_to_id("<bos>")
        self.eos_id = self.tokenizer.token_to_id("<eos>")

        if self.eos_id is None:
            raise ValueError("BPE tokenizer has no <eos> token")

        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids + [self.eos_id]

    def decode(self, ids: list[int]) -> str:
        ids = [
            token_id
            for token_id in ids
            if token_id
            not in {
                self.pad_id,
                self.bos_id,
                self.eos_id,
            }
        ]

        return self.tokenizer.decode(ids)

    def byte_length(self, text: str) -> int:
        """Return exact UTF-8 byte length of original text."""
        return len(text.encode("utf-8"))


class ByteTokenizer:
    """Lossless UTF-8 byte tokenizer."""

    EOS_ID = 256
    BOS_ID = 257
    PAD_ID = 258

    def __init__(self):
        self.pad_id = self.PAD_ID
        self.bos_id = self.BOS_ID
        self.eos_id = self.EOS_ID

        self.vocab_size = 259

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8")) + [self.eos_id]

    def decode(self, ids: list[int]) -> str:
        ids = [
            token_id
            for token_id in ids
            if token_id
            not in {
                self.pad_id,
                self.bos_id,
                self.eos_id,
            }
        ]

        return bytes(ids).decode("utf-8")

    def byte_length(self, text: str) -> int:
        """Return exact UTF-8 byte length of original text."""
        return len(text.encode("utf-8"))


def train_bpe(
    texts: list[str],
    vocab_size: int = 8192,
    output_path: str | Path = "artifacts/bpe/tokenizer.json",
) -> BPETokenizer:
    """Train a lossless ByteLevel BPE tokenizer."""

    tokenizer = Tokenizer(BPE())

    tokenizer.pre_tokenizer = ByteLevel(
        add_prefix_space=False,
    )

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=ByteLevel.alphabet(),
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    tokenizer.train_from_iterator(
        texts,
        trainer=trainer,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer.save(str(output_path))

    return BPETokenizer(output_path)
