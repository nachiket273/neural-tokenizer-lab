from tokenizers import Tokenizer

BYTE_EOS = 256
BYTE_VOCAB_SIZE = 257


class ByteTokenizer:
    vocab_size = BYTE_VOCAB_SIZE

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8")) + [BYTE_EOS]

    def decode(self, ids: list[int]) -> str:
        ids = [i for i in ids if i != BYTE_EOS]
        return bytes(ids).decode("utf-8", errors="replace")


class BPETokenizer:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.eos_id = tokenizer.token_to_id("<eos>")

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids + [self.eos_id]

    def decode(self, ids: list[int]) -> str:
        ids = [i for i in ids if i != self.eos_id]
        return self.tokenizer.decode(ids)
