import json
from pathlib import Path


class VocabIndex:
    """Provide lookup utilities for a model vocabulary."""

    def __init__(self, vocab_path: str) -> None:
        path = Path(vocab_path)

        if not path.exists():
            raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

        try:
            content = path.read_text(encoding="utf-8")
            vocab = json.loads(content)
        except json.JSONDecodeError as er:
            raise ValueError(f"Invalid JSON in vocab file: {er}") from er

        if not isinstance(vocab, dict):
            raise ValueError("Vocab file must contain a JSON object")

        self.token_to_id: dict[str, int] = vocab
        self.id_to_token = {
            token_id: token
            for token, token_id in vocab.items()
        }
        self.vocab_size = len(vocab)

    def get_numeric_token_ids(self) -> set[int]:
        """Return ids of tokens containing only digits."""
        return {
            token_id
            for token, token_id in self.token_to_id.items()
            if token.lstrip("Ġ").isdigit()
        }

    def get_token_id(self, token_str: str) -> int:
        """Return the id of a token, or -1 if it does not exist."""
        return self.token_to_id.get(token_str, -1)

    def get_number_end_token_ids(self) -> set[int]:
        """Return ids of tokens that can terminate a JSON number value."""
        candidates = [",", "}", " "]
        result: set[int] = set()

        for candidate in candidates:
            token_id = self.token_to_id.get(candidate)

            if token_id is not None:
                result.add(token_id)

        return result
