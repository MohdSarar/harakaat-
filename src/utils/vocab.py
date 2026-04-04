"""
Character-level vocabulary for the diacritization model.
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import Counter
from typing import Optional

from src.utils import (
    SPECIAL_TOKENS, PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN,
    WORD_BOUNDARY, ARABIC_LETTERS, HARAKAT, strip_diacritics,
    DIAC_LABELS, DIAC_LABEL_TO_IDX, NUM_DIAC_CLASSES,
)


class CharVocab:
    """
    Character vocabulary mapping chars → indices.
    Built from undiacritized text (input side).
    """

    def __init__(self):
        self.char2idx: dict[str, int] = {}
        self.idx2char: dict[int, str] = {}
        # Reserve special tokens
        for tok in SPECIAL_TOKENS:
            idx = len(self.char2idx)
            self.char2idx[tok] = idx
            self.idx2char[idx] = tok

    @property
    def pad_idx(self) -> int:
        return self.char2idx[PAD_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.char2idx[UNK_TOKEN]

    @property
    def bos_idx(self) -> int:
        return self.char2idx[BOS_TOKEN]

    @property
    def eos_idx(self) -> int:
        return self.char2idx[EOS_TOKEN]

    @property
    def wb_idx(self) -> int:
        return self.char2idx[WORD_BOUNDARY]

    def __len__(self) -> int:
        return len(self.char2idx)

    def build_from_texts(self, texts: list[str], min_count: int = 1) -> None:
        """Build vocab from a list of undiacritized texts."""
        counter: Counter[str] = Counter()
        for text in texts:
            clean = strip_diacritics(text)
            counter.update(clean)

        for ch, count in counter.most_common():
            if count >= min_count and ch not in self.char2idx:
                idx = len(self.char2idx)
                self.char2idx[ch] = idx
                self.idx2char[idx] = ch

    def encode(self, text: str) -> list[int]:
        """Encode undiacritized text to list of indices."""
        return [self.char2idx.get(ch, self.unk_idx) for ch in text]

    def decode(self, indices: list[int]) -> str:
        """Decode indices back to text."""
        return "".join(
            self.idx2char.get(idx, "?")
            for idx in indices
            if idx not in (self.pad_idx, self.bos_idx, self.eos_idx)
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.char2idx, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> CharVocab:
        vocab = cls()
        with open(path, "r", encoding="utf-8") as f:
            vocab.char2idx = json.load(f)
        vocab.idx2char = {v: k for k, v in vocab.char2idx.items()}
        return vocab


# Diacritic label vocabulary is fixed (defined in utils/__init__.py)
# Access via DIAC_LABELS, DIAC_LABEL_TO_IDX, NUM_DIAC_CLASSES
