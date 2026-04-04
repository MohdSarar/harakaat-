"""
PyTorch Dataset for Arabic diacritization.
Handles char-level encoding with diacritic labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from src.utils import (
    strip_diacritics, extract_diacritics, normalize_diac_sequence,
    DIAC_LABEL_TO_IDX, is_arabic_char, word_boundaries,
)
from src.utils.vocab import CharVocab


class DiacritizationDataset(Dataset):
    """
    Dataset that produces (input_chars, diac_labels, word_end_mask) tuples.
    
    - input_chars: indices of undiacritized characters
    - diac_labels: target diacritic class per character
    - word_end_mask: boolean mask marking word-final positions
    """

    def __init__(
        self,
        data_path: str | Path,
        vocab: CharVocab,
        max_length: int = 512,
        text_diac_field: str = "text_diac",
    ):
        self.vocab = vocab
        self.max_length = max_length
        self.samples: list[dict] = []

        data_path = Path(data_path)
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    if text_diac_field in record:
                        self.samples.append(record)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.samples[idx]
        text_diac = record["text_diac"]

        # Extract undiacritized text and diacritic labels
        text_undiac = strip_diacritics(text_diac)
        diac_strings = extract_diacritics(text_diac)

        # Truncate
        text_undiac = text_undiac[: self.max_length]
        diac_strings = diac_strings[: self.max_length]

        # Encode input characters
        char_indices = self.vocab.encode(text_undiac)

        # Encode diacritic labels
        diac_labels = []
        for d in diac_strings:
            d_norm = normalize_diac_sequence(d)
            label_idx = DIAC_LABEL_TO_IDX.get(d_norm, 0)
            diac_labels.append(label_idx)

        # Word-end mask
        wb = set(word_boundaries(text_undiac))
        word_end_mask = [1 if i in wb else 0 for i in range(len(text_undiac))]

        return {
            "input_ids": torch.tensor(char_indices, dtype=torch.long),
            "labels": torch.tensor(diac_labels, dtype=torch.long),
            "word_end_mask": torch.tensor(word_end_mask, dtype=torch.bool),
            "length": len(char_indices),
        }


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate function with padding."""
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    word_end_mask = [item["word_end_mask"] for item in batch]
    lengths = torch.tensor([item["length"] for item in batch])

    # Pad
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)
    word_end_padded = pad_sequence(word_end_mask, batch_first=True, padding_value=False)

    # Attention mask
    attention_mask = torch.zeros_like(input_ids_padded, dtype=torch.bool)
    for i, l in enumerate(lengths):
        attention_mask[i, :l] = True

    return {
        "input_ids": input_ids_padded,
        "labels": labels_padded,
        "word_end_mask": word_end_padded,
        "attention_mask": attention_mask,
        "lengths": lengths,
    }
