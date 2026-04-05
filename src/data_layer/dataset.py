"""
PyTorch Dataset for Arabic diacritization.
Supports both BiLSTM (char-level) and AraBERT (subword + char alignment) modes.
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
    DIAC_LABEL_TO_IDX, word_boundaries,
)
from src.utils.vocab import CharVocab


def _build_char_to_token_map(
    text: str,
    tokenizer,
    max_bert_length: int = 512,
) -> tuple[list[int], list[int], list[int]]:
    """
    Tokenize text with AraBERT and build a char→token alignment map.

    Returns:
        bert_input_ids:     BERT token ids (including [CLS]/[SEP])
        bert_attention_mask: 1 for real tokens, 0 for padding
        char_to_token_map:  for each char position, the index of its BERT token
    """
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        max_length=max_bert_length,
        truncation=True,
        padding=False,
    )
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    offset_mapping = encoding["offset_mapping"]  # [(start, end), ...]

    # Build char → token index map
    char_to_token = [-1] * len(text)
    for tok_idx, (start, end) in enumerate(offset_mapping):
        if start == end:  # special token ([CLS], [SEP])
            continue
        for char_idx in range(start, min(end, len(text))):
            char_to_token[char_idx] = tok_idx

    # Forward-fill unmapped positions (spaces, punctuation between tokens)
    last_valid = 1  # token 1 = first real token after [CLS]
    for i in range(len(char_to_token)):
        if char_to_token[i] == -1:
            char_to_token[i] = last_valid
        else:
            last_valid = char_to_token[i]

    # Clamp to valid range (safety)
    max_tok = len(input_ids) - 1
    char_to_token = [min(t, max_tok) for t in char_to_token]

    return input_ids, attention_mask, char_to_token


class DiacritizationDataset(Dataset):
    """
    Dataset that produces per-character diacritic labels.

    BiLSTM mode (tokenizer=None):
        input_ids, labels, word_end_mask, attention_mask, lengths

    AraBERT mode (tokenizer provided):
        bert_input_ids, bert_attention_mask, char_to_token_map,
        labels, word_end_mask, attention_mask, lengths
    """

    def __init__(
        self,
        data_path: str | Path,
        vocab: CharVocab,
        max_length: int = 512,
        text_diac_field: str = "text_diac",
        excluded_genres: list[str] | None = None,
        tokenizer=None,
        max_bert_length: int = 512,
    ):
        self.vocab = vocab
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.max_bert_length = max_bert_length
        self.samples: list[dict] = []
        excluded = set(excluded_genres or [])

        data_path = Path(data_path)
        n_skipped = 0
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    if text_diac_field not in record:
                        continue
                    if excluded and record.get("genre", "") in excluded:
                        n_skipped += 1
                        continue
                    self.samples.append(record)
        if n_skipped:
            print(f"  Skipped {n_skipped} samples from excluded genres: {sorted(excluded)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.samples[idx]
        text_diac = record["text_diac"]

        text_undiac = strip_diacritics(text_diac)
        diac_strings = extract_diacritics(text_diac)

        # Truncate to max char length
        text_undiac = text_undiac[:self.max_length]
        diac_strings = diac_strings[:self.max_length]

        # Diacritic labels
        diac_labels = [
            DIAC_LABEL_TO_IDX.get(normalize_diac_sequence(d), 0)
            for d in diac_strings
        ]

        # Word-end mask
        wb = set(word_boundaries(text_undiac))
        word_end_mask = [1 if i in wb else 0 for i in range(len(text_undiac))]

        item = {
            "labels": torch.tensor(diac_labels, dtype=torch.long),
            "word_end_mask": torch.tensor(word_end_mask, dtype=torch.bool),
            "length": len(text_undiac),
        }

        if self.tokenizer is not None:
            # AraBERT mode
            bert_ids, bert_mask, char_to_token = _build_char_to_token_map(
                text_undiac, self.tokenizer, self.max_bert_length
            )
            # Trim char sequences to what BERT can cover
            max_char = len(char_to_token)
            item["labels"] = item["labels"][:max_char]
            item["word_end_mask"] = item["word_end_mask"][:max_char]
            item["length"] = min(item["length"], max_char)
            item["bert_input_ids"] = torch.tensor(bert_ids, dtype=torch.long)
            item["bert_attention_mask"] = torch.tensor(bert_mask, dtype=torch.long)
            item["char_to_token_map"] = torch.tensor(char_to_token[:max_char], dtype=torch.long)
        else:
            # BiLSTM mode
            char_indices = self.vocab.encode(text_undiac)
            item["input_ids"] = torch.tensor(char_indices, dtype=torch.long)

        return item


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate function — handles both BiLSTM and AraBERT batches."""
    labels = [item["labels"] for item in batch]
    word_end_mask = [item["word_end_mask"] for item in batch]
    lengths = torch.tensor([item["length"] for item in batch])

    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)
    word_end_padded = pad_sequence(word_end_mask, batch_first=True, padding_value=False)

    # Char-level attention mask (for CRF)
    char_seq_len = labels_padded.size(1)
    attention_mask = torch.zeros(len(batch), char_seq_len, dtype=torch.bool)
    for i, l in enumerate(lengths):
        attention_mask[i, :l] = True

    result = {
        "labels": labels_padded,
        "word_end_mask": word_end_padded,
        "attention_mask": attention_mask,
        "lengths": lengths,
    }

    if "bert_input_ids" in batch[0]:
        # AraBERT mode — pad BERT sequences and char_to_token_map
        bert_input_ids = [item["bert_input_ids"] for item in batch]
        bert_attention_mask = [item["bert_attention_mask"] for item in batch]
        char_to_token_map = [item["char_to_token_map"] for item in batch]

        result["bert_input_ids"] = pad_sequence(bert_input_ids, batch_first=True, padding_value=0)
        result["bert_attention_mask"] = pad_sequence(bert_attention_mask, batch_first=True, padding_value=0)
        result["char_to_token_map"] = pad_sequence(char_to_token_map, batch_first=True, padding_value=0)
    else:
        # BiLSTM mode
        input_ids = [item["input_ids"] for item in batch]
        result["input_ids"] = pad_sequence(input_ids, batch_first=True, padding_value=0)

    return result
