"""
Evaluation metrics for Arabic diacritization.

Implements:
- DER (Diacritic Error Rate) — char-level
- WER (Word Error Rate) — word-level
- DER without case endings
- DER for case endings only
- Per-genre / per-variety breakdowns
- OOV analysis
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

from src.utils import (
    strip_diacritics, extract_diacritics, normalize_diac_sequence,
    DIAC_LABEL_TO_IDX, DIAC_LABELS, word_boundaries, is_arabic_char,
)


def compute_der(
    predictions: list[list[int]],
    references: list[list[int]],
    ignore_index: int = -1,
) -> float:
    """
    Compute Diacritic Error Rate (character-level).
    
    DER = (# wrong diacritic predictions) / (# total diacritizable characters)
    """
    total = 0
    errors = 0
    
    for pred_seq, ref_seq in zip(predictions, references):
        for p, r in zip(pred_seq, ref_seq):
            if r == ignore_index:
                continue
            total += 1
            if p != r:
                errors += 1
    
    return errors / max(total, 1)


def compute_wer(
    predictions: list[list[int]],
    references: list[list[int]],
    ignore_index: int = -1,
) -> float:
    """
    Compute Word Error Rate.
    
    A word is wrong if ANY of its diacritic predictions differ from reference.
    Words are delimited by label sequences between space characters.
    """
    total_words = 0
    wrong_words = 0
    
    for pred_seq, ref_seq in zip(predictions, references):
        # Group into words (split on wherever we'd have space in text)
        word_correct = True
        in_word = False
        
        for p, r in zip(pred_seq, ref_seq):
            if r == ignore_index:
                if in_word:
                    total_words += 1
                    if not word_correct:
                        wrong_words += 1
                    word_correct = True
                    in_word = False
                continue
            
            in_word = True
            if p != r:
                word_correct = False
        
        # Last word
        if in_word:
            total_words += 1
            if not word_correct:
                wrong_words += 1
    
    return wrong_words / max(total_words, 1)


def compute_der_case_endings(
    text_pred_diac: str,
    text_ref_diac: str,
) -> tuple[float, float]:
    """
    Compute DER separately for:
    1. Case endings (word-final diacritics)
    2. Non-case-ending diacritics
    
    Returns (der_case_endings, der_without_case_endings)
    """
    pred_undiac = strip_diacritics(text_pred_diac)
    ref_undiac = strip_diacritics(text_ref_diac)
    
    if pred_undiac != ref_undiac:
        raise ValueError("Undiacritized texts must match for DER computation")
    
    pred_diacs = extract_diacritics(text_pred_diac)
    ref_diacs = extract_diacritics(text_ref_diac)
    
    wb = set(word_boundaries(pred_undiac))
    
    case_total = case_errors = 0
    non_case_total = non_case_errors = 0
    
    for i, (pd, rd) in enumerate(zip(pred_diacs, ref_diacs)):
        pd_norm = normalize_diac_sequence(pd)
        rd_norm = normalize_diac_sequence(rd)
        
        if i in wb:
            case_total += 1
            if pd_norm != rd_norm:
                case_errors += 1
        else:
            non_case_total += 1
            if pd_norm != rd_norm:
                non_case_errors += 1
    
    der_case = case_errors / max(case_total, 1)
    der_no_case = non_case_errors / max(non_case_total, 1)
    
    return der_case, der_no_case


@dataclass
class EvaluationReport:
    """Complete evaluation report with breakdowns."""
    
    overall_der: float = 0.0
    overall_wer: float = 0.0
    der_case_endings: float = 0.0
    der_without_case_endings: float = 0.0
    
    # Breakdowns
    by_genre: dict[str, dict] = field(default_factory=dict)
    by_variety: dict[str, dict] = field(default_factory=dict)
    by_length: dict[str, dict] = field(default_factory=dict)
    
    # OOV analysis
    oov_der: float = 0.0
    in_vocab_der: float = 0.0
    
    total_samples: int = 0
    total_chars: int = 0
    total_words: int = 0
    
    def to_dict(self) -> dict:
        return {
            "overall_der": round(self.overall_der, 6),
            "overall_wer": round(self.overall_wer, 6),
            "der_case_endings": round(self.der_case_endings, 6),
            "der_without_case_endings": round(self.der_without_case_endings, 6),
            "by_genre": self.by_genre,
            "by_variety": self.by_variety,
            "by_length": self.by_length,
            "oov_der": round(self.oov_der, 6),
            "in_vocab_der": round(self.in_vocab_der, 6),
            "total_samples": self.total_samples,
            "total_chars": self.total_chars,
            "total_words": self.total_words,
        }
    
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  Arabic Diacritization — Evaluation Report",
            "=" * 60,
            f"  Overall DER:                {self.overall_der:.4f}",
            f"  Overall WER:                {self.overall_wer:.4f}",
            f"  DER (case endings):         {self.der_case_endings:.4f}",
            f"  DER (without case endings): {self.der_without_case_endings:.4f}",
            f"  OOV DER:                    {self.oov_der:.4f}",
            f"  In-vocab DER:               {self.in_vocab_der:.4f}",
            f"  Total samples:              {self.total_samples}",
            f"  Total chars:                {self.total_chars}",
            f"  Total words:                {self.total_words}",
        ]
        
        if self.by_genre:
            lines.append("\n  --- By Genre ---")
            for genre, metrics in self.by_genre.items():
                lines.append(f"  {genre:20s}  DER={metrics.get('der', 0):.4f}  WER={metrics.get('wer', 0):.4f}")
        
        if self.by_variety:
            lines.append("\n  --- By Variety ---")
            for var, metrics in self.by_variety.items():
                lines.append(f"  {var:20s}  DER={metrics.get('der', 0):.4f}")
        
        if self.by_length:
            lines.append("\n  --- By Sentence Length ---")
            for bucket, metrics in self.by_length.items():
                lines.append(f"  {bucket:20s}  DER={metrics.get('der', 0):.4f}  n={metrics.get('count', 0)}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


def run_full_evaluation(
    predictions: list[dict],
    references: list[dict],
    vocab_words: Optional[set[str]] = None,
) -> EvaluationReport:
    """
    Run comprehensive evaluation.
    
    Each item in predictions/references should have:
        - text_diac: diacritized text
        - text_undiac: undiacritized text
        - genre (optional)
        - variety (optional)
    
    Args:
        predictions: list of {"text_diac": ..., "text_undiac": ..., ...}
        references: list of {"text_diac": ..., "text_undiac": ..., ...}
        vocab_words: set of known (in-vocabulary) undiacritized words
        
    Returns:
        EvaluationReport
    """
    report = EvaluationReport()
    report.total_samples = len(predictions)
    
    genre_data = defaultdict(lambda: {"preds": [], "refs": []})
    variety_data = defaultdict(lambda: {"preds": [], "refs": []})
    length_data = defaultdict(lambda: {"preds": [], "refs": []})
    
    all_preds_labels = []
    all_refs_labels = []
    
    case_total_err = case_total_count = 0
    nocase_total_err = nocase_total_count = 0
    oov_total = oov_errors = 0
    iv_total = iv_errors = 0
    
    for pred, ref in zip(predictions, references):
        pred_diac = pred["text_diac"]
        ref_diac = ref["text_diac"]
        text_undiac = ref.get("text_undiac", strip_diacritics(ref_diac))
        
        report.total_chars += len(text_undiac)
        report.total_words += len(text_undiac.split())
        
        # Extract labels
        pred_d = extract_diacritics(pred_diac)
        ref_d = extract_diacritics(ref_diac)
        
        pred_labels = [DIAC_LABEL_TO_IDX.get(normalize_diac_sequence(d), 0) for d in pred_d]
        ref_labels = [DIAC_LABEL_TO_IDX.get(normalize_diac_sequence(d), 0) for d in ref_d]
        
        # Align lengths
        min_len = min(len(pred_labels), len(ref_labels))
        pred_labels = pred_labels[:min_len]
        ref_labels = ref_labels[:min_len]
        
        all_preds_labels.append(pred_labels)
        all_refs_labels.append(ref_labels)
        
        # Case ending analysis
        try:
            der_ce, der_noce = compute_der_case_endings(pred_diac, ref_diac)
            wb = word_boundaries(text_undiac)
            case_total_count += len(wb)
            nocase_total_count += min_len - len(wb)
        except ValueError:
            pass
        
        # Genre / variety breakdown
        genre = ref.get("genre", "unknown")
        variety = ref.get("variety", "unknown")
        genre_data[genre]["preds"].append(pred_labels)
        genre_data[genre]["refs"].append(ref_labels)
        variety_data[variety]["preds"].append(pred_labels)
        variety_data[variety]["refs"].append(ref_labels)
        
        # Length bucket
        wc = len(text_undiac.split())
        if wc <= 10:
            bucket = "short (≤10)"
        elif wc <= 30:
            bucket = "medium (11-30)"
        else:
            bucket = "long (>30)"
        length_data[bucket]["preds"].append(pred_labels)
        length_data[bucket]["refs"].append(ref_labels)
        
        # OOV analysis
        if vocab_words is not None:
            for word in text_undiac.split():
                for p, r in zip(pred_labels, ref_labels):
                    if word in vocab_words:
                        iv_total += 1
                        if p != r:
                            iv_errors += 1
                    else:
                        oov_total += 1
                        if p != r:
                            oov_errors += 1
    
    # Compute overall metrics
    report.overall_der = compute_der(all_preds_labels, all_refs_labels)
    report.overall_wer = compute_wer(all_preds_labels, all_refs_labels)
    report.oov_der = oov_errors / max(oov_total, 1)
    report.in_vocab_der = iv_errors / max(iv_total, 1)
    
    # Breakdowns
    for genre, data in genre_data.items():
        report.by_genre[genre] = {
            "der": compute_der(data["preds"], data["refs"]),
            "wer": compute_wer(data["preds"], data["refs"]),
            "count": len(data["preds"]),
        }
    
    for variety, data in variety_data.items():
        report.by_variety[variety] = {
            "der": compute_der(data["preds"], data["refs"]),
            "count": len(data["preds"]),
        }
    
    for bucket, data in length_data.items():
        report.by_length[bucket] = {
            "der": compute_der(data["preds"], data["refs"]),
            "count": len(data["preds"]),
        }
    
    return report
