"""
Hybrid decoder — combines neural model output with:
- Lexicon-based correction
- Morphological constraints
- Word-ending re-ranking
- Confidence scoring

This is where the magic happens: pure neural predictions
are filtered and corrected using linguistic knowledge.
"""

from __future__ import annotations

import torch
from typing import Optional
from dataclasses import dataclass

from src.utils import (
    DIAC_LABELS, NUM_DIAC_CLASSES, strip_diacritics,
    is_arabic_char, word_boundaries, HARAKAT,
)
from src.linguistic.lexicon import FrequencyLexicon


@dataclass
class DecodingResult:
    """Result of hybrid decoding."""
    text_diac: str
    confidence: float
    per_word_confidence: list[float]
    corrections_applied: int
    flagged_for_review: bool


# Invalid diacritic combinations (character → disallowed diacritics)
# These are hard phonological constraints
INVALID_COMBOS = {
    "\u0627": {5, 6},     # alif: no dammatan/kasratan typically
    "\u0629": {7},         # taa marbuta: no sukun (it's always voweled or silent)
    " ": set(range(1, NUM_DIAC_CLASSES)),  # spaces: no diacritics
}


class HybridDecoder:
    """
    Hybrid decoding pipeline:
    
    1. Take raw neural predictions (label indices per character)
    2. Apply morphological constraints (filter impossible combos)
    3. Apply lexicon corrections (high-frequency forms)
    4. Re-rank word endings using specialized head output
    5. Compute confidence scores
    6. Flag low-confidence segments for HITL review
    """

    def __init__(
        self,
        lexicon: Optional[FrequencyLexicon] = None,
        use_morphological_constraints: bool = True,
        use_lexicon: bool = True,
        use_reranking: bool = True,
        confidence_threshold: float = 0.85,
    ):
        self.lexicon = lexicon
        self.use_morph = use_morphological_constraints
        self.use_lex = use_lexicon and lexicon is not None
        self.use_reranking = use_reranking
        self.confidence_threshold = confidence_threshold

    def decode(
        self,
        text_undiac: str,
        pred_labels: list[int],
        emissions: Optional[torch.Tensor] = None,
        word_ending_logits: Optional[torch.Tensor] = None,
    ) -> DecodingResult:
        """
        Full hybrid decoding pipeline.
        
        Args:
            text_undiac: undiacritized input text
            pred_labels: predicted diacritic label indices (from model/CRF)
            emissions: raw model logits (for confidence), shape (seq_len, num_classes)
            word_ending_logits: word-ending head logits, same shape
            
        Returns:
            DecodingResult with diacritized text and metadata
        """
        chars = list(text_undiac)
        labels = list(pred_labels[:len(chars)])
        corrections = 0

        # Step 1: Morphological constraints
        if self.use_morph:
            labels, n_morph = self._apply_morphological_constraints(chars, labels)
            corrections += n_morph

        # Step 2: Word-ending re-ranking
        if self.use_reranking and word_ending_logits is not None:
            labels, n_rerank = self._rerank_word_endings(
                chars, labels, word_ending_logits
            )
            corrections += n_rerank

        # Step 3: Lexicon correction (word-level)
        if self.use_lex:
            labels, n_lex = self._apply_lexicon_corrections(chars, labels)
            corrections += n_lex

        # Step 4: Reconstruct diacritized text
        text_diac = self._reconstruct(chars, labels)

        # Step 5: Compute confidence
        confidence, per_word = self._compute_confidence(chars, labels, emissions)

        return DecodingResult(
            text_diac=text_diac,
            confidence=confidence,
            per_word_confidence=per_word,
            corrections_applied=corrections,
            flagged_for_review=confidence < self.confidence_threshold,
        )

    def _apply_morphological_constraints(
        self, chars: list[str], labels: list[int]
    ) -> tuple[list[int], int]:
        """Filter impossible diacritic-character combinations."""
        corrections = 0
        for i, (ch, lbl) in enumerate(zip(chars, labels)):
            invalid = INVALID_COMBOS.get(ch, set())
            if lbl in invalid:
                labels[i] = 0  # fallback to no diacritic
                corrections += 1
            
            # Non-Arabic characters should have no diacritics
            if not is_arabic_char(ch) and ch not in (" ", "\n"):
                if lbl != 0:
                    labels[i] = 0
                    corrections += 1
        
        return labels, corrections

    def _rerank_word_endings(
        self,
        chars: list[str],
        labels: list[int],
        we_logits: torch.Tensor,
    ) -> tuple[list[int], int]:
        """
        Re-rank predictions at word-final positions using
        the specialized word-ending head.
        """
        corrections = 0
        wb = set(word_boundaries("".join(chars)))
        
        for pos in wb:
            if pos >= len(labels) or pos >= we_logits.size(0):
                continue
            
            we_probs = torch.softmax(we_logits[pos], dim=-1)
            we_pred = we_probs.argmax().item()
            we_conf = we_probs[we_pred].item()
            
            # Only override if word-ending head is more confident
            if we_conf > 0.6 and we_pred != labels[pos]:
                labels[pos] = we_pred
                corrections += 1
        
        return labels, corrections

    def _apply_lexicon_corrections(
        self, chars: list[str], labels: list[int]
    ) -> tuple[list[int], int]:
        """
        Word-level lexicon correction.
        If the model's diacritized word doesn't match any known form,
        but the lexicon has a high-frequency form, use it.
        """
        corrections = 0
        
        # Reconstruct current prediction
        text = "".join(chars)
        words = text.split()
        
        char_offset = 0
        for word in words:
            word_len = len(word)
            
            # Skip spaces
            while char_offset < len(chars) and chars[char_offset] == " ":
                char_offset += 1
            
            if char_offset + word_len > len(labels):
                break
            
            # Get current predicted diacritized form
            word_labels = labels[char_offset:char_offset + word_len]
            predicted_form = ""
            for ch, lbl in zip(word, word_labels):
                predicted_form += ch + DIAC_LABELS[lbl]
            
            # Check lexicon
            if self.lexicon and word in self.lexicon:
                best = self.lexicon.best_form(word)
                if best and best != predicted_form:
                    # Decode best form back to labels
                    new_labels = self._form_to_labels(best, word)
                    if new_labels and len(new_labels) == word_len:
                        for j, nl in enumerate(new_labels):
                            if labels[char_offset + j] != nl:
                                labels[char_offset + j] = nl
                                corrections += 1
            
            char_offset += word_len
        
        return labels, corrections

    def _form_to_labels(self, diac_form: str, undiac_word: str) -> list[int]:
        """Convert a diacritized form back to label indices."""
        from src.utils import extract_diacritics, normalize_diac_sequence, DIAC_LABEL_TO_IDX
        
        diacs = extract_diacritics(diac_form)
        labels = []
        for d in diacs:
            d_norm = normalize_diac_sequence(d)
            labels.append(DIAC_LABEL_TO_IDX.get(d_norm, 0))
        return labels

    def _reconstruct(self, chars: list[str], labels: list[int]) -> str:
        """Reconstruct diacritized text from characters and labels."""
        result = []
        for ch, lbl in zip(chars, labels):
            result.append(ch)
            if lbl > 0 and lbl < len(DIAC_LABELS):
                result.append(DIAC_LABELS[lbl])
        return "".join(result)

    def _compute_confidence(
        self,
        chars: list[str],
        labels: list[int],
        emissions: Optional[torch.Tensor],
    ) -> tuple[float, list[float]]:
        """
        Compute overall and per-word confidence scores.
        Uses softmax probabilities from emissions if available.
        """
        if emissions is None:
            return 1.0, []
        
        probs = torch.softmax(emissions, dim=-1)
        
        # Per-character confidence
        char_conf = []
        for i, lbl in enumerate(labels):
            if i < probs.size(0):
                char_conf.append(probs[i, lbl].item())
            else:
                char_conf.append(1.0)
        
        # Aggregate to per-word
        text = "".join(chars)
        words = text.split()
        per_word = []
        offset = 0
        for word in words:
            while offset < len(chars) and chars[offset] == " ":
                offset += 1
            word_confs = char_conf[offset:offset + len(word)]
            per_word.append(min(word_confs) if word_confs else 1.0)
            offset += len(word)
        
        overall = sum(char_conf) / max(len(char_conf), 1)
        return overall, per_word
