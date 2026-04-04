"""
Arabic text normalizer for the diacritization pipeline.

Handles:
- Unicode normalization (NFC)
- Tatweel removal
- Alif/Hamza unification
- Punctuation normalization
- Whitespace cleanup
- Controlled diacritic preservation
"""

from __future__ import annotations

import re
import unicodedata

from src.utils import (
    TATWEEL, ALIF, ALIF_MADDA, ALIF_HAMZA_ABOVE, ALIF_HAMZA_BELOW,
    ALIF_WASLA, HARAKAT_PATTERN, strip_diacritics, SUPERSCRIPT_ALEF,
)


class ArabicNormalizer:
    """
    Configurable Arabic text normalizer.
    
    Each normalization step can be toggled on/off.
    Order matters and follows best practices for Arabic NLP.
    """

    def __init__(
        self,
        remove_tatweel: bool = True,
        normalize_alif: bool = True,
        normalize_hamza: bool = True,
        normalize_taa_marbuta: bool = False,
        normalize_punctuation: bool = True,
        strip_extra_whitespace: bool = True,
        preserve_rare_marks: bool = False,
    ):
        self.remove_tatweel = remove_tatweel
        self.normalize_alif = normalize_alif
        self.normalize_hamza = normalize_hamza
        self.normalize_taa_marbuta = normalize_taa_marbuta
        self.normalize_punctuation = normalize_punctuation
        self.strip_extra_whitespace = strip_extra_whitespace
        self.preserve_rare_marks = preserve_rare_marks

        # Precompile patterns
        self._multi_space = re.compile(r"\s+")
        self._tatweel_re = re.compile(TATWEEL)
        
        # Arabic punctuation → standard
        self._punct_map = str.maketrans({
            "،": ",",
            "؛": ";",
            "؟": "?",
            "٪": "%",
            "٫": ".",
            "٬": ",",
            "\u200c": "",   # zero-width non-joiner
            "\u200d": "",   # zero-width joiner
            "\u00a0": " ",  # non-breaking space
            "\ufeff": "",   # BOM
        })
        
        # Rare diacritical marks to strip if not preserved
        self._rare_marks = re.compile(
            r"[\u0610-\u061A\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]"
        )

    def __call__(self, text: str) -> str:
        return self.normalize(text)

    def normalize(self, text: str) -> str:
        """Apply all enabled normalization steps in order."""
        if not text:
            return text

        # Step 0: Unicode NFC normalization
        text = unicodedata.normalize("NFC", text)

        # Step 1: Remove tatweel (kashida)
        if self.remove_tatweel:
            text = self._tatweel_re.sub("", text)

        # Step 2: Normalize alif forms
        if self.normalize_alif:
            text = self._normalize_alif(text)

        # Step 3: Normalize hamza (careful — this can change meaning)
        if self.normalize_hamza:
            text = self._normalize_hamza(text)

        # Step 4: Normalize taa marbuta ↔ haa (usually OFF)
        if self.normalize_taa_marbuta:
            text = text.replace("\u0629", "\u0647")  # ة → ه

        # Step 5: Normalize punctuation
        if self.normalize_punctuation:
            text = text.translate(self._punct_map)

        # Step 6: Strip rare diacritical marks
        if not self.preserve_rare_marks:
            text = self._rare_marks.sub("", text)

        # Step 7: Clean whitespace
        if self.strip_extra_whitespace:
            text = self._multi_space.sub(" ", text).strip()

        return text

    def _normalize_alif(self, text: str) -> str:
        """
        Normalize alif variants.
        أ إ آ ٱ → ا
        
        NOTE: This is applied to the BASE character only.
        Diacritics on these characters are preserved.
        """
        result = []
        for ch in text:
            if ch in (ALIF_HAMZA_ABOVE, ALIF_HAMZA_BELOW, ALIF_MADDA, ALIF_WASLA):
                result.append(ALIF)
            else:
                result.append(ch)
        return "".join(result)

    def _normalize_hamza(self, text: str) -> str:
        """
        Light hamza normalization.
        Only normalizes alif-based hamza (handled above).
        Waw-hamza (ؤ) and ya-hamza (ئ) are kept as-is,
        since they carry morphological meaning.
        """
        # Already handled by _normalize_alif for alif-hamza
        return text


def normalize_text(text: str, **kwargs) -> str:
    """Convenience function: normalize with default or custom settings."""
    normalizer = ArabicNormalizer(**kwargs)
    return normalizer(text)
