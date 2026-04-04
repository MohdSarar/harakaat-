"""
Frequency-based diacritization lexicon.

Maps undiacritized words to their most frequent diacritized forms.
Used in hybrid decoding for disambiguation and post-correction.
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional

from src.utils import strip_diacritics, has_diacritics


class FrequencyLexicon:
    """
    Lexicon mapping undiacritized words → ranked diacritized forms.
    
    Structure:
    {
        "كتب": [
            {"form": "كَتَبَ", "count": 1500, "pos": "VERB"},
            {"form": "كُتُب", "count": 800, "pos": "NOUN"},
            {"form": "كُتِبَ", "count": 200, "pos": "VERB"},
        ]
    }
    """

    def __init__(self):
        self._data: dict[str, list[dict]] = defaultdict(list)
        self._counts: dict[str, Counter] = defaultdict(Counter)

    def add(self, diacritized_word: str, pos: str = "UNK") -> None:
        """Add a diacritized word occurrence."""
        undiac = strip_diacritics(diacritized_word)
        self._counts[undiac][diacritized_word] += 1

    def build_from_corpus(
        self,
        texts: list[str],
        min_frequency: int = 3,
        max_entries: int = 500_000,
    ) -> None:
        """Build lexicon from a list of diacritized texts."""
        for text in texts:
            for word in text.split():
                if has_diacritics(word):
                    self.add(word)

        self._finalize(min_frequency, max_entries)

    def _finalize(self, min_frequency: int, max_entries: int) -> None:
        """Convert raw counts to ranked entries."""
        self._data.clear()
        
        # Sort undiacritized forms by total frequency
        all_undiac = sorted(
            self._counts.keys(),
            key=lambda u: sum(self._counts[u].values()),
            reverse=True,
        )[:max_entries]

        for undiac in all_undiac:
            forms = []
            for diac_form, count in self._counts[undiac].most_common():
                if count >= min_frequency:
                    forms.append({"form": diac_form, "count": count})
            if forms:
                self._data[undiac] = forms

    def lookup(self, undiacritized_word: str) -> list[dict]:
        """
        Look up possible diacritized forms for an undiacritized word.
        Returns list sorted by frequency (most frequent first).
        """
        return self._data.get(undiacritized_word, [])

    def best_form(self, undiacritized_word: str) -> Optional[str]:
        """Return the most frequent diacritized form, or None."""
        forms = self.lookup(undiacritized_word)
        return forms[0]["form"] if forms else None

    def is_ambiguous(self, undiacritized_word: str) -> bool:
        """Check if a word has multiple diacritized forms."""
        return len(self._data.get(undiacritized_word, [])) > 1

    def ambiguity_score(self, undiacritized_word: str) -> float:
        """
        Entropy-like ambiguity score.
        0 = unambiguous, higher = more ambiguous.
        """
        forms = self.lookup(undiacritized_word)
        if len(forms) <= 1:
            return 0.0
        total = sum(f["count"] for f in forms)
        import math
        entropy = -sum(
            (f["count"] / total) * math.log2(f["count"] / total)
            for f in forms
            if f["count"] > 0
        )
        return entropy

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, word: str) -> bool:
        return word in self._data

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dict(self._data), f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path: str | Path) -> FrequencyLexicon:
        lex = cls()
        with open(path, "r", encoding="utf-8") as f:
            lex._data = defaultdict(list, json.load(f))
        return lex
