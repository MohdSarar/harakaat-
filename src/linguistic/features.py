"""
Linguistic feature extractor for the diacritization pipeline.

Provides auxiliary features:
- POS tags
- Morphological analysis
- Lemmatization
- Word-level features

Can optionally integrate SinaTools when available.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from src.utils import is_arabic_char, strip_diacritics


@dataclass
class WordFeatures:
    """Linguistic features for a single word."""
    word: str
    pos: str = "UNK"
    lemma: str = ""
    morphology: dict = field(default_factory=dict)
    is_stopword: bool = False
    has_definite_article: bool = False
    is_preposition: bool = False
    word_length: int = 0
    
    def __post_init__(self):
        self.word_length = len(strip_diacritics(self.word))
        self.has_definite_article = strip_diacritics(self.word).startswith("ال")


@dataclass
class SentenceFeatures:
    """Linguistic features for a full sentence."""
    words: list[WordFeatures]
    sentence_length: int = 0
    
    def __post_init__(self):
        self.sentence_length = len(self.words)


# Common Arabic stopwords (undiacritized)
ARABIC_STOPWORDS = frozenset([
    "في", "من", "على", "إلى", "عن", "مع", "هذا", "هذه", "ذلك", "تلك",
    "التي", "الذي", "الذين", "اللذين", "اللتين", "هو", "هي", "هم", "هن",
    "أن", "إن", "لا", "ما", "لم", "لن", "قد", "كان", "كانت", "يكون",
    "بين", "حتى", "ثم", "أو", "و", "ف", "ب", "ل", "ك",
])

# Common Arabic prepositions (undiacritized)
ARABIC_PREPOSITIONS = frozenset([
    "في", "من", "على", "إلى", "عن", "مع", "بين", "حتى",
    "خلال", "حول", "عند", "بعد", "قبل", "تحت", "فوق", "أمام", "وراء",
])


class LinguisticAnalyzer:
    """
    Linguistic analyzer with optional SinaTools integration.
    
    When SinaTools is not available, uses a rule-based fallback
    for basic features (definite article, prepositions, stopwords).
    """

    def __init__(self, use_sinatools: bool = False):
        self.use_sinatools = use_sinatools
        self._sinatools = None

        if use_sinatools:
            try:
                import sinatools
                self._sinatools = sinatools
            except ImportError:
                print(
                    "WARNING: sinatools not installed. "
                    "Falling back to rule-based features.\n"
                    "Install: pip install sinatools\n"
                    "Docs: https://sina.birzeit.edu/sinatools/"
                )
                self.use_sinatools = False

    def analyze_sentence(self, text: str) -> SentenceFeatures:
        """Analyze a sentence and return linguistic features."""
        clean = strip_diacritics(text)
        words = clean.split()
        
        word_features = []
        for w in words:
            wf = self._analyze_word(w)
            word_features.append(wf)
        
        return SentenceFeatures(words=word_features)

    def _analyze_word(self, word: str) -> WordFeatures:
        """Analyze a single word."""
        clean = strip_diacritics(word)
        
        if self.use_sinatools and self._sinatools is not None:
            return self._analyze_with_sinatools(word, clean)
        
        return self._analyze_rule_based(word, clean)

    def _analyze_rule_based(self, word: str, clean: str) -> WordFeatures:
        """Rule-based fallback analysis."""
        features = WordFeatures(word=word)
        
        # Stopword
        features.is_stopword = clean in ARABIC_STOPWORDS
        
        # Preposition
        features.is_preposition = clean in ARABIC_PREPOSITIONS
        
        # Simple POS heuristics
        if features.is_preposition:
            features.pos = "PREP"
        elif clean.startswith("ال"):
            features.pos = "NOUN"  # likely noun with definite article
        elif clean.endswith("ة"):
            features.pos = "NOUN"  # taa marbuta → likely noun/adj
        elif clean.endswith("ون") or clean.endswith("ين"):
            features.pos = "NOUN"  # plural
        elif len(clean) >= 4 and clean[0] in "يتنا":
            features.pos = "VERB"  # imperfect prefix
        else:
            features.pos = "UNK"
        
        features.lemma = clean
        return features

    def _analyze_with_sinatools(self, word: str, clean: str) -> WordFeatures:
        """Analysis using SinaTools (when available)."""
        features = WordFeatures(word=word)
        
        try:
            # SinaTools morphological analysis
            # API may vary — adapt to actual SinaTools version
            from sinatools.morphology import analyze
            result = analyze(clean)
            if result:
                features.pos = result.get("pos", "UNK")
                features.lemma = result.get("lemma", clean)
                features.morphology = result.get("features", {})
        except Exception:
            # Fallback
            return self._analyze_rule_based(word, clean)
        
        features.is_stopword = clean in ARABIC_STOPWORDS
        features.is_preposition = clean in ARABIC_PREPOSITIONS
        return features

    def get_feature_vector(self, word_features: WordFeatures) -> list[float]:
        """
        Convert word features to a numeric vector for model input.
        Returns a fixed-size feature vector.
        """
        pos_map = {"NOUN": 0, "VERB": 1, "PREP": 2, "ADJ": 3, "ADV": 4, "UNK": 5}
        
        return [
            pos_map.get(word_features.pos, 5) / 5.0,
            1.0 if word_features.is_stopword else 0.0,
            1.0 if word_features.has_definite_article else 0.0,
            1.0 if word_features.is_preposition else 0.0,
            min(word_features.word_length / 20.0, 1.0),
        ]
