"""
Arabic character constants, diacritic mappings, and shared utilities.
"""

import re
import unicodedata
from typing import Optional

# ============================================================
# Arabic Unicode Ranges
# ============================================================

# Basic Arabic block
ARABIC_START = 0x0600
ARABIC_END = 0x06FF

# Diacritical marks (tashkeel / harakat)
FATHAH = "\u064E"        # فتحة  َ
DAMMAH = "\u064F"        # ضمة   ُ
KASRAH = "\u0650"        # كسرة  ِ
FATHATAN = "\u064B"      # فتحتان ً
DAMMATAN = "\u064C"      # ضمتان ٌ
KASRATAN = "\u064D"      # كسرتان ٍ
SUKUN = "\u0652"         # سكون  ْ
SHADDAH = "\u0651"       # شدة   ّ
SUPERSCRIPT_ALEF = "\u0670"  # ألف خنجرية

# All diacritics
HARAKAT = frozenset([
    FATHAH, DAMMAH, KASRAH,
    FATHATAN, DAMMATAN, KASRATAN,
    SUKUN, SHADDAH, SUPERSCRIPT_ALEF,
])

# Primary diacritics (without shaddah — shaddah is a consonant doubler)
PRIMARY_HARAKAT = frozenset([
    FATHAH, DAMMAH, KASRAH,
    FATHATAN, DAMMATAN, KASRATAN,
    SUKUN,
])

# All harakat as a regex character class
HARAKAT_PATTERN = re.compile(r"[\u064B-\u0652\u0670]")

# Diacritic label vocabulary
# Index 0 = no diacritic; the rest follow a fixed order.
DIAC_LABELS = [
    "",              # 0: no diacritic
    FATHAH,          # 1
    DAMMAH,          # 2
    KASRAH,          # 3
    FATHATAN,        # 4
    DAMMATAN,        # 5
    KASRATAN,        # 6
    SUKUN,           # 7
    SHADDAH,         # 8: shaddah alone
    SHADDAH + FATHAH,    # 9
    SHADDAH + DAMMAH,    # 10
    SHADDAH + KASRAH,    # 11
    SHADDAH + FATHATAN,  # 12
    SHADDAH + DAMMATAN,  # 13
    SHADDAH + KASRATAN,  # 14
    SHADDAH + SUKUN,     # 15
]

DIAC_LABEL_TO_IDX = {label: idx for idx, label in enumerate(DIAC_LABELS)}
NUM_DIAC_CLASSES = len(DIAC_LABELS)

# Arabic letters (consonants + long vowels)
ARABIC_LETTERS = frozenset(
    chr(c)
    for c in range(ARABIC_START, ARABIC_END + 1)
    if unicodedata.category(chr(c)).startswith("L")
)

# Alif variants
ALIF = "\u0627"           # ا
ALIF_MADDA = "\u0622"     # آ
ALIF_HAMZA_ABOVE = "\u0623"  # أ
ALIF_HAMZA_BELOW = "\u0625"  # إ
ALIF_WASLA = "\u0671"     # ٱ

# Hamza variants
HAMZA = "\u0621"          # ء
WAW_HAMZA = "\u0624"      # ؤ
YA_HAMZA = "\u0626"       # ئ

# Tatweel
TATWEEL = "\u0640"        # ـ

# Taa Marbuta / Haa
TAA_MARBUTA = "\u0629"    # ة
HAA = "\u0647"            # ه

# ============================================================
# Special tokens for model vocabulary
# ============================================================

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
WORD_BOUNDARY = "<WB>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, WORD_BOUNDARY]

# ============================================================
# Utility functions
# ============================================================


def strip_diacritics(text: str) -> str:
    """Remove all Arabic diacritical marks from text."""
    return HARAKAT_PATTERN.sub("", text)


def extract_diacritics(text: str) -> list[str]:
    """
    Extract diacritic labels for each base character.
    Returns a list of diacritic strings (one per base char).
    """
    result = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch not in HARAKAT:
            # Collect diacritics following this character
            diac = ""
            j = i + 1
            while j < len(text) and text[j] in HARAKAT:
                diac += text[j]
                j += 1
            result.append(diac)
            i = j
        else:
            # Orphan diacritic at start — skip
            i += 1
    return result


def is_arabic_char(ch: str) -> bool:
    """Check if a character is an Arabic letter."""
    return ch in ARABIC_LETTERS


def is_arabic_word(word: str) -> bool:
    """Check if a word contains at least one Arabic letter."""
    return any(is_arabic_char(ch) for ch in word)


def has_diacritics(text: str) -> bool:
    """Check if text contains any diacritical marks."""
    return bool(HARAKAT_PATTERN.search(text))


def diacritic_density(text: str) -> float:
    """Fraction of base characters that have diacritics."""
    base = strip_diacritics(text)
    if not base:
        return 0.0
    diacs = extract_diacritics(text)
    return sum(1 for d in diacs if d) / len(diacs)


def normalize_diac_sequence(diac: str) -> str:
    """
    Normalize a diacritic sequence: shaddah always first,
    then at most one primary diacritic.
    """
    has_shaddah = SHADDAH in diac
    primary = [ch for ch in diac if ch in PRIMARY_HARAKAT]
    out = ""
    if has_shaddah:
        out += SHADDAH
    if primary:
        out += primary[0]  # keep first primary only
    return out


def word_boundaries(text: str) -> list[int]:
    """
    Return indices of characters that are at word-final position
    (last Arabic letter before a space or end of string).
    Operates on stripped (undiacritized) text.
    """
    clean = strip_diacritics(text)
    boundaries = []
    for i, ch in enumerate(clean):
        if is_arabic_char(ch):
            # Check if next non-diacritic char is space/end
            if i + 1 >= len(clean) or not is_arabic_char(clean[i + 1]):
                boundaries.append(i)
    return boundaries
