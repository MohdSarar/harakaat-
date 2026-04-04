"""
Data loaders for various Arabic diacritized corpus sources.

Supported:
- Tashkeela (local text files)
- Sadeed_Tashkeela (HuggingFace)
- Quranic Arabic Corpus (TSV morphology file from corpus.quran.com)
- Generic JSONL / Parquet
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator
from collections import defaultdict

from src.utils import has_diacritics


def load_tashkeela(path: str | Path) -> Iterator[dict]:
    """
    Load Tashkeela corpus from a directory of text files.
    Each file contains diacritized Arabic text.
    
    Yields dicts: {"text_diac": str, "source": "tashkeela", "variety": "msa"}
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Tashkeela directory not found: {path}\n"
            "Download from: https://sourceforge.net/projects/tashkeela/"
        )
    
    for fpath in sorted(path.rglob("*.txt")):
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and has_diacritics(line) and len(line) > 10:
                    yield {
                        "text_diac": line,
                        "source": "tashkeela",
                        "variety": "msa",
                        "genre": fpath.parent.name or "general",
                        "file": fpath.name,
                    }


def load_huggingface_dataset(
    repo: str = "Misraj/Sadeed_Tashkeela",
    split: str = "train",
    text_column: str = "text",
    variety: str = "msa",
) -> Iterator[dict]:
    """
    Load a diacritized dataset from HuggingFace.
    Requires `datasets` library.
    
    Yields dicts: {"text_diac": str, "source": repo, "variety": variety}
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    ds = load_dataset(repo, split=split)
    for row in ds:
        text = row.get(text_column, "")
        if text and has_diacritics(text):
            yield {
                "text_diac": text,
                "source": repo,
                "variety": variety,
                "genre": row.get("genre", "general"),
            }


# ============================================================
# Buckwalter-to-Arabic transliteration map
# Used to convert the Quranic Arabic Corpus morphology file
# (which uses Buckwalter encoding) back to Arabic script.
# ============================================================

_BUCKWALTER_TO_ARABIC = {
    "'": "\u0621",   # ء hamza
    "|": "\u0622",   # آ alef madda
    ">": "\u0623",   # أ alef hamza above
    "&": "\u0624",   # ؤ waw hamza
    "<": "\u0625",   # إ alef hamza below
    "}": "\u0626",   # ئ ya hamza
    "A": "\u0627",   # ا alef
    "b": "\u0628",   # ب ba
    "p": "\u0629",   # ة ta marbuta
    "t": "\u062A",   # ت ta
    "v": "\u062B",   # ث tha
    "j": "\u062C",   # ج jim
    "H": "\u062D",   # ح ha
    "x": "\u062E",   # خ kha
    "d": "\u062F",   # د dal
    "*": "\u0630",   # ذ dhal
    "r": "\u0631",   # ر ra
    "z": "\u0632",   # ز zay
    "s": "\u0633",   # س sin
    "$": "\u0634",   # ش shin
    "S": "\u0635",   # ص sad
    "D": "\u0636",   # ض dad
    "T": "\u0637",   # ط ta emphatic
    "Z": "\u0638",   # ظ dha
    "E": "\u0639",   # ع ain
    "g": "\u063A",   # غ ghain
    "_": "\u0640",   # ـ tatweel
    "f": "\u0641",   # ف fa
    "q": "\u0642",   # ق qaf
    "k": "\u0643",   # ك kaf
    "l": "\u0644",   # ل lam
    "m": "\u0645",   # م mim
    "n": "\u0646",   # ن nun
    "h": "\u0647",   # ه ha
    "w": "\u0648",   # و waw
    "Y": "\u0649",   # ى alef maqsura
    "y": "\u064A",   # ي ya
    "F": "\u064B",   # ً fathatan
    "N": "\u064C",   # ٌ dammatan
    "K": "\u064D",   # ٍ kasratan
    "a": "\u064E",   # َ fatha
    "u": "\u064F",   # ُ damma
    "i": "\u0650",   # ِ kasra
    "~": "\u0651",   # ّ shadda
    "o": "\u0652",   # ْ sukun
    "`": "\u0670",   # ٰ superscript alef (dagger alef)
    "{": "\u0671",   # ٱ alef wasla
    # Some corpus files use these extras
    "P": "\u067E",   # پ pe (loanwords)
    "J": "\u0686",   # چ che (loanwords)
    "V": "\u06A4",   # ڤ ve (loanwords)
    "G": "\u06AF",   # گ gaf (loanwords)
}


def buckwalter_to_arabic(bw_text: str) -> str:
    """Convert Buckwalter transliteration to Arabic Unicode."""
    return "".join(_BUCKWALTER_TO_ARABIC.get(ch, ch) for ch in bw_text)


def _parse_quran_location(loc: str) -> tuple[int, int, int, int]:
    """
    Parse a Quranic corpus location string.
    Format: (sura:aya:word:segment) e.g. (1:1:1:1)
    Returns (sura, aya, word, segment) as ints.
    """
    loc = loc.strip("()")
    parts = loc.split(":")
    return tuple(int(p) for p in parts)


def load_quran_corpus(path: str | Path) -> Iterator[dict]:
    """
    Load Quranic Arabic Corpus morphology file (TSV format).
    
    File format from https://corpus.quran.com/download/:
        LOCATION    FORM    TAG    FEATURES
        (1:1:1:1)   bi      P      PREFIX|bi+
        (1:1:1:2)   somi    N      STEM|POS:N|LEM:{som|ROOT:smw|M|GEN
        ...
    
    The FORM column is in Buckwalter transliteration.
    We reconstruct verse-level diacritized Arabic text by:
    1. Grouping segments by (sura, aya, word)
    2. Concatenating segments within each word
    3. Joining words with spaces to form ayat (verses)
    
    Yields dicts with verse-level diacritized text:
        {"text_diac": "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ",
         "source": "quran", "variety": "classical", "genre": "quran",
         "sura": "1", "aya": "1"}
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Quran corpus not found: {path}\n"
            "Download from: https://corpus.quran.com/download/\n"
            "Expected: TSV morphology file (quranic-corpus-morphology-*.txt)"
        )

    # Parse all morpheme entries
    # Structure: (sura, aya) → word_idx → list of segment forms (Buckwalter)
    verses: dict[tuple[int, int], dict[int, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            
            # Skip comments and header
            if not line or line.startswith("#") or line.startswith("LOCATION"):
                continue
            
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            
            location_str = parts[0].strip()
            form_bw = parts[1].strip()
            tag = parts[2].strip()
            features = parts[3].strip() if len(parts) > 3 else ""
            
            try:
                sura, aya, word_idx, segment_idx = _parse_quran_location(location_str)
            except (ValueError, TypeError):
                continue
            
            # Convert Buckwalter form to Arabic
            form_arabic = buckwalter_to_arabic(form_bw)
            
            verses[(sura, aya)][word_idx].append(form_arabic)

    # Reconstruct verses
    for (sura, aya) in sorted(verses.keys()):
        word_dict = verses[(sura, aya)]
        
        # Build each word by concatenating its segments
        words_arabic = []
        for word_idx in sorted(word_dict.keys()):
            segments = word_dict[word_idx]
            word = "".join(segments)
            words_arabic.append(word)
        
        verse_text = " ".join(words_arabic)
        
        if verse_text.strip() and has_diacritics(verse_text):
            yield {
                "text_diac": verse_text,
                "source": "quran",
                "variety": "classical",
                "genre": "quran",
                "sura": str(sura),
                "aya": str(aya),
            }


def load_quran_tanzil_text(path: str | Path) -> Iterator[dict]:
    """
    Load a Tanzil plain-text Quran file (alternative format).
    
    Some Tanzil downloads are simple text files with one verse per line,
    optionally prefixed with sura|aya numbers.
    
    Supports formats:
        - "بِسْمِ ٱللَّهِ ..." (just text)
        - "1|1|بِسْمِ ٱللَّهِ ..." (sura|aya|text)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Tanzil text not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            # Try sura|aya|text format
            if "|" in line:
                parts = line.split("|", maxsplit=2)
                if len(parts) == 3:
                    sura, aya, text = parts[0], parts[1], parts[2]
                else:
                    text = line
                    sura, aya = "", ""
            else:
                text = line
                sura, aya = "", ""
            
            text = text.strip()
            if text and has_diacritics(text):
                yield {
                    "text_diac": text,
                    "source": "quran_tanzil",
                    "variety": "classical",
                    "genre": "quran",
                    "sura": sura.strip(),
                    "aya": aya.strip(),
                }


def load_jsonl(path: str | Path, text_field: str = "text_diac") -> Iterator[dict]:
    """Load from JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                if text_field in record and has_diacritics(record[text_field]):
                    yield record


def load_parquet(path: str | Path, text_field: str = "text_diac") -> Iterator[dict]:
    """Load from Parquet file."""
    import pandas as pd
    df = pd.read_parquet(path)
    for _, row in df.iterrows():
        record = row.to_dict()
        if text_field in record and has_diacritics(str(record[text_field])):
            yield record
