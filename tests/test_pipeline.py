"""
Tests for the Arabic diacritization pipeline.
"""

import pytest
from src.utils import (
    strip_diacritics, extract_diacritics, has_diacritics,
    diacritic_density, normalize_diac_sequence, word_boundaries,
    is_arabic_char, FATHAH, DAMMAH, KASRAH, SHADDAH, SUKUN,
    DIAC_LABELS, DIAC_LABEL_TO_IDX, NUM_DIAC_CLASSES,
)
from src.normalization import normalize_text, ArabicNormalizer
from src.utils.vocab import CharVocab
from src.linguistic.lexicon import FrequencyLexicon


# ---- Utils tests ----

class TestStripDiacritics:
    def test_basic(self):
        assert strip_diacritics("كَتَبَ") == "كتب"
    
    def test_no_diacritics(self):
        assert strip_diacritics("كتب") == "كتب"
    
    def test_empty(self):
        assert strip_diacritics("") == ""
    
    def test_mixed(self):
        result = strip_diacritics("الْعَرَبِيَّة")
        assert result == "العربية"


class TestExtractDiacritics:
    def test_basic(self):
        diacs = extract_diacritics("كَتَبَ")
        assert len(diacs) == 3
        assert diacs[0] == FATHAH
        assert diacs[1] == FATHAH
        assert diacs[2] == FATHAH
    
    def test_no_diacritics(self):
        diacs = extract_diacritics("كتب")
        assert all(d == "" for d in diacs)
    
    def test_shaddah_combo(self):
        diacs = extract_diacritics("شَدَّ")
        assert SHADDAH in diacs[2]


class TestHasDiacritics:
    def test_with_diacritics(self):
        assert has_diacritics("كَتَبَ") is True
    
    def test_without(self):
        assert has_diacritics("كتب") is False


class TestDiacriticDensity:
    def test_full(self):
        d = diacritic_density("كَتَبَ")
        assert d == 1.0
    
    def test_empty(self):
        d = diacritic_density("كتب")
        assert d == 0.0


class TestNormalizeDiacSequence:
    def test_single(self):
        assert normalize_diac_sequence(FATHAH) == FATHAH
    
    def test_shaddah_first(self):
        result = normalize_diac_sequence(FATHAH + SHADDAH)
        assert result == SHADDAH + FATHAH
    
    def test_empty(self):
        assert normalize_diac_sequence("") == ""


class TestWordBoundaries:
    def test_basic(self):
        wb = word_boundaries("كتب الكتاب")
        assert len(wb) > 0


# ---- Normalization tests ----

class TestNormalization:
    def test_tatweel_removal(self):
        result = normalize_text("عـربـي")
        assert "ـ" not in result
    
    def test_alif_normalization(self):
        result = normalize_text("أحمد إبراهيم آمنة", normalize_alif=True)
        assert "أ" not in result
        assert "إ" not in result
        assert "آ" not in result
    
    def test_whitespace(self):
        result = normalize_text("كلمة   كلمة")
        assert "   " not in result
    
    def test_preserves_diacritics(self):
        text = "كَتَبَ"
        result = normalize_text(text)
        assert has_diacritics(result)


# ---- Vocabulary tests ----

class TestCharVocab:
    def test_build_and_encode(self):
        vocab = CharVocab()
        vocab.build_from_texts(["كتب الكتاب"])
        encoded = vocab.encode("كتب")
        assert len(encoded) == 3
        assert all(isinstance(i, int) for i in encoded)
    
    def test_special_tokens(self):
        vocab = CharVocab()
        assert vocab.pad_idx == 0
        assert vocab.unk_idx == 1
    
    def test_save_load(self, tmp_path):
        vocab = CharVocab()
        vocab.build_from_texts(["مرحبا"])
        path = tmp_path / "vocab.json"
        vocab.save(path)
        loaded = CharVocab.load(path)
        assert len(loaded) == len(vocab)


# ---- Lexicon tests ----

class TestFrequencyLexicon:
    def test_build_and_lookup(self):
        lex = FrequencyLexicon()
        texts = ["كَتَبَ الْكِتَابَ"] * 5
        lex.build_from_corpus(texts, min_frequency=1)
        assert len(lex) > 0
    
    def test_best_form(self):
        lex = FrequencyLexicon()
        for _ in range(10):
            lex.add("كَتَبَ")
        for _ in range(5):
            lex.add("كُتُبٌ")
        lex._finalize(min_frequency=1, max_entries=100)
        best = lex.best_form("كتب")
        assert best == "كَتَبَ"
    
    def test_ambiguity(self):
        lex = FrequencyLexicon()
        for _ in range(10):
            lex.add("كَتَبَ")
        for _ in range(5):
            lex.add("كُتُبٌ")
        lex._finalize(min_frequency=1, max_entries=100)
        assert lex.is_ambiguous("كتب") is True


# ---- Diacritic label mapping tests ----

class TestDiacLabels:
    def test_count(self):
        assert NUM_DIAC_CLASSES == 16
    
    def test_no_diac_is_zero(self):
        assert DIAC_LABEL_TO_IDX[""] == 0
    
    def test_roundtrip(self):
        for idx, label in enumerate(DIAC_LABELS):
            assert DIAC_LABEL_TO_IDX[label] == idx


# ---- Buckwalter / Quran loader tests ----

class TestBuckwalterConversion:
    def test_bismillah(self):
        from src.data_layer.loaders import buckwalter_to_arabic
        # bi → بِ
        result = buckwalter_to_arabic("bi")
        assert result == "بِ"
    
    def test_allah(self):
        from src.data_layer.loaders import buckwalter_to_arabic
        result = buckwalter_to_arabic("{ll~ahi")
        assert "لل" in strip_diacritics(result)  # contains lam-lam
    
    def test_shadda(self):
        from src.data_layer.loaders import buckwalter_to_arabic
        result = buckwalter_to_arabic("r~aHiymi")
        assert has_diacritics(result)
    
    def test_empty(self):
        from src.data_layer.loaders import buckwalter_to_arabic
        assert buckwalter_to_arabic("") == ""
    
    def test_non_buckwalter_passthrough(self):
        from src.data_layer.loaders import buckwalter_to_arabic
        # digits and unknown chars pass through
        assert buckwalter_to_arabic("123") == "123"


class TestQuranLocationParsing:
    def test_parse(self):
        from src.data_layer.loaders import _parse_quran_location
        sura, aya, word, seg = _parse_quran_location("(1:2:3:4)")
        assert sura == 1
        assert aya == 2
        assert word == 3
        assert seg == 4


class TestQuranCorpusLoader:
    def test_load_from_sample(self, tmp_path):
        """Test loading a minimal Quran corpus sample."""
        sample = """# Comment line
LOCATION\tFORM\tTAG\tFEATURES
(1:1:1:1)\tbi\tP\tPREFIX|bi+
(1:1:1:2)\tsomi\tN\tSTEM|POS:N|LEM:{som|ROOT:smw|M|GEN
(1:1:2:1)\t{ll~ahi\tPN\tSTEM|POS:PN|LEM:{ll~ah|ROOT:Alh|GEN
(1:1:3:1)\t{l\tDET\tPREFIX|Al+
(1:1:3:2)\tr~aHoma`ni\tADJ\tSTEM|POS:ADJ|LEM:r~aHoma`n|ROOT:rHm|MS|GEN
(1:1:4:1)\t{l\tDET\tPREFIX|Al+
(1:1:4:2)\tr~aHiymi\tADJ\tSTEM|POS:ADJ|LEM:r~aHiym|ROOT:rHm|MS|GEN
"""
        corpus_file = tmp_path / "quran_morphology.txt"
        corpus_file.write_text(sample, encoding="utf-8")
        
        from src.data_layer.loaders import load_quran_corpus
        results = list(load_quran_corpus(corpus_file))
        
        assert len(results) == 1  # one verse (1:1)
        verse = results[0]
        assert verse["source"] == "quran"
        assert verse["variety"] == "classical"
        assert verse["sura"] == "1"
        assert verse["aya"] == "1"
        assert has_diacritics(verse["text_diac"])
        # Should contain 4 words (bismillah)
        words = verse["text_diac"].split()
        assert len(words) == 4
