# Layer 1: Data management
from src.data_layer.corpus import CorpusManager
from src.data_layer.dataset import DiacritizationDataset, collate_fn
from src.data_layer.loaders import (
    load_tashkeela, load_huggingface_dataset,
    load_quran_corpus, load_quran_tanzil_text,
    buckwalter_to_arabic,
)
