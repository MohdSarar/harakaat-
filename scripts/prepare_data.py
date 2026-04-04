#!/usr/bin/env python3
"""
Prepare data: ingest → normalize → align → build vocab → build lexicon → split.

Usage:
    python scripts/prepare_data.py --config configs/default.yaml
"""

import argparse
import json
from pathlib import Path

from src.utils.config import Config
from src.utils.vocab import CharVocab
from src.data_layer.corpus import CorpusManager
from src.data_layer.loaders import load_tashkeela, load_huggingface_dataset, load_quran_corpus
from src.linguistic.lexicon import FrequencyLexicon


def main():
    parser = argparse.ArgumentParser(description="Prepare diacritization data")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    dc = config.data.to_dict()
    nc = config.normalization.to_dict()

    # Initialize corpus manager
    corpus = CorpusManager(
        aligned_dir=dc.get("aligned_dir", "data/aligned"),
        metadata_db=dc.get("metadata_dir", "data/metadata") + "/corpus.duckdb",
        normalization_config=nc,
    )

    # Ingest sources
    for source in dc.get("sources", []):
        name = source["name"]
        variety = source.get("variety", "msa")
        src_type = source.get("type", "local")
        src_path = source.get("path", "")
        
        print(f"\n--- Ingesting: {name} ({src_type}) ---")
        
        try:
            if src_type == "local" and source.get("format") == "txt":
                iterator = load_tashkeela(src_path)
            elif src_type == "local" and source.get("format") == "quran_morphology":
                # Quranic Arabic Corpus morphology TSV (Buckwalter)
                morph_files = list(Path(src_path).glob("*.txt"))
                if morph_files:
                    iterator = load_quran_corpus(morph_files[0])
                else:
                    print(f"  No .txt morphology files found in {src_path}, skipping.")
                    continue
            elif src_type == "local" and source.get("format") == "quran_tanzil":
                # Tanzil plain-text Quran
                text_files = list(Path(src_path).glob("*.txt"))
                if text_files:
                    from src.data_layer.loaders import load_quran_tanzil_text
                    iterator = load_quran_tanzil_text(text_files[0])
                else:
                    print(f"  No .txt files found in {src_path}, skipping.")
                    continue
            elif src_type == "huggingface":
                iterator = load_huggingface_dataset(
                    repo=source.get("repo", ""),
                    variety=variety,
                )
            else:
                print(f"  Unknown source type: {src_type}, skipping.")
                continue
            
            count = corpus.ingest(iterator, source_name=name)
            print(f"  Ingested {count} sentences from {name}")
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")
        except Exception as e:
            print(f"  ERROR: {e}")

    # Show stats
    stats = corpus.stats()
    print(f"\n--- Corpus Stats ---")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    if stats["total"] == 0:
        print("\nWARNING: No data ingested. See docs/MANUAL_STEPS.md for data download instructions.")
        return

    # Create splits
    print("\n--- Creating splits ---")
    split_ratios = dc.get("split_ratios", {})
    counts = corpus.create_splits(
        output_dir=dc.get("splits_dir", "data/splits"),
        train_ratio=split_ratios.get("train", 0.85),
        valid_ratio=split_ratios.get("valid", 0.10),
    )
    print(f"  Splits: {counts}")

    # Build character vocabulary from training data
    print("\n--- Building character vocabulary ---")
    train_path = Path(dc.get("splits_dir", "data/splits")) / "train" / "data.jsonl"
    texts = []
    diac_texts = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            texts.append(record["text_undiac"])
            diac_texts.append(record["text_diac"])

    vocab = CharVocab()
    vocab.build_from_texts(texts)
    vocab_path = Path(dc.get("lexicons_dir", "data/lexicons")) / "char_vocab.json"
    vocab.save(vocab_path)
    print(f"  Vocabulary size: {len(vocab)} (saved to {vocab_path})")

    # Build frequency lexicon
    print("\n--- Building frequency lexicon ---")
    ling_config = config.linguistic.to_dict()
    lexicon = FrequencyLexicon()
    lexicon.build_from_corpus(
        diac_texts,
        min_frequency=ling_config.get("lexicon", {}).get("min_frequency", 3),
        max_entries=ling_config.get("lexicon", {}).get("max_entries", 500_000),
    )
    lex_path = Path(dc.get("lexicons_dir", "data/lexicons")) / "frequency_lexicon.json"
    lexicon.save(lex_path)
    print(f"  Lexicon entries: {len(lexicon)} (saved to {lex_path})")

    print("\n✓ Data preparation complete!")
    print(f"  Next step: python scripts/train.py --config {args.config}")


if __name__ == "__main__":
    main()
