#!/usr/bin/env python3
"""
Download and prepare datasets for the Arabic diacritization pipeline.

Usage:
    python scripts/download_data.py --dataset tashkeela
    python scripts/download_data.py --dataset sadeed
    python scripts/download_data.py --dataset all
"""

import argparse
import os
import sys
from pathlib import Path


def download_tashkeela(output_dir: Path):
    """Download Tashkeela corpus from SourceForge."""
    print("=" * 60)
    print("Tashkeela Corpus")
    print("=" * 60)
    print(
        "Tashkeela must be downloaded manually from:\n"
        "  https://sourceforge.net/projects/tashkeela/\n\n"
        "Steps:\n"
        "  1. Download the archive from the link above\n"
        "  2. Extract to: data/raw/msa/tashkeela/\n"
        "  3. The directory should contain .txt files with diacritized text\n"
    )
    target = output_dir / "raw" / "msa" / "tashkeela"
    target.mkdir(parents=True, exist_ok=True)
    print(f"  Target directory created: {target}")


def download_sadeed(output_dir: Path):
    """Download Sadeed_Tashkeela from HuggingFace."""
    print("=" * 60)
    print("Sadeed_Tashkeela (HuggingFace)")
    print("=" * 60)
    try:
        from datasets import load_dataset
        
        print("Downloading Misraj/Sadeed_Tashkeela...")
        ds = load_dataset("Misraj/Sadeed_Tashkeela")
        
        target = output_dir / "raw" / "msa" / "sadeed"
        target.mkdir(parents=True, exist_ok=True)
        
        # Save as text files
        for split_name in ds:
            split_path = target / f"{split_name}.txt"
            with open(split_path, "w", encoding="utf-8") as f:
                for row in ds[split_name]:
                    text = row.get("text", "")
                    if text.strip():
                        f.write(text.strip() + "\n")
            print(f"  Saved {split_name}: {split_path}")
        
        print("Done!")
    except ImportError:
        print("ERROR: Install datasets library: pip install datasets")
    except Exception as e:
        print(f"ERROR: {e}")


def download_quran(output_dir: Path):
    """Instructions for Quranic Arabic Corpus."""
    print("=" * 60)
    print("Quranic Arabic Corpus")
    print("=" * 60)
    print(
        "Download manually from:\n"
        "  https://corpus.quran.com/download/\n\n"
        "The file is a TSV morphology file in Buckwalter transliteration.\n"
        "Format: LOCATION  FORM  TAG  FEATURES\n"
        "  e.g.  (1:1:1:1)  bi  P  PREFIX|bi+\n\n"
        "Steps:\n"
        "  1. Download the morphology file (quranic-corpus-morphology-0.4.txt)\n"
        "  2. Place in: data/raw/classical/quran/\n"
        "  3. The loader will convert Buckwalter to Arabic automatically\n"
    )
    target = output_dir / "raw" / "classical" / "quran"
    target.mkdir(parents=True, exist_ok=True)
    print(f"  Target directory created: {target}")


def main():
    parser = argparse.ArgumentParser(description="Download Arabic diacritization datasets")
    parser.add_argument(
        "--dataset",
        choices=["tashkeela", "sadeed", "quran", "all"],
        default="all",
        help="Dataset to download",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory (default: data/)",
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.dataset in ("tashkeela", "all"):
        download_tashkeela(output_dir)
        print()
    
    if args.dataset in ("sadeed", "all"):
        download_sadeed(output_dir)
        print()
    
    if args.dataset in ("quran", "all"):
        download_quran(output_dir)
        print()
    
    print("\nNext step: python scripts/prepare_data.py --config configs/default.yaml")


if __name__ == "__main__":
    main()
