"""
Corpus manager — ingests, indexes, and manages diacritized text data.
Uses DuckDB for metadata and JSONL for the actual text pairs.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Iterator, Optional

import duckdb
from tqdm import tqdm

from src.utils import strip_diacritics, has_diacritics, diacritic_density
from src.normalization import normalize_text


class CorpusManager:
    """
    Central corpus manager.
    
    - Ingests raw diacritized text from various loaders
    - Normalizes and generates undiacritized pairs
    - Stores aligned pairs as JSONL
    - Tracks metadata in DuckDB
    """

    def __init__(
        self,
        aligned_dir: str | Path = "data/aligned",
        metadata_db: str | Path = "data/metadata/corpus.duckdb",
        normalization_config: Optional[dict] = None,
    ):
        self.aligned_dir = Path(aligned_dir)
        self.aligned_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_db = Path(metadata_db)
        self.metadata_db.parent.mkdir(parents=True, exist_ok=True)
        self.norm_config = normalization_config or {}
        self._init_db()

    def _init_db(self):
        """Initialize metadata database."""
        self.db = duckdb.connect(str(self.metadata_db))
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS sentences (
                id VARCHAR PRIMARY KEY,
                source VARCHAR,
                variety VARCHAR,
                genre VARCHAR,
                char_count INTEGER,
                word_count INTEGER,
                diac_density FLOAT,
                split VARCHAR DEFAULT 'unassigned',
                quality_score FLOAT DEFAULT 1.0,
                annotation_mode VARCHAR DEFAULT 'automatic'
            )
        """)

    def ingest(
        self,
        data_iterator: Iterator[dict],
        source_name: str = "unknown",
        max_records: Optional[int] = None,
    ) -> int:
        """
        Ingest diacritized records from a data loader.
        
        Each record must have 'text_diac' field.
        Returns number of records ingested.
        """
        output_file = self.aligned_dir / f"{source_name}.jsonl"
        count = 0

        with open(output_file, "a", encoding="utf-8") as f:
            for record in tqdm(data_iterator, desc=f"Ingesting {source_name}"):
                if max_records and count >= max_records:
                    break

                text_diac = record.get("text_diac", "")
                if not text_diac or not has_diacritics(text_diac):
                    continue

                # Normalize the diacritized text
                text_diac_norm = normalize_text(text_diac, **self.norm_config)
                if not text_diac_norm:
                    continue

                # Generate undiacritized version
                text_undiac = strip_diacritics(text_diac_norm)

                # Compute ID
                record_id = hashlib.sha256(
                    text_diac_norm.encode("utf-8")
                ).hexdigest()[:16]

                # Write aligned pair
                aligned = {
                    "id": record_id,
                    "text_diac": text_diac_norm,
                    "text_undiac": text_undiac,
                    "source": record.get("source", source_name),
                    "variety": record.get("variety", "msa"),
                    "genre": record.get("genre", "general"),
                }
                f.write(json.dumps(aligned, ensure_ascii=False) + "\n")

                # Insert metadata
                density = diacritic_density(text_diac_norm)
                try:
                    self.db.execute(
                        """INSERT OR IGNORE INTO sentences 
                        (id, source, variety, genre, char_count, word_count, diac_density)
                        VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        [
                            record_id,
                            aligned["source"],
                            aligned["variety"],
                            aligned["genre"],
                            len(text_undiac),
                            len(text_undiac.split()),
                            density,
                        ],
                    )
                except Exception:
                    pass  # duplicate ID — skip

                count += 1

        return count

    def create_splits(
        self,
        output_dir: str | Path = "data/splits",
        train_ratio: float = 0.85,
        valid_ratio: float = 0.10,
        seed: int = 42,
    ) -> dict[str, int]:
        """
        Split corpus into train/valid/test.
        Returns counts per split.
        """
        output_dir = Path(output_dir)
        for split in ("train", "valid", "test"):
            (output_dir / split).mkdir(parents=True, exist_ok=True)

        # Load all aligned data
        all_records = []
        for jsonl_file in self.aligned_dir.glob("*.jsonl"):
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        all_records.append(json.loads(line))

        # Shuffle deterministically
        import random
        rng = random.Random(seed)
        rng.shuffle(all_records)

        n = len(all_records)
        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)

        splits = {
            "train": all_records[:n_train],
            "valid": all_records[n_train : n_train + n_valid],
            "test": all_records[n_train + n_valid :],
        }

        counts = {}
        for split_name, records in splits.items():
            out_path = output_dir / split_name / "data.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts[split_name] = len(records)

            # Update metadata
            for rec in records:
                self.db.execute(
                    "UPDATE sentences SET split = ? WHERE id = ?",
                    [split_name, rec["id"]],
                )

        return counts

    def stats(self) -> dict[str, Any]:
        """Return corpus statistics."""
        result = self.db.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(DISTINCT source) as sources,
                AVG(char_count) as avg_chars,
                AVG(word_count) as avg_words,
                AVG(diac_density) as avg_density
            FROM sentences
        """).fetchone()
        
        split_counts = self.db.execute("""
            SELECT split, COUNT(*) FROM sentences GROUP BY split
        """).fetchall()
        
        variety_counts = self.db.execute("""
            SELECT variety, COUNT(*) FROM sentences GROUP BY variety
        """).fetchall()

        return {
            "total": result[0],
            "sources": result[1],
            "avg_chars": round(result[2] or 0, 1),
            "avg_words": round(result[3] or 0, 1),
            "avg_diac_density": round(result[4] or 0, 3),
            "splits": dict(split_counts),
            "varieties": dict(variety_counts),
        }
