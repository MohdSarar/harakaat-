"""
Human-in-the-Loop (HITL) review system.

Manages a queue of low-confidence predictions for human review,
stores corrections, and exports corrected data for retraining.

Uses SQLite for the review queue.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class ReviewItem:
    """An item pending human review."""
    id: int
    text_undiac: str
    text_diac_auto: str
    confidence: float
    source: str
    status: str  # "pending", "approved", "corrected", "rejected"
    text_diac_corrected: Optional[str] = None
    reviewer: Optional[str] = None
    created_at: float = 0.0
    reviewed_at: Optional[float] = None


class HITLManager:
    """
    Human-in-the-Loop correction manager.
    
    Workflow:
    1. Auto-diacritized text with low confidence → enqueued for review
    2. Human reviewer approves, corrects, or rejects
    3. Corrected data exported as JSONL for retraining
    """

    def __init__(self, db_path: str | Path = "data/metadata/hitl.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = self._connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS review_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text_undiac TEXT NOT NULL,
                text_diac_auto TEXT NOT NULL,
                confidence REAL NOT NULL,
                source TEXT DEFAULT '',
                status TEXT DEFAULT 'pending',
                text_diac_corrected TEXT,
                reviewer TEXT,
                created_at REAL NOT NULL,
                reviewed_at REAL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_status ON review_queue(status)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_confidence ON review_queue(confidence)
        """)
        conn.commit()
        conn.close()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def enqueue(
        self,
        text_undiac: str,
        text_diac_auto: str,
        confidence: float,
        source: str = "",
    ) -> int:
        """Add an item to the review queue. Returns the item ID."""
        conn = self._connect()
        cursor = conn.execute(
            """INSERT INTO review_queue 
            (text_undiac, text_diac_auto, confidence, source, created_at)
            VALUES (?, ?, ?, ?, ?)""",
            (text_undiac, text_diac_auto, confidence, source, time.time()),
        )
        item_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return item_id

    def get_pending(self, limit: int = 50, sort_by: str = "confidence") -> list[ReviewItem]:
        """Get pending items, sorted by confidence (lowest first)."""
        conn = self._connect()
        order = "confidence ASC" if sort_by == "confidence" else "created_at ASC"
        rows = conn.execute(
            f"SELECT * FROM review_queue WHERE status = 'pending' ORDER BY {order} LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return [self._row_to_item(r) for r in rows]

    def approve(self, item_id: int, reviewer: str = "human") -> None:
        """Approve the automatic diacritization (it's correct)."""
        conn = self._connect()
        conn.execute(
            """UPDATE review_queue 
            SET status = 'approved', reviewer = ?, reviewed_at = ? 
            WHERE id = ?""",
            (reviewer, time.time(), item_id),
        )
        conn.commit()
        conn.close()

    def correct(self, item_id: int, corrected_text: str, reviewer: str = "human") -> None:
        """Submit a human correction."""
        conn = self._connect()
        conn.execute(
            """UPDATE review_queue 
            SET status = 'corrected', text_diac_corrected = ?, reviewer = ?, reviewed_at = ?
            WHERE id = ?""",
            (corrected_text, reviewer, time.time(), item_id),
        )
        conn.commit()
        conn.close()

    def reject(self, item_id: int, reviewer: str = "human") -> None:
        """Reject the item (bad data, skip)."""
        conn = self._connect()
        conn.execute(
            """UPDATE review_queue 
            SET status = 'rejected', reviewer = ?, reviewed_at = ?
            WHERE id = ?""",
            (reviewer, time.time(), item_id),
        )
        conn.commit()
        conn.close()

    def export_corrections(self, output_path: str | Path) -> int:
        """
        Export approved and corrected items as JSONL for retraining.
        Returns count of exported items.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM review_queue WHERE status IN ('approved', 'corrected')"
        ).fetchall()
        conn.close()

        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for row in rows:
                item = self._row_to_item(row)
                # Use corrected text if available, otherwise auto
                text_diac = item.text_diac_corrected or item.text_diac_auto
                record = {
                    "text_diac": text_diac,
                    "text_undiac": item.text_undiac,
                    "source": f"hitl_{item.source}",
                    "annotation_mode": "corrected" if item.text_diac_corrected else "approved",
                    "variety": "msa",
                    "genre": "hitl",
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        return count

    def stats(self) -> dict:
        """Return queue statistics."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT status, COUNT(*) FROM review_queue GROUP BY status"
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM review_queue").fetchone()[0]
        avg_conf = conn.execute(
            "SELECT AVG(confidence) FROM review_queue WHERE status = 'pending'"
        ).fetchone()[0]
        conn.close()

        return {
            "total": total,
            "by_status": dict(rows),
            "avg_pending_confidence": round(avg_conf or 0, 4),
        }

    def _row_to_item(self, row: tuple) -> ReviewItem:
        return ReviewItem(
            id=row[0],
            text_undiac=row[1],
            text_diac_auto=row[2],
            confidence=row[3],
            source=row[4],
            status=row[5],
            text_diac_corrected=row[6],
            reviewer=row[7],
            created_at=row[8],
            reviewed_at=row[9],
        )
