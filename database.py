"""
SQLite layer for the Complaint Analyzer.

Tables:
- complaints (id, complaint, summary, category, severity, sentiment, response,
              status, created_at, customer_id)
- customers  (id, name, email, lifetime_value, notes, created_at)
- complaint_embeddings (complaint_id PK, embedding BLOB, model)

init_db() runs idempotent migrations so existing DBs pick up new columns/tables
without losing data. Safe to call on every app start.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import numpy as np

DB_PATH = "complaints.db"


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Init / migrations
# ---------------------------------------------------------------------------
def init_db() -> None:
    conn = _conn()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS customers (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            name            TEXT NOT NULL,
            email           TEXT,
            lifetime_value  REAL DEFAULT 0,
            notes           TEXT,
            created_at      TEXT
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS complaints (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            complaint   TEXT,
            summary     TEXT,
            category    TEXT,
            severity    TEXT,
            sentiment   TEXT,
            response    TEXT,
            status      TEXT DEFAULT 'Needs Review',
            created_at  TEXT,
            customer_id INTEGER REFERENCES customers(id) ON DELETE SET NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS complaint_embeddings (
            complaint_id INTEGER PRIMARY KEY,
            embedding    BLOB,
            model        TEXT,
            FOREIGN KEY (complaint_id) REFERENCES complaints(id) ON DELETE CASCADE
        )
        """
    )

    # Migrations: add columns to complaints if they don't exist
    cur.execute("PRAGMA table_info(complaints)")
    cols = {row[1] for row in cur.fetchall()}
    if "status" not in cols:
        cur.execute("ALTER TABLE complaints ADD COLUMN status TEXT DEFAULT 'Needs Review'")
    if "created_at" not in cols:
        cur.execute("ALTER TABLE complaints ADD COLUMN created_at TEXT")
    if "customer_id" not in cols:
        cur.execute("ALTER TABLE complaints ADD COLUMN customer_id INTEGER")

    # Backfills (ALTER TABLE doesn't apply DEFAULT to existing rows in SQLite)
    cur.execute("UPDATE complaints SET status = 'Needs Review' WHERE status IS NULL")
    cur.execute(
        "UPDATE complaints SET created_at = ? WHERE created_at IS NULL OR created_at = ''",
        (_now_iso(),),
    )

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Complaints
# ---------------------------------------------------------------------------
def save_complaint(data: dict) -> int:
    conn = _conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO complaints
            (complaint, summary, category, severity, sentiment, response,
             status, created_at, customer_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            data["complaint"],
            data["summary"],
            data["category"],
            data["severity"],
            data["sentiment"],
            data["response"],
            data.get("status", "Needs Review"),
            _now_iso(),
            data.get("customer_id"),
        ),
    )
    conn.commit()
    cid = cur.lastrowid
    conn.close()
    return cid


def fetch_all_complaints() -> list[dict]:
    conn = _conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT c.id, c.complaint, c.summary, c.category, c.severity, c.sentiment,
               c.response, c.status, c.created_at, c.customer_id, cu.name
        FROM complaints c
        LEFT JOIN customers cu ON c.customer_id = cu.id
        ORDER BY c.id DESC
        """
    )
    rows = cur.fetchall()
    conn.close()

    return [
        {
            "id": r[0],
            "complaint": r[1],
            "summary": r[2],
            "category": r[3],
            "severity": r[4],
            "sentiment": r[5],
            "response": r[6],
            "status": r[7] or "Needs Review",
            "created_at": r[8],
            "customer_id": r[9],
            "customer_name": r[10],
        }
        for r in rows
    ]


def fetch_complaint(cid: int) -> dict | None:
    conn = _conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT c.id, c.complaint, c.summary, c.category, c.severity, c.sentiment,
               c.response, c.status, c.created_at, c.customer_id, cu.name
        FROM complaints c
        LEFT JOIN customers cu ON c.customer_id = cu.id
        WHERE c.id = ?
        """,
        (cid,),
    )
    r = cur.fetchone()
    conn.close()
    if not r:
        return None
    return {
        "id": r[0], "complaint": r[1], "summary": r[2], "category": r[3],
        "severity": r[4], "sentiment": r[5], "response": r[6],
        "status": r[7] or "Needs Review", "created_at": r[8],
        "customer_id": r[9], "customer_name": r[10],
    }


def update_complaint(data: dict) -> None:
    conn = _conn()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE complaints
        SET complaint=?, summary=?, category=?, severity=?, sentiment=?, response=?
        WHERE id=?
        """,
        (
            data["complaint"], data["summary"], data["category"],
            data["severity"], data["sentiment"], data["response"],
            data["id"],
        ),
    )
    conn.commit()
    conn.close()


def update_status(complaint_id: int, status: str) -> None:
    conn = _conn()
    cur = conn.cursor()
    cur.execute("UPDATE complaints SET status=? WHERE id=?", (status, complaint_id))
    conn.commit()
    conn.close()


def delete_complaint(complaint_id: int) -> None:
    conn = _conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM complaints WHERE id=?", (complaint_id,))
    conn.commit()
    conn.close()


def clear_complaints() -> None:
    conn = _conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM complaints")
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Customers
# ---------------------------------------------------------------------------
def create_customer(name: str, email: str | None = None,
                    lifetime_value: float = 0.0, notes: str | None = None) -> int:
    conn = _conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO customers (name, email, lifetime_value, notes, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (name, email, lifetime_value, notes, _now_iso()),
    )
    conn.commit()
    cid = cur.lastrowid
    conn.close()
    return cid


def fetch_all_customers() -> list[dict]:
    conn = _conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT c.id, c.name, c.email, c.lifetime_value, c.notes, c.created_at,
               COUNT(co.id) AS complaint_count
        FROM customers c
        LEFT JOIN complaints co ON co.customer_id = c.id
        GROUP BY c.id
        ORDER BY c.name COLLATE NOCASE
        """
    )
    rows = cur.fetchall()
    conn.close()
    return [
        {
            "id": r[0], "name": r[1], "email": r[2],
            "lifetime_value": r[3] or 0.0, "notes": r[4],
            "created_at": r[5], "complaint_count": r[6],
        }
        for r in rows
    ]


def fetch_customer(cid: int) -> dict | None:
    conn = _conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, name, email, lifetime_value, notes, created_at
        FROM customers WHERE id = ?
        """,
        (cid,),
    )
    r = cur.fetchone()
    conn.close()
    if not r:
        return None
    return {
        "id": r[0], "name": r[1], "email": r[2],
        "lifetime_value": r[3] or 0.0, "notes": r[4], "created_at": r[5],
    }


def fetch_customer_complaints(cid: int, limit: int = 10) -> list[dict]:
    conn = _conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, complaint, summary, category, severity, sentiment,
               response, status, created_at
        FROM complaints
        WHERE customer_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (cid, limit),
    )
    rows = cur.fetchall()
    conn.close()
    return [
        {
            "id": r[0], "complaint": r[1], "summary": r[2],
            "category": r[3], "severity": r[4], "sentiment": r[5],
            "response": r[6], "status": r[7], "created_at": r[8],
        }
        for r in rows
    ]


def update_customer(cid: int, name: str, email: str | None,
                    lifetime_value: float, notes: str | None) -> None:
    conn = _conn()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE customers SET name=?, email=?, lifetime_value=?, notes=?
        WHERE id=?
        """,
        (name, email, lifetime_value, notes, cid),
    )
    conn.commit()
    conn.close()


def delete_customer(cid: int) -> None:
    conn = _conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM customers WHERE id=?", (cid,))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def save_embedding(complaint_id: int, vector, model: str = "nomic-embed-text") -> None:
    arr = np.asarray(vector, dtype=np.float32)
    blob = arr.tobytes()
    conn = _conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO complaint_embeddings (complaint_id, embedding, model)
        VALUES (?, ?, ?)
        """,
        (complaint_id, blob, model),
    )
    conn.commit()
    conn.close()


def fetch_all_embeddings() -> dict[int, np.ndarray]:
    """Return {complaint_id: vector} for all stored embeddings."""
    conn = _conn()
    cur = conn.cursor()
    cur.execute("SELECT complaint_id, embedding FROM complaint_embeddings")
    rows = cur.fetchall()
    conn.close()

    out: dict[int, np.ndarray] = {}
    for cid, blob in rows:
        if blob:
            out[cid] = np.frombuffer(blob, dtype=np.float32)
    return out


def fetch_complaints_without_embeddings() -> list[dict]:
    conn = _conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT c.id, c.complaint
        FROM complaints c
        LEFT JOIN complaint_embeddings e ON c.id = e.complaint_id
        WHERE e.complaint_id IS NULL AND c.complaint IS NOT NULL
        """
    )
    rows = cur.fetchall()
    conn.close()
    return [{"id": r[0], "complaint": r[1]} for r in rows]


def count_complaints_without_embeddings() -> int:
    conn = _conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COUNT(*)
        FROM complaints c
        LEFT JOIN complaint_embeddings e ON c.id = e.complaint_id
        WHERE e.complaint_id IS NULL
        """
    )
    n = cur.fetchone()[0]
    conn.close()
    return n