"""
SQLite database for Xpert session & interaction persistence.

Tables:
  - sessions: tracks login sessions (student/doctor)
  - interactions: stores each image-analysis result
  - messages: stores chat messages for conversational context
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent / "xpert.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    conn = _connect()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id          TEXT PRIMARY KEY,
            student_id  TEXT NOT NULL DEFAULT 'anonymous',
            role        TEXT NOT NULL DEFAULT 'student',
            created_at  TEXT NOT NULL,
            ended_at    TEXT
        );

        CREATE TABLE IF NOT EXISTS interactions (
            id                      TEXT PRIMARY KEY,
            session_id              TEXT NOT NULL,
            image_filename          TEXT,
            prediction              TEXT,
            pneumonia_score         REAL,
            student_input           TEXT,
            competency_gap          TEXT,
            adaptive_summary        TEXT,
            adaptive_recommendation TEXT,
            quality_label           TEXT,
            quality_score           REAL,
            gradcam_regions         TEXT,
            created_at              TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );

        CREATE TABLE IF NOT EXISTS messages (
            id           TEXT PRIMARY KEY,
            session_id   TEXT NOT NULL,
            sender       TEXT NOT NULL,
            content      TEXT NOT NULL,
            message_type TEXT NOT NULL DEFAULT 'text',
            metadata     TEXT,
            created_at   TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );
    """)
    conn.commit()
    conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uid() -> str:
    return uuid.uuid4().hex[:12]


# ── Sessions ──────────────────────────────────────────────────────────────── #

def create_session(role: str = "student", student_id: str = "anonymous") -> str:
    sid = _uid()
    conn = _connect()
    conn.execute(
        "INSERT INTO sessions (id, student_id, role, created_at) VALUES (?, ?, ?, ?)",
        (sid, student_id, role, _now()),
    )
    conn.commit()
    conn.close()
    return sid


def end_session(session_id: str) -> None:
    conn = _connect()
    conn.execute(
        "UPDATE sessions SET ended_at = ? WHERE id = ?",
        (_now(), session_id),
    )
    conn.commit()
    conn.close()


def get_session(session_id: str) -> dict | None:
    conn = _connect()
    row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if not row:
        conn.close()
        return None
    session = dict(row)
    session["interactions"] = [
        _parse_interaction(r)
        for r in conn.execute(
            "SELECT * FROM interactions WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        ).fetchall()
    ]
    session["messages"] = [
        _parse_message(r)
        for r in conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        ).fetchall()
    ]
    conn.close()
    return session


def list_sessions(role: str | None = None) -> list[dict]:
    conn = _connect()
    if role:
        rows = conn.execute(
            "SELECT s.*, COUNT(i.id) as interaction_count "
            "FROM sessions s LEFT JOIN interactions i ON s.id = i.session_id "
            "WHERE s.role = ? GROUP BY s.id ORDER BY s.created_at DESC",
            (role,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT s.*, COUNT(i.id) as interaction_count "
            "FROM sessions s LEFT JOIN interactions i ON s.id = i.session_id "
            "GROUP BY s.id ORDER BY s.created_at DESC",
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Interactions ──────────────────────────────────────────────────────────── #

def add_interaction(
    session_id: str,
    *,
    image_filename: str | None = None,
    prediction: str | None = None,
    pneumonia_score: float | None = None,
    student_input: str | None = None,
    competency_gap: str | None = None,
    adaptive_summary: str | None = None,
    adaptive_recommendation: str | None = None,
    quality_label: str | None = None,
    quality_score: float | None = None,
    gradcam_regions: list | None = None,
) -> str:
    iid = _uid()
    conn = _connect()
    conn.execute(
        """INSERT INTO interactions
           (id, session_id, image_filename, prediction, pneumonia_score,
            student_input, competency_gap, adaptive_summary, adaptive_recommendation,
            quality_label, quality_score, gradcam_regions, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            iid, session_id, image_filename, prediction, pneumonia_score,
            student_input, competency_gap, adaptive_summary, adaptive_recommendation,
            quality_label, quality_score, json.dumps(gradcam_regions or []),
            _now(),
        ),
    )
    conn.commit()
    conn.close()
    return iid


def _parse_interaction(row: sqlite3.Row) -> dict:
    d = dict(row)
    try:
        d["gradcam_regions"] = json.loads(d.get("gradcam_regions") or "[]")
    except (json.JSONDecodeError, TypeError):
        d["gradcam_regions"] = []
    return d


# ── Messages ──────────────────────────────────────────────────────────────── #

def add_message(
    session_id: str,
    sender: str,
    content: str,
    message_type: str = "text",
    metadata: dict | None = None,
) -> str:
    mid = _uid()
    conn = _connect()
    conn.execute(
        """INSERT INTO messages
           (id, session_id, sender, content, message_type, metadata, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (mid, session_id, sender, content, message_type, json.dumps(metadata or {}), _now()),
    )
    conn.commit()
    conn.close()
    return mid


def get_session_messages(session_id: str, limit: int = 50) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
        (session_id, limit),
    ).fetchall()
    conn.close()
    return [_parse_message(r) for r in reversed(rows)]


def _parse_message(row: sqlite3.Row) -> dict:
    d = dict(row)
    try:
        d["metadata"] = json.loads(d.get("metadata") or "{}")
    except (json.JSONDecodeError, TypeError):
        d["metadata"] = {}
    return d


# ── Reports ───────────────────────────────────────────────────────────────── #

def get_session_report(session_id: str) -> dict | None:
    """Generate a summary report for a session."""
    session = get_session(session_id)
    if not session:
        return None

    interactions = session["interactions"]
    total = len(interactions)

    if total == 0:
        return {
            "session_id": session_id,
            "student_id": session["student_id"],
            "role": session["role"],
            "created_at": session["created_at"],
            "total_analyses": 0,
            "accuracy_percent": 0,
            "avg_pneumonia_score": 0,
            "competency_gaps": [],
            "recommendations": [],
            "interactions": [],
            "message_count": len(session["messages"]),
        }

    correct = sum(
        1
        for i in interactions
        if i.get("student_input")
        and i.get("prediction")
        and i["student_input"].strip().lower() == i["prediction"].strip().lower()
    )
    accuracy = (correct / total * 100) if total > 0 else 0

    scores = [i["pneumonia_score"] for i in interactions if i.get("pneumonia_score") is not None]
    avg_score = (sum(scores) / len(scores)) if scores else 0

    gaps = [
        i["competency_gap"]
        for i in interactions
        if i.get("competency_gap") and "no competency gap" not in i["competency_gap"].lower()
    ]
    recommendations = [
        i["adaptive_recommendation"]
        for i in interactions
        if i.get("adaptive_recommendation")
    ]

    return {
        "session_id": session_id,
        "student_id": session["student_id"],
        "role": session["role"],
        "created_at": session["created_at"],
        "ended_at": session.get("ended_at"),
        "total_analyses": total,
        "accuracy_percent": round(accuracy, 1),
        "avg_pneumonia_score": round(avg_score, 4),
        "competency_gaps": gaps,
        "recommendations": recommendations,
        "interactions": interactions,
        "message_count": len(session["messages"]),
    }
