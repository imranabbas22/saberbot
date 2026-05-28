"""
Feedback & Session Tracking — SQLite-backed.
Tracks per-visitor request counts, NPS feedback, session timeouts.
"""
import os
import sqlite3
import uuid
import time
import shutil
from pathlib import Path

DB_PATH = os.getenv("FEEDBACK_DB", str(Path(__file__).parent.parent / "feedback.db"))
MAX_REQUESTS_PER_SESSION = int(os.getenv("MAX_REQUESTS_PER_SESSION", "5"))
SESSION_TIMEOUT_SECONDS = int(os.getenv("SESSION_TIMEOUT_SECONDS", "180"))  # 3 min


def _get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = _get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            requests_used INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            last_active REAL NOT NULL,
            expired INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            nps_score INTEGER NOT NULL CHECK(nps_score >= 0 AND nps_score <= 10),
            comment_text TEXT DEFAULT '',
            created_at REAL NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );
        CREATE TABLE IF NOT EXISTS stats (
            key TEXT PRIMARY KEY,
            value INTEGER NOT NULL DEFAULT 0
        );
        INSERT OR IGNORE INTO stats (key, value) VALUES ('total_requests', 0);
    """)
    conn.commit()
    conn.close()


def get_or_create_session(session_id: str = None) -> dict:
    conn = _get_db()
    now = time.time()

    if not session_id:
        session_id = str(uuid.uuid4())

    row = conn.execute(
        "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()

    if not row:
        conn.execute(
            "INSERT INTO sessions (session_id, requests_used, created_at, last_active) VALUES (?, 0, ?, ?)",
            (session_id, now, now),
        )
        conn.commit()
        conn.close()
        return {"session_id": session_id, "requests_used": 0, "can_chat": True, "expired": False}

    elapsed = now - row["last_active"]
    is_expired = elapsed > SESSION_TIMEOUT_SECONDS

    if is_expired and not row["expired"]:
        conn.execute("UPDATE sessions SET expired = 1 WHERE session_id = ?", (session_id,))
        conn.commit()
        cleanup_tmp_files(session_id)

    conn.execute(
        "UPDATE sessions SET last_active = ? WHERE session_id = ?", (now, session_id)
    )
    conn.commit()

    used = row["requests_used"]
    expired = row["expired"] or is_expired
    conn.close()

    return {
        "session_id": session_id,
        "requests_used": used,
        "can_chat": used < MAX_REQUESTS_PER_SESSION and not expired,
        "expired": expired,
    }


def heartbeat(session_id: str) -> dict:
    """Refresh session timeout. Returns time remaining."""
    conn = _get_db()
    row = conn.execute(
        "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()

    if not row:
        conn.close()
        return {"active": False, "remaining_seconds": 0}

    now = time.time()
    elapsed = now - row["last_active"]
    remaining = max(0, SESSION_TIMEOUT_SECONDS - elapsed)

    if elapsed > SESSION_TIMEOUT_SECONDS:
        conn.execute("UPDATE sessions SET expired = 1 WHERE session_id = ?", (session_id,))
        conn.commit()
        cleanup_tmp_files(session_id)
        conn.close()
        return {"active": False, "remaining_seconds": 0, "expired": True}

    # Refresh if more than half the time has passed
    if elapsed > SESSION_TIMEOUT_SECONDS / 2:
        conn.execute("UPDATE sessions SET last_active = ? WHERE session_id = ?", (now, session_id))
        conn.commit()

    conn.close()
    return {
        "active": True,
        "remaining_seconds": int(remaining),
        "requests_used": row["requests_used"],
        "remaining_queries": max(0, MAX_REQUESTS_PER_SESSION - row["requests_used"]),
    }


def extend_session(session_id: str) -> dict:
    """User chose to continue — reset timeout."""
    conn = _get_db()
    now = time.time()
    conn.execute(
        "UPDATE sessions SET last_active = ?, expired = 0 WHERE session_id = ?",
        (now, session_id),
    )
    conn.commit()
    row = conn.execute(
        "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    conn.close()

    if not row:
        return {"active": False}
    return {
        "active": True,
        "remaining_seconds": SESSION_TIMEOUT_SECONDS,
        "requests_used": row["requests_used"],
    }


def expire_session(session_id: str) -> dict:
    """User chose to exit — clean up."""
    conn = _get_db()
    conn.execute("UPDATE sessions SET expired = 1 WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()
    cleanup_tmp_files(session_id)
    return {"expired": True}


def cleanup_tmp_files(session_id: str):
    """Delete any uploaded temp files for this session."""
    tmp_dir = Path(f"/tmp/uaelaw_uploads/{session_id}")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    # Also clean up any uploaded docs in the upload folder
    upload_dir = Path(f"/tmp/uaelaw_docs/{session_id}")
    if upload_dir.exists():
        shutil.rmtree(upload_dir)


def increment_request(session_id: str) -> dict:
    conn = _get_db()
    now = time.time()
    conn.execute(
        "UPDATE sessions SET requests_used = requests_used + 1, last_active = ? WHERE session_id = ?",
        (now, session_id),
    )
    conn.execute("UPDATE stats SET value = value + 1 WHERE key = 'total_requests'")
    conn.commit()
    row = conn.execute(
        "SELECT requests_used FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    used = row["requests_used"] if row else 0
    conn.close()
    return {"requests_used": used, "remaining": max(0, MAX_REQUESTS_PER_SESSION - used)}


def submit_feedback(session_id: str, nps_score: int, comment: str = "") -> bool:
    if nps_score < 0 or nps_score > 10:
        return False
    conn = _get_db()
    conn.execute(
        "INSERT INTO feedback (session_id, nps_score, comment_text, created_at) VALUES (?, ?, ?, ?)",
        (session_id, nps_score, comment.strip(), time.time()),
    )
    conn.commit()
    conn.close()
    return True


def get_global_stats() -> dict:
    conn = _get_db()
    total = conn.execute("SELECT value FROM stats WHERE key = 'total_requests'").fetchone()
    total_requests = total["value"] if total else 0
    active_sessions = conn.execute(
        "SELECT COUNT(*) as c FROM sessions WHERE expired = 0"
    ).fetchone()["c"]
    fb_count = conn.execute("SELECT COUNT(*) as c FROM feedback").fetchone()["c"]
    fb_avg = conn.execute("SELECT AVG(nps_score) as avg FROM feedback").fetchone()["avg"]
    fb_dist = conn.execute("""
        SELECT
            SUM(CASE WHEN nps_score >= 9 THEN 1 ELSE 0 END) as promoters,
            SUM(CASE WHEN nps_score BETWEEN 7 AND 8 THEN 1 ELSE 0 END) as passives,
            SUM(CASE WHEN nps_score <= 6 THEN 1 ELSE 0 END) as detractors
        FROM feedback
    """).fetchone()
    conn.close()

    nps = 0
    if fb_count > 0 and fb_dist:
        promoters = fb_dist["promoters"] or 0
        detractors = fb_dist["detractors"] or 0
        nps = round((promoters - detractors) / fb_count * 100)

    return {
        "total_requests": total_requests,
        "active_sessions": active_sessions,
        "feedback_count": fb_count,
        "nps_score": nps,
        "avg_rating": round(fb_avg, 1) if fb_avg else 0,
        "max_per_session": MAX_REQUESTS_PER_SESSION,
    }
