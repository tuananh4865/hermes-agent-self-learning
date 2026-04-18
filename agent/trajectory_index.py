"""
Trajectory Index — SQLite-backed storage and query for conversation trajectories.

Provides persistent storage for both successful and failed trajectories,
with similarity search capabilities for the proactive self-healing system.

Usage:
    index = TrajectoryIndex(hermes_home="/path/to/.hermes")
    
    # Index a completed conversation
    index.index(signals=task_signals, messages=full_messages,
                failure_reason=None, completed=True)
    
    # Find similar past tasks
    matches = index.find_similar(task_message="refactor auth module")
    
    # Get failure patterns for a task type
    patterns = index.get_failure_patterns(task_type="refactor")
"""

import hashlib
import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.failure_classifier import FailureReason, TaskSignals

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryMatch:
    """A past trajectory matched to a new task."""
    id: int
    task_hash: str
    task_message: str
    task_type: str
    complexity: str
    failure_reason: Optional[str]
    completed: bool
    correction_count: int
    tool_call_count: int
    context_util_final: float
    similarity_score: float = 0.0
    created_at: str = ""


@dataclass
class FailurePattern:
    """Aggregate failure pattern for a task type."""
    task_type: str
    complexity: str
    failure_reason: str
    count: int
    percentage: float
    avg_correction_count: float
    avg_tool_calls: float


@dataclass
class SuccessPattern:
    """What succeeded for a given task type."""
    task_type: str
    complexity: str
    count: int
    avg_correction_count: float
    avg_tool_calls: float
    avg_files_affected: float


class TrajectoryIndex:
    """
    SQLite-backed trajectory storage with similarity search.
    
    Schema:
        trajectories (
            id INTEGER PRIMARY KEY,
            task_hash TEXT NOT NULL,        -- SHA of normalized task
            task_message TEXT NOT NULL,     -- original user message
            task_type TEXT,
            complexity TEXT,
            failure_reason TEXT,
            completed INTEGER,
            correction_count INTEGER,
            tool_call_count INTEGER,
            context_util_final REAL,
            files_affected INTEGER,
            multi_repo INTEGER,
            has_tests INTEGER,
            exit_reason TEXT,
            trajectory_json TEXT,            -- full message history
            created_at TEXT,
            
            UNIQUE(task_hash)
        )
        
        -- Indexes for fast queries
        CREATE INDEX idx_task_type ON trajectories(task_type)
        CREATE INDEX idx_complexity ON trajectories(complexity)
        CREATE INDEX idx_failure_reason ON trajectories(failure_reason)
        CREATE INDEX idx_task_hash ON trajectories(task_hash)
        CREATE INDEX idx_created_at ON trajectories(created_at)
    """

    def __init__(self, hermes_home: Optional[str] = None):
        if hermes_home is None:
            from hermes_constants import get_hermes_home
            hermes_home = get_hermes_home()
        
        self.hermes_home = Path(hermes_home)
        self.db_path = self.hermes_home / "trajectory_index.db"
        self._ensure_db()
        
        # Auto-cleanup: run once per day (threshold stored alongside DB)
        self._cleanup_flag_path = self.hermes_home / ".trajectory_cleanup_flag"
        self._maybe_cleanup()

    def _maybe_cleanup(self):
        """Run cleanup once per day based on flag file timestamp."""
        try:
            if self._cleanup_flag_path.exists():
                import time
                age_hours = (time.time() - self._cleanup_flag_path.stat().st_mtime) / 3600
                if age_hours < 24:
                    return  # Already ran today
            
            self.cleanup_old_trajectories(retention_days=30)
            
            # Touch flag file
            self._cleanup_flag_path.parent.mkdir(parents=True, exist_ok=True)
            self._cleanup_flag_path.touch()
        except Exception as e:
            logger.debug("Cleanup skipped: %s", e)

    # ─── Initialization ─────────────────────────────────────────────────

    def _ensure_db(self):
        """Create database and tables if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trajectories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_hash TEXT NOT NULL UNIQUE,
                task_message TEXT NOT NULL,
                task_type TEXT DEFAULT 'unknown',
                complexity TEXT DEFAULT 'low',
                failure_reason TEXT,
                completed INTEGER DEFAULT 0,
                correction_count INTEGER DEFAULT 0,
                tool_call_count INTEGER DEFAULT 0,
                context_util_final REAL DEFAULT 0.0,
                files_affected INTEGER DEFAULT 0,
                multi_repo INTEGER DEFAULT 0,
                has_tests INTEGER DEFAULT 0,
                exit_reason TEXT DEFAULT '',
                trajectory_json TEXT,
                created_at TEXT,
                
                UNIQUE(task_hash)
            )
        """)
        
        # Create indexes (ignore if exist)
        for idx_sql in [
            "CREATE INDEX IF NOT EXISTS idx_task_type ON trajectories(task_type)",
            "CREATE INDEX IF NOT EXISTS idx_complexity ON trajectories(complexity)",
            "CREATE INDEX IF NOT EXISTS idx_failure_reason ON trajectories(failure_reason)",
            "CREATE INDEX IF NOT EXISTS idx_task_hash ON trajectories(task_hash)",
            "CREATE INDEX IF NOT EXISTS idx_created_at ON trajectories(created_at)",
        ]:
            try:
                cursor.execute(idx_sql)
            except sqlite3.Error:
                pass
        
        # Create FTS5 virtual table for full-text similarity search
        try:
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS trajectories_fts USING fts5(
                    task_message,
                    content='trajectories',
                    content_rowid='id',
                    tokenize='porter unicode61'
                )
            """)
            # Populate FTS index if empty (first run)
            cursor.execute("SELECT COUNT(*) FROM trajectories_fts")
            if cursor.fetchone()[0] == 0:
                cursor.execute("""
                    INSERT INTO trajectories_fts(rowid, task_message)
                    SELECT id, task_message FROM trajectories WHERE task_message IS NOT NULL
                """)
            # Triggers to keep FTS in sync
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS trajectories_ai AFTER INSERT ON trajectories BEGIN
                    INSERT INTO trajectories_fts(rowid, task_message) VALUES (new.id, new.task_message);
                END
            """)
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS trajectories_ad AFTER DELETE ON trajectories BEGIN
                    INSERT INTO trajectories_fts(trajectories_fts, rowid, task_message) VALUES('delete', old.id, old.task_message);
                END
            """)
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS trajectories_au AFTER UPDATE ON trajectories BEGIN
                    INSERT INTO trajectories_fts(trajectories_fts, rowid, task_message) VALUES('delete', old.id, old.task_message);
                    INSERT INTO trajectories_fts(rowid, task_message) VALUES (new.id, new.task_message);
                END
            """)
        except sqlite3.Error:
            pass

        conn.commit()
        conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    # ─── Core operations ───────────────────────────────────────────────

    def index(
        self,
        signals: TaskSignals,
        messages: List[Dict[str, Any]],
        failure_reason: Optional[FailureReason],
        completed: bool,
        task_message: Optional[str] = None
    ):
        """
        Index a completed conversation.
        
        Uses task_hash as unique key — if same task is retried,
        it updates the existing entry rather than creating duplicates.
        """
        if not task_message:
            # Extract from messages
            for msg in messages:
                if msg.get("role") == "user":
                    task_message = msg.get("content", "")[:1000]
                    break
            if not task_message:
                task_message = "unknown"
        
        task_hash = self._compute_task_hash(task_message)
        failure_str = failure_reason.value if failure_reason else None
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO trajectories (
                    task_hash, task_message, task_type, complexity,
                    failure_reason, completed, correction_count,
                    tool_call_count, context_util_final, files_affected,
                    multi_repo, has_tests, exit_reason, trajectory_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_hash) DO UPDATE SET
                    completed = excluded.completed,
                    failure_reason = excluded.failure_reason,
                    correction_count = excluded.correction_count,
                    tool_call_count = excluded.tool_call_count,
                    context_util_final = excluded.context_util_final,
                    exit_reason = excluded.exit_reason,
                    trajectory_json = excluded.trajectory_json,
                    created_at = excluded.created_at
            """, (
                task_hash,
                task_message[:2000],  # Limit message size
                signals.task_type,
                signals.complexity,
                failure_str,
                1 if completed else 0,
                signals.correction_count,
                signals.tool_call_count,
                signals.context_util_final,
                signals.files_affected,
                1 if signals.multi_repo else 0,
                1 if signals.has_tests else 0,
                signals.exit_reason,
                json.dumps(messages[-50:]),  # Last 50 messages only
                datetime.now().isoformat(),
            ))
            conn.commit()
            logger.debug("Indexed trajectory %s (completed=%s)", task_hash[:12], completed)
        except sqlite3.Error as e:
            logger.error("Failed to index trajectory: %s", e)
        finally:
            conn.close()

    def find_similar(
        self,
        task_message: str,
        task_type: Optional[str] = None,
        complexity: Optional[str] = None,
        limit: int = 5,
        include_failed_only: bool = False
    ) -> List[TrajectoryMatch]:
        """
        Find similar past tasks using FTS5 full-text search + keyword scoring hybrid.
        
        FTS5 BM25 ranking finds candidates, then keyword overlap re-ranks.
        """
        query_words = self._extract_significant_words(task_message)
        if not query_words and not task_message.strip():
            return []
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Build optional filters
        conditions = []
        params = []
        
        if task_type:
            conditions.append("t.task_type = ?")
            params.append(task_type)
        if complexity:
            conditions.append("t.complexity = ?")
            params.append(complexity)
        if include_failed_only:
            conditions.append("t.completed = 0")
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        # Try FTS5 MATCH first; fall back to plain keyword scan if FTS unavailable
        fts_results = None
        try:
            # BM25 search on FTS5
            fts_query = " OR ".join(f'"{w}"' for w in query_words if w)
            if fts_query:
                cursor.execute(f"""
                    SELECT t.id, t.task_hash, t.task_message, t.task_type, t.complexity,
                           t.failure_reason, t.completed, t.correction_count,
                           t.tool_call_count, t.context_util_final, t.created_at,
                           bm25(trajectories_fts) as rank
                    FROM trajectories_fts fts
                    JOIN trajectories t ON t.id = fts.rowid
                    WHERE trajectories_fts MATCH ? AND {where_clause}
                    ORDER BY rank
                    LIMIT 200
                """, (fts_query,) + tuple(params))
                fts_results = cursor.fetchall()
        except sqlite3.Error:
            fts_results = None
        
        # If FTS had no results, fall back to full table scan with keyword scoring
        if not fts_results:
            cursor.execute(f"""
                SELECT id, task_hash, task_message, task_type, complexity,
                       failure_reason, completed, correction_count,
                       tool_call_count, context_util_final, created_at
                FROM trajectories t
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT 200
            """, params)
            fts_results = cursor.fetchall()
        
        conn.close()
        
        # Score and rank by keyword overlap
        scored = []
        for row in fts_results:
            match = TrajectoryMatch(
                id=row[0],
                task_hash=row[1],
                task_message=row[2],
                task_type=row[3] or "unknown",
                complexity=row[4] or "low",
                failure_reason=row[5],
                completed=bool(row[6]),
                correction_count=row[7],
                tool_call_count=row[8],
                context_util_final=row[9] or 0.0,
                created_at=row[10],
            )
            
            # Keyword overlap score (Jaccard)
            match_words = self._extract_significant_words(row[2])
            overlap = len(query_words & match_words)
            union = len(query_words | match_words)
            match.similarity_score = overlap / union if union > 0 else 0.0
            
            if match.similarity_score > 0.05:  # Lowered threshold since FTS pre-filtered
                scored.append(match)
        
        scored.sort(key=lambda m: m.similarity_score, reverse=True)
        return scored[:limit]

    def get_failure_patterns(
        self,
        task_type: Optional[str] = None,
        complexity: Optional[str] = None,
        max_age_days: int = 30
    ) -> List[FailurePattern]:
        """
        Get aggregate failure patterns by task type/complexity.
        
        Useful for answering: "What kinds of tasks fail most often?"
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        conditions = ["completed = 0"]
        params = []
        
        if task_type:
            conditions.append("task_type = ?")
            params.append(task_type)
        
        if complexity:
            conditions.append("complexity = ?")
            params.append(complexity)
        
        cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()
        conditions.append("created_at > ?")
        params.append(cutoff)
        
        where_clause = " AND ".join(conditions)
        
        cursor.execute(f"""
            SELECT 
                task_type, complexity, failure_reason,
                COUNT(*) as count,
                AVG(correction_count) as avg_corrections,
                AVG(tool_call_count) as avg_tool_calls
            FROM trajectories
            WHERE {where_clause}
            GROUP BY task_type, complexity, failure_reason
            ORDER BY count DESC
        """, params)
        
        rows = cursor.fetchall()
        conn.close()
        
        # Get total for percentage calculation
        total = sum(r[3] for r in rows)
        
        patterns = []
        for row in rows:
            patterns.append(FailurePattern(
                task_type=row[0] or "unknown",
                complexity=row[1] or "low",
                failure_reason=row[2] or "unknown",
                count=row[3],
                percentage=(row[3] / total * 100) if total > 0 else 0,
                avg_correction_count=row[4] or 0,
                avg_tool_calls=row[5] or 0,
            ))
        
        return patterns

    def get_success_patterns(
        self,
        task_type: Optional[str] = None,
        complexity: Optional[str] = None,
        max_age_days: int = 30
    ) -> List[SuccessPattern]:
        """Get what succeeded for given task characteristics."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        conditions = ["completed = 1"]
        params = []
        
        if task_type:
            conditions.append("task_type = ?")
            params.append(task_type)
        
        if complexity:
            conditions.append("complexity = ?")
            params.append(complexity)
        
        cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()
        conditions.append("created_at > ?")
        params.append(cutoff)
        
        where_clause = " AND ".join(conditions)
        
        cursor.execute(f"""
            SELECT 
                task_type, complexity,
                COUNT(*) as count,
                AVG(correction_count) as avg_corrections,
                AVG(tool_call_count) as avg_tool_calls,
                AVG(files_affected) as avg_files
            FROM trajectories
            WHERE {where_clause}
            GROUP BY task_type, complexity
            ORDER BY count DESC
        """, params)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            SuccessPattern(
                task_type=row[0] or "unknown",
                complexity=row[1] or "low",
                count=row[2],
                avg_correction_count=row[3] or 0,
                avg_tool_calls=row[4] or 0,
                avg_files_affected=row[5] or 0,
            )
            for row in rows
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get overall trajectory statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(completed) as completed,
                SUM(CASE WHEN completed = 0 THEN 1 ELSE 0 END) as failed,
                AVG(context_util_final) as avg_context_util
            FROM trajectories
        """)
        row = cursor.fetchone()
        
        cursor.execute("SELECT COUNT(DISTINCT task_type) FROM trajectories")
        distinct_types = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_conversations": row[0] or 0,
            "completed": row[1] or 0,
            "failed": row[2] or 0,
            "success_rate": (row[1] or 0) / max(1, row[0] or 1),
            "avg_context_utilization": row[3] or 0,
            "distinct_task_types": distinct_types,
        }

    def prune_old_entries(self, max_age_days: int = 90) -> int:
        """Remove entries older than max_age_days. Returns count deleted."""
        cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM trajectories WHERE created_at < ?", (cutoff,))
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        logger.info("Pruned %d old trajectory entries", deleted)
        return deleted

    def get_adjustment(
        self,
        task_type: str,
        max_age_days: int = 30
    ) -> Optional[Dict[str, float]]:
        """
        Get complexity adjustment for a task type based on historical failure rates.

        Analyzes past trajectories for this task_type to determine if it
        historically has higher failure rates, suggesting we should upgrade
        complexity classification.

        Returns:
            Dict with 'high_kw_boost' (float) if historical failures found, else None.
            A high_kw_boost of 1.0 means: treat this as having 1 extra high keyword.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()

        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN completed = 0 THEN 1 ELSE 0 END) as failed,
                SUM(CASE WHEN complexity = 'high' THEN 1 ELSE 0 END) as high_count,
                SUM(CASE WHEN complexity = 'medium' THEN 1 ELSE 0 END) as med_count
            FROM trajectories
            WHERE task_type = ? AND created_at > ?
        """, (task_type, cutoff))

        row = cursor.fetchone()
        conn.close()

        if not row or row[0] == 0:
            return None  # No history for this task type

        total = row[0]
        failed = row[1] or 0
        high_count = row[2] or 0
        med_count = row[3] or 0

        # If most tasks of this type fail AND they're not already marked high complexity,
        # we should boost the effective complexity for future tasks
        if total >= 3:  # Need at least 3 samples
            failure_rate = failed / total

            # If failure rate > 50% and mostly medium complexity, boost
            if failure_rate > 0.5 and med_count > high_count:
                # Compute boost: how many extra "high keywords" to add
                # More failures + more medium tasks = bigger boost
                boost = min(2.0, (failure_rate - 0.5) * 4 + 0.5)
                return {"high_kw_boost": boost}

            # If very high failure rate even for high complexity tasks, moderate boost
            if failure_rate > 0.7 and high_count > 0:
                return {"high_kw_boost": 0.5}

        return None

    # ─── Utility methods ───────────────────────────────────────────────

    @staticmethod
    def _compute_task_hash(task_message: str) -> str:
        """Compute stable hash of normalized task message."""
        normalized = task_message.lower().strip()
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = ' '.join(normalized.split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]

    @staticmethod
    def _extract_significant_words(text: str) -> set:
        """
        Extract significant words from text for similarity matching.
        
        Filters out common stopwords and short words.
        """
        stopwords = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'or', 'and',
            'the', 'this', 'that', 'these', 'those', 'it', 'its', 'i',
            'you', 'we', 'they', 'he', 'she', 'my', 'your', 'our', 'their',
            'what', 'which', 'who', 'when', 'where', 'why', 'how',
            'all', 'each', 'every', 'both', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'not', 'only', 'same', 'so',
            'than', 'too', 'very', 'just', 'but', 'if', 'then', 'because',
            'while', 'although', 'though', 'after', 'before', 'above',
            'below', 'between', 'into', 'through', 'during', 'about',
            'under', 'again', 'further', 'once', 'here', 'there', 'any',
        }
        
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        return {w for w in words if w not in stopwords}

    def cleanup_old_trajectories(self, retention_days: int = 30) -> int:
        """
        Delete trajectories older than retention_days.
        
        Called automatically on init (once per day threshold) to keep the
        database lean. Returns the number of deleted rows.
        
        Also purges orphaned FTS entries.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cutoff = (datetime.now() - timedelta(days=retention_days)).isoformat()
        
        try:
            # Count before deleting (for logging)
            cursor.execute(
                "SELECT COUNT(*) FROM trajectories WHERE created_at < ?",
                (cutoff,)
            )
            count = cursor.fetchone()[0]
            
            if count > 0:
                # Delete from FTS first (via trigger), then from trajectories
                cursor.execute(
                    "DELETE FROM trajectories WHERE created_at < ?",
                    (cutoff,)
                )
                conn.commit()
                logger.info(
                    "Cleaned up %d trajectories older than %d days", count, retention_days
                )
            return count
            
        except sqlite3.Error as e:
            logger.error("Failed to cleanup old trajectories: %s", e)
            return 0
        finally:
            conn.close()
