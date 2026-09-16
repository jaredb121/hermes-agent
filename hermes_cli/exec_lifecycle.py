"""Execution lifecycle — normalised presentation of Hermes state.

A pure read/adapter layer that maps existing persisted statuses (kanban task
status, session state, run records) onto the operator lifecycle:

    Ready -> Running -> Ready for review -> Verified done

Plus: Waiting on you, Retrying, Stalled, Failed, Cancelled.

Does NOT migrate, rewrite, or alter any production state. All data is derived
from live queries of the state.db and kanban.db databases.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifecycle states (the normalised set)
# ---------------------------------------------------------------------------

LIFECYCLE_ORDER = [
    "running",
    "ready_for_review",
    "retrying",
    "waiting_on_you",
    "stalled",
    "ready",
    "verified_done",
    "failed",
    "cancelled",
]

LIFECYCLE_LABELS: Dict[str, str] = {
    "running": "Running",
    "ready_for_review": "Ready for review",
    "retrying": "Retrying",
    "waiting_on_you": "Waiting on you",
    "stalled": "Stalled",
    "ready": "Ready",
    "verified_done": "Verified done",
    "failed": "Failed",
    "cancelled": "Cancelled",
    "blocked": "Waiting on you",
}

# Kanban task status -> lifecycle mapping
_KANBAN_TO_LIFECYCLE: Dict[str, str] = {
    "triage": "waiting_on_you",
    "todo": "waiting_on_you",
    "scheduled": "ready",
    "ready": "ready",
    "running": "running",
    "blocked": "waiting_on_you",
    "review": "ready_for_review",
    "done": "verified_done",
    "failed": "failed",
    "cancelled": "cancelled",
    "archived": "verified_done",
}

_STALLED_SECONDS = 300  # 5 minutes without activity -> stalled


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class WorkItem:
    """A single unit of work in the operator view."""

    id: str
    title: str
    lifecycle: str  # one of LIFECYCLE_ORDER
    source: str  # "kanban" | "session"
    board: Optional[str] = None
    project: Optional[str] = None
    assignee: Optional[str] = None
    stage: Optional[str] = None  # e.g. "Inspect", "Implement", "Verify"
    stage_detail: Optional[str] = None  # e.g. "2/4", "testing accessibility"
    profile: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    last_updated: Optional[float] = None
    changed_files: List[str] = field(default_factory=list)
    commit: Optional[str] = None
    branch: Optional[str] = None
    diff_path: Optional[str] = None
    verification_result: Optional[str] = None
    unresolved_risks: List[str] = field(default_factory=list)
    has_review: bool = False
    consecutive_failures: int = 0
    heartbeat_age: Optional[float] = None


@dataclass
class WorkSnapshot:
    """Snapshot of all work across all boards at one moment."""

    items: List[WorkItem] = field(default_factory=list)
    captured_at: float = 0.0

    def sorted(self) -> List[WorkItem]:
        """Return items sorted by lifecycle priority, then recency."""
        priority = {s: i for i, s in enumerate(LIFECYCLE_ORDER)}

        def _key(item: WorkItem) -> Tuple[int, float, str]:
            p = priority.get(item.lifecycle, 99)
            # recency: most recent first within same lifecycle
            recency = -(item.last_updated or 0.0)
            return (p, recency, item.id)

        return sorted(self.items, key=_key)

    def by_lifecycle(self) -> Dict[str, List[WorkItem]]:
        result: Dict[str, List[WorkItem]] = {}
        for item in self.items:
            result.setdefault(item.lifecycle, []).append(item)
        return result


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def _hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home

        return get_hermes_home()
    except ImportError:
        return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))


def _kanban_db_path(board: Optional[str] = None) -> Path:
    """Resolve a board DB through Hermes' profile-safe shared-board rules."""
    from hermes_cli.kanban_db import kanban_db_path

    return kanban_db_path(board=board)


def _state_db_path() -> Path:
    return _hermes_home() / "state.db"


def _connect_read_only(path: Path) -> Optional[sqlite3.Connection]:
    """Open an existing SQLite database without creating or migrating it."""
    if not path.exists():
        return None
    # A normal read-only connection sees WAL data. Sandboxed/read-only mounts
    # may prohibit SQLite from opening the adjacent -shm file; immutable mode
    # is a safe last-resort snapshot for those environments.
    for query in ("mode=ro", "mode=ro&immutable=1"):
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = sqlite3.connect(f"file:{path}?{query}", uri=True, timeout=2.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only=ON")
            conn.execute("PRAGMA busy_timeout=2000")
            conn.execute("SELECT 1 FROM sqlite_schema LIMIT 1").fetchone()
            return conn
        except (sqlite3.Error, OSError):
            if conn is not None:
                conn.close()
            logger.debug(
                "operator view could not open %s with %s", path, query, exc_info=True
            )
    return None


def _connect_kanban(board: Optional[str] = None) -> Optional[sqlite3.Connection]:
    return _connect_read_only(_kanban_db_path(board))


def _connect_state() -> Optional[sqlite3.Connection]:
    return _connect_read_only(_state_db_path())


# ---------------------------------------------------------------------------
# Kanban adapters
# ---------------------------------------------------------------------------


def _resolve_kanban_task_lifecycle(
    status: str,
    run: Optional[Dict[str, Any]],
    consecutive_failures: int,
    now: float,
) -> str:
    """Map a kanban task's persisted state to the normalised lifecycle.

    Liveness check: a task whose kanban status is "running" but whose worker
    has not heartbeated recently is shown as "stalled", not "running". A task
    whose ended_at is set must not be shown as "running".
    """
    lc = _KANBAN_TO_LIFECYCLE.get(status, "ready")

    if lc == "running":
        # Confirm liveness
        if run is None:
            return "stalled"  # no run record -> can't confirm liveness
        ended_at = run.get("ended_at")
        if ended_at is not None and ended_at > 0:
            return "ready_for_review"
        heartbeat = run.get("last_heartbeat_at")
        if heartbeat is not None and heartbeat > 0:
            age = now - heartbeat
            if age > _STALLED_SECONDS:
                return "stalled"
        else:
            # No heartbeat ever recorded for this run
            started_at = run.get("started_at")
            if started_at is not None and started_at > 0:
                age = now - started_at
                if age > _STALLED_SECONDS:
                    return "stalled"
                return "running"
            return "running"

    if lc in ("ready", "waiting_on_you") and consecutive_failures > 0:
        # Was tried before, now waiting again -> retrying or waiting
        return "retrying" if consecutive_failures < 3 else "waiting_on_you"

    return lc


def _determine_stage(
    run: Optional[Dict[str, Any]],
) -> Tuple[Optional[str], Optional[str]]:
    """Derive a human-readable stage from a kanban run's metadata.

    Returns (stage_name, stage_detail) where stage_name is one of:
    "Inspect", "Implement", "Verify", "Review", "Done" and stage_detail
    provides granularity like "2/4".
    """
    if run is None:
        return (None, None)

    metadata = run.get("metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (json.JSONDecodeError, TypeError):
            metadata = None
    if not isinstance(metadata, dict):
        metadata = {}

    summary = metadata.get("summary", "")
    if "inspect" in summary.lower() or "orient" in summary.lower():
        return ("Inspect", None)

    changed_files = metadata.get("changed_files", [])
    tests_run = metadata.get("tests_run", 0)
    total_tests = metadata.get("total_tests", None) or tests_run

    if changed_files and tests_run > 0:
        detail = f"{tests_run}/{total_tests}" if total_tests else str(tests_run)
        return ("Verify", detail)
    if changed_files:
        return ("Implement", f"{len(changed_files)} files")
    if tests_run > 0:
        return ("Verify", str(tests_run))

    return ("Implement", None)


def _fetch_kanban_items(now: float, board: Optional[str] = None) -> List[WorkItem]:
    """Fetch all non-archived tasks from a kanban board.

    Args:
        now: Current timestamp for age calculations.
        board: Board slug. When None, uses the current/default board.
    """
    items: List[WorkItem] = []
    conn = _connect_kanban(board)
    if conn is None:
        return items

    try:
        rows = conn.execute("""
            SELECT id, title, status, assignee, project_id, tenant, created_at,
                   started_at, completed_at, result,
                   consecutive_failures, last_heartbeat_at,
                   current_run_id
            FROM tasks
            WHERE status != 'archived'
            ORDER BY
                CASE status
                    WHEN 'running' THEN 0
                    WHEN 'ready' THEN 1
                    WHEN 'blocked' THEN 2
                    ELSE 3
                END,
                COALESCE(started_at, created_at) DESC
        """).fetchall()

        for row in rows:
            tid = str(row["id"])
            run = None
            current_run_id = row["current_run_id"]

            if current_run_id is not None:
                run_row = conn.execute(
                    "SELECT * FROM task_runs WHERE id = ?",
                    (current_run_id,),
                ).fetchone()
                if run_row is not None:
                    run = dict(run_row)

            lifecycle = _resolve_kanban_task_lifecycle(
                row["status"],
                run,
                row["consecutive_failures"],
                now,
            )

            stage, stage_detail = _determine_stage(run)

            started_at = row["started_at"]
            completed_at = row["completed_at"]
            elapsed: Optional[float] = None
            if started_at is not None and started_at > 0:
                end = (
                    completed_at
                    if completed_at is not None and completed_at > 0
                    else now
                )
                elapsed = end - started_at

            heartbeat = row["last_heartbeat_at"]
            heartbeat_age: Optional[float] = None
            if heartbeat is not None and heartbeat > 0:
                heartbeat_age = now - heartbeat

            item = WorkItem(
                id=tid,
                title=row["title"] or tid,
                lifecycle=lifecycle,
                source="kanban",
                board=board,  # tracks which board this came from
                project=row["project_id"],
                assignee=row["assignee"],
                stage=stage,
                stage_detail=stage_detail,
                elapsed_seconds=elapsed,
                last_updated=started_at or row["created_at"],
                heartbeat_age=heartbeat_age,
                consecutive_failures=row["consecutive_failures"],
            )

            # Check if the task result contains review metadata
            result_str = row["result"]
            if result_str and isinstance(result_str, str):
                item.has_review = True
                try:
                    result_data = json.loads(result_str)
                except (json.JSONDecodeError, TypeError):
                    result_data = None
                if isinstance(result_data, dict):
                    item.changed_files = result_data.get("changed_files", [])
                    item.commit = result_data.get("commit")
                    item.branch = result_data.get("branch")
                    item.diff_path = result_data.get("diff_path")
                    item.verification_result = result_data.get(
                        "verification_result", result_data.get("summary")
                    )
                    item.unresolved_risks = result_data.get("unresolved_risks", [])

            items.append(item)
    finally:
        conn.close()

    return items


# ---------------------------------------------------------------------------
# Session adapters
# ---------------------------------------------------------------------------


def _resolve_session_lifecycle(
    ended_at: Optional[float],
    last_activity_at: Optional[float],
    now: float,
    end_reason: Optional[str] = None,
) -> str:
    """Map a session row to the normalised lifecycle.

    A session with ended_at unset must NOT automatically be shown as Running;
    confirm liveness using worker/run/delegation evidence and recent heartbeat.
    """
    if ended_at is not None and ended_at > 0:
        reason = (end_reason or "").casefold()
        if any(token in reason for token in ("error", "failed", "crash")):
            return "failed"
        # An ended session proves execution stopped, not that a reviewer
        # accepted the outcome. Only persisted Kanban ``done`` is verified.
        return "ready_for_review"

    if last_activity_at is not None and last_activity_at > 0:
        age = now - last_activity_at
        if age < 30:
            return "running"
        elif age < _STALLED_SECONDS:
            return "ready_for_review"
        else:
            return "stalled"

    return "ready"


def _fetch_session_items(now: float) -> List[WorkItem]:
    """Fetch recent sessions from state.db."""
    items: List[WorkItem] = []
    conn = _connect_state()
    if conn is None:
        return items

    try:
        rows = conn.execute("""
            SELECT id, title, source, profile_name, started_at, ended_at, end_reason,
                   last_activity_at, last_activity_description
            FROM sessions
            WHERE archived = 0
            ORDER BY COALESCE(last_activity_at, started_at) DESC
            LIMIT 50
        """).fetchall()

        for row in rows:
            sid = str(row["id"])
            ended_at = row["ended_at"]
            last_activity = row["last_activity_at"]

            lifecycle = _resolve_session_lifecycle(
                ended_at, last_activity, now, row["end_reason"]
            )

            started = row["started_at"]
            elapsed: Optional[float] = None
            if started is not None and started > 0:
                end = ended_at if ended_at is not None and ended_at > 0 else now
                elapsed = end - started

            item = WorkItem(
                id=sid,
                title=row["title"] or sid,
                lifecycle=lifecycle,
                source="session",
                profile=row["profile_name"] or row["source"],
                stage=row["last_activity_description"],
                elapsed_seconds=elapsed,
                last_updated=last_activity or started,
            )
            items.append(item)
    finally:
        conn.close()

    return items


def get_board_counts(board_slug: str) -> Dict[str, int]:
    """Return task counts per status for a specific board.

    Safe to call on any board directory, even if its DB doesn't exist yet.
    """
    try:
        path = _kanban_db_path(board_slug)
        if not path.exists():
            return {}
        conn = _connect_read_only(path)
        if conn is None:
            return {}
        try:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM tasks GROUP BY status"
            ).fetchall()
            return {r[0]: int(r[1]) for r in rows}
        finally:
            conn.close()
    except (sqlite3.Error, OSError):
        return {}


def list_all_boards() -> List[Dict[str, Any]]:
    """Enumerate all configured kanban boards with slug, name, and archived flag."""
    try:
        from hermes_cli.kanban_db import get_current_board, list_boards

        boards = list_boards(include_archived=True)
        current = get_current_board()
        for b in boards:
            b["is_current"] = b["slug"] == current
            b["counts"] = get_board_counts(b["slug"])
            b["total"] = sum(b["counts"].values())
        return boards
    except (ImportError, OSError, sqlite3.Error):
        # Fallback: read from filesystem
        home = _hermes_home()
        boards_dir = home / "kanban" / "boards"
        if not boards_dir.exists():
            return []
        result = []
        for child in sorted(boards_dir.iterdir()):
            if child.is_dir():
                slug = child.name
                if (child / "kanban.db").exists():
                    counts = get_board_counts(slug)
                else:
                    counts = {}
                result.append({
                    "slug": slug,
                    "name": slug.replace("-", " ").title(),
                    "is_current": False,
                    "archived": False,
                    "counts": counts,
                    "total": sum(counts.values()),
                })
        return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_work_snapshot() -> WorkSnapshot:
    """Return a snapshot of all work across all boards and sessions."""
    now = time.time()
    items: List[WorkItem] = []
    # Iterate every board
    for b in list_all_boards():
        slug = b["slug"]
        try:
            items.extend(_fetch_kanban_items(now, board=slug))
        except (KeyError, TypeError, sqlite3.Error):
            logger.debug("operator view could not read board %s", slug, exc_info=True)
    try:
        items.extend(_fetch_session_items(now))
    except (KeyError, TypeError, sqlite3.Error):
        logger.debug("operator view could not read sessions", exc_info=True)
    return WorkSnapshot(items=items, captured_at=now)


def get_health() -> Dict[str, Any]:
    """Return basic health info: DB connectivity, board enumeration, and active work counts."""
    now = time.time()
    state = _connect_state()

    # Enumerate all boards
    all_boards = list_all_boards()
    board_health = []
    for b in all_boards:
        slug = b["slug"]
        path = _kanban_db_path(slug)
        board_health.append({
            "slug": slug,
            "name": b.get("name", slug),
            "current": b.get("is_current", True),
            "archived": b.get("archived", False),
            "db_exists": path.exists() if path else False,
            "task_counts": b.get("counts", {}),
        })

    state_available = state is not None
    boards_available = all(b["db_exists"] for b in board_health if not b["archived"])
    health: Dict[str, Any] = {
        "status": "healthy" if state_available and boards_available else "degraded",
        "boards": board_health,
        "databases": {
            "state": state_available,
        },
        "active_sessions": 0,
        "timestamp": now,
    }

    if state:
        try:
            cnt = state.execute(
                "SELECT COUNT(*) as cnt FROM sessions WHERE archived = 0"
            ).fetchone()
            health["active_sessions"] = cnt["cnt"] if cnt else 0
        except sqlite3.Error:
            health["active_sessions"] = 0
        state.close()

    return health


def get_review_bundle(item: WorkItem) -> Dict[str, Any]:
    """Return a structured review bundle for a work item that has finished
    execution but needs review. Returns changed files, verification results,
    commit/branch info, and unresolved risks."""
    return {
        "id": item.id,
        "title": item.title,
        "lifecycle": item.lifecycle,
        "source": item.source,
        "changed_files": item.changed_files,
        "commit": item.commit,
        "branch": item.branch,
        "diff_path": item.diff_path,
        "verification_result": item.verification_result,
        # Keep the original short key while exposing the persisted field name
        # used by task results. Consumers can migrate without a flag day.
        "risks": item.unresolved_risks,
        "unresolved_risks": item.unresolved_risks,
        "assignee": item.assignee,
        "elapsed_seconds": item.elapsed_seconds,
    }
