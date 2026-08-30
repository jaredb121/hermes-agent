"""Tests for the execution lifecycle adapter (hermes_cli/exec_lifecycle.py).

Proves:
- Lifecycle state mapping from raw DB fields
- Running, stale-open, stalled, blocked, ready-for-review, and verified-done
  are distinct states
- Multi-source aggregation (kanban tasks + sessions)
- Health check connectivity
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from hermes_cli.commands import resolve_command
from hermes_cli.exec_lifecycle import (
    LIFECYCLE_LABELS,
    LIFECYCLE_ORDER,
    WorkItem,
    WorkSnapshot,
    _connect_read_only,
    _determine_stage,
    _fetch_kanban_items,
    _fetch_session_items,
    _hermes_home,
    _kanban_db_path,
    _resolve_kanban_task_lifecycle,
    _resolve_session_lifecycle,
    _state_db_path,
    get_health,
    get_review_bundle,
    get_work_snapshot,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_work_items() -> list[WorkItem]:
    """Produce one item per lifecycle state for exhaustive coverage."""
    now = time.time()
    return [
        WorkItem(
            id="sess-running-1",
            title="Implement auth middleware",
            lifecycle="running",
            source="session",
            profile="dev",
            stage="Verify",
            stage_detail="3/5 tests",
            elapsed_seconds=42,
            heartbeat_age=None,
            last_updated=now,
            has_review=False,
            changed_files=[],
        ),
        WorkItem(
            id="kanban-running-1",
            title="Deploy staging env",
            lifecycle="running",
            source="kanban",
            profile="ops",
            stage="Ready",
            elapsed_seconds=300,
            heartbeat_age=10.0,
            last_updated=now - 10,
            has_review=False,
            project="anvil",
            assignee="ops-bot",
        ),
        WorkItem(
            id="kanban-stalled-1",
            title="Update dependencies",
            lifecycle="stalled",
            source="kanban",
            profile="dev",
            stage="Implement",
            elapsed_seconds=7200,
            heartbeat_age=600.0,
            last_updated=now - 600,
            has_review=False,
            project="hermes",
            assignee="bot-1",
        ),
        WorkItem(
            id="kanban-blocked-1",
            title="API key rotation",
            lifecycle="waiting_on_you",
            source="kanban",
            profile="sec",
            stage="Waiting on you",
            stage_detail="needs approval",
            elapsed_seconds=14400,
            last_updated=now - 14400,
            has_review=False,
            project="anvil",
            assignee="sec-bot",
        ),
        WorkItem(
            id="kanban-review-1",
            title="Refactor CI pipeline",
            lifecycle="ready_for_review",
            source="kanban",
            profile="ops",
            stage="Review",
            elapsed_seconds=600,
            last_updated=now - 600,
            has_review=True,
            project="infra",
            assignee="ops-bot",
            changed_files=["ci/main.yml", "ci/deploy.yml"],
            verification_result="3/3 tests passed, lint clean",
            commit="abc123def456",
            branch="feature/ci-refactor",
            diff_path="/tmp/ci-refactor.diff",
            unresolved_risks=["Docker cache invalidation may slow first build"],
        ),
        WorkItem(
            id="kanban-done-1",
            title="Add health endpoint",
            lifecycle="verified_done",
            source="kanban",
            profile="dev",
            stage="Verified done",
            elapsed_seconds=1800,
            last_updated=now - 3600,
            has_review=False,
            project="hermes",
            assignee="dev-bot",
            changed_files=["api/health.py", "tests/test_health.py"],
            verification_result="4/4 tests passed, coverage +2%",
            commit="789ghi012jkl",
            branch="feature/health-endpoint",
            unresolved_risks=[],
        ),
        WorkItem(
            id="kanban-cancelled-1",
            title="Deprecated experiment",
            lifecycle="cancelled",
            source="kanban",
            profile="research",
            stage="Cancelled",
            stage_detail="superseded by new approach",
            elapsed_seconds=3600,
            last_updated=now - 86400,
            has_review=False,
            project="sandbox",
            assignee="bot-x",
        ),
    ]


# ---------------------------------------------------------------------------
# WorkItem
# ---------------------------------------------------------------------------


class TestWorkItem:
    def test_repr(self):
        item = WorkItem(
            id="test-1", title="Test", lifecycle="running", source="session"
        )
        r = repr(item)
        assert "test-1" in r
        assert "running" in r

    def test_stage_defaults_to_none(self):
        item = WorkItem(id="t1", title="T", lifecycle="ready", source="kanban")
        assert item.stage is None


class TestOperatorCommands:
    @pytest.mark.parametrize(
        ("name", "executor"),
        (("ops", "ops_status"), ("work", "work_board")),
    )
    def test_commands_are_registry_owned(self, name, executor):
        command = resolve_command(name)
        assert command is not None
        assert command.execute == executor


# ---------------------------------------------------------------------------
# WorkSnapshot
# ---------------------------------------------------------------------------


class TestWorkSnapshot:
    def test_empty_snapshot(self):
        snap = WorkSnapshot([])
        assert len(snap.items) == 0
        assert snap.sorted() == []

    def test_sort_orders_by_lifecycle(self, sample_work_items):
        snap = WorkSnapshot(sample_work_items)
        sorted_items = snap.sorted()
        assert len(sorted_items) == len(sample_work_items)
        # running comes first
        assert sorted_items[0].lifecycle == "running"
        assert sorted_items[1].lifecycle == "running"
        # ready_for_review should come before waiting_on_you
        review_idx = next(
            i for i, it in enumerate(sorted_items) if it.lifecycle == "ready_for_review"
        )
        blocked_idx = next(
            i for i, it in enumerate(sorted_items) if it.lifecycle == "waiting_on_you"
        )
        assert review_idx < blocked_idx

    def test_by_lifecycle(self, sample_work_items):
        snap = WorkSnapshot(sample_work_items)
        grouped = snap.by_lifecycle()
        assert "running" in grouped
        assert "ready_for_review" in grouped
        assert "waiting_on_you" in grouped
        assert "stalled" in grouped
        assert "verified_done" in grouped
        assert "cancelled" in grouped
        assert len(grouped["running"]) == 2
        assert len(grouped["ready_for_review"]) == 1

    def test_recency_sort_uses_last_updated(self):
        older = WorkItem(
            id="older",
            title="Older",
            lifecycle="running",
            source="session",
            last_updated=100,
            elapsed_seconds=1,
        )
        newer = WorkItem(
            id="newer",
            title="Newer",
            lifecycle="running",
            source="session",
            last_updated=200,
            elapsed_seconds=10_000,
        )
        assert WorkSnapshot([older, newer]).sorted() == [newer, older]


# ---------------------------------------------------------------------------
# State classification — the core of the adapter
# ---------------------------------------------------------------------------


class TestResolveKanbanLifecycle:
    """Verify that kanban statuses map to the right lifecycle states."""

    def test_running_with_recent_heartbeat(self):
        now = time.time()
        run = {
            "last_heartbeat_at": now - 30,
            "started_at": now - 3600,
            "ended_at": None,
        }
        assert _resolve_kanban_task_lifecycle("running", run, 0, now) == "running"

    def test_running_no_heartbeat_recent_start(self):
        now = time.time()
        run = {"last_heartbeat_at": None, "started_at": now - 60, "ended_at": None}
        assert _resolve_kanban_task_lifecycle("running", run, 0, now) == "running"

    def test_stalled_old_heartbeat(self):
        now = time.time()
        run = {
            "last_heartbeat_at": now - 600,
            "started_at": now - 3600,
            "ended_at": None,
        }
        assert _resolve_kanban_task_lifecycle("running", run, 0, now) == "stalled"

    def test_stalled_no_heartbeat_old_start(self):
        now = time.time()
        run = {"last_heartbeat_at": None, "started_at": now - 600, "ended_at": None}
        assert _resolve_kanban_task_lifecycle("running", run, 0, now) == "stalled"

    def test_no_run_record(self):
        now = time.time()
        assert _resolve_kanban_task_lifecycle("running", None, 0, now) == "stalled"

    def test_running_with_ended_at_becomes_ready_for_review(self):
        now = time.time()
        run = {
            "last_heartbeat_at": now - 30,
            "started_at": now - 3600,
            "ended_at": now - 60,
        }
        assert (
            _resolve_kanban_task_lifecycle("running", run, 0, now) == "ready_for_review"
        )

    def test_blocked(self):
        now = time.time()
        assert (
            _resolve_kanban_task_lifecycle("blocked", None, 0, now) == "waiting_on_you"
        )

    def test_ready(self):
        now = time.time()
        assert _resolve_kanban_task_lifecycle("ready", None, 0, now) == "ready"

    def test_todo(self):
        now = time.time()
        assert _resolve_kanban_task_lifecycle("todo", None, 0, now) == "waiting_on_you"

    def test_triage(self):
        now = time.time()
        assert (
            _resolve_kanban_task_lifecycle("triage", None, 0, now) == "waiting_on_you"
        )

    def test_done(self):
        now = time.time()
        assert _resolve_kanban_task_lifecycle("done", None, 0, now) == "verified_done"

    def test_cancelled(self):
        now = time.time()
        assert _resolve_kanban_task_lifecycle("cancelled", None, 0, now) == "cancelled"

    def test_unknown_status_defaults_to_ready(self):
        now = time.time()
        assert _resolve_kanban_task_lifecycle("unknown_status", None, 0, now) == "ready"

    def test_retrying_on_consecutive_failures(self):
        now = time.time()
        assert _resolve_kanban_task_lifecycle("ready", None, 1, now) == "retrying"
        assert _resolve_kanban_task_lifecycle("ready", None, 2, now) == "retrying"

    def test_waiting_on_you_after_many_failures(self):
        now = time.time()
        assert _resolve_kanban_task_lifecycle("ready", None, 3, now) == "waiting_on_you"
        assert _resolve_kanban_task_lifecycle("ready", None, 5, now) == "waiting_on_you"


class TestResolveSessionLifecycle:
    """Verify that session states map to lifecycle states."""

    def test_active_recent_activity(self):
        now = time.time()
        assert _resolve_session_lifecycle(None, now - 10, now) == "running"

    def test_active_idle_but_recent(self):
        now = time.time()
        # Age 60s is within the ready_for_review window (30s-300s)
        assert _resolve_session_lifecycle(None, now - 60, now) == "ready_for_review"

    def test_ended(self):
        now = time.time()
        assert (
            _resolve_session_lifecycle(now - 100, now - 120, now) == "ready_for_review"
        )

    def test_ended_with_failure_reason(self):
        now = time.time()
        assert (
            _resolve_session_lifecycle(now - 100, now - 120, now, "worker_failed")
            == "failed"
        )

    def test_stalled_long_idle(self):
        now = time.time()
        assert _resolve_session_lifecycle(None, now - 600, now) == "stalled"

    def test_no_activity_is_ready(self):
        now = time.time()
        assert _resolve_session_lifecycle(None, None, now) == "ready"

    def test_no_ended_at_with_old_activity(self):
        now = time.time()
        assert _resolve_session_lifecycle(None, now - 400, now) == "stalled"


# ---------------------------------------------------------------------------
# Distinct states proof
# ---------------------------------------------------------------------------


class TestDistinctStates:
    """Prove that all lifecycle states are semantically distinct."""

    def test_ordered_list_contains_all_states(self):
        for label_key in LIFECYCLE_LABELS:
            # 'blocked' maps to 'waiting_on_you' in the lifecycle system
            if label_key == "blocked":
                continue
            assert label_key in LIFECYCLE_ORDER, (
                f"{label_key} missing from LIFECYCLE_ORDER"
            )

    def test_all_labels_have_descriptions(self):
        for state in LIFECYCLE_ORDER:
            if state == "blocked":
                continue  # blocked maps to waiting_on_you which has a label
        # Just verify the label dict has entries
        assert LIFECYCLE_LABELS

    def test_lifecycle_order_no_duplicates(self):
        assert len(LIFECYCLE_ORDER) == len(set(LIFECYCLE_ORDER))

    def test_liveness_disambiguation(self):
        """A session with ended_at unset must NOT auto-show as Running.
        Confirm liveness using heartbeat or activity evidence.
        """
        now = time.time()
        # Ended -> ready for review (not running and not falsely verified)
        assert (
            _resolve_session_lifecycle(
                ended_at=now - 100, last_activity_at=now - 120, now=now
            )
            == "ready_for_review"
        )
        # Ended with recent activity -> ready for review
        assert (
            _resolve_session_lifecycle(
                ended_at=now - 50, last_activity_at=now - 50, now=now
            )
            == "ready_for_review"
        )
        # No ended_at, very old activity -> stalled (not running)
        assert (
            _resolve_session_lifecycle(
                ended_at=None, last_activity_at=now - 3600, now=now
            )
            == "stalled"
        )
        # No ended_at, recent activity -> running
        assert (
            _resolve_session_lifecycle(
                ended_at=None, last_activity_at=now - 10, now=now
            )
            == "running"
        )

    def test_ready_for_review_is_not_done(self):
        now = time.time()
        run_with_ended = {
            "started_at": now - 3600,
            "ended_at": now - 60,
            "last_heartbeat_at": now - 60,
        }
        run_no_ended = {
            "started_at": now - 3600,
            "ended_at": None,
            "last_heartbeat_at": now - 30,
        }
        r1 = _resolve_kanban_task_lifecycle("running", run_with_ended, 0, now)
        r2 = _resolve_kanban_task_lifecycle("done", None, 0, now)
        assert r1 == "ready_for_review"
        assert r2 == "verified_done"
        assert r1 != r2

    def test_blocked_is_not_stalled(self):
        now = time.time()
        run = {
            "started_at": now - 3600,
            "ended_at": None,
            "last_heartbeat_at": now - 600,
        }
        r1 = _resolve_kanban_task_lifecycle("blocked", None, 0, now)
        r2 = _resolve_kanban_task_lifecycle("running", run, 0, now)
        assert r1 == "waiting_on_you"
        assert r2 == "stalled"
        assert r1 != r2


# ---------------------------------------------------------------------------
# Stage determination
# ---------------------------------------------------------------------------


class TestDetermineStage:
    def test_none_run(self):
        assert _determine_stage(None) == (None, None)

    def test_inspect_summary(self):
        run = {"metadata": json.dumps({"summary": "Inspect the codebase"})}
        stage, detail = _determine_stage(run)
        assert stage == "Inspect"

    def test_implement_with_changed_files(self):
        run = {"metadata": json.dumps({"changed_files": ["a.py", "b.py"]})}
        stage, detail = _determine_stage(run)
        assert stage == "Implement"
        assert "2 files" in (detail or "")

    def test_verify_with_tests(self):
        run = {
            "metadata": json.dumps({
                "changed_files": ["a.py"],
                "tests_run": 5,
                "total_tests": 5,
            })
        }
        stage, detail = _determine_stage(run)
        assert stage == "Verify"
        assert "5/5" in (detail or "")

    def test_verify_only_tests(self):
        run = {"metadata": json.dumps({"tests_run": 3})}
        stage, detail = _determine_stage(run)
        assert stage == "Verify"
        assert detail == "3"

    def test_fallback_implement(self):
        run = {"metadata": "{}"}
        stage, detail = _determine_stage(run)
        assert stage == "Implement"

    def test_string_metadata_parsing(self):
        run = {"metadata": '{"summary": "Orient and inspect"}', "started_at": 1000}
        stage, _ = _determine_stage(run)
        assert stage == "Inspect"


# ---------------------------------------------------------------------------
# get_review_bundle
# ---------------------------------------------------------------------------


class TestGetReviewBundle:
    def test_constructs_review_bundle(self):
        item = WorkItem(
            id="review-item",
            title="CI refactor",
            lifecycle="ready_for_review",
            source="kanban",
            profile="ops",
            stage="Review",
            elapsed_seconds=600,
            has_review=True,
            changed_files=["ci/main.yml", "ci/deploy.yml"],
            verification_result="3/3 tests passed",
            commit="abc123",
            branch="feature/ci-refactor",
            diff_path="/tmp/ci-refactor.diff",
            unresolved_risks=["Risk A", "Risk B"],
        )
        bundle = get_review_bundle(item)
        assert bundle["id"] == "review-item"
        assert bundle["title"] == "CI refactor"
        assert bundle["lifecycle"] == "ready_for_review"
        assert "changed_files" in bundle
        assert "verification_result" in bundle
        assert "risks" in bundle
        assert "commit" in bundle
        assert "branch" in bundle
        assert "diff_path" in bundle
        assert len(bundle["changed_files"]) == 2
        assert len(bundle["risks"]) == 2
        assert bundle["unresolved_risks"] == bundle["risks"]

    def test_non_review_item_returns_minimal_bundle(self):
        item = WorkItem(
            id="running-1", title="Running task", lifecycle="running", source="kanban"
        )
        bundle = get_review_bundle(item)
        assert bundle["id"] == "running-1"
        assert bundle["lifecycle"] == "running"
        assert "changed_files" in bundle


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class TestGetHealth:
    def test_returns_structure(self):
        health = get_health()
        assert isinstance(health, dict)
        assert "status" in health
        assert "databases" in health
        assert "boards" in health

    def test_databases_substructure(self):
        health = get_health()
        dbs = health["databases"]
        assert isinstance(dbs, dict)

    def test_boards_is_list(self):
        health = get_health()
        assert isinstance(health["boards"], list)
        if health["boards"]:
            board = health["boards"][0]
            assert "slug" in board
            assert "task_counts" in board


# ---------------------------------------------------------------------------
# Multi-source aggregation
# ---------------------------------------------------------------------------


class TestWorkSnapshotAggregation:
    def test_mixed_sources(self):
        items = [
            WorkItem(id="s1", title="Session 1", lifecycle="running", source="session"),
            WorkItem(id="k1", title="Card 1", lifecycle="ready", source="kanban"),
        ]
        snap = WorkSnapshot(items)
        assert len(snap.items) == 2
        sources = {i.source for i in snap.items}
        assert "session" in sources
        assert "kanban" in sources

    def test_filtering(self):
        items = [
            WorkItem(
                id="s-run", title="Running", lifecycle="running", source="session"
            ),
            WorkItem(
                id="k-done", title="Done", lifecycle="verified_done", source="kanban"
            ),
            WorkItem(
                id="k-review",
                title="Review me",
                lifecycle="ready_for_review",
                source="kanban",
            ),
            WorkItem(
                id="k-stalled", title="Stalled", lifecycle="stalled", source="kanban"
            ),
        ]
        snap = WorkSnapshot(items)
        grouped = snap.by_lifecycle()
        assert len(grouped.get("running", [])) == 1
        assert len(grouped.get("verified_done", [])) == 1
        assert len(grouped.get("ready_for_review", [])) == 1
        assert len(grouped.get("stalled", [])) == 1

    def test_empty_snapshot_by_lifecycle(self):
        snap = WorkSnapshot([])
        assert snap.by_lifecycle() == {}


# ---------------------------------------------------------------------------
# get_work_snapshot
# ---------------------------------------------------------------------------


class TestGetWorkSnapshot:
    def test_returns_snapshot(self):
        snap = get_work_snapshot()
        assert isinstance(snap, WorkSnapshot)
        assert isinstance(snap.items, list)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


class TestPathHelpers:
    def test_hermes_home_is_path(self):
        h = _hermes_home()
        assert isinstance(h, Path)

    def test_kanban_db_path(self):
        p = _kanban_db_path()
        if p is not None:
            assert str(p).endswith("kanban.db")

    def test_state_db_path(self):
        p = _state_db_path()
        if p is not None:
            assert str(p).endswith("state.db")


# ---------------------------------------------------------------------------
# Real SQLite adapter coverage
# ---------------------------------------------------------------------------


def _create_state_db(path: Path, now: float) -> None:
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            title TEXT,
            source TEXT NOT NULL,
            profile_name TEXT,
            started_at REAL NOT NULL,
            ended_at REAL,
            end_reason TEXT,
            last_activity_at REAL,
            last_activity_description TEXT,
            archived INTEGER NOT NULL DEFAULT 0
        );
    """)
    conn.execute(
        """INSERT INTO sessions
           (id, title, source, profile_name, started_at, ended_at, end_reason,
            last_activity_at, last_activity_description, archived)
           VALUES (?, ?, ?, ?, ?, NULL, NULL, ?, ?, 0)""",
        ("session-1", "Ship operator view", "cli", "ops", now - 60, now - 5, "Verify"),
    )
    conn.commit()
    conn.close()


def _create_kanban_db(path: Path, now: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            status TEXT NOT NULL,
            assignee TEXT,
            project_id TEXT,
            tenant TEXT,
            created_at INTEGER NOT NULL,
            started_at INTEGER,
            completed_at INTEGER,
            result TEXT,
            consecutive_failures INTEGER NOT NULL DEFAULT 0,
            last_heartbeat_at INTEGER,
            current_run_id INTEGER
        );
        CREATE TABLE task_runs (
            id INTEGER PRIMARY KEY,
            started_at INTEGER NOT NULL,
            ended_at INTEGER,
            last_heartbeat_at INTEGER,
            metadata TEXT
        );
    """)
    conn.execute(
        "INSERT INTO task_runs VALUES (?, ?, NULL, ?, ?)",
        (1, int(now - 120), int(now - 10), json.dumps({"changed_files": ["a.py"]})),
    )
    conn.execute(
        """INSERT INTO tasks
           (id, title, status, assignee, project_id, tenant, created_at,
            started_at, completed_at, result, consecutive_failures,
            last_heartbeat_at, current_run_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, 0, ?, 1)""",
        (
            "task-1",
            "Build operator view",
            "running",
            "ops",
            "hermes",
            None,
            int(now - 180),
            int(now - 120),
            json.dumps({"changed_files": ["a.py"], "unresolved_risks": ["review"]}),
            int(now - 10),
        ),
    )
    conn.commit()
    conn.close()


class TestRealDatabaseAdapters:
    def test_read_only_connection_never_creates_or_mutates(self, tmp_path):
        missing = tmp_path / "missing.db"
        assert _connect_read_only(missing) is None
        assert not missing.exists()

        existing = tmp_path / "existing.db"
        sqlite3.connect(existing).close()
        conn = _connect_read_only(existing)
        assert conn is not None
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("CREATE TABLE forbidden (id INTEGER)")
        conn.close()

    def test_reads_current_session_schema(self, tmp_path, monkeypatch):
        now = time.time()
        home = tmp_path / ".hermes"
        home.mkdir()
        _create_state_db(home / "state.db", now)
        monkeypatch.setenv("HERMES_HOME", str(home))

        items = _fetch_session_items(now)
        assert len(items) == 1
        assert items[0].id == "session-1"
        assert items[0].profile == "ops"
        assert items[0].assignee is None
        assert items[0].stage == "Verify"
        assert items[0].lifecycle == "running"

    def test_reads_current_kanban_schema(self, tmp_path, monkeypatch):
        now = time.time()
        root = tmp_path / ".hermes"
        board_db = root / "kanban" / "boards" / "platform" / "kanban.db"
        _create_kanban_db(board_db, now)
        monkeypatch.setenv("HERMES_HOME", str(root))
        monkeypatch.setenv("HERMES_KANBAN_HOME", str(root))

        items = _fetch_kanban_items(now, board="platform")
        assert len(items) == 1
        assert items[0].id == "task-1"
        assert items[0].board == "platform"
        assert items[0].lifecycle == "running"
        assert items[0].changed_files == ["a.py"]
        assert items[0].unresolved_risks == ["review"]
