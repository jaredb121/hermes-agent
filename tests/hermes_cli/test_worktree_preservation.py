"""Cleanup must keep ignored evidence and any untracked bytes not safely archived.

Real Git repositories and native modules exercise classification and cleanup. Destructive
Git commands are intercepted so the assertions observe decisions without deleting fixtures.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def cleanup_repo(tmp_path, monkeypatch):
    from hermes_cli import worktree_gc as gc, worktree_ops as ops
    import cli

    for key in (
        "GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_CEILING_DIRECTORIES", "GIT_NAMESPACE",
        "GIT_PREFIX", "GIT_CONFIG", "GIT_CONFIG_PARAMETERS", "GIT_CONFIG_COUNT",
        "GIT_SHALLOW_FILE",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in {
        "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_AUTHOR_NAME": "test", "GIT_COMMITTER_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@example.invalid", "GIT_COMMITTER_EMAIL": "test@example.invalid",
    }.items():
        monkeypatch.setenv(key, value)

    real_run = subprocess.run
    attempts = []
    on_unlock = []

    def guarded_run(command, *args, **kwargs):
        argv = list(command) if isinstance(command, (list, tuple)) else []
        if argv[:1] == ["git"] and (
            argv[1:3] in (["worktree", "remove"], ["worktree", "prune"], ["worktree", "unlock"])
            or argv[1:2] == ["clean"]
            or (argv[1:2] == ["branch"] and any(flag in argv for flag in ("-D", "-d", "--delete")))
        ):
            attempts.append(argv[1:])
            if argv[1:3] == ["worktree", "unlock"]:
                for callback in on_unlock:
                    callback()
            return subprocess.CompletedProcess(command, 1, "", "cleanup intercepted")
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", guarded_run)
    monkeypatch.setattr(cli, "release_lsp_clients", lambda _: None)
    monkeypatch.setattr(cli, "_cprint", lambda _: None)
    monkeypatch.setattr(cli, "_active_worktree", None)

    repo = tmp_path / "repo"

    def git(cwd, *args):
        return subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True,
                              check=True).stdout.strip()

    git(tmp_path, "init", "-b", "main", str(repo))
    (repo / ".gitignore").write_text(".env\nignored/\ncache/\n", encoding="utf-8")
    (repo / "tracked.txt").write_text("committed\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "fixture")
    git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    tree = repo / ".worktrees" / "hermes-preserve"
    branch = "hermes/preserve"
    git(repo, "worktree", "add", "-b", branch, str(tree))
    info = {"path": str(tree), "branch": branch, "repo_root": str(repo)}
    return repo, tree, info, attempts, on_unlock, git, cli, gc, ops


@pytest.mark.parametrize("artifact", [
    ".env", "ignored/evidence.json", "cache/output.bin",
    pytest.param('ignored/report "quoted"\nname.txt', marks=pytest.mark.platforms("posix")),
    "inventory-failure",
])
def test_ignored_or_unreadable_files_survive_every_cleanup_path(cleanup_repo, monkeypatch, artifact):
    repo, tree, info, attempts, _, git, cli, gc, ops = cleanup_repo
    data = b"local evidence\x00must survive\n"
    if artifact == "inventory-failure":
        real_run = subprocess.run

        def fail_status(command, *args, **kwargs):
            if list(command)[:2] == ["git", "status"] and str(kwargs.get("cwd")) == str(tree):
                return subprocess.CompletedProcess(command, 128, "", "inventory unavailable")
            return real_run(command, *args, **kwargs)

        monkeypatch.setattr(subprocess, "run", fail_status)
    else:
        path = tree / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    assert ops._worktree_is_dirty(str(tree))
    candidates = [(tree, 0, True)]
    assert ops._classify_prune_candidates(str(repo), candidates)[0][3] == "dirty"
    assert gc.audit_worktrees(str(repo), with_sizes=False)[0].verdict == "keep"
    # A verdict frozen while the tree was clean must not override today's contents.
    preserved, _ = ops._reap_prune_verdicts(str(repo), [(tree, 0, True, "reap", None)], 1)
    assert preserved
    frozen = gc.TreeRecord(tree.name, str(tree), info["branch"], 10, None, "reap", "old audit")
    assert any(action.startswith("kept ") for action in gc.reclaim_worktrees(str(repo), records=[frozen]))
    cli._cleanup_worktree(info)
    assert not any(command[:2] == ["worktree", "remove"] for command in attempts)
    assert not any(command[:2] == ["branch", "-D"] for command in attempts)
    assert git(repo, "rev-parse", "--verify", info["branch"])
    if artifact != "inventory-failure":
        assert (tree / artifact).read_bytes() == data
    from hermes_constants import get_hermes_home
    assert not (get_hermes_home() / "archive" / "worktree-prune").exists()


@pytest.mark.parametrize("change", [
    "clean", "visible-files",
    pytest.param("quoted-paths", marks=pytest.mark.platforms("posix")),
    pytest.param("byte-filenames", marks=pytest.mark.platforms("linux")),
    "ignored-after-audit", "tracked-after-audit", "ignored-during-copy",
    "bytes-during-copy", "bytes-during-unlock", "ignored-during-unlock",
    pytest.param("symlink", marks=pytest.mark.platforms("posix")),
    "live-lock-after-audit", "exit-release", "startup-unlock",
])
def test_removal_requires_current_files_and_verified_archive(cleanup_repo, monkeypatch, change):
    repo, tree, info, attempts, on_unlock, git, cli, gc, ops = cleanup_repo
    from hermes_constants import get_hermes_home

    clean_changes = {"clean", "ignored-after-audit", "tracked-after-audit",
                     "live-lock-after-audit", "exit-release", "startup-unlock"}
    files = {}
    if change not in clean_changes:
        files = {"drafts/sibling.txt": b"sibling bytes\n", "drafts/evidence.txt": b"before copy\n"}
        if change == "quoted-paths":
            files['drafts/report "quoted"\nname.txt'] = files.pop("drafts/evidence.txt")
        if change == "byte-filenames":
            files["drafts/bad-\udcff.txt"] = b"non-UTF8 pathname bytes\n"
            files["drafts/bad-\ufffd.txt"] = b"literal replacement character\n"
        for rel, data in files.items():
            path = tree / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    if change == "symlink":
        (tree / "drafts/link").symlink_to("evidence.txt")

    records = gc.audit_worktrees(str(repo), with_sizes=False)
    assert records[0].verdict in {"reap", "reap-archive"}

    def add_ignored():
        (tree / ".env").write_bytes(b"late ignored evidence\n")

    def edit_bytes():
        (tree / "drafts/evidence.txt").write_bytes(b"changed after copying\n")

    def edit_tracked():
        (tree / "tracked.txt").write_bytes(b"new tracked work\n")

    def live_lock():
        git(repo, "worktree", "lock", "--reason", f"hermes pid={os.getpid()}", str(tree))

    def change_during_copy():
        copy = shutil.copy2

        def copy_then_change(src, dest, *args, **kwargs):
            result = copy(src, dest, *args, **kwargs)
            (add_ignored if change == "ignored-during-copy" else edit_bytes)()
            return result

        monkeypatch.setattr(shutil, "copy2", copy_then_change)

    def change_during_unlock():
        on_unlock.append(edit_bytes if change == "bytes-during-unlock" else add_ignored)

    def change_during_release():
        monkeypatch.setattr(cli, "release_lsp_clients", lambda _: add_ignored())

    changes = {
        "ignored-after-audit": add_ignored, "tracked-after-audit": edit_tracked,
        "live-lock-after-audit": live_lock,
        "ignored-during-copy": change_during_copy, "bytes-during-copy": change_during_copy,
        "bytes-during-unlock": change_during_unlock, "ignored-during-unlock": change_during_unlock,
        "startup-unlock": change_during_unlock, "exit-release": change_during_release,
    }
    if change in changes:
        changes[change]()

    if change == "exit-release":
        cli._cleanup_worktree(info)
    elif change == "startup-unlock":
        preserved, _ = ops._reap_prune_verdicts(str(repo), [(tree, 0, True, "reap", "dead")], 1)
        assert preserved
    else:
        gc.reclaim_worktrees(str(repo), records=records)

    eligible = change in {"clean", "visible-files", "quoted-paths", "byte-filenames"}
    assert any(command[:2] == ["worktree", "remove"] for command in attempts) == eligible
    assert not any(command[:2] == ["branch", "-D"] for command in attempts)
    assert tree.exists() and git(repo, "rev-parse", "--verify", info["branch"])

    if eligible and files:
        archives = list((get_hermes_home() / "archive" / "worktree-prune").iterdir())
        assert len(archives) == 1
        for rel, data in files.items():
            assert (archives[0] / rel).read_bytes() == (tree / rel).read_bytes() == data
        # The archive destination must be fresh even if the tree and timestamp repeat.
        monkeypatch.setattr(gc.time, "strftime", lambda _: "fixed-time")
        first = gc._archive_untracked(tree, list(files))
        changed = next(iter(files))
        (tree / changed).write_bytes(b"second archive bytes\n")
        second = gc._archive_untracked(tree, list(files))
        assert first is not None and second is not None and first != second
        assert (first / changed).read_bytes() == files[changed]
        assert (second / changed).read_bytes() == b"second archive bytes\n"
    if change in {"bytes-during-copy", "bytes-during-unlock"}:
        assert (tree / "drafts/evidence.txt").read_bytes() == b"changed after copying\n"
    if change == "symlink":
        assert (tree / "drafts/link").is_symlink()
    if change.startswith("ignored-") or change in {"exit-release", "startup-unlock"}:
        assert (tree / ".env").read_bytes() == b"late ignored evidence\n"
        for archived in (get_hermes_home() / "archive" / "worktree-prune").glob("**/.env"):
            pytest.fail(f"ignored file was archived: {archived}")

    if change == "clean":
        # Exit also must retain the branch when Git refuses a clean-tree removal.
        attempts.clear()
        cli._cleanup_worktree(info)
        assert any(command[:2] == ["worktree", "remove"] for command in attempts)
        assert not any(command[:2] == ["branch", "-D"] for command in attempts)
