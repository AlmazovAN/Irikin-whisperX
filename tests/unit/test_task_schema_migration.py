"""Fast, offline regression tests for PR14's additive SQLite migrations."""
from __future__ import annotations

from pathlib import Path
import sqlite3
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backend.db.task.migrations import (  # noqa: E402
    COMPONENT, MIGRATION_TABLE, NEW_COLUMNS, SCHEMA_VERSION,
    TaskSchemaError, migrate_task_database, migrate_task_schema,
)


def make_legacy_db(conn):
    conn.execute("""
        CREATE TABLE tasks (
            id INTEGER PRIMARY KEY, uuid VARCHAR,
            status VARCHAR, task_type VARCHAR,
            file_name VARCHAR, result JSON, task_params JSON,
            created_at DATETIME, updated_at DATETIME
        )
    """)
    conn.execute("""
        INSERT INTO tasks (id, uuid, status, task_type, file_name,
                           result, task_params, created_at, updated_at)
        VALUES (7, 'old-id', 'COMPLETED', 'TRANSCRIPTION', 'old.wav',
                '{"segments":[{"text":"hello"}]}',
                '{"legacy":true}', '2026-01-01', '2026-01-02')
    """)
    conn.execute("CREATE TABLE unrelated (id INTEGER PRIMARY KEY, note TEXT)")
    conn.execute("INSERT INTO unrelated VALUES (9, 'preserve')")
    conn.commit()


def test_fresh_profile_bootstraps_schema_and_default_attempt():
    with sqlite3.connect(":memory:") as conn:
        report = migrate_task_schema(conn)
        assert report.created_table and report.new_version == SCHEMA_VERSION
        names = {row[1] for row in conn.execute("PRAGMA table_info(tasks)")}
        assert set(NEW_COLUMNS).issubset(names)
        conn.execute("""
            INSERT INTO tasks (uuid, status, task_type, created_at, updated_at)
            VALUES ('new-id', 'QUEUED', 'TRANSCRIPTION', 'now', 'now')
        """)
        assert conn.execute(
            "SELECT attempt FROM tasks WHERE uuid='new-id'"
        ).fetchone()[0] == 0


def test_existing_task_and_unrelated_table_preserved():
    with sqlite3.connect(":memory:") as conn:
        make_legacy_db(conn)
        old = conn.execute(
            "SELECT * FROM tasks WHERE id=7"
        ).fetchone()
        report = migrate_task_schema(conn)
        assert set(report.added_columns) == set(NEW_COLUMNS)
        new = conn.execute(
            "SELECT id, uuid, status, task_type, file_name, result, "
            "task_params, created_at, updated_at FROM tasks WHERE id=7"
        ).fetchone()
        assert old == new
        assert conn.execute(
            "SELECT note FROM unrelated WHERE id=9"
        ).fetchone() == ("preserve",)
        assert conn.execute(
            "SELECT stage, attempt, input_path FROM tasks WHERE id=7"
        ).fetchone() == (None, 0, None)


def test_repeated_migration_does_not_reset_metadata():
    with sqlite3.connect(":memory:") as conn:
        make_legacy_db(conn)
        migrate_task_schema(conn)
        conn.execute("""
            UPDATE tasks SET stage='ASR', attempt=3,
                             model_name='small', model_version='3.4.2'
            WHERE id=7
        """)
        conn.commit()
        repeat = migrate_task_schema(conn)
        assert repeat.added_columns == ()
        assert not repeat.created_table
        assert conn.execute(
            "SELECT stage, attempt, model_name, model_version "
            "FROM tasks WHERE id=7"
        ).fetchone() == ("ASR", 3, "small", "3.4.2")


def test_unknown_legacy_schema_refused_and_rolled_back():
    with sqlite3.connect(":memory:") as conn:
        conn.execute("CREATE TABLE tasks (id INTEGER PRIMARY KEY)")
        conn.commit()
        with pytest.raises(TaskSchemaError, match="Unrecognized"):
            migrate_task_schema(conn)
        names = {row[1] for row in conn.execute("PRAGMA table_info(tasks)")}
        assert names == {"id"}
        assert not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name=?", (MIGRATION_TABLE,)
        ).fetchone()


def test_newer_schema_refused_without_downgrade():
    with sqlite3.connect(":memory:") as conn:
        migrate_task_schema(conn)
        conn.execute(
            "UPDATE whisperx_schema_migrations SET version=999 "
            "WHERE component=?", (COMPONENT,)
        )
        conn.commit()
        with pytest.raises(TaskSchemaError, match="Newer"):
            migrate_task_schema(conn)
        assert conn.execute(
            "SELECT version FROM whisperx_schema_migrations "
            "WHERE component=?", (COMPONENT,)
        ).fetchone()[0] == 999


def test_old_file_backed_up_before_upgrade_and_not_on_second_run(tmp_path):
    path = tmp_path / "existing.db"
    with sqlite3.connect(path) as conn:
        make_legacy_db(conn)
    upgraded = migrate_task_database(path)
    assert upgraded.backup_path and upgraded.backup_path.is_file()
    with sqlite3.connect(upgraded.backup_path) as original:
        assert original.execute(
            "SELECT file_name FROM tasks WHERE id=7"
        ).fetchone() == ("old.wav",)
        assert "attempt" not in {
            col[1] for col in original.execute("PRAGMA table_info(tasks)")
        }
    with sqlite3.connect(path) as conn:
        assert conn.execute(
            "SELECT status, attempt FROM tasks WHERE id=7"
        ).fetchone() == ("COMPLETED", 0)
    again = migrate_task_database(path)
    assert again.backup_path is None and again.added_columns == ()


def test_brand_new_on_disk_profile_has_no_backup(tmp_path):
    path = tmp_path / "data" / "jobs" / "jobs.sqlite3"
    report = migrate_task_database(path)
    assert path.is_file() and report.created_table
    assert report.backup_path is None


def test_file_symlink_is_not_followed(tmp_path):
    target = tmp_path / "legacy.db"
    target.touch()
    alias = tmp_path / "link.db"
    try:
        alias.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks not supported on test system")
    with pytest.raises(ValueError, match="regular"):
        migrate_task_database(alias)


def test_task_model_declares_private_paths_but_public_response_does_not():
    source = (ROOT / "backend/db/task/models.py").read_text(encoding="utf-8")
    for field in NEW_COLUMNS:
        assert f"{field}:" in source
    public = source.split("class TaskStatusResponse")[1].split("class Task(")[0]
    for private in ("input_path", "output_json_path", "output_txt_path"):
        assert private not in public
