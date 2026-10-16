"""Versioned additive SQLite migrations for WhisperX Task jobs (P13 / PR14).

Standard-library only; never drop or rewrite existing Task rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

SCHEMA_VERSION = 1
COMPONENT = "whisperx_tasks"
MIGRATION_TABLE = "whisperx_schema_migrations"
NEW_COLUMNS = {
    "stage": "VARCHAR(24)",
    "attempt": "INTEGER NOT NULL DEFAULT 0",
    "input_path": "VARCHAR",
    "output_json_path": "VARCHAR",
    "output_txt_path": "VARCHAR",
    "model_name": "VARCHAR",
    "model_version": "VARCHAR",
    "started_at": "DATETIME",
    "finished_at": "DATETIME",
    "error_code": "VARCHAR",
}
REQUIRED_LEGACY_COLUMNS = {"id", "uuid", "status", "task_type"}


class TaskSchemaError(RuntimeError):
    """Unknown/incompatible Task schema: refuse a destructive migration."""


@dataclass(frozen=True)
class MigrationReport:
    old_version: int
    new_version: int
    created_table: bool
    added_columns: tuple[str, ...]
    backup_path: Path | None = None


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _columns(conn: sqlite3.Connection) -> set[str]:
    return {row[1] for row in conn.execute("PRAGMA table_info(tasks)")}


def _stored_version(conn: sqlite3.Connection) -> int:
    if not _has_table(conn, MIGRATION_TABLE):
        return 0
    row = conn.execute(
        "SELECT version FROM whisperx_schema_migrations WHERE component=?",
        (COMPONENT,),
    ).fetchone()
    return int(row[0]) if row else 0


def _needs_backup(conn: sqlite3.Connection) -> bool:
    return _has_table(conn, "tasks") and (
        _stored_version(conn) < SCHEMA_VERSION
        or not set(NEW_COLUMNS).issubset(_columns(conn))
    )


def migrate_task_schema(conn: sqlite3.Connection) -> MigrationReport:
    """Run an atomic, idempotent SQLite migration with a fixed column set."""
    try:
        conn.execute("BEGIN IMMEDIATE")
        previous = _stored_version(conn)
        if previous > SCHEMA_VERSION:
            raise TaskSchemaError("Newer Task schema; downgrade refused")
        created = not _has_table(conn, "tasks")
        if created:
            conn.execute("""
                CREATE TABLE tasks (
                    id INTEGER PRIMARY KEY,
                    uuid VARCHAR NOT NULL,
                    status VARCHAR(16),
                    result JSON,
                    result_type VARCHAR(16),
                    file_name VARCHAR,
                    url VARCHAR,
                    audio_duration FLOAT,
                    language VARCHAR,
                    task_type VARCHAR(24),
                    task_params JSON,
                    duration FLOAT,
                    error VARCHAR,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL,
                    progress FLOAT
                )
            """)
        elif REQUIRED_LEGACY_COLUMNS - _columns(conn):
            raise TaskSchemaError("Unrecognized legacy Task schema; upgrade refused")
        known = _columns(conn)
        added: list[str] = []
        for name, sql_type in NEW_COLUMNS.items():
            if name not in known:
                # Identifiers/types come only from fixed module constants.
                conn.execute(f"ALTER TABLE tasks ADD COLUMN {name} {sql_type}")
                added.append(name)
        conn.execute("CREATE INDEX IF NOT EXISTS ix_jobs_task_uuid ON tasks(uuid)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_jobs_status_created ON tasks(status, created_at)"
        )
        conn.execute("""
            CREATE TABLE IF NOT EXISTS whisperx_schema_migrations (
                component TEXT PRIMARY KEY,
                version INTEGER NOT NULL,
                applied_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            INSERT INTO whisperx_schema_migrations (component, version, applied_at)
            VALUES (?, ?, ?)
            ON CONFLICT(component) DO UPDATE SET
                version=excluded.version, applied_at=excluded.applied_at
        """, (COMPONENT, SCHEMA_VERSION, datetime.now(timezone.utc).isoformat()))
        conn.commit()
        return MigrationReport(previous, SCHEMA_VERSION, created, tuple(added))
    except Exception:
        conn.rollback()
        raise


def migrate_task_database(path: str | Path) -> MigrationReport:
    """Upgrade a file and create a consistent SQLite backup when required."""
    database = Path(path).expanduser()
    if database == Path(":memory:") or database.is_symlink():
        raise ValueError("Expected a regular on-disk SQLite database")
    database.parent.mkdir(parents=True, exist_ok=True)
    existed = database.is_file() and database.stat().st_size > 0
    with sqlite3.connect(str(database), timeout=30) as conn:
        backup: Path | None = None
        if existed and _needs_backup(conn):
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            backup = database.with_name(
                f"{database.name}.pre-job-schema-v{SCHEMA_VERSION}-{stamp}.bak"
            )
            with sqlite3.connect(str(backup)) as dest:
                conn.backup(dest)
        report = migrate_task_schema(conn)
    return MigrationReport(
        report.old_version, report.new_version, report.created_table,
        report.added_columns, backup
    )
