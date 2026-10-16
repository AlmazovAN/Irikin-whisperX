import functools
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from functools import wraps
from sqlalchemy.exc import SQLAlchemyError
from fastapi import HTTPException
from sqlmodel import SQLModel
from backend.db.task.migrations import migrate_task_database, migrate_task_schema
from dotenv import load_dotenv

from backend.common.config_loader import read_env


@functools.lru_cache
def init_db():
    db_url = read_env("DB_URL", "sqlite:///backend/records.db")
    sqlite_url = db_url.startswith("sqlite:")
    engine = create_engine(db_url, connect_args={"check_same_thread": False} if sqlite_url else {})
    if engine.dialect.name == "sqlite":
        if engine.url.database and engine.url.database != ":memory:":
            # Automatic backup precedes any upgrade of a legacy SQLite file.
            migrate_task_database(engine.url.database)
        else:
            raw = engine.raw_connection()
            try:
                migrate_task_schema(raw)
            finally:
                raw.close()
    SQLModel.metadata.create_all(engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db_session():
    db_instance = init_db()
    return db_instance()


def handle_database_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        session = None
        try:
            session = get_db_session()
            kwargs['session'] = session

            return func(*args, **kwargs)
        except Exception as e:
            print(f"Database error has occurred: {e}")
            raise
        finally:
            if session:
                session.close()
    return wrapper
