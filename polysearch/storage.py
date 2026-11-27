"""Transactional persistence shared by the workspace, pilots and account service."""
import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from threading import local

# Application-wide transaction lock, per PostgreSQL database.
LOCK_ID = 581723104283


class Session:
    def __init__(self, connection, postgres):
        self.connection, self.postgres = connection, postgres

    def execute(self, sql, args=()):
        # All SQL templates are application constants. Never pass user SQL.
        return self.connection.execute(sql.replace("?", "%s") if self.postgres else sql, args)


class Database:
    def __init__(self, location):
        self.location = str(location)
        self.postgres = self.location.startswith(("postgresql://", "postgres://")) or self.location.startswith("dbname=")
        self._local = local()
        if not self.postgres:
            if self.location == ":memory:":
                raise ValueError("Use a file-backed SQLite database")
            Path(self.location).parent.mkdir(parents=True, exist_ok=True)

    def connect(self):
        if self.postgres:
            import psycopg
            options = {"connect_timeout": 5}
            secret = os.environ.get("POLYSEARCH_PG_PASSWORD_FILE")
            if secret:
                options["password"] = Path(secret).read_text(encoding="utf-8").rstrip("\r\n")
            return psycopg.connect(self.location, **options)
        connection = sqlite3.connect(self.location, timeout=30)
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @contextmanager
    def transaction(self):
        existing = getattr(self._local, "session", None)
        if existing is not None:
            yield existing
            return
        connection = self.connect()
        session = Session(connection, self.postgres)
        try:
            if self.postgres:
                session.execute("SELECT pg_advisory_xact_lock(?)", (LOCK_ID,))
            else:
                session.execute("BEGIN IMMEDIATE")
            self._local.session = session
            yield session
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            self._local.session = None
            connection.close()

    @property
    def lock(self):
        return self.transaction()

    def ping(self):
        # Do not queue health checks behind model training.
        connection = self.connect()
        try:
            return connection.execute("SELECT 1").fetchone()[0] == 1
        finally:
            connection.close()


class Store:
    def __init__(self, location):
        self.db = Database(location)
        with self.db.transaction() as db:
            db.execute("CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS pilot_campaigns (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS integration_inbox (id TEXT PRIMARY KEY, source_system TEXT NOT NULL, external_id TEXT NOT NULL, payload TEXT NOT NULL, UNIQUE(source_system, external_id))")

            db.execute("CREATE TABLE IF NOT EXISTS research_runs (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, summary TEXT NOT NULL, payload TEXT NOT NULL)")

    @property
    def lock(self):
        return self.db.lock

    def load(self):
        with self.db.transaction() as db:
            row = db.execute("SELECT payload FROM state WHERE id=1").fetchone()
        return json.loads(row[0]) if row else {"config": None, "trials": [], "recommendations": [],
                                               "report": None, "revision": 0}

    def save(self, state):
        with self.db.transaction() as db:
            db.execute("INSERT INTO state VALUES (1, ?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                       (json.dumps(state, ensure_ascii=False, allow_nan=False),))


def database_location():
    return os.environ.get("POLYSEARCH_POSTGRES_DSN") or os.environ.get("POLYSEARCH_DB", "polysearch.sqlite3")
