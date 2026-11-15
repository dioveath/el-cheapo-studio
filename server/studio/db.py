import os
from contextlib import contextmanager
from pathlib import Path
import psycopg
from psycopg.rows import dict_row

DATABASE_URL = os.getenv("DATABASE_URL")
DATA_ROOT = Path(os.getenv("DATA_ROOT", "/data"))


@contextmanager
def connection():
    options = {"row_factory": dict_row}
    if DATABASE_URL:
        options["conninfo"] = DATABASE_URL
    else:
        password = os.getenv("POSTGRES_PASSWORD")
        if not password:
            raise RuntimeError("Set DATABASE_URL or POSTGRES_PASSWORD before connecting to Postgres.")
        options.update(host=os.getenv("POSTGRES_HOST", "postgres"), port=5432,
                       dbname="studio", user="studio", password=password)
    with psycopg.connect(**options) as conn:
        yield conn


def initialize():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    with connection() as conn:
        conn.execute("SELECT pg_advisory_xact_lock(871204)")
        conn.execute(Path(__file__).with_name("schema.sql").read_text())
        for migration in sorted(Path(__file__).with_name("migrations").glob("*.sql")):
            if not conn.execute("SELECT 1 FROM schema_migrations WHERE name=%s", (migration.name,)).fetchone():
                conn.execute(migration.read_text())
                conn.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (migration.name,))


def get_job(job_id):
    with connection() as conn:
        return conn.execute("SELECT j.*, COALESCE(a.filename, left(j.settings->>'prompt', 80)) AS filename, COALESCE(a.kind, 'text') AS kind FROM jobs j LEFT JOIN assets a ON a.id=j.asset_id WHERE j.id=%s", (job_id,)).fetchone()
