"""Direct PostgreSQL access for the embedding pipeline.

Uses psycopg (see AGENTS.md §8.3: no psql locally) against the configured
DATABASE_URL. All job-state mutations go through the plpgsql functions from
migration 057 (queue/claim/finish/fail/reset) so the job-claiming pattern is
identical to the scraper worker (FOR UPDATE SKIP LOCKED).
"""
import logging
import os
from pathlib import Path
from typing import Any

import psycopg

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except Exception:
    pass

logger = logging.getLogger("pasal.embed.db")

DEFAULT_DATABASE_URL = "postgresql://postgres:postgres@127.0.0.1:54322/postgres"


def get_database_url() -> str:
    """Return DATABASE_URL env (local fallback from AGENTS.md §8.1)."""
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


def connect() -> psycopg.Connection:
    return psycopg.connect(get_database_url(), autocommit=True)


# ---------------------------------------------------------------------------
# Job-state operations (mirror scripts/crawler/state.py)
# ---------------------------------------------------------------------------

def queue_jobs(
    limit: int,
    model: str,
    reg_type: str | None = None,
    conn: psycopg.Connection | None = None,
) -> int:
    """Enqueue unembedded pasal rows into embedding_jobs. Returns count."""
    own = conn is None
    c = conn or connect()
    try:
        with c.cursor() as cur:
            cur.execute(
                "SELECT queue_embedding_jobs(%s, %s, %s)",
                (limit, model, reg_type),
            )
            row = cur.fetchone()
            return int(row[0]) if row else 0
    finally:
        if own:
            c.close()


def claim_jobs(limit: int, conn: psycopg.Connection | None = None) -> list[dict[str, Any]]:
    """Atomically claim a batch of pending embedding jobs."""
    own = conn is None
    c = conn or connect()
    try:
        with c.cursor() as cur:
            cur.execute("SELECT * FROM claim_embedding_jobs(%s)", (limit,))
            cols = [d.name for d in cur.description] if cur.description else []
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        if own:
            c.close()


def finish_job(job_id: int, embedding_text: str, model: str, conn=None) -> int | None:
    """Write embedding to document_nodes + mark job done atomically."""
    own = conn is None
    c = conn or connect()
    try:
        with c.cursor() as cur:
            cur.execute(
                "SELECT finish_embedding(%s, %s::text, %s)",
                (job_id, embedding_text, model),
            )
            row = cur.fetchone()
            return int(row[0]) if row and row[0] is not None else None
    finally:
        if own:
            c.close()


def fail_job(job_id: int, error: str, conn=None) -> None:
    own = conn is None
    c = conn or connect()
    try:
        with c.cursor() as cur:
            cur.execute("SELECT fail_embedding(%s, %s)", (job_id, error))
    finally:
        if own:
            c.close()


def reset_model(model: str, conn=None) -> int:
    """Clear embeddings for a model so it can be re-embedded. Returns count."""
    own = conn is None
    c = conn or connect()
    try:
        with c.cursor() as cur:
            cur.execute("SELECT reset_embedding_jobs(%s)", (model,))
            row = cur.fetchone()
            return int(row[0]) if row else 0
    finally:
        if own:
            c.close()


def embedding_stats(model: str | None = None, conn=None) -> list[tuple]:
    """Per-model/status counts for the `stats` command."""
    own = conn is None
    c = conn or connect()
    try:
        with c.cursor() as cur:
            if model:
                cur.execute(
                    """
                    SELECT status, count(*) FROM embedding_jobs
                    WHERE model = %s GROUP BY status ORDER BY status
                    """,
                    (model,),
                )
            else:
                cur.execute(
                    """
                    SELECT model, status, count(*) FROM embedding_jobs
                    GROUP BY model, status ORDER BY model, status
                    """
                )
            return cur.fetchall()
    finally:
        if own:
            c.close()


def pasal_embedding_coverage(conn=None) -> list[tuple]:
    """How many pasal rows carry an embedding (per model)."""
    own = conn is None
    c = conn or connect()
    try:
        with c.cursor() as cur:
            cur.execute(
                """
                SELECT COALESCE(embedding_model, '(none)'),
                       count(*) FILTER (WHERE embedding IS NOT NULL),
                       count(*)
                FROM document_nodes
                WHERE node_type = 'pasal'
                GROUP BY 1 ORDER BY 1
                """
            )
            return cur.fetchall()
    finally:
        if own:
            c.close()
