"""Embedding pipeline CLI for pasal nodes (AGENTS.md §4.2).

Usage:
    .venv/bin/python -m scripts.embed.embed_pasals seed --reg-type UU --limit 5000
    .venv/bin/python -m scripts.embed.embed_pasals run --batch-size 32 --max-jobs 5000
    .venv/bin/python -m scripts.embed.embed_pasals stats
    .venv/bin/python -m scripts.embed.embed_pasals reset

The run command loops atomically-claimed batches (FOR UPDATE SKIP LOCKED, via
claim_embedding_jobs) and is resumable: crash mid-batch → the claimed jobs are
auto-reclaimed after the stale timeout and re-embedded. Batching + backoff keeps
the model under control even on CPU.

Requires scripts/.env with DATABASE_URL (local default is AGENTS.md §8.1).
"""
import argparse
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(Path(__file__).parent.parent / ".env")

from scripts.embed import db as db  # noqa: E402
from scripts.embed.context_builder import (  # noqa: E402
    build_contexts,
    fetch_batch_contexts,
)
from scripts.embed.model import EmbeddingModel, get_model_name  # noqa: E402

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("pasal.embed")

MAX_BATCH_SIZE = 128
CLAIM_TIMEOUT_RETRY_S = 5.0


def _vector_literal(vec: list[float]) -> str:
    """Serialize a vector for the SQL finish_embedding(_, text, _) call."""
    return "[" + ",".join(f"{v:.6f}" for v in vec) + "]"


def cmd_seed(args: argparse.Namespace) -> int:
    conn = db.connect()
    try:
        n = db.queue_jobs(args.limit, args.model, args.reg_type, conn=conn)
    finally:
        conn.close()
    logger.info("Queued %d new embedding jobs (model=%s reg_type=%s)",
                n, args.model, args.reg_type)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    model = EmbeddingModel()
    conn = db.connect()
    done_total = 0
    failed_total = 0
    claimed_rounds = 0

    try:
        while True:
            if args.max_jobs and done_total + failed_total >= args.max_jobs:
                break

            jobs = db.claim_jobs(args.batch_size, conn=conn)
            if not jobs:
                # Nothing pending right now — either finished or all mid-flight.
                logger.info("No pending jobs. Done=%d failed=%d",
                            done_total, failed_total)
                if args.max_rounds and claimed_rounds >= args.max_rounds:
                    break
                if args.loop:
                    time.sleep(CLAIM_TIMEOUT_RETRY_S)
                    continue
                break

            claimed_rounds += 1
            node_ids = [j["node_id"] for j in jobs]
            job_by_node = {j["node_id"]: j for j in jobs}

            pasal_nodes, works, ancestors, ayat = fetch_batch_contexts(conn, node_ids)
            contexts = build_contexts(pasal_nodes, works, ancestors, ayat)

            # Embed only what we have context for; fail the rest loudly.
            embeddable = []
            for nid in node_ids:
                text = contexts.get(nid)
                if text:
                    embeddable.append((nid, text))
                else:
                    db.fail_job(job_by_node[nid]["id"], "no context text", conn=conn)
                    failed_total += 1

            texts = [t for _, t in embeddable]
            try:
                vecs = model.encode(texts, batch_size=args.batch_size,
                                    show_progress=False)
            except Exception as e:
                logger.error("encode failed for batch of %d: %s", len(texts), e)
                for nid, _ in embeddable:
                    db.fail_job(job_by_node[nid]["id"], f"encode error: {e}", conn=conn)
                    failed_total += 1
                continue

            batch_failed = 0
            for (nid, _), vec in zip(embeddable, vecs):
                try:
                    db.finish_job(job_by_node[nid]["id"], _vector_literal(vec),
                                  args.model, conn=conn)
                    done_total += 1
                except Exception as e:
                    logger.error("finish_embedding failed for node %s: %s", nid, e)
                    db.fail_job(job_by_node[nid]["id"], f"finish error: {e}", conn=conn)
                    batch_failed += 1
                    failed_total += 1

            logger.info("Batch %d: %d embedded, %d failed "
                        "(cumulative %d done / %d failed)",
                        claimed_rounds, len(embeddable) - batch_failed, batch_failed,
                        done_total, failed_total)
    finally:
        conn.close()
        model.unload()

    logger.info("Run finished: %d embedded, %d failed", done_total, failed_total)
    return 0 if failed_total == 0 else 1


def cmd_stats(_args: argparse.Namespace) -> int:
    conn = db.connect()
    try:
        print("=== embedding_jobs by model/status ===")
        for row in db.embedding_stats(conn=conn):
            print("  ", row)
        print("=== pasal embedding coverage ===")
        for row in db.pasal_embedding_coverage(conn=conn):
            print("  ", row)
    finally:
        conn.close()
    return 0


def cmd_reset(args: argparse.Namespace) -> int:
    conn = db.connect()
    try:
        n = db.reset_model(args.model, conn=conn)
    finally:
        conn.close()
    logger.info("Cleared embeddings for model %s (%d pasal rows). "
                "Re-run `seed` then `run` to re-embed.", args.model, n)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embedding pipeline for pasal nodes (BGE-M3 / pgvector).",
    )
    parser.add_argument("--model", default=None,
                        help="Embedding model name (default: EMBEDDING_MODEL env / BAAI/bge-m3)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_seed = sub.add_parser("seed", help="Enqueue unembedded pasal rows")
    p_seed.add_argument("--reg-type", default=None,
                        help="Regulation type code filter, e.g. UU (default: all)")
    p_seed.add_argument("--limit", type=int, default=1000000,
                        help="Max rows to enqueue (default: unlimited)")

    p_run = sub.add_parser("run", help="Claim + embed pending jobs")
    p_run.add_argument("--batch-size", type=int, default=32,
                       help="Jobs per batch (default: 32)")
    p_run.add_argument("--max-jobs", type=int, default=None,
                       help="Stop after this many total jobs")
    p_run.add_argument("--max-rounds", type=int, default=None,
                       help="Stop after this many claimed batches")
    p_run.add_argument("--loop", action="store_true",
                       help="Keep polling when no jobs are pending")

    sub.add_parser("stats", help="Show embedding job/pipeline stats")

    p_reset = sub.add_parser("reset", help="Clear a model's embeddings for re-embed")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.model:
        import os
        os.environ["EMBEDDING_MODEL"] = args.model
    else:
        args.model = get_model_name()

    if args.command == "seed":
        return cmd_seed(args)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "stats":
        return cmd_stats(args)
    if args.command == "reset":
        return cmd_reset(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
