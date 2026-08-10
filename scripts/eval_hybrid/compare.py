"""Compare fts_only vs hybrid search on the eval set (AGENTS.md §6).

Runs every query in queries.py through search_hybrid() twice:
  - mode='fts_only'   (legacy behavior; query embedding not needed)
  - mode='hybrid'     (RRF fusion; query embedded locally via scripts/embed)
and writes a comparison report (JSON + human-readable table).

Usage:
    .venv/bin/python -m scripts.eval_hybrid.compare --output data/eval_report.json
    .venv/bin/python -m scripts.eval_hybrid.compare --categories pasal,mixed
    .venv/bin/python -m scripts.eval_hybrid.compare --skip-embedding   # hybrid may degrade

Requires scripts/.env with DATABASE_URL. Embedding model defaults to
EMBEDDING_MODEL env / BAAI/bge-m3 — must match the model that embedded the DB.
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from psycopg.types.json import Jsonb

load_dotenv(Path(__file__).parent.parent / ".env")

from scripts.embed import db  # noqa: E402
from scripts.eval_hybrid.queries import QUERIES, by_category  # noqa: E402

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger("pasal.eval")

TOP_K = 5


def _vector_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in vec) + "]"


def _run_query(conn, query: str, mode: str, embedding: list[float] | None, limit: int):
    payload = {
        "query_text": query,
        "match_count": limit,
        "metadata_filter": {},
        "mode": mode,
    }
    if embedding is not None:
        payload["query_embedding"] = _vector_literal(embedding)
    with conn.cursor() as cur:
        cur.execute("select * from search_hybrid(%s, %s, %s, %s, %s)",
                    (query, limit, Jsonb({}), mode,
                     _vector_literal(embedding) if embedding else None))
        rows = cur.fetchall()
    cols = ["id", "work_id", "content", "metadata", "score", "snippet"]
    return [dict(zip(cols, r)) for r in rows]


def _overlap(fts_ids: list, hyb_ids: list) -> tuple[int, int]:
    fts5, hyb5 = set(fts_ids[:TOP_K]), set(hyb_ids[:TOP_K])
    return (len(fts5 & hyb5), len(set(fts_ids) & set(hyb_ids)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare fts_only vs hybrid on eval set")
    ap.add_argument("--output", default="data/eval_report.json", help="JSON report path")
    ap.add_argument("--categories", default="pasal,case,mixed",
                    help="Comma-separated categories to run (default: all)")
    ap.add_argument("--limit", type=int, default=10, help="Results per query per mode")
    ap.add_argument("--skip-embedding", action="store_true",
                    help="Do not embed queries (hybrid leg then degrades to FTS)")
    args = ap.parse_args(argv)

    cats = set(args.categories.split(","))
    queries = [q for q in QUERIES if q["category"] in cats]

    conn = db.connect()
    model = None
    if not args.skip_embedding:
        try:
            from scripts.embed.model import EmbeddingModel
            model = EmbeddingModel()
        except Exception as e:
            logger.warning("Embedding model unavailable (%s); hybrid may degrade.", e)

    report: dict = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mode": "skip_embedding" if args.skip_embedding else "embedding",
        "queries_run": 0,
        "by_category": {},
        "queries": [],
    }
    totals = {"fts": 0, "hybrid": 0, "top5_overlap": 0, "all_overlap": 0}

    try:
        for q in queries:
            t0 = time.time()
            vec = None
            if model is not None:
                try:
                    vec = model.encode([q["query"]], batch_size=1)["dense_vecs"][0].tolist()
                except Exception as e:
                    logger.error("encode failed for %s: %s", q["id"], e)

            fts = _run_query(conn, q["query"], "fts_only", None, args.limit)
            hyb = _run_query(conn, q["query"], "hybrid", vec, args.limit)
            o5, oall = _overlap([r["id"] for r in fts], [r["id"] for r in hyb])

            item = {
                "id": q["id"],
                "category": q["category"],
                "query": q["query"],
                "note": q["note"],
                "fts_count": len(fts),
                "hybrid_count": len(hyb),
                "top5_overlap": o5,
                "total_overlap": oall,
                "ms": round((time.time() - t0) * 1000),
                "fts_top": [r["metadata"] for r in fts[:TOP_K]],
                "hybrid_top": [r["metadata"] for r in hyb[:TOP_K]],
            }
            report["queries"].append(item)
            report["queries_run"] += 1
            totals["fts"] += item["fts_count"]
            totals["hybrid"] += item["hybrid_count"]
            totals["top5_overlap"] += o5
            totals["all_overlap"] += oall

            flag = "SAME" if fts[:TOP_K] == hyb[:TOP_K] else "diff"
            logger.info("%-5s [%s] fts=%d hyb=%d top5overlap=%d/5 %s (%dms)",
                        q["id"], flag, item["fts_count"], item["hybrid_count"], o5,
                        q["query"][:45], item["ms"])

        for cat in sorted(cats):
            cat_items = [i for i in report["queries"] if i["category"] == cat]
            report["by_category"][cat] = {
                "n": len(cat_items),
                "avg_top5_overlap": (sum(i["top5_overlap"] for i in cat_items) / len(cat_items))
                                    if cat_items else 0.0,
            }
    finally:
        conn.close()
        if model is not None:
            model.unload()

    report["summary"] = {
        "total_queries": report["queries_run"],
        "total_fts_results": totals["fts"],
        "total_hybrid_results": totals["hybrid"],
        "avg_top5_overlap": totals["top5_overlap"] / max(report["queries_run"], 1),
        "avg_total_overlap": totals["all_overlap"] / max(report["queries_run"], 1),
        "queries_with_more_hybrid_results": sum(
            1 for i in report["queries"] if i["hybrid_count"] > i["fts_count"]),
        "queries_with_fewer_hybrid_results": sum(
            1 for i in report["queries"] if i["hybrid_count"] < i["fts_count"]),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"Report written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
