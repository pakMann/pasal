"""Optional query-embedding for semantic search (search_laws_semantic).

BGE-M3 runs in-process via FlagEmbedding, loaded lazily on first use so the
base MCP server keeps its current light footprint. If the heavy deps
(torch + FlagEmbedding, see requirements-semantic.txt) are not installed, or
the model fails to load, embed_query() returns None and server.py degrades
the semantic tool to the existing full-text path (AGENTS.md §4.4).

The embedding model must match the one used by the batch pipeline
(scripts/embed/, default EMBEDDING_MODEL env / BAAI/bge-m3) so query and
document vectors share a space.
"""
import logging
import os
from typing import Optional

logger = logging.getLogger("pasal.mcp.semantic")

DEFAULT_MODEL = "BAAI/bge-m3"

_model = None
_model_name = None


def _get_model():
    """Lazily load the FlagEmbedding model singleton (thread-unsafe, ok here)."""
    global _model, _model_name
    if _model is not None:
        return _model
    name = os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL)
    try:
        from FlagEmbedding import BGEM3FlagModel

        logger.info("Loading embedding model %s (first call, may take a while)...", name)
        _model = BGEM3FlagModel(
            name,
            use_fp16=False,
            device="cpu",
        )
        _model_name = name
        return _model
    except Exception as e:
        logger.warning("Embedding model %s unavailable (%s); semantic search will degrade to FTS.", name, e)
        return None


def embed_query(text: str) -> Optional[list[float]]:
    """Embed a single query as a dense vector, or None on any failure."""
    model = _get_model()
    if model is None:
        return None
    try:
        out = model.encode(
            [text],
            batch_size=1,
            max_length=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        vec = out["dense_vecs"][0].tolist()
        return vec
    except Exception as e:
        logger.error("embed_query failed: %s", e)
        return None


def model_name() -> str:
    """Return the active model name, or the configured/default one."""
    return _model_name or os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL)
