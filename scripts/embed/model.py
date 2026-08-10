"""BGE-M3 embedding model wrapper (config-driven, AGENTS.md §4.2).

The model name is read from the EMBEDDING_MODEL env var (default BAAI/bge-m3)
so the choice can be re-evaluated without code changes. Only dense vectors are
produced for v1; BGE-M3's sparse / ColBERT outputs are left unused (sparse
lexical matching is already covered by the FTS path fused via RRF in §4.3).
"""
import logging
import os

logger = logging.getLogger("pasal.embed")

DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_MAX_LENGTH = 2048  # legal pasals fit comfortably; BGE-M3 supports 8192
DEFAULT_BATCH_SIZE = 32

_IMPORT_ERROR = (
    "FlagEmbedding is not installed. Install pipeline deps first:\n"
    "  .venv/bin/pip install -r scripts/embed/requirements.txt"
)


def get_model_name() -> str:
    """Return the configured embedding model name."""
    return os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL)


class EmbeddingModel:
    """Lazy-loaded singleton around FlagEmbedding's BGEM3FlagModel."""

    def __init__(self, model_name: str | None = None, max_length: int = DEFAULT_MAX_LENGTH):
        self.model_name = model_name or get_model_name()
        self.max_length = max_length
        self._model = None

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------
    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as e:
            raise RuntimeError(_IMPORT_ERROR) from e
        logger.info("Loading embedding model %s ...", self.model_name)
        self._model = BGEM3FlagModel(
            self.model_name,
            use_fp16=False,  # CPU inference in local/dev environments
            devices="cpu",
        )
        logger.info("Embedding model %s loaded.", self.model_name)
        return self._model

    def unload(self) -> None:
        """Release the model (frees CPU memory)."""
        self._model = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def encode(
        self,
        texts: list[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """Embed a list of texts, returning L2-normalized dense vectors.

        Vectors are normalized so cosine distance (pgvector `<=>`) and the
        dot-product scoring in the hybrid function behave consistently.
        """
        if not texts:
            return []
        model = self._load()
        clean = [t or "" for t in texts]
        out = model.encode(
            clean,
            batch_size=batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense = out["dense_vecs"]
        # L2-normalize in place (BGE-M3 does not always emit unit vectors)
        import numpy as np

        dense = np.asarray(dense, dtype=np.float32)
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        dense = dense / norms
        return dense.tolist()


_shared_model: EmbeddingModel | None = None


def get_embedding_model() -> EmbeddingModel:
    """Return the process-wide shared EmbeddingModel singleton."""
    global _shared_model
    if _shared_model is None:
        _shared_model = EmbeddingModel()
    return _shared_model
