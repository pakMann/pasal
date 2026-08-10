-- Migration 056: Hybrid search schema — pgvector extension + embedding columns
--
-- Adds semantic/vector search on top of the existing full-text search (AGENTS.md §4.1).
--   1. Enable pgvector (extension `vector`)
--   2. Add embedding columns to document_nodes (derived data, NOT legal content)
--   3. Partial HNSW index over pasal rows only (937K rows → hnsw for recall at scale)
--   4. RLS: public read continues to be covered by the existing "Public read nodes"
--      policy (migration 007). No write path for anon/authenticated is added.
--
-- Idempotent: safe to run on a clean environment.

-- ============================================================
-- Step 1: pgvector extension
-- ============================================================
-- Local environments may need the extension created manually (like `ltree`,
-- see AGENTS.md §8.3). CREATE EXTENSION IF NOT EXISTS is still safe here.
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;


-- ============================================================
-- Step 2: Embedding columns on document_nodes
-- ============================================================
-- BGE-M3 dense embeddings are 1024-dimensional (see PR description for the
-- model rationale). Only `pasal` rows are embedded (one enriched pasal = one
-- chunk); structural rows (bab, bagian, ...) keep NULL.
ALTER TABLE document_nodes
    ADD COLUMN IF NOT EXISTS embedding VECTOR(1024),
    ADD COLUMN IF NOT EXISTS embedding_model TEXT,
    ADD COLUMN IF NOT EXISTS embedding_generated_at TIMESTAMPTZ;


-- ============================================================
-- Step 3: HNSW index (partial — pasal rows with an embedding only)
-- ============================================================
-- Cosine distance matches the normalized dense vectors produced by BGE-M3.
-- m = 16, ef_construction = 64 (pgvector defaults, good quality/build-time
-- trade-off for ~937K rows).
CREATE INDEX IF NOT EXISTS idx_nodes_embedding_hnsw
    ON document_nodes USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE node_type = 'pasal' AND embedding IS NOT NULL;


-- ============================================================
-- Step 4: RLS — ensure public read, no public write on new columns
-- ============================================================
-- The new columns live on document_nodes, which already has a public read
-- policy from migration 007. We guard it here with a DO-block (pg_policies)
-- because `CREATE POLICY IF NOT EXISTS` is not valid Postgres.
-- `UPDATE ... SET embedding = ...` stays a service-role-only operation:
-- there is deliberately NO policy granting UPDATE/INSERT to anon/authenticated.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE schemaname = 'public'
          AND tablename = 'document_nodes'
          AND policyname = 'Public read nodes'
    ) THEN
        CREATE POLICY "Public read nodes" ON document_nodes
            FOR SELECT TO anon, authenticated USING (true);
    END IF;
END;
$$;
