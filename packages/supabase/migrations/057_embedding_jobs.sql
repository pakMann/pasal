-- Migration 057: Embedding pipeline — atomic job claiming for pasal embeddings
--
-- Mirrors the crawl_jobs pattern (AGENTS.md §4.2): a dedicated job table lets
-- multiple embedding workers claim disjoint batches atomically via
-- FOR UPDATE SKIP LOCKED, with self-healing of jobs stuck mid-flight, so a
-- 937K-row re-embed never has to restart from zero.
--
-- All mutation goes through these functions (service-role only). Anon /
-- authenticated get no write path. Embeddings are derived data written to
-- document_nodes with full audit (embedding_model + embedding_generated_at).

-- ============================================================
-- Step 1: embedding_jobs table
-- ============================================================

CREATE TABLE IF NOT EXISTS embedding_jobs (
    id BIGSERIAL PRIMARY KEY,
    node_id INTEGER NOT NULL UNIQUE REFERENCES document_nodes(id) ON DELETE CASCADE,
    work_id INTEGER,
    model TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'embedding', 'done', 'failed')),
    attempts INT NOT NULL DEFAULT 0,
    last_error TEXT,
    claimed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Partial index for stale-job recovery query (mirrors idx_crawl_status_updated)
CREATE INDEX IF NOT EXISTS idx_embedding_status_claimed
    ON embedding_jobs(status, claimed_at)
    WHERE status = 'embedding';

-- Tunable autovacuum, like crawl_jobs (032)
ALTER TABLE embedding_jobs SET (
    autovacuum_vacuum_threshold = 100,
    autovacuum_vacuum_scale_factor = 0.05,
    autovacuum_analyze_threshold = 50,
    autovacuum_analyze_scale_factor = 0.02
);


-- ============================================================
-- Step 2: queue_embedding_jobs() — enqueue unembedded pasal rows
-- ============================================================
-- Idempotent: ON CONFLICT DO NOTHING on node_id. Filters by regulation type
-- code (e.g. 'UU') so operators can seed a subset first.
CREATE OR REPLACE FUNCTION queue_embedding_jobs(
    p_limit INT DEFAULT 1000000,
    p_model TEXT DEFAULT 'BAAI/bge-m3',
    p_reg_type TEXT DEFAULT NULL
)
RETURNS INT
LANGUAGE plpgsql
AS $$
DECLARE
    v_inserted INT;
BEGIN
    INSERT INTO embedding_jobs (node_id, work_id, model, status)
    SELECT dn.id, dn.work_id, p_model, 'pending'
    FROM document_nodes dn
    JOIN works w ON w.id = dn.work_id
    JOIN regulation_types rt ON rt.id = w.regulation_type_id
    WHERE dn.node_type = 'pasal'
      AND dn.embedding IS NULL
      AND (p_reg_type IS NULL OR rt.code = p_reg_type)
    ORDER BY dn.id
    LIMIT p_limit
    ON CONFLICT (node_id) DO NOTHING;

    GET DIAGNOSTICS v_inserted = ROW_COUNT;
    RETURN v_inserted;
END;
$$;


-- ============================================================
-- Step 3: claim_embedding_jobs() — atomic claim + self-healing
-- ============================================================
CREATE OR REPLACE FUNCTION claim_embedding_jobs(p_limit INT DEFAULT 50)
RETURNS SETOF embedding_jobs
LANGUAGE plpgsql
AS $$
BEGIN
    -- Survive transient DB pressure without orphaning jobs
    SET LOCAL statement_timeout = '30s';

    -- Poison-pill: claimed 3+ times without completing → failed
    UPDATE embedding_jobs
    SET status = 'failed',
        last_error = 'reclaimed ' || attempts || ' times without completing',
        updated_at = NOW()
    WHERE status = 'embedding'
      AND claimed_at < NOW() - INTERVAL '15 minutes'
      AND attempts >= 3;

    -- Reclaim stale jobs that still have retries left
    UPDATE embedding_jobs
    SET status = 'pending',
        attempts = attempts + 1,
        claimed_at = NULL,
        updated_at = NOW()
    WHERE status = 'embedding'
      AND claimed_at < NOW() - INTERVAL '15 minutes'
      AND attempts < 3;

    -- Claim pending jobs atomically
    RETURN QUERY
    UPDATE embedding_jobs
    SET status = 'embedding',
        attempts = attempts + 1,
        claimed_at = NOW(),
        updated_at = NOW()
    WHERE id IN (
        SELECT id FROM embedding_jobs
        WHERE status = 'pending'
        ORDER BY id ASC
        LIMIT p_limit
        FOR UPDATE SKIP LOCKED
    )
    RETURNING *;
END;
$$;


-- ============================================================
-- Step 4: finish_embedding() — write embedding + mark done atomically
-- ============================================================
-- p_embedding is the vector as a text literal, e.g. '[0.1,0.2,...]'
-- (passed as text so PostgREST never has to serialize a `vector` param).
-- Casting to vector(1024) fails loudly on wrong dimensionality.
CREATE OR REPLACE FUNCTION finish_embedding(
    p_job_id BIGINT,
    p_embedding TEXT,
    p_model TEXT
)
RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    v_node_id INTEGER;
    v_work_id INTEGER;
BEGIN
    SELECT node_id, work_id INTO v_node_id, v_work_id
    FROM embedding_jobs WHERE id = p_job_id;

    IF v_node_id IS NULL THEN
        RETURN NULL;
    END IF;

    UPDATE document_nodes
    SET embedding = p_embedding::vector(1024),
        embedding_model = p_model,
        embedding_generated_at = NOW()
    WHERE id = v_node_id;

    UPDATE embedding_jobs
    SET status = 'done',
        last_error = NULL,
        claimed_at = NULL,
        updated_at = NOW()
    WHERE id = p_job_id;

    RETURN v_work_id;
END;
$$;


-- ============================================================
-- Step 5: fail_embedding() — record a failed batch item
-- ============================================================
CREATE OR REPLACE FUNCTION fail_embedding(p_job_id BIGINT, p_error TEXT)
RETURNS VOID
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE embedding_jobs
    SET status = 'failed',
        last_error = LEFT(COALESCE(p_error, 'unknown error'), 500),
        claimed_at = NULL,
        updated_at = NOW()
    WHERE id = p_job_id;
END;
$$;


-- ============================================================
-- Step 6: reset_embedding_jobs() — requeue a model's embeddings (re-embed)
-- ============================================================
-- Clears the derived embedding columns for a given model and deletes its jobs
-- so the next queue_embedding_jobs() run re-embeds from scratch.
CREATE OR REPLACE FUNCTION reset_embedding_jobs(p_model TEXT)
RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    v_cleared INT;
BEGIN
    DELETE FROM embedding_jobs WHERE model = p_model;
    UPDATE document_nodes
    SET embedding = NULL,
        embedding_model = NULL,
        embedding_generated_at = NULL
    WHERE embedding_model = p_model;
    GET DIAGNOSTICS v_cleared = ROW_COUNT;
    RETURN v_cleared;
END;
$$;


-- ============================================================
-- Step 7: Privileges — service_role only (follows migration 051)
-- ============================================================

REVOKE EXECUTE ON FUNCTION queue_embedding_jobs(INT, TEXT, TEXT) FROM PUBLIC, anon, authenticated;
REVOKE EXECUTE ON FUNCTION claim_embedding_jobs(INT) FROM PUBLIC, anon, authenticated;
REVOKE EXECUTE ON FUNCTION finish_embedding(BIGINT, TEXT, TEXT) FROM PUBLIC, anon, authenticated;
REVOKE EXECUTE ON FUNCTION fail_embedding(BIGINT, TEXT) FROM PUBLIC, anon, authenticated;
REVOKE EXECUTE ON FUNCTION reset_embedding_jobs(TEXT) FROM PUBLIC, anon, authenticated;

GRANT EXECUTE ON FUNCTION queue_embedding_jobs(INT, TEXT, TEXT) TO service_role;
GRANT EXECUTE ON FUNCTION claim_embedding_jobs(INT) TO service_role;
GRANT EXECUTE ON FUNCTION finish_embedding(BIGINT, TEXT, TEXT) TO service_role;
GRANT EXECUTE ON FUNCTION fail_embedding(BIGINT, TEXT) TO service_role;
GRANT EXECUTE ON FUNCTION reset_embedding_jobs(TEXT) TO service_role;

-- Pin search_path on all new functions (follows migration 049)
ALTER FUNCTION queue_embedding_jobs(INT, TEXT, TEXT) SET search_path = 'public', 'extensions';
ALTER FUNCTION claim_embedding_jobs(INT) SET search_path = 'public', 'extensions';
ALTER FUNCTION finish_embedding(BIGINT, TEXT, TEXT) SET search_path = 'public', 'extensions';
ALTER FUNCTION fail_embedding(BIGINT, TEXT) SET search_path = 'public', 'extensions';
ALTER FUNCTION reset_embedding_jobs(TEXT) SET search_path = 'public', 'extensions';
