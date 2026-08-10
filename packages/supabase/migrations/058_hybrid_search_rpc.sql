-- Migration 058: Hybrid search — vector leg + Reciprocal Rank Fusion RPC
--
-- Adds two functions on top of the existing 3-layer FTS (AGENTS.md §4.3):
--   1. search_vector_chunks() — pure pgvector cosine search over embedded
--      pasal rows. Embeds only exist on node_type='pasal', so the HNSW
--      partial index (056) is used directly.
--   2. search_hybrid() — fuse the full 3-layer FTS (search_legal_chunks) with
--      the vector leg using Reciprocal Rank Fusion (RRF). k=60 is the
--      standard value from Cormack et al.
--
-- Mode parameter (backward compatible by default):
--   'fts_only'    → exact delegate to search_legal_chunks (old behaviour)
--   'vector_only' → vector leg only
--   'hybrid'      → RRF fusion of both (default). If query_embedding is NULL
--                   (e.g. embedding model unavailable), degrades to fts_only.
--
-- query_embedding is passed as a text literal like '[0.1,0.2,...]' (same
-- convention as finish_embedding in 057) so the embedding model stays in the
-- application layer, never in Postgres.
--
-- Return shape matches search_legal_chunks so MCP consumers keep their
-- existing result-handling code.

-- ============================================================
-- Step 1: search_vector_chunks() — vector leg
-- ============================================================

DROP FUNCTION IF EXISTS search_vector_chunks(TEXT, INT, JSONB);

CREATE FUNCTION search_vector_chunks(
    query_embedding TEXT,
    match_count INT DEFAULT 10,
    metadata_filter JSONB DEFAULT '{}'::jsonb
)
RETURNS TABLE (
    id BIGINT,
    work_id INTEGER,
    content TEXT,
    metadata JSONB,
    score FLOAT,
    snippet TEXT
)
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    v_qvec VECTOR(1024);
    v_type_filters TEXT[];
    v_status_filters TEXT[];
    v_year_filter INT := CASE WHEN metadata_filter ? 'year'
        THEN (metadata_filter ->> 'year')::int ELSE NULL END;
    v_year_from INT := CASE WHEN metadata_filter ? 'year_from'
        THEN (metadata_filter ->> 'year_from')::int ELSE NULL END;
BEGIN
    v_type_filters := CASE
        WHEN metadata_filter ->> 'type' IS NOT NULL AND metadata_filter ->> 'type' != ''
        THEN string_to_array(metadata_filter ->> 'type', ',')
        ELSE NULL
    END;

    v_status_filters := CASE
        WHEN metadata_filter ->> 'status' IS NOT NULL AND metadata_filter ->> 'status' != ''
        THEN string_to_array(metadata_filter ->> 'status', ',')
        ELSE NULL
    END;

    -- Casting to vector(1024) fails loudly on malformed/garbage input, same
    -- philosophy as the `[^a-zA-Z0-9 ]` stripping on FTS query_text.
    v_qvec := query_embedding::vector(1024);

    RETURN QUERY
    SELECT
        dn.id::bigint,
        dn.work_id,
        dn.content_text,
        jsonb_build_object(
            'type', rt.code,
            'number', w.number,
            'year', w.year::text,
            'pasal', dn.number
        ),
        (1.0 / (1.0 + (dn.embedding <=> v_qvec)))::float AS cosine_sim,
        LEFT(dn.content_text, 200)
    FROM document_nodes dn
    JOIN works w ON w.id = dn.work_id
    JOIN regulation_types rt ON rt.id = w.regulation_type_id
    WHERE dn.node_type = 'pasal'
      AND dn.embedding IS NOT NULL
      AND (v_type_filters IS NULL OR rt.code = ANY(v_type_filters))
      AND (v_year_filter IS NULL OR w.year = v_year_filter)
      AND (v_year_from IS NULL OR w.year >= v_year_from)
      AND (v_status_filters IS NULL OR w.status = ANY(v_status_filters))
    ORDER BY dn.embedding <=> v_qvec
    LIMIT match_count;
END;
$$;


-- ============================================================
-- Step 2: search_hybrid() — RRF fusion RPC
-- ============================================================

DROP FUNCTION IF EXISTS search_hybrid(TEXT, INT, JSONB, TEXT, TEXT);

CREATE FUNCTION search_hybrid(
    query_text TEXT,
    match_count INT DEFAULT 10,
    metadata_filter JSONB DEFAULT '{}'::jsonb,
    mode TEXT DEFAULT 'hybrid',
    query_embedding TEXT DEFAULT NULL
)
RETURNS TABLE (
    id BIGINT,
    work_id INTEGER,
    content TEXT,
    metadata JSONB,
    score FLOAT,
    snippet TEXT
)
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    v_candidates INT;
BEGIN
    IF mode NOT IN ('fts_only', 'vector_only', 'hybrid') THEN
        mode := 'hybrid';
    END IF;

    IF mode = 'fts_only' OR (mode = 'hybrid' AND query_embedding IS NULL) THEN
        -- Exact legacy behaviour (identity fast-path + works FTS + content FTS)
        -- and graceful degradation when no embedding is available.
        RETURN QUERY
        SELECT * FROM search_legal_chunks(query_text, match_count, metadata_filter);
        RETURN;
    END IF;

    IF mode = 'vector_only' THEN
        IF query_embedding IS NULL THEN
            RETURN;
        END IF;
        RETURN QUERY
        SELECT * FROM search_vector_chunks(query_embedding, match_count, metadata_filter);
        RETURN;
    END IF;

    -- ================================================================
    -- hybrid: Reciprocal Rank Fusion over the two legs
    -- k = 60 (Cormack et al.). Each leg returns match_count*2 candidates
    -- so the fused top-k is not starved by a single leg's ordering.
    -- ================================================================
    v_candidates := GREATEST(match_count, 10) * 2;

    RETURN QUERY
    WITH fts_leg AS (
        SELECT r.id AS rid, r.work_id AS rwork_id, r.content AS rcontent,
               r.metadata AS rmeta,
               row_number() OVER (ORDER BY r.score DESC) AS rk
        FROM search_legal_chunks(query_text, v_candidates, metadata_filter) AS r
    ),
    vec_leg AS (
        SELECT r.id AS rid, r.work_id AS rwork_id, r.content AS rcontent,
               r.metadata AS rmeta,
               row_number() OVER (ORDER BY r.score DESC) AS rk
        FROM search_vector_chunks(query_embedding, v_candidates, metadata_filter) AS r
    ),
    fused AS (
        SELECT rid, rwork_id, rcontent, rmeta,
               SUM(1.0 / (60.0 + rk)) AS fused_score
        FROM (
            SELECT rid, rwork_id, rcontent, rmeta, rk FROM fts_leg
            UNION ALL
            SELECT rid, rwork_id, rcontent, rmeta, rk FROM vec_leg
        ) AS both_legs
        GROUP BY rid, rwork_id, rcontent, rmeta
    )
    SELECT
        f.rid,
        f.rwork_id,
        f.rcontent,
        f.rmeta,
        f.fused_score::float,
        LEFT(f.rcontent, 200)
    FROM fused f
    ORDER BY f.fused_score DESC
    LIMIT match_count;
END;
$$;

-- ============================================================
-- Step 3: search_path hardening (follows 049/057) + privileges
-- ============================================================

ALTER FUNCTION search_vector_chunks(text, int, jsonb) SET search_path = 'public', 'extensions';
ALTER FUNCTION search_hybrid(text, int, jsonb, text, text) SET search_path = 'public', 'extensions';

-- Public read: search functions are readable by anon/authenticated (same as
-- search_legal_chunks). No REVOKE is applied — defaults stay PUBLIC.
GRANT EXECUTE ON FUNCTION search_vector_chunks(text, int, jsonb) TO anon, authenticated;
GRANT EXECUTE ON FUNCTION search_hybrid(text, int, jsonb, text, text) TO anon, authenticated;
