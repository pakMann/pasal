-- Migration 000: Ensure ltree extension exists before document_nodes uses it.
-- Idempotent: no-op if already installed (locally created manually, see AGENTS.md).
CREATE EXTENSION IF NOT EXISTS ltree WITH SCHEMA public;
