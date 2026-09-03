-- backend/sql/init.sql
-- PostgreSQL 16 schema for SIH26146.
-- This file is run automatically by the postgres:16 Docker image on first boot.
-- It is idempotent: safe to run multiple times.

-- ── Custom enum types ────────────────────────────────────────────────────────

DO $$ BEGIN
    CREATE TYPE cluster_confidence AS ENUM ('HIGH', 'MEDIUM', 'LOW');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE risk_label AS ENUM ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE score_type AS ENUM ('raw_ranking');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;


-- ── Table 1: address_clusters ────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS address_clusters (
    id            BIGSERIAL       PRIMARY KEY,
    address_hash  VARCHAR(128)    NOT NULL,
    cluster_id    VARCHAR(64)     NOT NULL,
    confidence    cluster_confidence NOT NULL,
    contains_seed BOOLEAN         NOT NULL DEFAULT FALSE,
    snapshot_tag  VARCHAR(32)     NOT NULL,
    created_at    TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_clusters_address ON address_clusters (address_hash);
CREATE INDEX IF NOT EXISTS idx_clusters_cluster  ON address_clusters (cluster_id);


-- ── Table 2: node_scores ─────────────────────────────────────────────────────
-- risk_score is a 0–100 model-derived ranking from raw GCN sigmoid output.
-- It is NOT a calibrated probability. score_type='raw_ranking' makes this
-- explicit at the data layer.

CREATE TABLE IF NOT EXISTS node_scores (
    id            BIGSERIAL       PRIMARY KEY,
    address_hash  VARCHAR(128)    NOT NULL,
    risk_score    INTEGER         NOT NULL CHECK (risk_score BETWEEN 0 AND 100),
    risk_label    risk_label      NOT NULL,
    score_type    score_type      NOT NULL DEFAULT 'raw_ranking',
    pipeline      VARCHAR(64)     NOT NULL,
    snapshot_tag  VARCHAR(32)     NOT NULL,
    created_at    TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_scores_address ON node_scores (address_hash);
CREATE INDEX IF NOT EXISTS idx_scores_label   ON node_scores (risk_label);


-- ── Table 3: node_reasons ────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS node_reasons (
    id            BIGSERIAL       PRIMARY KEY,
    address_hash  VARCHAR(128)    NOT NULL,
    reason        TEXT            NOT NULL,
    rank          INTEGER         NOT NULL CHECK (rank BETWEEN 1 AND 3),
    layer         VARCHAR(16)     NOT NULL CHECK (layer IN ('deterministic', 'gnnexplainer')),
    snapshot_tag  VARCHAR(32)     NOT NULL,
    created_at    TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_reasons_address ON node_reasons (address_hash);


-- ── Table 4: audit_log ───────────────────────────────────────────────────────
-- Append-only. No UPDATE or DELETE should ever touch this table.
-- Tracks every API query that returns risk score data.

CREATE TABLE IF NOT EXISTS audit_log (
    id             BIGSERIAL       PRIMARY KEY,
    queried_hash   VARCHAR(128)    NOT NULL,
    endpoint       VARCHAR(128)    NOT NULL,
    investigator   VARCHAR(128),                -- NULL until Phase 2 MFA is added
    response_code  INTEGER         NOT NULL,
    queried_at     TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_hash ON audit_log (queried_hash);
CREATE INDEX IF NOT EXISTS idx_audit_time ON audit_log (queried_at);
