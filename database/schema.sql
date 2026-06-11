-- PostgreSQL schema for the Arabic customer support agent.

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS orders (
    order_id VARCHAR(64) PRIMARY KEY,
    customer_id VARCHAR(64) NOT NULL,
    status VARCHAR(32) NOT NULL,
    items JSONB NOT NULL DEFAULT '[]'::jsonb,
    total_amount DOUBLE PRECISION NOT NULL DEFAULT 0,
    currency VARCHAR(8) NOT NULL DEFAULT 'SAR',
    tracking_number VARCHAR(64),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    expected_delivery_date TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_orders_customer_id ON orders (customer_id);

CREATE TABLE IF NOT EXISTS triage_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    customer_id VARCHAR(64) NOT NULL,
    message TEXT NOT NULL,

    intent VARCHAR(64) NOT NULL,
    intent_confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
    urgency VARCHAR(16) NOT NULL,
    sentiment VARCHAR(16) NOT NULL,
    dialect VARCHAR(32) NOT NULL,

    entities JSONB NOT NULL DEFAULT '{}'::jsonb,
    requires_human BOOLEAN NOT NULL DEFAULT FALSE,
    routed_team VARCHAR(64) NOT NULL,
    draft_response_ar TEXT NOT NULL DEFAULT '',

    reasoning_trace JSONB NOT NULL DEFAULT '[]'::jsonb,
    tools_used TEXT[] NOT NULL DEFAULT '{}',

    latency_ms INTEGER NOT NULL DEFAULT 0,
    est_cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0,

    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_triage_runs_customer_id ON triage_runs (customer_id);
CREATE INDEX IF NOT EXISTS idx_triage_runs_created_at ON triage_runs (created_at);
CREATE INDEX IF NOT EXISTS idx_triage_runs_intent ON triage_runs (intent);
