CREATE TABLE IF NOT EXISTS assets (
    id UUID PRIMARY KEY,
    filename TEXT NOT NULL,
    kind TEXT NOT NULL CHECK(kind IN ('image','mesh')),
    path TEXT NOT NULL,
    thumbnail TEXT,
    sha256 TEXT NOT NULL,
    size_bytes BIGINT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS reconstructions (
    key TEXT PRIMARY KEY,
    original_path TEXT NOT NULL,
    prepared_path TEXT,
    stats JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY,
    asset_id UUID NOT NULL REFERENCES assets(id),
    parent_id UUID REFERENCES jobs(id),
    batch_id UUID NOT NULL,
    settings JSONB NOT NULL,
    operation TEXT NOT NULL DEFAULT 'mesh' CHECK(operation IN ('mesh','rig','animation')),
    motion JSONB,
    status TEXT NOT NULL DEFAULT 'queued',
    phase TEXT NOT NULL DEFAULT 'Waiting for worker',
    progress INT NOT NULL DEFAULT 0,
    cancel_requested BOOLEAN NOT NULL DEFAULT false,
    dispatched BOOLEAN NOT NULL DEFAULT false,
    reconstruction_key TEXT,
    artifacts JSONB NOT NULL DEFAULT '{}',
    stats JSONB NOT NULL DEFAULT '{}',
    error TEXT,
    attempt INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS jobs_created ON jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS jobs_pending ON jobs(created_at) WHERE NOT dispatched AND status='queued' AND NOT cancel_requested;
CREATE INDEX IF NOT EXISTS jobs_asset ON jobs(asset_id);
CREATE INDEX IF NOT EXISTS jobs_parent ON jobs(parent_id) WHERE parent_id IS NOT NULL;
CREATE TABLE IF NOT EXISTS schema_migrations (
    name TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
