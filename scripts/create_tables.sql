CREATE TABLE IF Not EXISTS drug_master(
    drug_name VARCHAR(200) PRIMARY KEY,
    rxcui VARCHAR(50),
    normalized_name VARCHAR(200),
    verified         BOOLEAN DEFAULT FALSE,
    lookup_count     INTEGER DEFAULT 1,
    last_verified_at TIMESTAMP WITH TIME ZONE,
    created_at       TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_drug_master_last_verified
ON drug_master(last_verified_at);