ALTER TABLE jobs DROP CONSTRAINT jobs_operation_check;
ALTER TABLE jobs ADD CONSTRAINT jobs_operation_check CHECK(operation IN ('mesh','rig','animation'));
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS motion JSONB;
UPDATE jobs SET settings=jsonb_set(settings, '{rig}', 'false') WHERE NOT settings ? 'rig';
