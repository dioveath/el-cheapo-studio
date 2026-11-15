ALTER TABLE jobs ALTER COLUMN asset_id DROP NOT NULL;
ALTER TABLE jobs DROP CONSTRAINT jobs_operation_check;
ALTER TABLE jobs ADD CONSTRAINT jobs_operation_check CHECK(operation IN ('mesh','rig','animation','video'));
ALTER TABLE jobs ADD CONSTRAINT jobs_source_check CHECK(asset_id IS NOT NULL OR operation='video');
