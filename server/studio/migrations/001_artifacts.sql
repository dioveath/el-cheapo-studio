ALTER TABLE jobs ADD COLUMN IF NOT EXISTS operation TEXT NOT NULL DEFAULT 'mesh' CHECK(operation IN ('mesh','rig'));
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS artifacts JSONB NOT NULL DEFAULT '{}';
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema='public' AND table_name='jobs' AND column_name='result_path') THEN
        UPDATE jobs SET artifacts=jsonb_strip_nulls(jsonb_build_object(
            'glb', result_path,
            'obj', obj_path,
            'geometry', obj_path,
            'thumbnail', regexp_replace(result_path, '[^/]+$', 'preview.png')
        ));
        ALTER TABLE jobs DROP COLUMN result_path, DROP COLUMN obj_path;
    END IF;
END $$;
