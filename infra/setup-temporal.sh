#!/bin/sh
set -eu
export PGPASSWORD="$POSTGRES_PWD"
export SQL_PASSWORD="$POSTGRES_PWD"
for db in temporal temporal_visibility; do
  initialized=$(psql -h "$POSTGRES_SEEDS" -U "$POSTGRES_USER" -d "$db" -tAc "SELECT to_regclass('public.schema_version') IS NOT NULL")
  if [ "$initialized" != "t" ]; then
    temporal-sql-tool --plugin postgres12 --ep "$POSTGRES_SEEDS" -u "$POSTGRES_USER" -p 5432 --db "$db" setup-schema -v 0.0
  fi
  schema=temporal
  if [ "$db" = "temporal_visibility" ]; then schema=visibility; fi
  temporal-sql-tool --plugin postgres12 --ep "$POSTGRES_SEEDS" -u "$POSTGRES_USER" -p 5432 --db "$db" update-schema -d "/etc/temporal/schema/postgresql/v12/$schema/versioned"
done
