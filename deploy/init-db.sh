#!/bin/sh
set -eu
# Executed only on initialization of a new PostgreSQL volume.
app_password="$(cat /run/secrets/db_app_password)"
test -n "$app_password"
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" --set=app_password="$app_password" <<'SQL'
CREATE ROLE polysearch_app LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD :'app_password';
ALTER DATABASE polysearch OWNER TO polysearch_app;
ALTER SCHEMA public OWNER TO polysearch_app;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
SQL
