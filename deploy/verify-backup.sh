#!/bin/sh
set -eu
dump="${1:?Specify a backup .dump file}"
test -f "$dump"
if [ -f "$dump.sha256" ]; then sha256sum -c "$dump.sha256"; fi
# Restore only into a newly created disposable database, never production.
name="polysearch_restore_$(date +%s)_$$"
docker compose exec -T db createdb -U postgres "$name"
trap 'docker compose exec -T db dropdb -U postgres "$name"' EXIT
docker compose exec -T db pg_restore -U postgres --exit-on-error --no-owner --no-acl -d "$name" < "$dump"
docker compose exec -T db psql -U postgres -d "$name" -v ON_ERROR_STOP=1 -c "SELECT count(*) AS workspace_rows FROM state; SELECT count(*) AS campaigns FROM pilot_campaigns; SELECT count(*) AS inbox_items FROM integration_inbox; SELECT count(*) AS research_runs FROM research_runs;"
echo "Restore validation succeeded in disposable database: $name"
