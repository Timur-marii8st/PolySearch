#!/bin/sh
set -eu
umask 077
out="${1:-backups/polysearch-$(date -u +%Y%m%dT%H%M%SZ).dump}"
mkdir -p "$(dirname "$out")"
test ! -e "$out" || { echo "Refusing to overwrite backup" >&2; exit 1; }
tmp="$out.part"
trap 'rm -f "$tmp"' EXIT HUP INT TERM
docker compose exec -T db pg_dump -U postgres -d polysearch -Fc --no-owner --no-acl > "$tmp"
docker compose exec -T db pg_restore --list < "$tmp" >/dev/null
mv "$tmp" "$out"
sha256sum "$out" > "$out.sha256"
printf 'Backup created: %s\n' "$out"
