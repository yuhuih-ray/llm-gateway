#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
lock_project
trap 'rmdir "$STATE/lock"' EXIT
if [[ ! -f "$STATE/env" ]]; then
  echo "No load-test environment has been configured."
  exit 0
fi
compose down --volumes --remove-orphans >"$STATE/teardown.log" 2>&1
# Keys are invalid once their isolated database is gone.
rm -f "$STATE/keys.json" "$STATE/env" "$STATE/baseline.json"
echo "Removed the load-test Compose project and its volumes."
