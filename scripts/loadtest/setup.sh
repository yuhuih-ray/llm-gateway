#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
lock_project
trap 'rmdir "$STATE/lock"' EXIT
command -v uv >/dev/null
command -v docker >/dev/null
uv run --project "$ROOT" python "$LOADTEST_DIR/manage.py" init
load_environment
compose up -d --wait --wait-timeout 90 >"$STATE/setup.log" 2>&1
"$PYTHON" -m alembic -c "$ROOT/alembic.ini" upgrade head >>"$STATE/setup.log" 2>&1
"$PYTHON" "$LOADTEST_DIR/manage.py" seed
printf 'Load-test services ready. Keys: %s/keys.json (private, gitignored).\n' "$STATE"
