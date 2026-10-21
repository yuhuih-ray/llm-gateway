#!/usr/bin/env bash
# Sourced by the entry points; paths and project identity are deliberately fixed.
LOADTEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$LOADTEST_DIR/../.." && pwd)"
STATE="$LOADTEST_DIR/.state"
umask 077
mkdir -p "$STATE"

lock_project() {
  if ! mkdir "$STATE/lock" 2>/dev/null; then
    echo "Another loadtest command owns .state/lock; do not run concurrently." >&2
    exit 1
  fi
}

load_environment() {
  if [[ ! -f "$STATE/env" ]]; then
    echo "Run scripts/loadtest/setup.sh first." >&2
    exit 1
  fi
  set -a
  source "$STATE/env"
  set +a
  # Avoid loading the repository's dev .env via Pydantic's relative env_file.
  cd "$STATE"
  PYTHON="$(uv run --project "$ROOT" python -c 'import sys; print(sys.executable)')"
  "$PYTHON" "$LOADTEST_DIR/manage.py" validate
}

compose() {
  docker compose --project-name llm-gateway-loadtest \
    --project-directory "$LOADTEST_DIR" --env-file "$STATE/env" \
    -f "$LOADTEST_DIR/compose.yaml" "$@"
}
