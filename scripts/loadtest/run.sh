#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
SCENARIO="${1:-}"
SMOKE=0
case "$SCENARIO" in
  s1) SCRIPT=s1_overhead.js; DELAY=0; TIMING=false ;;
  s2) SCRIPT=s2_breakdown.js; DELAY=0; TIMING=true ;;
  s3) SCRIPT=s3_inflight.js; DELAY=2000; TIMING=false ;;
  *) echo "Usage: $0 <s1|s2|s3> [--smoke (s1 only)]" >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then
  if [[ $# -ne 2 || "$2" != --smoke || "$SCENARIO" != s1 ]]; then
    echo "Only s1 accepts --smoke (20 RPS for 10 seconds)." >&2
    exit 2
  fi
  SMOKE=1
fi
command -v k6 >/dev/null
lock_project
GATEWAY_PID=""
WORKER_PID=""
K6_PID=""
cleanup() {
  # Only terminate children started by this invocation, never matching dev processes.
  for pid in "$K6_PID" "$GATEWAY_PID" "$WORKER_PID"; do
    if [[ -n "$pid" ]]; then kill -TERM "$pid" 2>/dev/null || true; fi
  done
  for pid in "$K6_PID" "$GATEWAY_PID" "$WORKER_PID"; do
    if [[ -n "$pid" ]]; then wait "$pid" 2>/dev/null || true; fi
  done
  rmdir "$STATE/lock"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
load_environment
export FAKE_DELAY_MS="$DELAY" SERVER_TIMING="$TIMING" LOADTEST_SMOKE="$SMOKE"
"$PYTHON" "$LOADTEST_DIR/manage.py" port
"$PYTHON" "$LOADTEST_DIR/manage.py" reset
if [[ "$SMOKE" == 1 ]]; then
  OUTPUT="$STATE/smoke"
else
  OUTPUT="$ROOT/docs/benchmarks/gateway-load"
fi
mkdir -p "$OUTPUT"
SUMMARY="$OUTPUT/$SCENARIO.json"
META="$OUTPUT/$SCENARIO-meta.json"
rm -f "$SUMMARY" "$META"
"$PYTHON" -m uvicorn llm_gateway.main:app --host 127.0.0.1 --port 18000 \
  --workers 1 --log-level warning --no-access-log --no-proxy-headers \
  >"$STATE/$SCENARIO-gateway.log" 2>&1 &
GATEWAY_PID=$!
"$PYTHON" -m arq llm_gateway.worker.WorkerSettings >"$STATE/$SCENARIO-worker.log" 2>&1 &
WORKER_PID=$!
"$PYTHON" "$LOADTEST_DIR/manage.py" ready
kill -0 "$GATEWAY_PID" "$WORKER_PID"
k6 run --quiet --summary-export "$SUMMARY" "$LOADTEST_DIR/$SCRIPT" >"$STATE/$SCENARIO-k6.log" 2>&1 &
K6_PID=$!
STATUS=0
wait "$K6_PID" || STATUS=$?
K6_PID=""
META_ARGS=()
if [[ "$SMOKE" == 1 ]]; then META_ARGS+=(--smoke); fi
if [[ -f "$SUMMARY" ]]; then
  "$PYTHON" "$LOADTEST_DIR/manage.py" finish --scenario "$SCENARIO" \
    --summary "$SUMMARY" --output "$META" --exit-code "$STATUS" ${META_ARGS[@]+"${META_ARGS[@]}"}
else
  echo "k6 produced no summary; inspect .state/$SCENARIO-k6.log." >&2
  exit 1
fi
printf 'Summary: %s\nMetadata: %s\n' "$SUMMARY" "$META"
exit "$STATUS"
