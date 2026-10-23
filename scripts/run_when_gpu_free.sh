#!/usr/bin/env bash
# Wait for the GPU to be free, then run the full experiment for one model.
#
#   MODEL=llama31-8b bash scripts/run_when_gpu_free.sh
#
# The machine may be running other work: this waits for sustained free VRAM rather than
# competing for it, since two models that do not both fit will either fail to load or push each
# other into host memory and slow both jobs down.
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1
unset PYTHONPATH
PYTHON_BIN="${PYTHON_BIN:-/mnt/c/Users/Minh Nhat/AppData/Local/Programs/Python/Python311/python.exe}"
if ! "$PYTHON_BIN" -c 'import numpy' >/dev/null 2>&1; then
  echo "Python interpreter '$PYTHON_BIN' cannot import numpy; set PYTHON_BIN." >&2
  exit 1
fi
rv() { "$PYTHON_BIN" -m repurpose_verify "$@"; }
MODEL="${MODEL:-llama31-8b}"
NEED_MIB="${NEED_MIB:-6300}"       # free VRAM a 8B model at 16k context needs, with headroom
STABLE="${STABLE:-3}"              # consecutive checks that must pass, so a brief dip is ignored
INTERVAL="${INTERVAL:-300}"        # seconds between checks
MAX_WAIT_H="${MAX_WAIT_H:-24}"
LOG="${LOG:-runs/logs}"
mkdir -p "$LOG"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
status() { echo "[$(stamp)] $*" | tee -a "$LOG/gpu_wait.txt"; }

free_mib() {
  local used total
  # nvidia-smi on Windows emits CRLF; a stray carriage return breaks the arithmetic below.
  read -r used total < <(nvidia-smi --query-gpu=memory.used,memory.total \
    --format=csv,noheader,nounits | head -1 | tr -d ',\r')
  [ -n "${used:-}" ] && [ -n "${total:-}" ] || { echo 0; return; }
  echo $((total - used))
}

status "waiting for >= ${NEED_MIB} MiB free VRAM on $STABLE consecutive checks (every ${INTERVAL}s)"
deadline=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
ok=0
while [ "$(date +%s)" -lt "$deadline" ]; do
  f=$(free_mib)
  if [ "$f" -ge "$NEED_MIB" ]; then
    ok=$((ok + 1))
    status "free ${f} MiB  (${ok}/${STABLE})"
    [ "$ok" -ge "$STABLE" ] && break
  else
    [ "$ok" -gt 0 ] && status "free ${f} MiB - GPU busy again, resetting"
    ok=0
  fi
  sleep "$INTERVAL"
done

if [ "$ok" -lt "$STABLE" ]; then
  status "gave up after ${MAX_WAIT_H}h - GPU never freed"
  exit 1
fi

status "GPU free, starting $MODEL"

# The leakage probe comes first and is cheap: without it this model's scores cannot be compared
# with the others, and it would be wasteful to discover a loading problem 6 hours in.
PROBE_LOG="$LOG/${MODEL}_probe_$(date '+%Y%m%d_%H%M%S').log"
if ! rv probe --model "$MODEL" --controls 200 > "$PROBE_LOG" 2>&1; then
  status "ABORT - leakage probe failed; check $PROBE_LOG"
  exit 1
fi
status "probe done"

# Gate the long runs on the tool path actually working for this model. The probe exercises
# plain chat only, so a model whose tool-call format the harness mishandles would still pass it
# and then produce 200 empty rankings over six hours. Three diseases cost a few minutes.
SMOKE="${MODEL}__smoke"
rm -rf "runs/$SMOKE"
rv agent --model "$MODEL" --verifier revise --limit 3 --workers 1 --run-name "$SMOKE" \
  >> "$LOG/${MODEL}_smoke.log" 2>&1
PROPOSALS=$("$PYTHON_BIN" - "$SMOKE" <<'PY'
import json, pathlib, sys
path = pathlib.Path("runs") / sys.argv[1] / "rankings.jsonl"
rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] \
    if path.exists() else []
print(sum(len(r.get("proposals", [])) for r in rows))
PY
)
PROPOSALS=$(printf '%s' "$PROPOSALS" | tr -d '\r\n[:space:]')
status "smoke test: $PROPOSALS proposals over 3 diseases"
if ! [[ "${PROPOSALS:-}" =~ ^[0-9]+$ ]] || [ "$PROPOSALS" -lt 1 ]; then
  status "ABORT - the agent produced no proposals; check $LOG/${MODEL}_smoke.log before"
  status "       spending twelve hours on a model whose tool calls are not being parsed"
  exit 1
fi

MODEL="$MODEL" LOG="$LOG" bash scripts/run_full.sh >> "$LOG/${MODEL}_full.log" 2>&1
status "runs done"

rv validate all --runs runs/${MODEL}__rag_verify-revise runs/${MODEL}__rag_verify-none \
  runs/${MODEL}__closed_book --out runs/report_external_${MODEL}/report.md \
  >> "$LOG/${MODEL}_validate.log" 2>&1
status "external validation done"

rv evaluate --runs runs/${MODEL}__* runs/qwen25-7b__* runs/popularity runs/proximity \
  runs/complex --out runs/report_two_models >> "$LOG/${MODEL}_eval.log" 2>&1
status "DONE - runs/report_two_models/report.md"
