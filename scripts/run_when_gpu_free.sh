#!/usr/bin/env bash
# Wait for the GPU to be free, then run the full experiment for one model.
#
#   MODEL=llama31-8b bash scripts/run_when_gpu_free.sh
#
# The machine may be running other work: this waits for sustained free VRAM rather than
# competing for it, since two models that do not both fit will either fail to load or push each
# other into host memory and slow both jobs down.
set -u

export PYTHONPATH="${PYTHONPATH:-src}"
RV="python -m repurpose_verify"
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
$RV probe --model "$MODEL" --controls 200 >> "$LOG/${MODEL}_probe.log" 2>&1
status "probe done"

MODEL="$MODEL" LOG="$LOG" bash scripts/run_full.sh >> "$LOG/${MODEL}_full.log" 2>&1
status "runs done"

$RV validate all --runs runs/${MODEL}__rag_verify-revise runs/${MODEL}__rag_verify-none \
  runs/${MODEL}__closed_book --out runs/report_external_${MODEL}/report.md \
  >> "$LOG/${MODEL}_validate.log" 2>&1
status "external validation done"

$RV evaluate --runs runs/${MODEL}__* runs/qwen25-7b__* runs/popularity runs/proximity \
  runs/complex --out runs/report_two_models >> "$LOG/${MODEL}_eval.log" 2>&1
status "DONE - runs/report_two_models/report.md"
