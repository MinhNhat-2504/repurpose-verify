#!/usr/bin/env bash
# The reported LLM experiments: three systems over every evaluation disease, then the reports.
#
#   MODEL=qwen25-7b bash scripts/run_full.sh
#
# Systems run sequentially: a single local server does not serve requests in parallel, and
# concurrent turns against one Ollama instance were observed to stall (see README). Each system
# is attempted up to three times - runs are resumable, so a retry continues from the diseases
# already written rather than starting over. Expect roughly two minutes per disease per system
# on one 8 GB GPU.
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
MODEL="${MODEL:-qwen25-7b}"
LOG="${LOG:-runs/logs}"
mkdir -p "$LOG"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
status() { echo "[$(stamp)] $*" | tee -a "$LOG/status.txt"; }

run_system() {           # run_system <run-name> <extra args...>
  local name="$1"; shift
  local target done_count
  target=$(( $(wc -l < data/processed/groundtruth/eval_diseases.csv) - 1 ))
  for attempt in 1 2 3; do
    status "$name attempt $attempt"
    rv agent --model "$MODEL" --workers 1 --run-name "$name" "$@" >> "$LOG/$name.log" 2>&1
    done_count=$(wc -l < "runs/$name/rankings.jsonl" 2>/dev/null || echo 0)
    status "$name: $done_count/$target diseases after attempt $attempt"
    [ "$done_count" -ge "$target" ] && return 0
  done
  status "$name did not finish after 3 attempts - see $LOG/$name.log"
  return 1
}

status "START full runs for $MODEL"

# The paper's central contrast first: if the job is cut short, these two carry the
# fabricated-edge result.
run_system "${MODEL}__rag_verify-revise" --verifier revise
run_system "${MODEL}__rag_verify-none"   --verifier none
run_system "${MODEL}__closed_book"       --mode closed_book

status "runs finished, evaluating"

REF="${MODEL}__rag_verify-revise"
RUNS="runs/${MODEL}__rag_verify-revise runs/${MODEL}__rag_verify-none runs/${MODEL}__closed_book"
RUNS="$RUNS runs/popularity runs/proximity runs/complex"

# The manuscript's tables are read from runs/report_full, which belongs to the primary model.
# A second model running with the defaults would overwrite them, and the numbers in the draft
# would silently stop matching their artefacts, so reports are namespaced per model unless the
# model is the primary one.
if [ "$MODEL" = "${PRIMARY_MODEL:-qwen25-7b}" ]; then
  REPORT_PREFIX="${REPORT_PREFIX:-runs/report_full}"
  REVIEW_DIR="${REVIEW_DIR:-runs/review}"
else
  REPORT_PREFIX="${REPORT_PREFIX:-runs/report_${MODEL}}"
  REVIEW_DIR="${REVIEW_DIR:-runs/review_${MODEL}}"
fi
status "reports -> ${REPORT_PREFIX}*, review sheet -> ${REVIEW_DIR}"

rv evaluate --runs $RUNS --reference "$REF" --out "$REPORT_PREFIX"
rv evaluate --runs $RUNS --reference "$REF" --probe "runs/probes/${MODEL}.jsonl" \
  --out "${REPORT_PREFIX}_masked"
rv evaluate --runs $RUNS --reference "$REF" --min-tier phase3 --out "${REPORT_PREFIX}_phase3"
rv review export --runs runs/${MODEL}__* --out "$REVIEW_DIR" --n-items 60

status "DONE - ${REPORT_PREFIX}/report.md"
echo "Now update paper/draft.md from the new tables and re-run: python paper/check_numbers.py"
