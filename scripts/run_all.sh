#!/usr/bin/env bash
# Full experiment matrix. Run from the project root.
#   MODEL=local-llama LIMIT=50 bash scripts/run_all.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1
unset PYTHONPATH
PYTHON_BIN="${PYTHON_BIN:-/mnt/c/Users/Minh Nhat/AppData/Local/Programs/Python/Python311/python.exe}"
if ! "$PYTHON_BIN" -c 'import numpy' >/dev/null 2>&1; then
  echo "Python interpreter '$PYTHON_BIN' cannot import numpy; set PYTHON_BIN." >&2
  exit 1
fi
rv() { "$PYTHON_BIN" -m repurpose_verify "$@"; }
MODEL="${MODEL:-local-llama}"
LIMIT="${LIMIT:-}"                       # e.g. LIMIT=50 for a pilot
LIM=${LIMIT:+--limit $LIMIT}

# ---------------------------------------------------------------- data
rv download all                         # PrimeKG parts + MONDO + HGNC aliases
rv build-kg
rv build-gt
rv audit-mapping -n 200                 # sample for hand-checking entity resolution

# ---------------------------------------------------------------- baselines
rv baseline popularity $LIM
rv baseline proximity $LIM
rv baseline complex --epochs 5 $LIM
rv baseline transe --epochs 5 $LIM
# $RV import-rankings txgnn_scores.csv --run-name txgnn --knowledge-cutoff 2021-12-31

# ---------------------------------------------------------------- LLM systems
# For the reported runs over every evaluation disease, prefer scripts/run_full.sh: it runs the
# systems sequentially with retries, which a single local server needs.
rv agent --model "$MODEL" --mode closed_book $LIM
rv agent --model "$MODEL" --verifier none    $LIM
rv agent --model "$MODEL" --verifier filter  $LIM
rv agent --model "$MODEL" --verifier revise  $LIM        # full system

# ---------------------------------------------------------------- ablations
rv agent --model "$MODEL" --verifier revise --no-pubmed     $LIM
rv agent --model "$MODEL" --verifier revise --no-candidates $LIM
rv agent --model "$MODEL" --verifier filter --policy drop   $LIM

# ---------------------------------------------------------------- leakage probe
rv probe --model "$MODEL"

# ---------------------------------------------------------------- evaluation
REF="${MODEL}__rag_verify-revise"
rv evaluate --reference "$REF"
rv evaluate --probe "runs/probes/${MODEL}.jsonl" --out runs/report_masked --reference "$REF"
rv evaluate --min-tier phase3 --out runs/report_phase3 --reference "$REF"

# ---------------------------------------------------------------- expert review
rv review export --runs runs/"$MODEL"__* --out runs/review --n-items 50
echo "Done. Report: runs/report/report.md"
