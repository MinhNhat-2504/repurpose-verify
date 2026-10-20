#!/usr/bin/env bash
# Full experiment matrix. Run from the project root.
#   MODEL=local-llama LIMIT=50 bash scripts/run_all.sh
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-src}"
RV="python -m repurpose_verify"
MODEL="${MODEL:-local-llama}"
LIMIT="${LIMIT:-}"                       # e.g. LIMIT=50 for a pilot
LIM=${LIMIT:+--limit $LIMIT}

# ---------------------------------------------------------------- data
$RV download all                         # PrimeKG parts + MONDO + HGNC aliases
$RV build-kg
$RV build-gt
$RV audit-mapping -n 200                 # sample for hand-checking entity resolution

# ---------------------------------------------------------------- baselines
$RV baseline popularity $LIM
$RV baseline proximity $LIM
$RV baseline complex --epochs 5 $LIM
$RV baseline transe --epochs 5 $LIM
# $RV import-rankings txgnn_scores.csv --run-name txgnn --knowledge-cutoff 2021-12-31

# ---------------------------------------------------------------- LLM systems
# For the reported runs over every evaluation disease, prefer scripts/run_full.sh: it runs the
# systems sequentially with retries, which a single local server needs.
$RV agent --model "$MODEL" --mode closed_book $LIM
$RV agent --model "$MODEL" --verifier none    $LIM
$RV agent --model "$MODEL" --verifier filter  $LIM
$RV agent --model "$MODEL" --verifier revise  $LIM        # full system

# ---------------------------------------------------------------- ablations
$RV agent --model "$MODEL" --verifier revise --no-pubmed     $LIM
$RV agent --model "$MODEL" --verifier revise --no-candidates $LIM
$RV agent --model "$MODEL" --verifier filter --policy drop   $LIM

# ---------------------------------------------------------------- leakage probe
$RV probe --model "$MODEL"

# ---------------------------------------------------------------- evaluation
REF="${MODEL}__rag_verify-revise"
$RV evaluate --reference "$REF"
$RV evaluate --probe "runs/probes/${MODEL}.jsonl" --out runs/report_masked --reference "$REF"
$RV evaluate --min-tier phase3 --out runs/report_phase3 --reference "$REF"

# ---------------------------------------------------------------- expert review
$RV review export --runs runs/"$MODEL"__* --out runs/review --n-items 50
echo "Done. Report: runs/report/report.md"
