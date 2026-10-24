#!/usr/bin/env bash
# Re-run everything that depends on the hand-curated inputs, after they are filled.
#
#   bash scripts/refresh_after_curation.sh
#
# Adding approvals changes the ground truth, so every ranking number in the manuscript is
# recomputed from the stored runs. No model is called: the agents' proposals are already on
# disk, and only the scoring against the new positives changes.
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

APPROVALS=data/manual/approvals_after_cutoff.csv
AUDIT=data/manual/mapping_audit.csv

echo "== curated inputs"
"$PYTHON_BIN" - "$APPROVALS" "$AUDIT" <<'PY'
import sys, pandas as pd, pathlib
appr, audit = (pathlib.Path(p) for p in sys.argv[1:3])
n_appr = 0
if appr.exists() and appr.stat().st_size > 5:
    d = pd.read_csv(appr, encoding="utf-8-sig")
    n_appr = int(d["disease"].notna().sum()) if "disease" in d else 0
print(f"  approvals with a disease filled in: {n_appr}")
if audit.exists():
    d = pd.read_csv(audit, encoding="utf-8-sig")
    judged = sum(int(d[c].notna().sum()) for c in ("drug_correct_yn", "disease_correct_yn")
                 if c in d)
    print(f"  mapping-audit judgements: {judged}")
PY

echo "== rebuilding the ground truth"
rv build-gt || exit 1

echo "== re-verifying the stored proposals with the current verifier"
# A change to entity grounding changes which edges the verifier can support, and the edge
# statuses live in each run's rankings.jsonl. Re-scoring without this step would report new
# ranking metrics next to stale faithfulness metrics. No model is called: the proposals and
# their text are untouched, only the verification derived from them.
for MODEL in qwen25-7b llama31-8b; do
  [ -d "runs/${MODEL}__closed_book" ] || continue
  rv reverify "runs/${MODEL}__rag_verify-revise" "runs/${MODEL}__rag_verify-none" \
    "runs/${MODEL}__closed_book" --in-place || exit 1
done

echo "== rescoring every run against the new positives"
for MODEL in qwen25-7b llama31-8b; do
  RUNS="runs/${MODEL}__rag_verify-revise runs/${MODEL}__rag_verify-none runs/${MODEL}__closed_book"
  [ -d "runs/${MODEL}__closed_book" ] || { echo "  skipping $MODEL (no runs)"; continue; }
  REF="${MODEL}__rag_verify-revise"
  if [ "$MODEL" = qwen25-7b ]; then PREFIX=runs/report_full; else PREFIX=runs/report_${MODEL}; fi
  rv evaluate --runs $RUNS runs/popularity runs/proximity runs/complex \
    --reference "$REF" --out "$PREFIX"
  rv evaluate --runs $RUNS runs/popularity runs/proximity runs/complex \
    --reference "$REF" --probe "runs/probes/${MODEL}.jsonl" --out "${PREFIX}_masked"
  rv evaluate --runs $RUNS runs/popularity runs/proximity runs/complex \
    --reference "$REF" --min-tier phase3 --out "${PREFIX}_phase3"
done

rv evaluate --runs runs/popularity runs/proximity runs/complex --out runs/report_baselines
rv evaluate --runs runs/qwen25-7b__* runs/llama31-8b__* runs/popularity runs/proximity \
  runs/complex --out runs/report_two_models
rv evaluate --runs runs/popularity runs/proximity runs/complex \
  --probe runs/probes/qwen25-7b.jsonl --out runs/report_masked

echo "== strict re-verification (Section 7.6)"
rv --config configs/strict.toml reverify runs/qwen25-7b__rag_verify-revise \
  runs/qwen25-7b__rag_verify-none runs/qwen25-7b__closed_book --suffix _strict
rv evaluate --runs runs/qwen25-7b__rag_verify-revise runs/qwen25-7b__rag_verify-revise_strict \
  runs/qwen25-7b__rag_verify-none runs/qwen25-7b__rag_verify-none_strict \
  runs/qwen25-7b__closed_book_strict --out runs/report_strict

echo "== mapping-audit precision"
rv audit-mapping --score "$AUDIT" | tee runs/mapping_audit_score.json

echo "== figures"
"$PYTHON_BIN" paper/make_figures.py

echo "== checking the manuscript against the new artefacts"
"$PYTHON_BIN" paper/check_numbers.py
echo
echo "Every claim the checker now rejects is a number in paper/draft.md that the new ground"
echo "truth has changed. Update the text from the regenerated tables, then re-run the checker."
