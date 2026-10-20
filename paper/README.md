# Manuscript

| File | What it is |
|---|---|
| `draft.md` | the full draft. A status block at the top marks what is final and what is still a pilot |
| `check_numbers.py` | re-derives every headline number in the draft from the generated artefacts |

## Before any commit that touches the draft

```bash
python paper/check_numbers.py
```

It compares the draft against `data/processed/groundtruth/report.json`,
`runs/report_*/summary.csv` and `runs/probes/*.jsonl`, and exits non-zero on a mismatch or a
missing artefact. It caught a rounding error (0.927 for a verified rate of 126/136) on its first
run, which is the whole reason it exists: a paper whose finding is that agents fabricate
citations cannot carry numbers nobody re-derived.

Each claim is printed with the file it came from, so a mismatch tells you which side is stale —
fix the draft if the runs are newer, re-read the artefact if the draft is.

## Outstanding before submission

1. **Expert review is no longer on the critical path.** Section 6.4 replaces the two-rater panel
   with three checks that need no pharmacological judgement:
   * `validate moa` — ChEMBL mechanism-of-action confirmation of the drug→target edges, with
     drugs mapped through UniChem by structure. This answers the objection a panel would not:
     that the verifier accepts edges from the same graph the agent reads.
   * `reverify --config strict.toml` with `verifier.require_target_first_edge = true` — how much
     of the verified rate survives requiring a curated target rather than a metabolising enzyme.
     Costs no inference.
   * `validate literature` — post-cutoff PubMed co-mention of the top proposals against a
     random-drug control.

   The blinded sheet still exists at `runs/review/review_sheet.csv` (60 anonymised, shuffled
   proposals) if a rater becomes available. A **20-item spot check by the pharmacologist
   co-author** is worth having for face validity — ten minutes, one rater, reported as a spot
   check with no κ — but it is not a primary outcome. A non-expert can instead fill
   `data/processed/groundtruth/mapping_audit.csv` (from `audit-mapping`), which asks only whether
   a trial's drug and condition were matched to the right entities; that covers the mapping
   precision limitation, which is what reviewers press on.
2. **Curated approvals.** `data/manual/approvals_after_cutoff.csv` is empty; FDA/EMA
   new-indication approvals after the cutoff would add the `approved` tier, the strongest
   positive label.
3. **References.** Entries marked `[verify]` have not been opened against the published record.
   Given the subject of the paper, none may be cited from a secondary summary.
4. **A second model family.** One open-weight 7B model is reported. The two headline results —
   faithfulness and accuracy decoupling, and no LLM configuration beating popularity — are the
   ones a reviewer will most want replicated on a stronger model. A late-cutoff hosted model
   must be probed (`probe --model …`) before its scores are compared with these.
5. **Figures.** The draft is table-only. Candidates: a faithfulness-versus-accuracy scatter over
   the six systems (which is the paper's argument in one picture), and the risk–coverage curve
   implied by `coverage` against Hit@20.

## Where the numbers come from

| Draft section | Artefact |
|---|---|
| §7.1 benchmark composition | `data/processed/groundtruth/report.json` |
| §7.2 baselines | `runs/report_baselines/summary.csv` |
| §7.3 faithfulness and ranking | `runs/report_full/{summary,comparisons}.csv` |
| §7.3 phase-3 stratum | `runs/report_full_phase3/summary.csv` |
| §7.4 leakage probe | `runs/probes/qwen25-7b.jsonl`, `runs/probes/qwen25-7b.md` |
| §7.5 cost | `runs/report_full/summary.csv` |
| masked scoring | `runs/report_masked/`, `runs/report_full_masked/` |
