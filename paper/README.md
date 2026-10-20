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

1. **Full 200-disease agent runs.** Sections and sentences depending on pilot scale are marked
   `[PILOT]`. Regenerate Table 3 and Section 7.5, then re-run `check_numbers.py`.
2. **Blinded expert review** (`[PENDING-REVIEW]`): `review export`, two pharmacists,
   `review score`.
3. **Curated approvals.** `data/manual/approvals_after_cutoff.csv` is empty; FDA/EMA new-indication
   approvals after the cutoff would add the `approved` tier, the strongest positive label.
4. **References.** Entries marked `[verify]` have not been opened against the published record.
   Given the subject of the paper, none may be cited from a secondary summary.
5. **A second model family.** One open-weight 7B model is reported; a second, and ideally a
   late-cutoff hosted model with its probe results, would separate model effects from the effect
   of verification.
