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

## Target venue: GigaScience

Chosen after reading the actual author instructions rather than from reputation. The deciding
facts:

* **Briefings in Bioinformatics was ruled out.** It lists no original-research article type; the
  closest routes (problem-solving protocol, case study) cap at 5,000 words and reviews at 7,000,
  against a 9,800-word manuscript whose contribution is the measurement design. Cutting by half
  would remove the argument and leave the negative results undefended.
* **GigaScience accepts Research articles with no stated word limit**, publishes benchmarks and
  negative results, and requires exactly what this project already has: data deposited with a
  persistent identifier, code in a repository under an OSI licence (MIT here), and software cited
  with versions.

Journal of Cheminformatics was dropped on scope — its remit is cheminformatics and molecular
modelling, and this work is knowledge graphs, trial records and agent evaluation. Journal of
Biomedical Informatics remains a reasonable fallback; its guidelines could not be retrieved
(Elsevier returns 403 to automated requests) and should be read before it is considered.

GigaScience's quartile should be confirmed on SCImago before submission; aggregators report Q1
but disagree on the impact factor (3.9 vs 4.33), and SCImago blocks automated access.

## Outstanding before submission

1. **Deposit in GigaDB** and paste the DOI into Section 10, replacing
   `[to be assigned on deposit]`: evaluation splits, cached source responses, per-run proposals
   with verification records, and the generated reports.
2. **Complete the declarations** in Section 11: funding, acknowledgements, and confirm the author
   contribution statement.
3. **Curated approvals.** `python -m repurpose_verify fda-approvals` downloads the Drugs@FDA
   bulk release and writes `data/manual/approvals_candidates.csv`: every submission tagged
   *Efficacy-New Indication*, approved in the evaluation window, whose drug matches the graph.
   265 such approvals exist for this window. Drugs@FDA records that an indication was added but
   never which one, so the `disease` column is filled by hand from the FDA record linked in each
   row; rows that turn out not to be a new indication for an existing drug are deleted. Copy the
   first five columns into `approvals_after_cutoff.csv` and re-run `build-gt` to add the
   `approved` tier.
4. **Mapping audit.** `data/processed/groundtruth/mapping_audit.csv` (from `audit-mapping`) holds
   200 sampled trials with a link to each record. Judging a row means checking whether the drug
   and condition were matched to the right entities - identity, not pharmacology - so it needs no
   clinical training. Fill `drug_correct_yn` / `disease_correct_yn`, then
   `audit-mapping --score <file>` reports precision with a bootstrap interval, counting only rows
   where a mapping was produced: an unmapped trial is a recall miss already reported as coverage,
   and scoring it twice would overstate the problem.
5. **Reference style.** The list is currently in a readable long form. Convert to the journal's
   style at submission; every entry has a DOI or arXiv identifier already.

No expert rating is planned. Section 6.4 sets out the three checks used in its place.

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
| strict target sensitivity | `configs/strict.toml`, `runs/report_strict/` |

## Figures

```bash
python paper/make_figures.py        # writes paper/figures/*.png and *.pdf
```

Every value plotted comes from an artefact `check_numbers.py` also verifies, so a figure cannot
drift from the text. Figure 1 is the paper's argument in one picture: faithfulness on one axis,
accuracy on the other, and no system good on both.
