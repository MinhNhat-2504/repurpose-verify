# Curated inputs

`approvals_after_cutoff.csv` — FDA/EMA approvals of a **new indication for an existing drug**
dated after the knowledge cutoff (default 2021-12-31). These are merged into the ground truth
with tier `approved`, the strongest positive label.

| column  | meaning |
|---------|---------|
| drug    | generic drug name or DrugBank id (e.g. `DB00331`) |
| disease | disease name, MONDO id (`MONDO:0005575`) or MeSH id (`MESH:D015179`) |
| date    | approval date, `YYYY-MM-DD` |
| source  | where it was checked: e.g. `FDA Drugs@FDA efficacy supplement`, `EMA EPAR variation` |
| note    | free text (application number, label section) |

Fill it from primary sources only (Drugs@FDA supplements with submission class "Efficacy -
New Indication", EMA EPAR "extension of indication" variations). Rows that cannot be mapped to
the KG are listed under `unmapped_manual_approvals` in `data/processed/groundtruth/report.json`.
