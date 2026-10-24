# GigaDB submission metadata

Fields the GigaDB wizard asks for. Paste them in when the database team makes contact; nothing
here needs to be settled before the manuscript is submitted, because the dataset DOI is assigned
on acceptance. Build the files first with:

    python scripts/make_gigadb_deposit.py

That writes `dist/gigadb/` — six archives, a `MANIFEST.tsv` with sizes and SHA-256 sums, and a
`README.md` restating the sections below.

---

## Dataset title

Supporting data for: a mechanism-verified, time-split benchmark for drug repurposing with large
language model agents

## Dataset type

Experiment / Evaluation data. Not a sequencing dataset, so no community-mandated database applies.

## Authors

| Order | Name | ORCID | Affiliation |
|---|---|---|---|
| 1 | Nhat Trinh Ngoc Minh | 0009-0006-9477-9642 | Faculty of Information Technology, Van Lang School of Technology, Van Lang University, 69/68 Dang Thuy Tram Street, Binh Loi Trung Ward, Ho Chi Minh City, Vietnam |
| 2 | Linh Nguyen Thi My | 0000-0003-3990-312X | as above |

**Submitting and corresponding author.** Linh Nguyen Thi My, linh.ntm@vlu.edu.vn

## Abstract

Evaluations of large language model agents for drug repurposing rarely establish two things a
reader needs: that the agent could not have seen the answer, and that the mechanism it offers is
real. This dataset supports a benchmark built to separate those questions. Every source an agent
may query is frozen at 2021-12-31 — a knowledge-graph snapshot of that period and PubMed
restricted by publication date — and agents are scored on drug–disease pairs whose first phase 2
or later trial, or whose regulatory approval of a new indication, falls after that cutoff.

The deposit contains the time-split ground truth (2,731 drug–disease pairs that were unknown at
the cutoff by three independent criteria, with tier, first date and trial identifiers, alongside
the pairs excluded as already known); a frozen 200-disease evaluation split; and the complete
output of twelve systems over that split — two open-weight models (`qwen2.5-7b`, `llama3.1-8b`) in
three agent configurations each, three knowledge-graph baselines, and the leakage probes. Each of
the 14,065 agent proposals carries the mechanism path it asserted and the per-edge verification
record for that path, so the faithfulness measurements can be recomputed from the stored evidence
without re-running a model.

Four curated files carry human judgement that the pipeline cannot re-derive: 114 FDA
new-indication approvals judged against the approval letters, the worksheet those were judged
from, a 200-trial entity-mapping audit with per-row judgements, and the alias overrides for cases
where the source ontologies disagree with pharmacology usage. The cached ClinicalTrials.gov and
Drugs@FDA responses are included because both registries are revised in place, so a record
refetched later is not the record that produced these labels.

## Keywords

drug repurposing; large language model agents; benchmark; data leakage; knowledge graph;
mechanism verification; hallucination; clinical trials; time split

## Related manuscript

Trinh NNM, Nguyen LTM. Can an LLM agent justify what it recommends? A mechanism-verified,
time-split benchmark for drug repurposing. Submitted to *GigaScience*.

## Related code

https://github.com/MinhNhat-2504/repurpose-verify — MIT licence. The repository rebuilds every
derived file in this deposit from the sources named below.

## File descriptions

| Archive | Contents |
|---|---|
| `curated_inputs.tar.gz` | The frozen evaluation split; 114 judged FDA new-indication approvals, each with the approval letter's operative clause as its note; the candidate worksheet those were judged from, with the proposed ontology term and a link to each FDA record; the 200-trial entity-mapping audit with per-row judgements; the curated alias overrides. |
| `ground_truth.tar.gz` | `positives.csv` (novel pairs with tier, date, trial identifiers, target counts and mechanism-reachability), `known_pairs.csv.gz` (pairs excluded as known at the cutoff, with the reason), the evaluation-disease table, the build report, and the grounding-correction difference. |
| `model_runs.tar.gz` | Per-disease records for all twelve systems: each proposal, its asserted mechanism path, the per-edge verification status, the ranking, and token accounting; the two leakage probes with their positive and negative controls; and every generated report. |
| `figures.tar.gz` | Manuscript figures, PNG and PDF at 600 dpi. |
| `cache_clinicaltrials.tar.gz` | Cached ClinicalTrials.gov API v2 responses, queried 2026-10-20. |
| `cache_fda_letters.tar.gz` | Cached Drugs@FDA approval letters, PDF with extracted text, retrieved 2026-10-24. |

## Licence of the deposited files

CC0, as GigaDB requires. Every deposited file is either our own output or a work of the United
States government, so the waiver is ours to give.

## Third-party sources NOT redistributed here

CC0 cannot be applied to data we did not create, so these are identified by version and access
date rather than included. `python -m repurpose_verify download all` refetches them into the
layout the code expects.

| Source | Version / access | Terms |
|---|---|---|
| PrimeKG | Harvard Dataverse doi:10.7910/DVN/IXA7BM, `nodes.tab` + `edges.csv` | per the Dataverse record |
| Mondo Disease Ontology | `mondo.obo`, retrieved 2026-10-20 | CC BY 4.0 |
| HGNC | `hgnc_complete_set.txt`, retrieved 2026-10-21 | per genenames.org |
| ChEMBL | REST API, 2026-10-21 and 2026-10-24 | CC BY-SA 3.0 |
| PubMed | E-utilities, 2026-10-20 to 2026-10-24 | abstracts carry publisher copyright |

## Ethics

No human subjects, no human or animal samples, and no personal data. Every record is a public
registry entry, a published abstract, a regulatory document or a public database release, so no
ethics approval or consent form applies.
