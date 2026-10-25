# Supplementary material

Can an LLM agent justify what it recommends? A mechanism-verified, time-split benchmark
for drug repurposing.

Nhat Trinh Ngoc Minh, Linh Nguyen Thi My

These tables support results stated in the article. Each is cited at the point in the main
text where its numbers are discussed.

| System | MRR | Hit@1 | Hit@5 | Hit@10 | Hit@20 |
|---|---|---|---|---|---|
| Popularity | 0.030 [0.015, 0.048] | 0.010 | 0.057 | 0.077 | 0.088 |
| Network proximity | 0.012 [0.005, 0.021] | 0.000 | 0.021 | 0.036 | 0.057 |
| ComplEx | 0.007 [0.003, 0.011] | 0.000 | 0.000 | 0.015 | 0.041 |

**Table S1.** Baselines over the 194 evaluation diseases and 546 positives; 95% CI from a
disease-level bootstrap. Full table with Recall@K in `runs/report_baselines/`.

| Model | Item | n | Said yes | Flagged as post-cutoff knowledge |
|---|---|---|---|---|
| qwen2.5-7b | Ground-truth positives | 565 | 0.117 | **0.021** |
| qwen2.5-7b | Known at the cutoff (positive control) | 200 | 0.310 | 0.050 |
| qwen2.5-7b | Random pairs (negative control) | 200 | 0.025 | 0.000 |
| llama3.1-8b | Ground-truth positives | 565 | 0.432 | **0.007** |
| llama3.1-8b | Known at the cutoff (positive control) | 200 | 0.565 | 0.000 |
| llama3.1-8b | Random pairs (negative control) | 201 | 0.134 | 0.005 |

**Table S2.** Closed-book probe, 965 items per model.

| Model | System | Input tokens | Output tokens | Correct hits | Tokens per correct hit |
|---|---|---|---|---|---|
| qwen2.5-7b | closed book | 83,398 | 487,960 | 13 | **43,951** |
| qwen2.5-7b | RAG, no verifier | 11,373,366 | 567,307 | 8 | 1,492,584 |
| qwen2.5-7b | RAG + verify-revise | 12,316,771 | 677,320 | 9 | 1,443,788 |
| llama3.1-8b | closed book | 80,716 | 597,643 | 11 | 61,669 |
| llama3.1-8b | RAG, no verifier | 1,844,408 | 508,827 | 4 | 588,309 |
| llama3.1-8b | RAG + verify-revise | 3,391,410 | 635,875 | 4 | 1,006,821 |
| — | Popularity | 0 | 0 | 17 | 0 |

**Table S3.** Cost over the 194 evaluation diseases. Hits are positives recovered in a top-20 list.

| Model | System | Pairs | Co-mention | Random control | Degree-matched | Lift vs matched |
|---|---|---|---|---|---|---|
| qwen2.5-7b | closed book | 955 | 0.308 | 0.034 | 0.063 | **+0.245** |
| qwen2.5-7b | RAG, no verifier | 893 | 0.078 | 0.021 | 0.038 | **+0.040** |
| qwen2.5-7b | RAG + verify-revise | 888 | 0.083 | 0.036 | 0.046 | **+0.037** |
| llama3.1-8b | closed book | 957 | 0.278 | 0.034 | 0.065 | **+0.213** |
| llama3.1-8b | RAG, no verifier | 394 | 0.089 | 0.040 | 0.030 | **+0.058** |
| llama3.1-8b | RAG + verify-revise | 399 | 0.090 | 0.041 | 0.045 | **+0.045** |

**Table S4.** Post-cutoff co-mention. The matched control pairs each proposed drug with another of
comparable knowledge-graph degree for the same disease. The two models agree closely, which is
some reassurance that this measure reflects the proposals rather than the model.

| File | Contents |
|---|---|
| `data/manual/eval_split.csv` | the frozen 200-disease evaluation split (Section 7.1) |
| `data/manual/approvals_after_cutoff.csv` | 114 author-curated FDA new-indication approvals, each with the approval letter's own clause as its note (Section 7.8) |
| `data/manual/approvals_candidates.csv` | the worksheet those were judged from: every candidate approval, its letter quote, the proposed ontology term and the link to the FDA record |
| `data/manual/mapping_audit.csv` | the 200-trial entity-mapping audit with its per-row judgements (Section 3.2) |
| `scripts/measure_grounding_correction.py` | regenerates every positive added or removed by the grounding correction, as a CSV and the counts quoted in Section 7.7 |

**Table S5.** Curated inputs released with the manuscript. Each holds a human judgement or a
frozen choice that the pipeline cannot re-derive.

| Resource | Version / access date |
|---|---|
| PrimeKG | Harvard Dataverse doi:10.7910/DVN/IXA7BM, `nodes.tab` + `edges.csv` |
| Mondo Disease Ontology | `mondo.obo`, retrieved 2026-10-20 |
| HGNC | `hgnc_complete_set.txt`, retrieved 2026-10-21 |
| ClinicalTrials.gov | API v2, queried 2026-10-20 |
| PubMed | E-utilities, queried 2026-10-20 to 2026-10-24 |
| ChEMBL | REST API, queried 2026-10-21 and 2026-10-24 |
| UniChem | API v1, same dates |
| Models | `qwen2.5:7b` and `llama3.1:8b` served by Ollama 0.35.0, each rebuilt with `num_ctx` 16384 |
| Hardware | one NVIDIA RTX 4060 Laptop GPU (8 GB) |

**Table S6.** Source versions. Each external resource is named by the release or
access date that produced the reported results.

