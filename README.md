# repurpose-verify

**Can an LLM agent justify what it recommends? A mechanism-verified, time-split benchmark for drug repurposing.**

This repository implements the benchmark, the agent and the baselines for the study. It has two parts:

1. **Prospective-style evaluation in hindsight.** Everything the agent can read is frozen at a knowledge cutoff (default **2021-12-31**): a PrimeKG snapshot compiled in 2021, plus PubMed queried with `maxdate=2021/12/31`. The agent is then scored on drug–disease pairs that **first entered a phase 2+ trial (ClinicalTrials.gov) or got an approval after the cutoff** (2022–2026). A pair only counts if nothing at the cutoff already linked the drug to the disease.
2. **Mechanism verification.** Every proposal has to come with a path `drug → target → (gene | pathway → gene) → disease`. A separate verifier grounds each node in the KG and checks each edge against KG relations, or against a cited pre-cutoff PubMed abstract. Edges with no evidence count as fabricated, and the **fabricated-edge rate** is reported as a primary metric for every system.

```
          cutoff 2021-12-31                                   2022 ........ 2026
 ┌────────────────────────────────┐                 ┌───────────────────────────────┐
 │ PrimeKG (2021)   PubMed ≤cutoff │──tools──► agent │ CT.gov phase 2+ trials started │
 │ known uses + pre-cutoff trials  │        │        │ after cutoff (+ FDA/EMA list)  │
 └────────────────────────────────┘        ▼        └──────────────┬────────────────┘
                                   proposals + paths               │ novel positives
                                           │                       ▼
                               mechanism verifier ──► filtered ranking ──► Hit@K, MRR,
                               (kg / literature /                         fabricated-edge rate,
                                unsupported / ungrounded)                 cost per correct hit
```

## What the benchmark looks like (measured, cutoff 2021-12-31)

| | |
|---|---|
| PrimeKG snapshot | 129,375 nodes · 4,050,249 undirected edges · 7,957 drugs · 17,080 diseases |
| Trials read | 172,939 started on/before the cutoff · 34,595 phase 2+ started after it |
| Pairs known at the cutoff | 11,831 from KG indications/off-label use + 68,631 from earlier trials |
| Post-cutoff pairs found | 6,556 → **2,776 novel** (phase 2: 1,602 · phase 3: 626 · phase 4: 548) |
| Diseases carrying a novel positive | 1,149 → 725 after requiring ≥1 associated gene → **200 sampled** (565 positives) |
| Leakage check | 0 positives have a drug–disease edge in the KG |
| Entity mapping coverage | a drug mapped in 65% of trials, a disease in 77% (`report.json`; audit a sample with `audit-mapping`) |

**The verifiable ceiling.** Only **59.3%** of the 565 evaluation positives are *mechanism-reachable*: for the rest the graph admits no drug → gene … → disease path within four edges, usually because the drug has no recorded target (79 positives) or the disease has few associated genes. A system that only reports verified proposals therefore cannot exceed ~0.59 recall, and that number belongs next to the results rather than in a footnote. `positives.csv` carries `mechanism_reachable` and `n_targets` per pair so the ceiling can be recomputed per stratum.

Requiring ≥1 associated gene per evaluation disease (`min_disease_genes`) drops 424 of 1,149 diseases. Without it, 39% of the evaluation set would be diseases on which *no* system could ever produce a verifiable path — an artefact of PrimeKG's coverage, not a property of any model. Set it to 0 to keep them and report the strata separately.

### Baselines on this split

| System | MRR | Hit@5 | Hit@10 | Hit@20 |
|---|---|---|---|---|
| popularity | 0.029 [0.014, 0.046] | 0.055 | 0.075 | 0.085 |
| network proximity | 0.012 [0.005, 0.020] | 0.020 | 0.035 | 0.055 |
| ComplEx (5 epochs, dim 128) | 0.007 [0.004, 0.010] | 0.000 | 0.020 | 0.045 |

Ranking ~8,000 drugs for an average of 2.8 unseen positives per disease is hard, and these numbers say so. A benchmark where the trivial baseline already scores well would not be measuring anything.

### What the verifier finds (12-disease pilot, qwen2.5-7b, 16k context)

| System | Proposals | Verified | **Fabricated** | Unresolvable | Invalid citations |
|---|---|---|---|---|---|
| closed book (no retrieval) | 213 | 0.000 | **0.144** | 0.670 | – (cited nothing) |
| RAG, no verifier | 173 | 0.705 | **0.228** | 0.000 | **1.000** (20 of 20) |
| RAG + verify-revise | 136 | 0.926 | **0.031** | 0.000 | – (cited nothing) |

Two results, both measured rather than assumed:

**Verification cuts fabricated edges 7×, from 0.228 to 0.031.** Giving the agent graph tools is not what makes it faithful — unverified, it still asserts 23% of its mechanism edges without support. Checking each edge and handing the rejects back for one revision round is what moves the number.

**Every citation the unverified agent produced was fake.** It attached 20 PMIDs across its proposals and never once called `search_pubmed`; the ids were sequential placeholders — `PMID12345`, `PMID12346`, `PMID12347` … — so all 20 failed the check. This is exactly the failure mode the benchmark exists to catch, and it is invisible to any evaluation that reads citations without resolving them.

A note on what this pilot cannot show: all six systems, baselines included, score Hit@K = 0 on these 12 diseases, which are mostly rare genetic disorders. Over the full 200-disease set the same baselines reach Hit@20 = 0.085, so ranking differences need the full run — faithfulness is measurable at pilot scale, ranking is not. Neither agent configuration called `search_pubmed` at all; both leaned entirely on `find_mechanism_paths` (144 and 161 calls).

Reading the fabricated and unresolvable columns apart is the point:

* **Fabricated** edges are the model's own claims about real entities that nothing supports. The closed-book model asserts `COX-2 → ataxia telangiectasia` nine times, `serotonin → Sandhoff disease`, and — in the same run — both `Metformin activates AMPK` and `Metformin inhibits AMPK`.
* **Unresolvable** edges are the larger failure and a different one: two thirds of closed-book mechanism endpoints are prose, not entities — "inflammation", "oxidative stress", "excitotoxicity", "bone growth and development". The claims may be true; they cannot be checked against any structured source, which is what a reviewer is entitled to ask for.

An earlier version of this table reported 0.953 "fabricated" for closed book. That number was wrong, and the way it was wrong is worth recording: "COX-2" is not an invented entity, it is PTGS2 under the name every pharmacologist uses. Grounding now loads HGNC alias symbols (44,495 genes, joined on NCBI gene id — never by name, because a substring search for "COX-2" returns COX2, a mitochondrial subunit of a different protein) and a small curated pin file for the few aliases where HGNC itself disagrees with drug-mechanism usage (`data/manual/gene_aliases.csv`; HGNC lists COX-2 as an alias of *PTGER2*). Mentions that still match several nodes are reported as *ambiguous* rather than silently resolved, because a wrong grounding could turn an invented edge into a supported one.

A caveat on the high verified rate: with `find_mechanism_paths` available, the agent largely submits paths the tool handed it, so verification is close to automatic for the proposals it keeps. The informative quantities are the differences between configurations — 0.228 → 0.031 fabricated with verification on, 0.670 unresolvable without tools — not the absolute rate.

## Systems and ablations

| System | What it is | Command |
|---|---|---|
| `popularity` | rank drugs by number of known indications | `baseline popularity` |
| `proximity` | network proximity: drug targets ∩ / PPI-adjacent to disease genes | `baseline proximity` |
| `transe` / `distmult` / `complex` | KG embedding trained on the cutoff KG, scored on `indication` | `baseline complex` |
| TxGNN | external; import its scores as CSV | `import-rankings` |
| `closed_book` | LLM, no retrieval | `agent --mode closed_book` |
| `rag_verify-none` | LLM + KG/PubMed tools, no verifier | `agent --verifier none` |
| `rag_verify-filter` | + verifier demotes (or `--policy drop` removes) unverified proposals | `agent --verifier filter` |
| `rag_verify-revise` (**full**) | + rejected proposals go back to the agent once for repair | `agent --verifier revise` |
| ablations | `--no-pubmed`, `--no-candidates`, `--no-kg`, `--policy drop`, `verifier.accept_literature=false` | |

All systems are verified afterwards, whatever their mode, so fabricated-edge rates can be compared across them.

## Metrics

* **Ranking** (filtered: drugs known at the cutoff are removed before scoring): Hit@K, Recall@K, MRR. 95% CIs come from a disease-level bootstrap (1,000 resamples), and pairs of systems are compared with a paired bootstrap.
* **Faithfulness**: verified-proposal rate; **fabricated**-edge rate (unsupported + circular: real entities, no admissible evidence) kept apart from **unresolvable** (ungrounded + ambiguous: not checkable in the graph's vocabulary); invalid-citation rate (the PMID does not exist or was published after the cutoff); malformed-path rate. A `kg:` id quoted next to another node's name counts as a fabricated entity, not as a lookup — models do cite plausible ids for drugs they invented, and that was caught in a live run during development.
* Saved runs can be re-scored with an improved verifier without paying for the LLM again: `reverify runs/<name> --in-place`.
* **Cost**: input/output tokens, USD, tokens and USD per correct hit.
* **Blinded expert review**: an anonymised sheet of top proposals for two pharmacists, scored with quadratic-weighted Cohen's κ (`review export|score`).
* **Tier sensitivity**: `--min-tier phase3` or `approved`.

## Leakage controls

| Channel | Control |
|---|---|
| KG contains the answer | Positives must have no indication/off-label edge in the KG. Held-out pairs are also masked from the KG before any run, and `report.json` shows the leakage check (`leakage_check_positives_in_kg` must be 0). |
| Pre-cutoff evidence of the pair | Any trial of the pair that started on or before the cutoff, in any phase, makes it "known". |
| Literature after the cutoff | PubMed search uses `datetype=pdat&maxdate=cutoff`. Every fetched record is re-checked, and citations to later papers are counted as invalid. |
| LLM weights | Use open-weight models with an early documented cutoff (`knowledge_cutoff` is printed next to every result). The **leakage probe** asks the model closed-book whether it knows each positive, with positive and negative controls. `evaluate --probe` re-scores on the pairs the model does *not* know. |
| External verifier evidence | Open Targets (live release, includes post-cutoff evidence) is **off** by default. When it is enabled, its edges are reported separately. |

These controls reduce leakage but do not remove it. Discuss the remaining limits in the paper (see *Limitations*).

## Quick start

```bash
pip install -e ".[all]"                 # or: export PYTHONPATH=src
python -m repurpose_verify download all # PrimeKG nodes.tab + edges.csv (395 MB) + MONDO + HGNC
                                        # add --kg-csv to use the 982 MB monolithic kg.csv instead
python -m repurpose_verify build-kg
python -m repurpose_verify build-gt     # ClinicalTrials.gov API v2, cached under data/cache
python -m repurpose_verify baseline popularity
python -m repurpose_verify baseline proximity
python -m repurpose_verify baseline complex --epochs 5
python -m repurpose_verify agent --model local-llama --mode closed_book
python -m repurpose_verify agent --model local-llama --verifier none
python -m repurpose_verify agent --model local-llama --verifier revise
python -m repurpose_verify probe --model local-llama
python -m repurpose_verify evaluate --reference local-llama__rag_verify-revise
python -m repurpose_verify evaluate --probe runs/probes/local-llama.jsonl --out runs/report_masked
```

Run the tests with `pytest`. They use a synthetic 11-node graph and a mock LLM, with no network access.

### Models

Model profiles live in `configs/default.toml` (`[models.<name>]`). Override them with `--config my.toml`.

* `backend = "openai_compat"`: vLLM, Ollama (`base_url = "http://localhost:11434/v1"`), llama.cpp or TGI. Set `tool_mode = "react"` for models without native function calling. The agent then uses a `<tool_call>{json}</tool_call>` text protocol.
* **Ollama truncates prompts silently.** It serves its own context default (4k) no matter what the model supports, so an agent turn carrying tool results is cut without an error and the run looks like a model failure. Raise it per model before measuring anything:
  ```bash
  printf 'FROM qwen2.5:7b\nPARAMETER num_ctx 16384\n' > Modelfile
  ollama create qwen2.5-7b-ctx16k -f Modelfile
  ```
  Ollama also serialises requests unless `OLLAMA_NUM_PARALLEL` is set, so `--workers` above 1 buys little against a single instance — and four concurrent turns were observed to end with four established sockets and no response ever arriving, leaving the run parked in the client timeout. `workers` therefore defaults to 1, and the client timeout to 240 s with retries, so a stalled server costs one request rather than the run.
* `backend = "anthropic"`: Claude through the official SDK (`pip install anthropic`, `ANTHROPIC_API_KEY`). Its knowledge cutoff is later than the benchmark cutoff, so report it as a leakage-prone upper bound together with its probe results. Server-side refusal fallbacks are **not** enabled: they would silently swap in a different model in the middle of a run.
* Set `price_in_per_mtok` / `price_out_per_mtok` in each profile to get cost per correct hit.

### External rankings (TxGNN etc.)

Write a CSV with columns `disease,drug,score`. Diseases and drugs may be given as KG ids (`kg:123`), MONDO / DrugBank ids, or names. Then import it:

```bash
python -m repurpose_verify import-rankings txgnn_scores.csv --run-name txgnn --knowledge-cutoff 2021-12-31
```

For TxGNN, train with `split='full_graph'` on the same PrimeKG, then export the disease-centric scores for the diseases in `data/processed/groundtruth/eval_diseases.csv`. Because held-out pairs are not in PrimeKG, this is a fair comparison.

## Outputs

| Path | Content |
|---|---|
| `data/processed/kg/` | `nodes.csv.gz`, `edges.npz`, `vocab.json` |
| `data/manual/gene_aliases.csv` | curated mention → node pins, with the reason for each |
| `data/processed/groundtruth/positives.csv` | novel post-cutoff pairs: tier, first start date, NCT ids |
| `data/processed/groundtruth/known_pairs.csv.gz` | pairs known at the cutoff (KG uses + pre-cutoff trials) |
| `data/processed/groundtruth/eval_diseases.csv` | the sampled evaluation diseases |
| `data/processed/groundtruth/report.json` | mapping coverage, counts by tier, leakage check |
| `runs/<system>/rankings.jsonl` | per disease: ranked drugs, raw proposals with full verification, token usage, tool trace |
| `runs/report/report.md` | the tables for the paper (+ `summary.csv`, `per_disease.csv`, `comparisons.csv`) |
| `data/cache/` | every HTTP response (CT.gov, PubMed, Open Targets), for exact re-runs and archiving |

## Limitations to report

* PrimeKG's sources have different release dates around 2021, so the snapshot is "about 2021", not exactly the cutoff date.
* A trial start is evidence of clinical interest, not of efficacy. Report tier-stratified results (`--min-tier phase3`, `approved`).
* Entity mapping from CT.gov (MeSH ids → MONDO xrefs, then drug names) is imperfect: a drug resolves in 65% of trials and a disease in 77%. Coverage is in `report.json`; audit a random sample with `audit-mapping`.
* Grounding is only as complete as the vocabulary. PrimeKG + MONDO + HGNC still miss informal receptor names ("AT1 receptor") and hold no node for many processes an author would name in prose, which is why *unresolvable* is its own category instead of being folded into fabrication.
* Only 59.3% of the evaluation positives are mechanism-reachable, so a verified-only system is capped well below 1.0 recall by the graph, not by the model.
* Literature evidence uses co-mention, which is necessary but not sufficient. Report results with `accept_literature = false` (KG-only verification) as a stricter variant.
* The leakage probe measures what the model *says* it knows. A model can still be helped by knowledge it does not verbalise.

## Layout

```
src/repurpose_verify/
  data/       kg.py (PrimeKG store, grounding, paths), ctgov.py, groundtruth.py, pubmed.py,
              mondo.py, opentargets.py, normalize.py
  verify/     schema.py (Proposal/PathEdge), verifier.py (MechanismVerifier)
  agent/      llm.py (OpenAI-compatible / Anthropic / mock), tools.py, prompts.py, agent.py
  baselines/  heuristics.py, kge.py, external.py
  eval/       metrics.py, evaluate.py, leakage_probe.py, review.py
  cli.py, context.py, config.py
configs/default.toml
tests/
```

## License

MIT
