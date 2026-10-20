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
* **Faithfulness**: verified-proposal rate; fabricated-edge rate, split into unsupported / ungrounded / circular; invalid-citation rate (the PMID does not exist or was published after the cutoff); malformed-path rate. A `kg:` id quoted next to another node's name counts as a fabricated entity, not as a lookup — models do cite plausible ids for drugs they invented, and that was caught in a live run during development.
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
python -m repurpose_verify download all # PrimeKG nodes.tab + edges.csv (395 MB) + MONDO
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
* Entity mapping from CT.gov (MeSH ids → MONDO xrefs, then drug names) is imperfect. Coverage is reported in `report.json`, and a manual audit of a random sample is recommended.
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
