# Can an LLM agent justify what it recommends? A mechanism-verified, time-split benchmark for drug repurposing

**Authors.** [Author 1]¹, [Pharmacy/pharmacology co-author]², [Senior author]¹
¹ [Affiliation], ² [Affiliation]
Corresponding author: [e-mail]

---

> **Draft status (2026-10-21).** Every number in this draft is measured at full scale — 200
> evaluation diseases, three LLM systems, three knowledge-graph baselines — and reproducible
> from the released code. `paper/check_numbers.py` re-derives each one from the stored
> artefacts and fails if the text disagrees.
>
> Every result is in. Section 6.4 replaces the conventional blinded rating panel with three
> checks that need no expert rater — independent ChEMBL confirmation of the drug–target claims,
> a stricter reading of which graph relations count as a mechanism, and post-cutoff literature
> corroboration against matched controls. A 20-item spot check by a pharmacologist co-author
> remains desirable for face validity and is marked `[PENDING-REVIEW]`; it is not a primary
> outcome and no conclusion rests on it.
>
> An earlier version of this draft reported a 12-disease pilot in which verification cut
> fabricated edges 7-fold. At full scale the reduction is 2.6-fold (Section 7.3). The pilot
> overestimated the effect by roughly three times, and a claim about citation behaviour drawn
> from it did not survive either. Both corrections are recorded in Section 7.3 rather than
> quietly replaced, because the discrepancy is itself a result about pilot-scale faithfulness
> estimates.
>
> Every reference has been resolved against an authoritative record; the audit and the one
> citation it corrected are documented at the end of the reference list.

---

## Abstract

**Background.** Large language model (LLM) agents are being applied to drug repurposing, but
evaluations of them rarely establish two things a reader needs: that the agent could not have
seen the answer, and that the mechanism it offers is real. Scores are commonly reported on
retrospective associations the agent's sources already contain, and mechanistic narratives are
accepted as written or graded by another LLM.

**Results.** We release a benchmark that separates these two questions. Every source an agent
may query is frozen at 2021-12-31: a knowledge graph snapshot of that period and PubMed
restricted by publication date. Agents are then scored on drug–disease pairs whose first phase 2
or later trial began after the cutoff. From 172,939 pre-cutoff and 34,595 post-cutoff trials we
obtain 2,776 pairs that were unknown at the cutoff by three independent criteria; 565 of them
fall in a 200-disease evaluation set, and none has a drug–disease edge in the graph. Each
proposal must carry a mechanism path, and a verifier checks every edge against graph relations
or a cited pre-cutoff abstract. We report *fabricated* edges (both endpoints real, nothing links
them) apart from *unresolvable* ones (the endpoint names no single entity), because merging them
overstates fabrication threefold.

Across 200 diseases and 14,483 proposals from two open-weight models of the same size class,
grounding and verification act strongly on faithfulness, in opposite directions depending on the
model, and not at all on accuracy. With `qwen2.5-7b`, a closed-book agent fabricates 21.1% of its
mechanism edges and leaves 69.5% unresolvable; graph tools take fabrication to 4.6% and
edge-level verification with one revision round to 1.8%. With `llama3.1-8b` the same tools take
fabrication the other way, from 0.289 closed book to 0.944, because that model reuses identifiers
it has seen instead of querying for a path — naming other *drugs* as targets in 688 of its edges.
Grounding did not make either model truthful; it moved their errors from unresolvable prose into
checkable claims, which one model then got right and the other wrong. Ranking is untouched by
verification in both (ΔMRR 0.0004, *p* = 0.70 and 0.60), the closed-book configuration ranks at
least as well as the grounded ones in both, and **no LLM configuration of either model matches a
popularity baseline** (MRR 0.029 against 0.006–0.020; *p* = 0.014 and 0.002). A weight-level
leakage probe with positive and negative controls finds no post-cutoff signal above its own
false-positive floor for either model.

**Conclusions.** Faithfulness and accuracy are decoupled in current repurposing agents: in both
models the least verifiable configuration ranks best, and neither beats ranking drugs by how many
indications they already have. Verification removes three fifths of one model's fabricated claims
for 9% more input tokens and no measurable loss in ranking, but it buys checkability, not
discovery — and what grounding does to a model's fabrication rate is a property of the model, so
single-model evidence about it is unsafe. Reporting either quantity alone, or either model alone,
would have supported a conclusion the rest of the data contradicts. We also quantify a limit of
mechanism-verified evaluation itself: only 59.3% of our positives admit any mechanism path in the
graph, which caps what a verified-only system can recover regardless of the model.

**Availability.** Code, splits, cached evidence and reports: https://github.com/MinhNhat-2504/repurpose-verify

**Keywords.** drug repurposing; large language model agents; benchmark; data leakage;
knowledge graphs; hallucination; evaluation

---

## 1 Introduction

Computational drug repurposing is an attractive target for LLM agents. The task rewards wide
reading, the evidence is public, and the output — a short ranked list with an argument for each
entry — is the kind of thing language models produce fluently. Recent reviews of agentic AI in
drug discovery identify evaluation as the field's weak point rather than capability [1].

Two specific weaknesses motivate this work.

**Retrospective evaluation does not establish prospective skill.** A system is usually scored on
held-out associations drawn from the same curated sources it reads. Those associations are
historical facts: the indication is in DrugBank, the gene–disease link is in DisGeNET, the trial
is in the literature. High accuracy under such a split is consistent with retrieval rather than
inference. Audits of adjacent benchmarks have found exactly this kind of contamination across
widely used splits, and have shown that which model leads a leaderboard can change once the test
set's composition is corrected [2]; evaluated inductively rather than transductively, drug–target
models generalise considerably worse than their reported figures suggest [3].

**A fluent mechanism is not a verified one.** When an agent explains that drug D hits target T
which drives disease S, the explanation is usually either taken at face value or scored by
another language model. Neither establishes that the edges exist. The failure is not
hypothetical: across 200 diseases an agent with a literature-search tool available called it
once, and every one of the twenty PubMed identifiers it volunteered was invented
(Section 7.3).

We address both by construction.

1. **A time split with the evidence base frozen.** The agent sees a 2021 knowledge graph and
   PubMed filtered by publication date; it is scored on what entered the clinic afterwards
   (Section 3).
2. **A mechanism verifier.** Every proposal must present a path from drug to disease through at
   least one gene, and every edge is checked against graph relations or a cited pre-cutoff
   abstract (Section 4).

Our contributions are:

* A leakage-audited, time-split ground truth of 2,776 novel drug–disease pairs, with the audit
  reported rather than asserted (Section 7.1).
* A mechanism verifier whose evidence taxonomy distinguishes a false claim from an uncheckable
  one, and the demonstration that conflating them inflates measured fabrication threefold
  (Sections 4, 7.3).
* A measurement of what verification buys and what it does not: fabricated edges fall 2.6-fold
  when rejected proposals are returned for one revision round, while ranking does not move
  (*p* = 0.70) and no configuration reaches a popularity baseline (Section 7.3).
* A *verifiable ceiling*: only 59.3% of our positives admit any mechanism path, a bound on
  mechanism-verified evaluation that is a property of the knowledge base, not of the model
  (Section 7.1).
* A weight-level leakage probe with positive and negative controls, so that "no leakage found"
  is distinguishable from "the probe cannot detect leakage" (Section 7.4).
* An open implementation in which every external response is cached, so a reported run can be
  re-scored, or re-run offline, by a reviewer (Section 8).

## 2 Related work

**Knowledge-graph repurposing.** Hetionet established the template of ranking drugs for a
disease by paths in a heterogeneous biomedical graph [4]. PrimeKG consolidates twenty primary
resources into a single graph aimed at precision medicine [5], and TxGNN trains a graph neural
network on it for zero-shot indication prediction, with a clinician-facing interface [6].
Network proximity methods score a drug by the graph distance between its targets and the
disease's genes [7]. General-purpose link predictors — TransE [8], DistMult [9], ComplEx [10] —
remain standard baselines. These systems rank; they do not produce or defend a mechanism, and
they are typically evaluated on associations already present in the graph.

**LLM agents in biomedical discovery.** Agentic systems now perform literature synthesis,
property prediction and candidate generation. Reviews of the area single out reliability and the
absence of agreed benchmarks as the limiting factors rather than raw capability [1].
Where mechanisms are reported, they are rarely machine-checked.

**Leakage and benchmark auditing.** Transductive drug–target interaction models have been shown to report inflated performance
under the evaluation setups used in the literature, and to generalise poorly to the inductive
setting that repurposing actually requires [3]. An audit of over fifty benchmark dataset
configurations found pervasive cross-split contamination, label conflicts and structural
redundancy, with drug–target interaction datasets singled out for extensive protein overlap
between splits [2]. Our ground truth is built to be auditable in the same
spirit, and we report the audit alongside the data.

**Position of this work.** We do not propose a new predictor. We propose an evaluation under
which a predictor's *justification* is checkable, and we measure existing systems under it.

## 3 Benchmark construction

### 3.1 Knowledge snapshot

We use PrimeKG [5] as the pre-cutoff knowledge base: 129,375 nodes and 4,050,249 undirected
edges, including 7,957 drugs, 27,671 genes or proteins, 17,080 diseases and 2,516 pathways. Its
constituent resources were compiled around 2021, which matches our cutoff. We read PrimeKG's
`nodes.tab` and `edges.csv` distributions (395 MB) rather than the monolithic `kg.csv` (982 MB),
which repeats every node name on both sides of each edge, and verified that both yield an
identical graph (test `test_build_from_parts_matches_kg_csv`).

Disease grounding uses MONDO exact synonyms and MeSH cross-references [11]; gene grounding adds
HGNC alias symbols for 44,495 genes [12], joined on NCBI gene identifiers. Section 4.2 explains
why that join must be on identifiers and never on names.

### 3.2 Time-split ground truth

We query the ClinicalTrials.gov API v2 for interventional studies with a drug or biological
intervention, partitioned by start date: 172,939 studies beginning on or before 2021-12-31 (any
phase) and 34,595 beginning afterwards with a maximum phase of 2, 3 or 4.

Interventions are mapped to graph drugs after stripping doses, salts and formulations and
splitting combination strings; generic arms (placebo, standard of care, saline) are discarded.
Conditions are mapped through the trial record's MeSH identifiers to MONDO cross-references,
falling back to name matching. A drug resolves in 64.9% of trials and a disease in 77.3%; the
unmatched remainder is dominated by novel compounds absent from DrugBank and by conditions with
no MONDO equivalent. The `audit-mapping` command samples trials with their mappings for manual
checking, which we recommend reporting for any derived split.

A post-cutoff pair becomes a **positive** when a phase 2+ trial of that pair started in
(2021-12-31, 2026-12-31], or a curated regulatory approval of the new indication falls in the
same window, **and** the pair was unknown at the cutoff under all three of the following:

1. no `indication` or `off-label use` edge in the graph snapshot;
2. no trial of the pair, in any phase, starting on or before the cutoff;
3. not present in the curated approvals list before the cutoff.

Criterion 2 is what distinguishes this split from one built on curated indications alone: 68,628
pairs were already in trials before the cutoff without appearing as graph indications, and
counting them as novel would reward retrieval. Only trials testing exactly one graph drug
contribute, so that combination arms do not create spurious pairs.

### 3.3 Leakage controls

| Channel | Control | Audit |
|---|---|---|
| The graph contains the answer | Criterion 1 above; held-out pairs' drug–disease edges are additionally removed before any run | 0 of 2,776 positives have such an edge; 27 contraindication edges were removed |
| The pair was already in trials | Criterion 2, any phase | 68,628 pairs excluded |
| Literature after the cutoff | PubMed queried with `datetype=pdat&maxdate=2021/12/31`; every fetched record re-checked; later citations counted invalid | Invalid-citation rate reported per system |
| Model weights | Open-weight models with documented cutoffs; closed-book probe with controls; masked re-scoring | Section 7.4 |
| Verifier evidence | Open Targets (a live release containing post-cutoff evidence) disabled by default; reported separately when enabled | Off in all reported runs |

### 3.4 Evaluation diseases and the verifiable ceiling

Of 1,149 diseases carrying at least one positive, 424 have no `disease_protein` edge at all. No
mechanism path can terminate at such a disease, so every proposal for it is unverifiable for a
reason that has nothing to do with the model. We therefore require at least one associated gene,
leaving 725 diseases, and sample 200 (seed 13) holding 565 positives.

Within those 565 positives, only **335 (59.3%)** are *mechanism-reachable*: the graph admits at
least one drug → gene (→ gene | → pathway → gene) → disease path of at most four edges. For 79
positives the drug has no recorded target at all. A system that reports only verified proposals
therefore cannot exceed 0.59 recall on this set. We release `mechanism_reachable` and `n_targets`
per pair so that the ceiling can be recomputed for any stratum, and we regard reporting it as
part of reporting the benchmark: without it, a verified-only system's recall is uninterpretable.

## 4 Mechanism verification

### 4.1 Proposal format and edge statuses

A proposal is a drug plus a path: a chain of (source, relation, target) edges beginning at the
drug, passing through at least one gene or protein, and ending at the queried disease (or a
graph-adjacent disease), of length 2 to 4 edges. Optional PubMed identifiers may be attached per
edge. Each edge receives one status:

| Status | Meaning |
|---|---|
| `kg` | the grounded endpoints are joined in the snapshot by a biological relation |
| `literature` | a cited record published on or before the cutoff co-mentions both endpoints |
| `opentargets` | optional; a gene–disease association above threshold in a live release |
| `circular` | the only link between the endpoints is a drug–disease relation — the prediction target itself, hence not evidence |
| `unsupported` | both endpoints resolve, no admissible evidence links them |
| `ungrounded` | an endpoint resolves to no node of an acceptable type |
| `ambiguous` | an endpoint's name matches several nodes and nothing disambiguates them |

A proposal is **verified** when the drug grounds, the path is a well-formed chain from drug to
disease through at least one gene, and every edge is supported. Excluding drug–disease relations
from the admissible evidence is essential: without it a model could "justify" an indication with
the indication edge.

### 4.2 Grounding, and why it decides the headline number

The verifier's grounding is the difference between a defensible fabrication rate and an
indefensible one. Three mechanisms matter.

**Identifier joins, not name matching.** An agent writing `COX-2` means PTGS2. A substring search
for `COX-2` in PrimeKG returns `COX2`, the mitochondrial cytochrome *c* oxidase subunit II — an
unrelated protein. We therefore attach HGNC aliases through NCBI gene identifiers (80,577 alias
symbols over 44,495 genes) and never resolve by string similarity to a primary name. Name
variants cover separator-free symbol forms (`COX-2`, `COX 2`, `COX2`; `IL-6`, `IL6`), and an
official node name outranks an alias so that a gene listing a symbol as an alias cannot outrank
the gene actually called that.

**Explicit pins where curated sources conflict with usage.** HGNC lists `COX-2` as an alias of
*PTGER2*, while drug-mechanism text uses it for PTGS2. Five such mentions are pinned in a
version-controlled file with the reason for each, rather than resolved silently
(`data/manual/gene_aliases.csv`).

**Ambiguity is reported, not resolved.** A mention matching several nodes yields `ambiguous`.
Choosing the best candidate instead would risk the worst outcome available to a verifier: an
invented edge marked as supported.

A fourth check concerns identifiers cited by the model. Agents may refer to nodes by graph
identifier (`kg:123`). If the accompanying name belongs to a different node, the endpoint is
treated as fabricated rather than resolved — a case observed in a live run, where a model
proposed invented drug names carrying real identifiers.

### 4.3 Separating fabrication from unverifiability

Our first implementation counted `unsupported`, `ungrounded`, `circular` and `ambiguous` together
as "fabricated", and reported 0.953 for a closed-book model. Inspecting the flagged endpoints
showed the figure was wrong in a way worth recording: most were real biology written informally
(`COX-2`, `VEGF`, `PDE5`, `AMPK`) or processes rather than entities (`inflammation`,
`oxidative stress`, `excitotoxicity`, `bone growth and development`). After identifier-based
alias expansion the same run scores 0.144 fabricated with 0.670 unresolvable.

We therefore report two quantities:

* **Fabricated** = `unsupported` + `circular`. Both endpoints are real entities and the evidence
  base records no link, or the only link is the label being predicted. These are checkable claims
  that the evidence contradicts.
* **Unresolvable** = `ungrounded` + `ambiguous`. The endpoint does not pick out one entity. The
  claim may be true; it is not checkable in this vocabulary.

Both are failures of a mechanism-verified system, and their sum (`unverified`) is reported too.
But they call for different remedies — better grounding versus better claims — and a single
merged number conceals which is which.

## 5 Systems evaluated

**Knowledge-graph baselines**, all restricted to the masked snapshot. *Popularity* ranks drugs by
their number of recorded indications. *Network proximity* scores a drug by overlap and
PPI-adjacency between its targets and the disease's genes, normalised by target count [7].
*TransE* [8], *DistMult* [9] and *ComplEx* [10] are trained on the snapshot with type-constrained
negative sampling and scored on the `indication` relation; we report ComplEx (dimension 128, five
epochs, 5 minutes on one RTX 4060). An importer accepts external rankings, e.g. TxGNN [6], as CSV
so that published systems can be placed on the same split.

**LLM systems**, all with the same prompt and tool budget (16 turns, top-20 output):

| System | Retrieval | Verification |
|---|---|---|
| closed book | none | none |
| RAG, no verifier | graph tools + date-capped PubMed | none |
| RAG + verify-filter | same | unverified proposals demoted (or dropped) |
| RAG + verify-revise | same | rejected proposals returned once for repair, then filtered |

Tools are: `search_entities`, `get_neighbors`, `find_mechanism_paths` (hub-penalised path
enumeration), `rank_candidate_drugs` (the proximity screen, with known indications removed) and
`search_pubmed` (date-capped). Ablations disable PubMed, the candidate screen or the graph tools
individually. Proposals from *every* system, including those whose ranking is not filtered, are
verified post hoc, so faithfulness is comparable across systems.

Two open-weight models of the same size class are reported, so that a difference between them
is a model difference and not a capacity one: `qwen2.5:7b` (documented cutoff 2023-10) and
`llama3.1:8b` (2023-12), both served locally. Two deployment
details affect reproducibility and are easy to get wrong: the server applies its own context
limit and truncates longer prompts silently, so runs use a derived model with the context raised
to 16,384 tokens; and concurrent requests against a single instance were observed to leave
established sockets with no response, so the default concurrency is one worker.

## 6 Metrics

### 6.1 Ranking

Rankings are *filtered*: drugs already used for the disease at the cutoff are removed before
scoring, as in disease-centric graph evaluation [6]. We report Hit@K and Recall@K for
K ∈ {1, 5, 10, 20} and MRR, with 95% confidence intervals from a disease-level bootstrap (1,000
resamples) and paired bootstrap comparisons against a reference system. A disease with no
ranking scores zero, so abstention is not rewarded; coverage is reported separately.

### 6.2 Faithfulness

Verified-proposal rate; fabricated- and unresolvable-edge rates with their components;
malformed-path rate; ungrounded-drug rate; and invalid-citation rate — the share of cited
identifiers that do not resolve or that postdate the cutoff.

### 6.3 Cost

Input and output tokens, monetary cost where a price is configured, and tokens or cost per
correct hit — the quantity that makes agentic and non-agentic systems comparable.

### 6.4 Validation that does not depend on expert raters

A blinded panel of pharmacists rating plausibility is the conventional check here, and it has two
problems for this paper. It is the method we argue against relying on — a human judgement that
another group cannot reproduce — and it does not address the objection a reviewer will actually
raise, which is that our verifier accepts an edge from the same graph the agent reads. We
therefore use three checks that are either automated or require no pharmacological judgement.

**(a) Independent confirmation of drug–target claims.** The first edge of every path is a claim
about a drug's protein target, accepted by the verifier because PrimeKG contains it. We re-check
those edges against ChEMBL's mechanism-of-action curation, a pipeline the agent never queried.
Drugs are mapped DrugBank → ChEMBL through UniChem by chemical structure (InChI), never by name,
for the reason given in Section 4.2. ChEMBL records no mechanism for many drugs; that is reported
as coverage, not counted as disagreement. The two resources are not fully independent — both
curate from labels and literature — so this is corroboration, not arbitration.

**(b) Role sensitivity of "mechanism".** PrimeKG merges four distinct relations into
`drug_protein`: target, enzyme, transporter and carrier. Only the first can be a mechanism of
action; a path through a drug's metabolising enzyme describes pharmacokinetics. We therefore
re-score every run under a stricter verifier that requires the drug→protein hop to be a curated
target, and report the change. This costs no model inference: the verifier re-reads stored
proposals.

**(c) Post-cutoff literature corroboration.** For each system's top-5 proposals we count PubMed
records published *after* the cutoff that co-mention the drug and the disease — an outcome signal
independent of ClinicalTrials.gov. Co-mention is weak evidence, and a system that names famous
drugs scores well on it, so each system is compared with a control of random drugs drawn for the
same diseases, as in the leakage probe.

A 20-item spot check by a pharmacologist co-author remains useful for face validity and is a
ten-minute task; we do not report inter-rater agreement on it and do not treat it as a primary
outcome. `[PENDING-REVIEW]`

## 7 Results

### 7.1 The benchmark

| Quantity | Value |
|---|---|
| Graph | 129,375 nodes; 4,050,249 edges; 7,957 drugs; 17,080 diseases |
| Trials read | 172,939 pre-cutoff; 34,595 post-cutoff phase 2+ |
| Pairs known at the cutoff | 11,831 graph indications/off-label; 68,628 pre-cutoff trial pairs |
| Post-cutoff pairs found | 6,556 |
| Novel positives | **2,776** (phase 2: 1,602; phase 3: 626; phase 4: 548) |
| Diseases with a positive | 1,149 → 725 with ≥1 associated gene → **200 sampled** (565 positives) |
| Leakage audit | **0** positives carry a drug–disease edge in the snapshot |
| Mechanism-reachable positives | **335 / 565 (59.3%)**; 79 positives' drugs have no recorded target |

**Table 1.** Benchmark composition at cutoff 2021-12-31.

Two features deserve emphasis. First, 58% of the post-cutoff pairs we found were discarded as
already known — the step that most distinguishes a time split from a random one. Second, the
59.3% reachability figure means that mechanism-verified evaluation on this knowledge base is
bounded well below perfect recall; a system reporting 0.5 recall under verification may be close
to the achievable maximum rather than far from it.

### 7.2 Knowledge-graph baselines

| System | MRR | Hit@1 | Hit@5 | Hit@10 | Hit@20 |
|---|---|---|---|---|---|
| Popularity | 0.029 [0.015, 0.047] | 0.010 | 0.055 | 0.075 | 0.085 |
| Network proximity | 0.012 [0.005, 0.020] | 0.000 | 0.020 | 0.035 | 0.055 |
| ComplEx | 0.007 [0.004, 0.010] | 0.000 | 0.000 | 0.020 | 0.045 |

**Table 2.** Baselines over 200 diseases and 565 positives; 95% CI from a disease-level
bootstrap. Full table with Recall@K in `runs/report_baselines/`.

The absolute numbers are low: the task is to rank ~8,000 drugs for an average of 2.8 unseen
positives per disease, and the positives are, by construction, pairs that nothing in the 2021
evidence base had connected. We take this as a property of an honest time split rather than a
defect — a benchmark on which indication popularity already scored well would not be measuring
inference. Notably, popularity outperforms both structural methods, which is a useful warning
for any work reporting graph-based repurposing gains without a frequency baseline.

Under probe-masked scoring (Section 7.4) the same systems move by at most 0.005 Hit@20
(199 diseases, 553 positives), indicating the split is not sensitive to the handful of pairs the
probed model claims to know.

### 7.3 Faithfulness, accuracy, and the gap between them

| Model | System | Proposals | Verified | **Fabricated** | Unresolvable | Malformed path |
|---|---|---|---|---|---|---|
| qwen2.5-7b | closed book | 3,516 | 0.003 | **0.211** | 0.695 | 0.676 |
| qwen2.5-7b | RAG, no verifier | 2,592 | 0.895 | **0.046** | 0.013 | 0.043 |
| qwen2.5-7b | RAG + verify-revise | 2,478 | 0.958 | **0.018** | 0.005 | 0.017 |
| llama3.1-8b | closed book | 3,835 | 0.009 | **0.289** | 0.552 | 0.555 |
| llama3.1-8b | RAG, no verifier | 1,081 | 0.000 | **0.944** | 0.023 | 0.851 |
| llama3.1-8b | RAG + verify-revise | 981 | 0.030 | **0.902** | 0.002 | 0.825 |

**Table 3.** Faithfulness over 200 evaluation diseases, 14,483 proposals from two models under
identical prompts, tools and budgets. Only `qwen2.5-7b`'s unverified configuration cited
anything: 20 PubMed identifiers, all invalid.

| Model | System | MRR | Hit@5 | Hit@20 | Coverage | Correct hits | Input tokens |
|---|---|---|---|---|---|---|---|
| — | **Popularity** | **0.029** | 0.055 | **0.085** | 1.000 | **17** | 0 |
| — | Network proximity | 0.012 | 0.020 | 0.055 | 1.000 | 11 | 0 |
| — | ComplEx | 0.007 | 0.000 | 0.045 | 1.000 | 9 | 0 |
| qwen2.5-7b | closed book | 0.016 | 0.040 | 0.055 | 0.995 | 13 | 85,925 |
| qwen2.5-7b | RAG, no verifier | 0.010 | 0.020 | 0.040 | 0.925 | 8 | 11,677,655 |
| qwen2.5-7b | RAG + verify-revise | 0.010 | 0.020 | 0.045 | 0.910 | 9 | 12,735,285 |
| llama3.1-8b | closed book | 0.020 | 0.035 | 0.050 | 0.960 | 11 | 83,162 |
| llama3.1-8b | RAG, no verifier | 0.006 | 0.010 | 0.020 | 0.620 | 4 | 1,884,196 |
| llama3.1-8b | RAG + verify-revise | 0.006 | 0.010 | 0.020 | 0.630 | 4 | 3,472,891 |

**Table 4.** Ranking over the same 200 diseases and 565 positives. "Correct hits" counts
positives recovered in a top-20 list, summed over diseases.

![Faithfulness against accuracy for all six systems](figures/fig1_faithfulness_vs_accuracy.png)

**Figure 1.** The two axes do not move together, in either model. Filled circles are the LLM
systems — green where proposals are verified, orange where they are not, darker shades for
`llama3.1-8b` — and open diamonds are the knowledge-graph baselines, which emit no mechanism and
therefore have no fabrication rate; they are placed in the shaded strip rather than at zero,
which would read as "never fabricates" instead of "makes no checkable claim". Within each model
the least faithful configuration ranks best, the two models' grounded configurations sit at
opposite ends of the fabrication axis, and the baseline that reads nothing ranks best of all.

![Effect of verification on three measures](figures/fig2_verification_effect.png)

**Figure 2.** The same intervention on two models. Grounding drives `qwen2.5-7b`'s fabricated
edges down (0.211 → 0.046 → 0.018) and `llama3.1-8b`'s up (0.289 → 0.944 → 0.902), while for
both it empties the unresolvable column — the errors move from uncheckable to checkable, and
which way they then land is the model's doing. Hit@20 is flat within each model, and all six
configurations sit below the popularity baseline (dashed).

**On one model, grounding and verification act on faithfulness, strongly.** With `qwen2.5-7b`,
the closed-book agent fabricates 21.1% of its mechanism edges, leaves 69.5% unresolvable,
produces malformed paths in 67.6% of proposals, and has 0.3% of its proposals verified — 12 of
3,516. Adding graph tools takes fabrication to 4.6% and unresolvable endpoints to 1.3%. Adding
edge-level verification with one revision round takes fabrication to 1.8%: three fifths of the
remaining fabricated claims disappear, for 9% more input tokens. The verified agent also submits
fewer proposals (2,478 against 2,592), part of the gain being claims withdrawn rather than
repaired, which is the intended behaviour.

**On the other model, the same intervention makes fabrication worse.** With `llama3.1-8b`,
grounding takes the fabricated-edge rate *up*, from 0.289 closed book to 0.944 with the full tool
set, and verification barely moves it (0.902). The mechanism is visible in the traces. Over 200
diseases `llama3.1-8b` called `find_mechanism_paths` eleven times, against `qwen2.5-7b`'s 2,645.
Instead of asking for a path it reused identifiers it had already seen — mostly from
`rank_candidate_drugs`, which returns *drugs* — and presented them as targets. Of the nodes it
named as a drug's target, 688 were drugs, 172 were genes and 11 were anatomical structures; among
the most frequent was `kg:12345`, a round number of exactly the kind that produced its
counterpart's invented `PMID12345`. Its paths are then malformed in 82.5% of proposals, usually
for having no gene on them at all.

That inversion is the result we would not have had from one model. **Grounding does not make a
model truthful; it changes the shape of its errors.** Identifiers let a model state claims whose
endpoints resolve, so a claim moves out of "unresolvable" and into "checkable" — and whether it
lands on "supported" or "fabricated" is a property of the model, not of the tooling. For
`qwen2.5-7b` the claims became checkable and largely true; for `llama3.1-8b` they became checkable
and false. A benchmark that reported only the unresolvable rate would have called the second
model's grounded configuration its best (0.002 against 0.552 closed book).

**Neither model's faithfulness buys accuracy.** For `qwen2.5-7b`, verification moves MRR by
0.0004 (paired bootstrap over diseases, *p* = 0.70) and Hit@20 by 0.005 (*p* = 0.70); for
`llama3.1-8b`, by 0.0004 (*p* = 0.60) and 0.000 (*p* = 1.00). In both models the closed-book
agent — the least verifiable configuration — ranks at least as well as the grounded ones: for
`qwen2.5-7b` the difference is not significant (*p* = 0.41), for `llama3.1-8b` it is (0.020
against 0.006, *p* = 0.048). And no LLM
configuration of either model reaches the popularity baseline. Against `qwen2.5-7b`'s verified
agent, ranking drugs by how many indications they already have scores MRR 0.029 and Hit@20 0.085
against 0.010 and 0.045, significant on MRR (*p* = 0.014) though not on Hit@20 (*p* = 0.138);
against `llama3.1-8b`'s, popularity is better on both at *p* = 0.002.

This decoupling is the central empirical result, and it now holds across two models. A paper
reporting only faithfulness would conclude that verification works; one reporting only ranking
would conclude that none of this helps and that a frequency heuristic wins. Both are true of the
same runs, which is why we argue a repurposing benchmark has to measure both.

**Coverage is part of the comparison.** `llama3.1-8b` returned no ranking at all for 37% of
diseases (coverage 0.62–0.63, against 0.91–0.93 for `qwen2.5-7b` and 0.96–1.00 for the
closed-book and baseline systems). Unanswered diseases score zero rather than being dropped, so
this is counted against it rather than hidden.

**The agent ignores the literature.** Across 200 diseases `qwen2.5-7b`'s unverified
configuration called `search_pubmed` once and made 2,496 `find_mechanism_paths` calls; the
verified one called it once against 2,645. The unverified agent nevertheless attached 20 PubMed
identifiers to its proposals,
drawn from just two invented placeholders (`PMID12345`, `PMID67890`); all 20 failed resolution or
the date check, and none of its citations was admissible. The honest reading is not that
fabricated citations are frequent here — 20 across 2,592 proposals is rare — but that an agent
given a literature tool declined to use it and still volunteered references, every one of them
invented. An evaluation that reads citations without resolving them cannot distinguish that from
genuine support.

**A correction we are recording rather than replacing.** An earlier 12-disease pilot of the same
three systems gave fabricated-edge rates of 0.144 (closed book), 0.228 (no verifier) and 0.031
(verified) — a 7.4-fold effect of verification against the 2.6-fold measured here, and a
citation rate of 20 identifiers per 173 proposals against 20 per 2,592. The pilot overstated the
effect roughly threefold and made rare behaviour look routine. We report this because
faithfulness is often measured on a handful of hand-picked cases, and a twelve-disease sample was
enough to be wrong by a factor of three in the direction that favours the method under test.

**A caveat on the absolute verified rate.** With a path-enumeration tool available, an agent
that uses it largely submits paths the tool returned, so verification of what it keeps is close
to automatic; `qwen2.5-7b`'s 95.8% says as much about the tool as about the model, and
`llama3.1-8b`'s 3.0% says that it did not use the tool. The informative quantities are the
differences between configurations and between models, not the absolute rate.

Stratifying to pairs that reached phase 3 or beyond (115 diseases) leaves the picture intact and
the LLM systems weaker still: the grounded configurations recover a single positive each in
their top 20 against popularity's eleven (`runs/report_full_phase3/`). Probe-masked scoring
(Section 7.4) changes no system by more than 0.001 on any ranking metric.

### 7.4 Weight-level leakage

| Model | Item | n | Said yes | Flagged as post-cutoff knowledge |
|---|---|---|---|---|
| qwen2.5-7b | Ground-truth positives | 565 | 0.117 | **0.021** |
| qwen2.5-7b | Known at the cutoff (positive control) | 200 | 0.310 | 0.050 |
| qwen2.5-7b | Random pairs (negative control) | 200 | 0.025 | 0.000 |
| llama3.1-8b | Ground-truth positives | 565 | 0.432 | **0.007** |
| llama3.1-8b | Known at the cutoff (positive control) | 200 | 0.565 | 0.000 |
| llama3.1-8b | Random pairs (negative control) | 201 | 0.134 | 0.005 |

**Table 5.** Closed-book probe, 965 items per model.

The model is asked, with no tools, whether each pair has reached a phase 2+ trial or approval and
since when; a claim carrying a post-cutoff year, or no year, is flagged.

The controls carry the inference. The yes-rate gap between pairs known at the cutoff and random
pairs is 0.285 for `qwen2.5-7b` and 0.431 for `llama3.1-8b`, so in both cases the probe
discriminates rather than defaulting to "no". Because the positive controls were already
established before the cutoff, their flag rate is the probe mistaking a confabulated date for
knowledge — a false-positive floor. For `qwen2.5-7b` that floor is 0.050 and the positives sit
*below* it at 0.021. For `llama3.1-8b` the positives are flagged at 0.007 against a floor of
0.005 from the negative controls: four pairs of 565, which is noise at this sample size. Neither
model shows a leakage signal above the probe's own.

The 66 positives `qwen2.5-7b` claimed are more informative still: **none** carried a post-2021
year. Fifty-four were dated before the cutoff (1957–2014) — the model believing a pair was long
established, which is confabulation rather than leaked knowledge — and the remaining 12 carried
no year, which is the only reason they are flagged. Masking those 12 leaves 553 pairs over 199
diseases and barely moves the baselines (Section 7.2).

`llama3.1-8b` says yes far more readily — to 43.2% of positives and 13.4% of random pairs,
against 11.7% and 2.5% — so its answers carry less information per item even though its control
gap is wider. That difference is itself a reason to probe each model rather than assume a rate
measured on one transfers.

We state the limit plainly: the probe recalls only 31.0% and 56.5% of pairs genuinely known at
the cutoff, so its flag rate is a lower bound and the masked evaluation is a sanity check, not a
guarantee. A model with a later cutoff must be probed before its scores are interpreted; the
probe writes a per-model report for that purpose.

### 7.5 Cost

| Model | System | Input tokens | Output tokens | Correct hits | Tokens per correct hit |
|---|---|---|---|---|---|
| qwen2.5-7b | closed book | 85,925 | 502,397 | 13 | **45,256** |
| qwen2.5-7b | RAG, no verifier | 11,677,655 | 590,537 | 8 | 1,533,524 |
| qwen2.5-7b | RAG + verify-revise | 12,735,285 | 701,714 | 9 | 1,493,000 |
| llama3.1-8b | closed book | 83,162 | 617,479 | 11 | 63,695 |
| llama3.1-8b | RAG, no verifier | 1,884,196 | 520,375 | 4 | 601,143 |
| llama3.1-8b | RAG + verify-revise | 3,472,891 | 654,970 | 4 | 1,031,965 |
| — | Popularity | 0 | 0 | 17 | 0 |

**Table 7.** Cost over 200 diseases. Hits are positives recovered in a top-20 list.

Tool use dominates the bill: the grounded configurations read roughly 136 times as many input
tokens as the closed-book agent, because every path enumeration and candidate screen returns to
the context. Verification adds 9% on top of that (12.74 M against 11.68 M input tokens) and is
the cheapest intervention here per unit of faithfulness gained. Per correct hit, however, the
ordering inverts: the closed-book agent is 33 times cheaper than the verified one and recovers
more positives, and the popularity baseline costs nothing at all and recovers most. A reader
deciding whether to deploy an agentic pipeline for *candidate generation*, as opposed to
candidate *justification*, should weigh that directly.

The ordering is the same for `llama3.1-8b` and the gap is wider: its closed-book agent recovers
eleven positives at 63,695 tokens each, its verified agent four at 1,031,965. Across both models
the pattern is that the cheapest configuration recovers the most, and the free baseline recovers
more than any of them.

On one RTX 4060 (8 GB) with these models served locally, a `qwen2.5-7b` disease takes 96.6 s
under verification, 65.3 s without it and 50.5 s closed book; `llama3.1-8b` takes 134 s, 228 s
and 177 s. Decoding dominates: the systems differ by two orders of magnitude in input tokens but
only by a factor of three in wall-clock, because every configuration generates two to three
thousand tokens per disease and a 7–8B model on this hardware decodes at tens of tokens per
second. Single-worker execution is not a limitation of the hardware but of the server: four
concurrent turns against one instance stalled, and one worker was also faster per disease.

### 7.6 How much does a "verified mechanism" mean?

**Role sensitivity.** Requiring the drug→protein hop to be a curated *target*, rather than any
of PrimeKG's four `drug_protein` roles, drops the verified-proposal rate from 0.958 to 0.843 for
the verified agent and from 0.895 to 0.786 without verification: about one in nine accepted
mechanisms runs through a metabolising enzyme, transporter or carrier instead of a target. The
malformed-path rate rises correspondingly (0.017 → 0.132 and 0.043 → 0.152), while fabricated-edge
rates and all ranking metrics are unchanged — a role objection is not a false claim, and it does
not alter which drugs were proposed.

This is a limitation of our own verifier, and of mechanism verification over this graph in
general: `find_mechanism_paths` enumerates over every `drug_protein` edge, so an agent can be
handed, and then credited with, a path that is pharmacokinetics. We report both readings and
provide the stricter one as a configuration flag (`verifier.require_target_first_edge`) rather
than silently adopting either.

**Independent confirmation.** Of the 1,690 distinct drug→gene edges asserted by verified
proposals of the full system, covering 1,471 drugs, 65.7% of the drugs map to ChEMBL and 23.5%
have a ChEMBL mechanism-of-action entry, leaving 406 edges on which the independent pipeline has
an opinion.
Of those, **43.1% are confirmed** — the agent's protein is among ChEMBL's recorded mechanism
targets for that drug. Split by the role PrimeKG assigns the edge:

| KG role of the drug→protein edge | Edges | Confirmed by ChEMBL |
|---|---|---|
| target | 1,442 | **0.535** |
| enzyme | 133 | **0.000** |
| transporter | 61 | **0.000** |
| carrier | 54 | **0.000** |

**Table 8.** ChEMBL mechanism-of-action confirmation by KG edge role, verified proposals of
`qwen2.5-7b`'s full system (its unverified system gives 0.541 / 0.000 / 0.000 / 0.000 on 1,705
edges).

For `llama3.1-8b` the same check is possible only on a handful of edges, because so few of its
proposals verify: 29 drug–gene edges from the full system, of which ChEMBL has an opinion on 6,
confirming 1. Its *closed-book* configuration is the interesting case — 27 edges, all 27 drugs
mapped to ChEMBL, 13 with a recorded mechanism, and **0.846 of those confirmed**, against
`qwen2.5-7b`'s 0.535 on target-role edges. Reading that as "the ungrounded agent is more
accurate about mechanism" would be a mistake at n = 13; the simpler explanation is that a model
answering from memory names well-known drugs with well-known mechanisms, which is exactly the
subset ChEMBL curates, while a tool-grounded agent is handed DrugBank interaction partners that
are often not the therapeutic target.

![Edges by graph role and their independent confirmation](figures/fig3_moa_by_role.png)

**Figure 3.** Edges asserted by verified proposals, by the role PrimeKG assigns them (left), and
the share each role's edges that ChEMBL confirms as a mechanism target (right). Rates are
computed over the edges on which ChEMBL has an opinion. No enzyme, transporter or carrier edge
is confirmed, which is what those roles mean.

Not one edge whose KG role is enzyme, transporter or carrier is confirmed as a mechanism target,
which is what those roles mean and is independent evidence that the role distinction above is the
right one to make. Among target-role edges, roughly half are confirmed: DrugBank records many
proteins a drug binds, while ChEMBL's mechanism curation records the therapeutic target, so an
unconfirmed target edge is usually a real interaction that is not the drug's mechanism of action
rather than a false claim. The honest summary is that passing our verifier establishes that a
claimed interaction is *recorded somewhere*, and for about half of the target-role edges — a
quarter of all edges on which ChEMBL has an opinion — it is also the drug's curated mechanism.
A reader should not read "verified" as "pharmacologically the mechanism".

**Post-cutoff literature.** For each system's top-5 proposals we asked whether any PubMed record
published after the cutoff co-mentions the drug and the disease.

| Model | System | Pairs | Co-mention | Random control | Degree-matched | Lift vs matched |
|---|---|---|---|---|---|---|
| qwen2.5-7b | closed book | 955 | 0.308 | 0.034 | 0.063 | **+0.245** |
| qwen2.5-7b | RAG, no verifier | 893 | 0.078 | 0.021 | 0.038 | **+0.040** |
| qwen2.5-7b | RAG + verify-revise | 888 | 0.083 | 0.036 | 0.046 | **+0.037** |
| llama3.1-8b | closed book | 957 | 0.278 | 0.034 | 0.065 | **+0.213** |
| llama3.1-8b | RAG, no verifier | 394 | 0.089 | 0.040 | 0.030 | **+0.058** |
| llama3.1-8b | RAG + verify-revise | 399 | 0.090 | 0.041 | 0.045 | **+0.045** |

**Table 9.** Post-cutoff co-mention. The matched control pairs each proposed drug with another of
comparable knowledge-graph degree for the same disease. The two models agree closely, which is
some reassurance that this measure reflects the proposals rather than the model.

The control matters more than the headline here. Against randomly drawn drugs, the closed-book
agent looks strongly corroborated — 0.308 against 0.034, an eight-fold lift. Its examples are
metformin, ibuprofen and amiodarone: drugs that co-occur with nearly any disease in the
literature. Matching the control on knowledge-graph degree, a proxy for how well studied a drug
is, nearly doubles its control rate to 0.063 and removes about a tenth of the apparent lift. Two
readings follow, and they differ in how much weight they can carry.

The **grounded systems' lift is small and that finding is robust**: their mechanism-verified
proposals are co-mentioned in post-cutoff literature barely more often than equally studied drugs
for the same disease (+0.037 and +0.040). Degree is an imperfect proxy and under-corrects for
publication volume, but under-correction can only inflate a lift, so the true values are no
higher than these.

The **closed-book advantage is an upper bound** and we do not read it as skill. The same
under-correction applies in its favour, its proposals are concentrated on widely published drugs,
and co-mention is not efficacy — a drug discussed alongside a disease may be discussed as a
contraindication or a confounder. Matching the control on each drug's post-cutoff publication
count, rather than its graph degree, would tighten this and is the obvious next measurement.

What this check does establish is a negative result that the ranking metrics support: a
mechanism-verified pipeline over this knowledge base did not surface drug–disease pairs that the
subsequent literature went on to discuss, relative to comparable drugs.

## 8 Discussion

**Verification is cheap and it works — on the thing it targets.** The comparison that matters is
not tools versus no tools but checked versus unchecked: the unverified agent had every tool the
verified one had. For 9% more input tokens, edge-level checking with one revision round removed
three fifths of the fabricated claims and left ranking statistically unchanged. Nothing in the
mechanism is model-specific; it applies to any system that emits structured claims over a
knowledge base. What it does *not* do is make the agent find more true repurposing candidates,
and we would caution against the inference — easy to make from a faithfulness table alone — that
a more faithful agent is a more useful one.

**Faithfulness and accuracy are decoupled, and that is a reason to measure both.** In both
models the least verifiable configuration ranked best among the LLM systems, and neither
model's most faithful configuration came close to ranking drugs by their existing indication
count. The mechanism path requirement plausibly constrains the grounded agents toward candidates
the graph can justify, which is a narrower set than the clinically interesting one — the 59.3%
reachability ceiling (Section 3.4) bounds that explicitly. We read the pattern as evidence that
reporting either quantity alone invites a conclusion the other contradicts.

**What grounding does to fabrication is a property of the model, not of the tooling.** This is
the finding we could not have had from one model, and it reverses sign between them: the same
tools took `qwen2.5-7b` from 0.211 fabricated edges to 0.046, and `llama3.1-8b` from 0.289 to
0.944. Identifiers do not make claims true, they make claims *checkable*; a model that queries
for a path produces checkable claims that hold, and a model that recycles identifiers it has
already seen produces checkable claims that do not. The practical consequence for anyone
building such a system is that the gain from grounding has to be measured on the model being
deployed, and the measurement has to distinguish unresolvable from fabricated, because the two
move in opposite directions here. A single-model result about retrieval-augmented faithfulness
should be read as a fact about that model.

**A popularity baseline belongs in every repurposing evaluation.** Ranking by indication count
beat two structural methods, three LLM configurations, and a knowledge-graph embedding, on a
split where the answers are by construction absent from the graph. It costs nothing to compute
and it is the first thing a reviewer should ask for.

**Small-sample faithfulness estimates are unreliable in a predictable direction.** Our own
12-disease pilot overstated verification's effect by roughly threefold and turned a rare
behaviour — volunteering invented citations — into an apparent default (Section 7.3). Hand-picked
case studies are the norm in this literature; a twelve-disease sample was enough to be wrong by a
factor of three in the direction that favoured our method.

**Measurement design decides the headline.** Our first figure for closed-book fabrication was
also threefold too high, because the grounding vocabulary lacked gene aliases (Section 4.3). Any
paper reporting hallucination rates against a knowledge base is making a claim about its
grounding as much as about the model, and the two failure modes should be reported separately as
a matter of course.

**What the benchmark cannot settle.** Trial initiation is evidence of clinical interest, not of
efficacy; tier-stratified results (`--min-tier phase3`, `approved`) are provided for that reason.
And the 59.3% reachability ceiling means a mechanism-verified system is scored against a
knowledge base that cannot express the mechanism for two of every five positives.

## 9 Limitations

* PrimeKG's constituent resources have differing release dates around 2021, so the snapshot
  approximates the cutoff rather than matching it exactly.
* Entity mapping from trial records is imperfect (64.9% of trials yield a drug, 77.3% a disease);
  coverage is reported and a sampling tool is provided, but a manual precision audit should
  accompany any derived split.
* Literature support is established by co-mention in a cited pre-cutoff abstract, which is
  necessary but not sufficient; a knowledge-graph-only variant
  (`verifier.accept_literature = false`) is provided as the stricter setting.
* The leakage probe measures what a model will state, not what it has internalised, and recalls
  only 31% of genuinely known pairs.
* Grounding is bounded by the vocabulary: informal receptor names and most biological processes
  have no node, which is why *unresolvable* is reported separately rather than folded into
  fabrication.
* The LLM results come from two open-weight models of the same size class. The decoupling of
  faithfulness and accuracy and the failure to beat a popularity baseline hold in both, but a
  larger or differently trained model may behave differently again — the sign reversal in what
  grounding does to fabrication is itself the warning. We report the harness and the split so
  the claim can be tested rather than argued. A late-cutoff hosted model should be probed
  (Section 7.4) before its scores are compared with these.
* The ChEMBL check on `llama3.1-8b` rests on 6 and 13 edges with a curated opinion. We report
  the counts beside the rates and draw no conclusion from the difference between models on this
  measure.
* Sixteen of the 200 diseases in one `llama3.1-8b` run were processed twice by two concurrent
  copies of the driver, and the two runs returned different proposal counts for them. The
  duplicates were collapsed to one record per disease, keeping the clean copy, but it is a
  reminder that these runs are not deterministic: the same model and prompt can return different
  proposals on a second pass.
* Ranking differences among the LLM systems are small relative to their confidence intervals, so
  the ordering within that group should not be over-read. The comparison that is significant is
  popularity against the verified agent on MRR (*p* = 0.014).
* No human expert has rated the proposals. Section 6.4 argues that the checks we report answer
  more of what a reviewer needs than a plausibility panel would, but they do not establish that
  a pharmacologist would find the surviving mechanisms sensible, and a 20-item spot check is
  outstanding. `[PENDING-REVIEW]`
* The literature control matches on knowledge-graph degree rather than publication volume, which
  under-corrects for how much a drug is written about (Section 7.6).

## 10 Data and code availability

Code, configuration, splits and reports: https://github.com/MinhNhat-2504/repurpose-verify.
The repository builds the graph and the ground truth from public sources (PrimeKG, MONDO, HGNC,
ClinicalTrials.gov API v2, PubMed E-utilities) and caches every external response, so a reported
run can be re-scored or re-executed offline. A `reverify` command re-scores saved proposals with
an updated verifier without re-running any model, which is how the revised taxonomy in
Section 4.3 was applied to runs already completed. The suite comprises 45 tests over a synthetic
graph and a mock model, requiring no network access.

## References

1. Huynh D, Seal S, Reid D, Carpenter A, Bender A, Spjuth O, AIA4S Consortium. AI agents in drug
   discovery: applications and case studies. *Drug Discovery Today* 2026;31:104650.
   doi:10.1016/j.drudis.2026.104650
2. Schuh MG, Daniluk A, Sieber SA. Auditing widely used biomolecular benchmarks reveals
   systematic data inconsistencies. *Chemical Science* 2026;17:17455–17467.
   doi:10.1039/d6sc01799a
3. de la Fuente J, Serrano G, Veleiro U, Casals M, Vera L, Pizurica M, Pineda-Lucena A, Ochoa I,
   Vicent S, Gevaert O, Hernaez M. Towards a more inductive world for drug repurposing
   approaches. arXiv:2311.12670 (2023).
4. Himmelstein DS, Lizee A, Hessler C, Brueggeman L, Chen SL, Hadley D, Green A, Khankhanian P,
   Baranzini SE. Systematic integration of biomedical knowledge prioritizes drugs for
   repurposing. *eLife* 2017;6:e26726. doi:10.7554/eLife.26726
5. Chandak P, Huang K, Zitnik M. Building a knowledge graph to enable precision medicine.
   *Scientific Data* 2023;10:67. doi:10.1038/s41597-023-01960-3
6. Huang K, Chandak P, Wang Q, Havaldar S, Vaid A, Leskovec J, Nadkarni GN, Glicksberg BS,
   Gehlenborg N, Zitnik M. A foundation model for clinician-centered drug repurposing.
   *Nature Medicine* 2024;30(12):3601–3613. doi:10.1038/s41591-024-03233-x
7. Guney E, Menche J, Vidal M, Barabási A-L. Network-based in silico drug efficacy screening.
   *Nature Communications* 2016;7:10331. doi:10.1038/ncomms10331
8. Bordes A, Usunier N, Garcia-Duran A, Weston J, Yakhnenko O. Translating embeddings for
   modeling multi-relational data. In: *Advances in Neural Information Processing Systems 26*
   (NIPS 2013).
9. Yang B, Yih W, He X, Gao J, Deng L. Embedding entities and relations for learning and
   inference in knowledge bases. arXiv:1412.6575 (2014); presented at ICLR 2015.
10. Trouillon T, Welbl J, Riedel S, Gaussier É, Bouchard G. Complex embeddings for simple link
    prediction. In: *Proceedings of the 33rd International Conference on Machine Learning*
    (ICML 2016). arXiv:1606.06357
11. Vasilevsky NA, Toro S, Matentzoglu N, Flack JE, et al. Mondo: integrating disease
    terminology across communities. *Genetics* 2026;232(4). doi:10.1093/genetics/iyaf215
12. Seal RL, Braschi B, Gray K, Jones TEM, et al. Genenames.org: the HGNC resources in 2023.
    *Nucleic Acids Research* 2023;51(D1):D1003–D1009. doi:10.1093/nar/gkac888
13. ClinicalTrials.gov. National Library of Medicine. API v2. Accessed 2026-10-20.
14. Wishart DS, Feunang YD, Guo AC, Lo EJ, et al. DrugBank 5.0: a major update to the DrugBank
    database for 2018. *Nucleic Acids Research* 2018;46(D1):D1074–D1082.
    doi:10.1093/nar/gkx1037

> **Reference audit (2026-10-21).** Every entry above was resolved against an authoritative
> record — Crossref, PubMed, arXiv, OpenAlex, the NIPS 2013 proceedings index or the publisher
> page — rather than a secondary summary. The audit changed three entries:
>
> * **Reference 3 was wrong.** An earlier draft cited arXiv:2311.12670 under the title "On the
>   reliability of drug–target interaction benchmarking and data leakage", which is not its
>   title. The identifier belongs to de la Fuente *et al.*, "Towards a more inductive world for
>   drug repurposing approaches". The claim it supports — that transductive DTI evaluation
>   inflates performance — is in that paper, but the citation as written was a fabrication of the
>   kind this manuscript measures, and it survived two drafts. It is recorded here rather than
>   quietly corrected.
> * **Reference 11** now cites the peer-reviewed version in *Genetics* (2026) in place of the
>   2022 medRxiv preprint.
> * **Reference 6** gained the volume and pages that were marked for checking.
>
> One wording check also came out of the audit: reference 2's abstract says "over fifty dataset
> configurations", so Section 2 says that rather than a precise count taken from a summary.

## Appendix A Reproduction

```bash
pip install -e ".[all]"
python -m repurpose_verify download all      # PrimeKG parts, MONDO, HGNC
python -m repurpose_verify build-kg
python -m repurpose_verify build-gt
python -m repurpose_verify baseline popularity
python -m repurpose_verify baseline proximity
python -m repurpose_verify baseline complex --epochs 5
python -m repurpose_verify agent --model qwen25-7b --mode closed_book
python -m repurpose_verify agent --model qwen25-7b --verifier none
python -m repurpose_verify agent --model qwen25-7b --verifier revise
python -m repurpose_verify probe --model qwen25-7b
python -m repurpose_verify evaluate --reference qwen25-7b__rag_verify-revise
python -m repurpose_verify evaluate --probe runs/probes/qwen25-7b.jsonl --out runs/report_masked
```

## Appendix B Prompts

Verbatim in `src/repurpose_verify/agent/prompts.py`; the system prompt, the task prompt and the
revision prompt are reproduced there unmodified from the runs reported here.
