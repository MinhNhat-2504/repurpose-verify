# Can an LLM agent justify what it recommends? A mechanism-verified, time-split benchmark for drug repurposing

**Authors.** [Author 1]¹, [Pharmacy/pharmacology co-author]², [Senior author]¹
¹ [Affiliation], ² [Affiliation]
Corresponding author: [e-mail]

---

> **Draft status (2026-10-20).** Every number in this draft is measured and reproducible from
> the released code; nothing is projected. Two things are *not* final:
> 1. **The LLM experiments are a 12-disease pilot** (Section 7.3). The full 200-disease runs
>    are queued; Tables 4 and 5 will be regenerated, and the §7.3 text that depends on pilot
>    scale is marked `[PILOT]`.
> 2. **Blinded expert review has not been run** (Section 6.4). Placeholders are marked
>    `[PENDING-REVIEW]`.
> Sections 3–6, 7.1, 7.2, 7.4 and 7.5 are complete at full scale.
> Every reference must be checked against the record before submission; those taken from
> secondary summaries rather than read in full are marked `[verify]`.

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
overstates fabrication by a factor of three. In a 12-disease pilot with an open-weight 7B model,
verification with one revision round reduces fabricated edges from 0.228 to 0.031, and the
unverified agent attached 20 PubMed identifiers without ever querying PubMed — all 20 were
sequential placeholders and all 20 failed the check. A weight-level leakage probe with positive
and negative controls finds no post-cutoff signal above its own false-positive floor.

**Conclusions.** Giving an agent retrieval tools does not make it faithful; checking its claims
does. We also quantify a limit of mechanism-verified evaluation itself: only 59.3% of our
positives admit any mechanism path in the graph, which caps what a verified-only system can
recover regardless of the model.

**Availability.** Code, splits, cached evidence and reports: https://github.com/MinhNhat-2504/repurpose-verify

**Keywords.** drug repurposing; large language model agents; benchmark; data leakage;
knowledge graphs; hallucination; evaluation

---

## 1 Introduction

Computational drug repurposing is an attractive target for LLM agents. The task rewards wide
reading, the evidence is public, and the output — a short ranked list with an argument for each
entry — is the kind of thing language models produce fluently. Recent reviews of agentic AI in
drug discovery identify evaluation as the field's weak point rather than capability [1 verify].

Two specific weaknesses motivate this work.

**Retrospective evaluation does not establish prospective skill.** A system is usually scored on
held-out associations drawn from the same curated sources it reads. Those associations are
historical facts: the indication is in DrugBank, the gene–disease link is in DisGeNET, the trial
is in the literature. High accuracy under such a split is consistent with retrieval rather than
inference. Benchmark audits in adjacent areas have shown that widely used splits leak in exactly
this way, and that reported progress shrinks once leakage is removed [2 verify, 3 verify].

**A fluent mechanism is not a verified one.** When an agent explains that drug D hits target T
which drives disease S, the explanation is usually either taken at face value or scored by
another language model. Neither establishes that the edges exist. The failure is not
hypothetical: in our pilot an agent with a literature-search tool available attached twenty
PubMed identifiers without calling that tool once, and every identifier was invented
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
* A measurement of what verification buys: fabricated edges fall 7× when rejected proposals are
  returned to the agent for one revision round `[PILOT]`.
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
absence of agreed benchmarks as the limiting factors rather than raw capability [1 verify].
Where mechanisms are reported, they are rarely machine-checked.

**Leakage and benchmark auditing.** Transductive evaluation of drug–target interaction models
has been shown to approach ceiling performance largely through leakage [3 verify], and a 2026
audit of widely used biomolecular benchmarks found systematic inconsistencies, including protein
duplication across splits [2 verify]. Our ground truth is built to be auditable in the same
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

The reported model is `qwen2.5:7b` (documented cutoff 2023-10) served locally. Two deployment
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

### 6.4 Blinded expert review `[PENDING-REVIEW]`

Top proposals from every system are exported anonymised and shuffled, with graph identifiers
rendered as entity names, for two pharmacists to rate plausibility and mechanism validity on
five-point scales. Agreement is quantified with quadratic-weighted Cohen's κ. `[PENDING-REVIEW]`

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

### 7.3 What verification buys `[PILOT]`

| System | Proposals | Verified | **Fabricated** | Unresolvable | Malformed path | Invalid citations |
|---|---|---|---|---|---|---|
| closed book | 213 | 0.000 | **0.144** | 0.670 | 0.775 | — (cited none) |
| RAG, no verifier | 173 | 0.705 | **0.228** | 0.000 | 0.012 | **1.000** (20/20) |
| RAG + verify-revise | 136 | 0.926 | **0.031** | 0.000 | 0.007 | — (cited none) |

**Table 3.** Faithfulness on 12 evaluation diseases, `qwen2.5:7b`. `[PILOT]`

Three findings.

**Retrieval tools do not confer faithfulness; checking does.** With the full tool set but no
verification, 22.8% of the agent's mechanism edges link two real entities that the evidence base
does not connect. Verifying each edge and returning the rejected proposals for one revision round
takes that to 3.1% — a 7-fold reduction, and the single largest effect we measure. The agent also
submits fewer proposals (136 against 173), i.e. part of the gain is the agent withdrawing claims
it cannot support, which is the intended behaviour.

**Every citation the unverified agent produced was fabricated.** It attached 20 PubMed
identifiers across its proposals while never calling `search_pubmed`; the identifiers were
sequential placeholders (`PMID12345`, `PMID12346`, `PMID12347`, …), and all 20 failed resolution
or the date check. Any evaluation that reads citations without resolving them is blind to this,
and it is not a rare corner case but the default behaviour of this configuration.

**Without tools the output is not false so much as uncheckable.** The closed-book model's
mechanism endpoints are 67.0% unresolvable and 77.5% of its paths are malformed: it writes
processes ("inflammation", "oxidative stress", "retinal cell death") where the format asks for
entities. Its *fabricated* rate, 0.144, is lower than the unverified agent's, because fewer of
its claims are checkable at all. Reporting a single merged number would have ranked it as the
most faithful system of the three on the "fabricated" column while hiding that none of its 213
proposals passed verification.

`[PILOT]` Ranking cannot be compared at this scale: all six systems, baselines included, score
Hit@K = 0 on these 12 diseases, which are predominantly rare genetic disorders, while the same
baselines reach Hit@20 = 0.085 over the full 200. Faithfulness is measurable in a pilot; ranking
is not. Neither agent configuration called `search_pubmed` at all (161 and 144
`find_mechanism_paths` calls respectively), a behavioural observation we expect to revisit at
full scale.

A caveat on the high verified rate: with a path-enumeration tool available, the agent largely
submits paths the tool returned, so verification of what it keeps is close to automatic. The
informative quantities are the differences between configurations, not the absolute rate.

### 7.4 Weight-level leakage

| Item | n | Said yes | Flagged as post-cutoff knowledge |
|---|---|---|---|
| Ground-truth positives | 565 | 0.117 | **0.021** |
| Pairs known at the cutoff (positive control) | 200 | 0.310 | 0.050 |
| Random unrelated pairs (negative control) | 200 | 0.025 | 0.000 |

**Table 4.** Closed-book probe of `qwen2.5:7b`, 965 items.

The model is asked, with no tools, whether each pair has reached a phase 2+ trial or approval and
since when; a claim carrying a post-cutoff year, or no year, is flagged.

The controls carry the inference. The yes-rate gap between pairs known at the cutoff and random
pairs is 0.285, so the probe discriminates rather than defaulting to "no". Because the positive
controls were already established before the cutoff, their 0.050 flag rate is the probe
mistaking a confabulated date for knowledge — a false-positive floor. The positives' flag rate,
0.021, lies *below* that floor: this probe finds no leakage signal above its own noise.

The 66 positives the model did claim are more informative still: **none** carried a post-2021
year. Fifty-four were dated before the cutoff (1957–2014) — the model believing a pair was long
established, which is confabulation rather than leaked knowledge — and the remaining 12 carried
no year, which is the only reason they are flagged. Masking those 12 leaves 553 pairs over 199
diseases and barely moves the baselines (Section 7.2).

We state the limit plainly: the probe recalls only 31.0% of pairs genuinely known at the cutoff,
so its flag rate is a lower bound and the masked evaluation is a sanity check, not a guarantee. A
model with a later cutoff must be probed before its scores are interpreted; the probe writes a
per-model report for that purpose.

### 7.5 Cost

Over 12 diseases the verify-revise agent consumed 795,312 input and 45,009 output tokens
(≈66,000 input tokens per disease); without verification, 673,110 and 40,298; closed book, 4,653
and 30,422. Verification costs roughly 18% more input tokens than the unverified agent for the
7-fold reduction in fabricated edges. On one RTX 4060 (8 GB) a disease takes two to three
minutes; single-worker execution completed the same 12 diseases faster than four concurrent
workers against one server (115 s versus 180 s per disease), the concurrency having caused
stalled requests. Cost per correct hit is reported by the harness but is not yet informative at
pilot scale. `[PILOT]`

## 8 Discussion

**Verification is a cheap intervention with a large effect.** The comparison that matters is not
tools versus no tools but checked versus unchecked: the unverified agent had every tool the
verified one had. At roughly 18% additional input tokens, edge-level checking with a single
revision round removed six sevenths of the fabricated claims. Nothing in the mechanism is
model-specific; it applies to any system that emits structured claims over a knowledge base.

**Measurement design decides the headline.** Our own first figure for closed-book fabrication was
three times too high because the grounding vocabulary lacked gene aliases. Any paper reporting
hallucination rates against a knowledge base is making a claim about its grounding as much as
about the model, and we would encourage reporting the two failure modes separately as a matter of
course.

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
* The LLM results are a 12-disease pilot with one open-weight 7B model. Scale, model family and
  the blinded expert review are pending. `[PILOT]` `[PENDING-REVIEW]`

## 10 Data and code availability

Code, configuration, splits and reports: https://github.com/MinhNhat-2504/repurpose-verify.
The repository builds the graph and the ground truth from public sources (PrimeKG, MONDO, HGNC,
ClinicalTrials.gov API v2, PubMed E-utilities) and caches every external response, so a reported
run can be re-scored or re-executed offline. A `reverify` command re-scores saved proposals with
an updated verifier without re-running any model, which is how the revised taxonomy in
Section 4.3 was applied to runs already completed. The suite comprises 45 tests over a synthetic
graph and a mock model, requiring no network access.

## References

1. AI agents in drug discovery: applications and case studies. *Drug Discovery Today*, 2026.
   `[verify — cited from a secondary summary; confirm authors, volume and pages]`
2. Auditing widely used biomolecular benchmarks reveals systematic data inconsistencies.
   *Chemical Science*, 2026. `[verify]`
3. On the reliability of drug–target interaction benchmarking and data leakage. arXiv:2311.12670.
   `[verify — confirm title and authors]`
4. Himmelstein DS, Lizee A, Hessler C, et al. Systematic integration of biomedical knowledge
   prioritizes drugs for repurposing. *eLife* 2017;6:e26726.
5. Chandak P, Huang K, Zitnik M. Building a knowledge graph to enable precision medicine.
   *Scientific Data* 2023;10:67.
6. Huang K, Chandak P, Wang Q, et al. A foundation model for clinician-centered drug repurposing.
   *Nature Medicine* 2024. `[verify volume and pages]`
7. Guney E, Menche J, Vidal M, Barabási A-L. Network-based in silico drug efficacy screening.
   *Nature Communications* 2016;7:10331.
8. Bordes A, Usunier N, Garcia-Durán A, Weston J, Yakhnenko O. Translating embeddings for
   modeling multi-relational data. *NIPS* 2013.
9. Yang B, Yih W, He X, Gao J, Deng L. Embedding entities and relations for learning and
   inference in knowledge bases. *ICLR* 2015.
10. Trouillon T, Welbl J, Riedel S, Gaussier É, Bouchard G. Complex embeddings for simple link
    prediction. *ICML* 2016.
11. Vasilevsky NA, Matentzoglu NA, Toro S, et al. Mondo: unifying diseases for the world, by the
    world. *medRxiv* 2022. `[verify — check for the peer-reviewed version]`
12. Seal RL, Braschi B, Gray K, et al. Genenames.org: the HGNC resources in 2023.
    *Nucleic Acids Research* 2023;51:D1003–D1009. `[verify pages]`
13. ClinicalTrials.gov. National Library of Medicine. API v2. Accessed 2026-10-20.
14. Wishart DS, Feunang YD, Guo AC, et al. DrugBank 5.0: a major update to the DrugBank database
    for 2018. *Nucleic Acids Research* 2018;46:D1074–D1082.

> **Reference policy for this manuscript.** Given that the paper reports fabricated citations as
> a finding, every entry above must be opened and checked against the published record before
> submission, and entries marked `[verify]` are not yet so checked. None should be copied from a
> summary.

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
