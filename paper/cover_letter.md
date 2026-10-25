# Cover letter

**To:** The Editors, *Artificial Intelligence in Medicine*
**Re:** Submission of an original research contribution
**Title:** Can an LLM agent justify what it recommends? A mechanism-verified, time-split
benchmark for drug repurposing

Dear Editors,

We submit the manuscript above for consideration as an original research contribution in
*Artificial Intelligence in Medicine*.

Large language model agents are increasingly applied to drug repurposing, but the evaluations
accompanying them rarely establish two things a reader needs: that the agent could not have read
the answer, and that the mechanism it offers is real. We built a benchmark that separates these
questions and ran it end to end.

Every source an agent may query is frozen at 31 December 2021 — a knowledge-graph snapshot of
that period and PubMed restricted by publication date — and systems are scored on drug–disease
pairs whose first phase 2 or later trial began afterwards. From 172,939 pre-cutoff and 34,595
post-cutoff trials we obtain 2,731 pairs that were unknown at the cutoff by three independent
criteria. Each proposal must carry a mechanism path, and a verifier checks every edge against
graph relations or a cited pre-cutoff abstract.

Three findings seem to us worth the journal's attention, and two of them are negative.

First, **faithfulness and accuracy are decoupled**. Verifying each edge and returning the
rejected proposals for one revision round reduces one model's fabricated-edge rate from 0.039 to
0.018, while leaving its ranking statistically unchanged (ΔMRR 0.0004, *p* = 0.72). In both
models we tested, the least verifiable configuration ranks at least as well as the most
carefully grounded one.

Second, **no configuration of either model beats a trivial baseline**. Ranking drugs by how many
indications they already have scores MRR 0.029 against 0.006–0.020 for six agent configurations,
significantly better in both models. We report this plainly, because a benchmark whose purpose
is to make claims checkable should not exempt its own.

Third, and only visible because we ran two models, **what retrieval does to a model's
fabrication rate reverses sign between them**. The same tools took `qwen2.5-7b` from 0.208
fabricated mechanism edges to 0.039, and `llama3.1-8b` from 0.290 to 0.942. Identifiers do not
make claims true; they make claims checkable, moving errors out of unverifiable prose. One model
then landed on supported claims and the other on false ones. A single-model result about
retrieval-augmented faithfulness is a fact about that model, and we would not have known that
from our first set of runs.

We believe the work suits *Artificial Intelligence in Medicine* specifically. The journal asks
for both a potential high impact in a medical domain and strong novelty of method in AI, and
requires that the medical domain be discussed at proper depth. Drug repurposing is the medical
domain here, and the methodological contribution is an evaluation design rather than another
predictor: a knowledge base and literature corpus frozen at a cutoff, a ground truth built from
what subsequently entered the clinic, and a verifier that checks every edge of a proposed
mechanism against a graph relation or a dated abstract. The clinical reading is that an agent's
mechanistic narrative, however fluent, is not evidence until each link in it is checked, and that
on this benchmark the configurations whose narratives check out are not the ones that rank
candidates best. A pipeline deployed to generate repurposing hypotheses and a pipeline deployed
to justify them are therefore not the same system, which is a distinction that matters before
such tools reach a clinical workflow.

All code is released under an MIT licence, the evaluation split is frozen in a version-controlled
file, every external response is cached so a reported run can be re-scored or re-executed offline
from archived evidence, and each of the 14,065 agent proposals is released with the mechanism
path it asserted and the per-edge verification record behind it. The repository carries 68 tests
that need no network access, two of which assert that entity grounding is reproducible across
interpreters, and a script that re-derives every number quoted in the manuscript from the stored
artefacts and fails if the text disagrees — a check that caught several of our own errors,
including a citation whose title the drafting assistant had reconstructed rather than verified.
We describe that episode in the manuscript rather than quietly correcting it, since it is an
instance of the failure the paper measures.

We also report the limits of our own instrument: only 61.4% of the evaluation positives admit
any mechanism path in the graph, which caps what a verified-only system can recover; about one
in nine accepted mechanisms runs through a drug's metabolising enzyme rather than a target; and
an earlier 12-disease pilot overstated the effect of verification threefold, in the direction
that favoured our method.

The manuscript is original, has not been published elsewhere, and is not under consideration by
another journal. Both authors have read and approved the submission and declare no competing
interests. The supporting data are prepared for deposit in a public repository and the
identifier will be added at acceptance.

We would be glad to supply any further material the editors or reviewers require.

Yours sincerely,

**Linh Nguyen Thi My** (corresponding author)
linh.ntm@vlu.edu.vn · ORCID 0000-0003-3990-312X

on behalf of the authors:
Nhat Trinh Ngoc Minh (ORCID 0009-0006-9477-9642), Linh Nguyen Thi My

Faculty of Information Technology, Van Lang School of Technology, Van Lang University,
69/68 Dang Thuy Tram Street, Binh Loi Trung Ward, Ho Chi Minh City, Vietnam
