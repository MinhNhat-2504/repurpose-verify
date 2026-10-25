# Clinical assessment — scaffold for NTML

## Why this section is needed

*Artificial Intelligence in Medicine* requires that a paper discuss its medical domain "at the
proper depth, from both the technical and the medical points of view", and states that "the
inclusion of a clinical assessment of the usefulness and potential impact of the submitted work
is strongly recommended". The manuscript is strong on the technical side and currently says
almost nothing from the pharmacological one. This is the most likely reason an editor would
desk-reject it.

## What is already written, and what is not

Everything below marked **[drafted]** is written from the paper's own measurements and needs only
a yes/no from you. Everything marked **[NTML]** is a pharmacological judgement that an author who
is not a pharmacist should not make. There are four of them, and a 12-item checklist.

Estimated time: **20–30 minutes.** If you have less, Q1 and Q2 are the two that matter; the
section can be shortened to those.

A note on consistency: Section 6.4 of the manuscript argues that no expert *rated individual
proposals*, and replaces a plausibility panel with automated checks. That stands. This section is
a different claim — a judgement about whether the *work* is useful and what it would take to use
it — which is an author-level statement, not a per-proposal rating. The drafted text says so
explicitly so that a reviewer does not read a contradiction.

---

# 9 Clinical assessment

**[drafted]** **What the benchmark offers a clinician is a triage procedure, not a drug list.**
A repurposing hypothesis produced by a language-model agent arrives as a narrative. The verifier
converts that narrative into a small set of assertions that can each be checked: that the drug
resolves to a known compound, that it has a recorded interaction with a named protein, that the
protein is associated with the disease, and that any supporting citation existed before the
knowledge cutoff. What survives is not a validated hypothesis. It is a hypothesis whose stated
reasoning is at least internally checkable, which is the minimum a reviewer needs before any
laboratory or clinical resource is committed to it.

**[drafted]** **The results argue against using such agents for candidate generation today.**
Two findings bear on practice. The configurations whose mechanisms check out do not rank
candidates better than the configuration whose mechanisms do not: the closed-book agent ranks at
least as well as the grounded ones in both models tested, and no configuration of either model
matches the accuracy of ranking drugs by how many indications they already carry. And the effect
of grounding on fabrication reversed sign between two open-weight models of the same size class,
so a team cannot carry a published hallucination rate over to the model it intends to deploy.
Taken together, the case for an agentic pipeline at present is for *auditing* a hypothesis, not
for *producing* one.

**[NTML — Q1]** **What mechanism verification does not establish.**
*Draft to react to:* A path through the graph records that an interaction has been reported. It
does not record the direction of that interaction — PrimeKG's drug–protein edges do not
distinguish inhibition from activation — nor whether the modulation runs in the direction the
disease biology would require. It says nothing about whether the drug reaches the relevant tissue
at a concentration achievable at a tolerated dose, nor about route, protein binding, metabolism
or interaction burden. Roughly one accepted mechanism in nine runs through a metabolising enzyme,
transporter or carrier rather than a target, which is a pharmacokinetic relationship presented as
a pharmacodynamic one.

> **Q1 for NTML:** Is "drug → target → disease-associated gene, every edge checked" the right
> *first* filter from a pharmacology standpoint? What is the single most important thing it
> misses that is not already listed above? *(2–4 sentences.)*

**[NTML — Q2]** **What the verified proposals look like to a pharmacist.**
*Draft to react to:* Verification confirms that the asserted edges exist; it does not confirm that
the proposal is a repurposing candidate at all. Among proposals the verifier accepted we find
experimental compounds that have never been administered to patients, endogenous molecules rather
than drugs, and antibodies whose development was discontinued. [NTML to confirm and characterise
after reviewing the sample.]

> **Q2 for NTML:** `paper/clinical_review_sample.md` holds twelve accepted proposals drawn at
> random. For each, tick sensible / not sensible / unsure, and for the ones that are not, write a
> few words on what is wrong. *(This is the part that cannot be automated, and it is the strongest
> evidence this section can carry.)*

**[NTML — Q3]** **What would have to be true before such a pipeline informed a clinical decision.**
*Draft to react to:* At minimum: a mechanism whose direction matches the disease biology; evidence
of exposure at the target site at a tolerated dose; a safety profile compatible with the intended
population; and a plausible route from hypothesis to a testable design.

> **Q3 for NTML:** Is that list right, and in the right order? What would you add or remove?
> *(2–4 sentences.)*

**[NTML — Q4]** **Where the benchmark's limits bite clinically.**
*Draft to react to:* Only 61.4% of the positives admit any mechanism path in this knowledge base,
so a system restricted to verified proposals is capped well below full recall for reasons that
have nothing to do with its reasoning. Diseases whose biology is poorly represented in the graph —
and these are disproportionately the rare and the neglected — are therefore the ones such a
pipeline would serve worst, which is the opposite of where repurposing is most needed.

> **Q4 for NTML:** Is that last inference fair, and is it the right clinical framing of the
> ceiling? *(2–3 sentences, or strike it if you disagree.)*

**[drafted]** **Overall assessment.** The contribution this work offers the clinical reader is a
way to tell a checkable mechanistic claim from an unverifiable one, and a measurement showing
that current agents do not yet earn a place in candidate generation. Its practical use today is
as an evaluation instrument for teams building such systems, and as a caution against accepting a
fluent mechanistic narrative as evidence.

---

## Checklist for NTML

- [ ] Q1 answered
- [ ] Q2: twelve proposals reviewed in `paper/clinical_review_sample.md`
- [ ] Q3 answered
- [ ] Q4 answered or struck
- [ ] The drafted paragraphs read correctly to a pharmacologist, or marked where they do not
