"""Prompt templates. Kept in one place so the paper can quote them verbatim."""

from __future__ import annotations

import json

SYSTEM_AGENT = """You are a drug-repurposing analyst. The current date is {cutoff}. You must reason \
only with knowledge available on or before {cutoff}; do not use or mention anything that \
happened later, including later trials or approvals.

Task: for the disease you are given, propose up to {k} existing drugs that are NOT already \
approved or in clinical use for it, ranked by how promising they are as repurposing candidates.

For every candidate, give a mechanism path: a chain of edges starting at the drug, passing \
through at least one gene/protein (optionally a second gene or a pathway), and ending at the \
disease, e.g. drug -[inhibits]-> GENE -[associated with]-> disease. Each edge must be a claim \
you can back up. Use the knowledge-graph ids (kg:<number>) of the nodes you looked up. Attach \
PMIDs only if they came back from search_pubmed and support that specific edge. Never invent \
entities, edges or citations: a shorter list of well-supported candidates is better than a long \
list of guesses.

Work efficiently: you have a budget of {max_steps} tool-use turns. Finish by calling \
submit_proposals exactly once with the full ranked list."""

SYSTEM_CLOSED_BOOK = """You are a drug-repurposing analyst. The current date is {cutoff}. You \
must reason only with knowledge available on or before {cutoff}.

Task: for the disease you are given, propose up to {k} existing drugs that are NOT already \
approved or in clinical use for it, ranked by how promising they are as repurposing candidates. \
For every candidate give a mechanism path from the drug through at least one gene/protein to the \
disease. Do not invent entities or citations.

Reply with JSON only, in this format:
{schema}"""

CLOSED_BOOK_SCHEMA = json.dumps({"proposals": [{
    "drug": "drug name", "confidence": 0.5, "rationale": "one sentence",
    "path": [{"source": "drug name", "relation": "inhibits", "target": "GENE"},
             {"source": "GENE", "relation": "associated with", "target": "disease name"}]}]},
    indent=1)

REACT_SUFFIX = """

You can call tools. To call one, reply with exactly one block and nothing after it:
<tool_call>{{"name": "<tool name>", "arguments": {{...}}}}</tool_call>
The result will come back in a <tool_result> block. Available tools (JSON schema):
{tools}"""

USER_TASK = """Disease: {disease_name}{disease_ref}
Drugs already used for this disease as of {cutoff} (do not propose these): {known}

Find and submit up to {k} repurposing candidates with mechanism paths."""

USER_REVISE = """The mechanism verifier checked your submission against the knowledge graph and \
pre-{cutoff} literature. These proposals failed:
{failures}

An edge is accepted only if the two endpoints are directly linked in the knowledge graph \
(check with get_neighbors / find_mechanism_paths) or a cited pre-cutoff PubMed abstract \
mentions both. Repair the failed paths with supported edges, or drop / replace those drugs. \
Then call submit_proposals again with the complete ranked list (including the proposals that \
passed)."""

NUDGE_SUBMIT = "Please call submit_proposals now with your ranked list."
BUDGET_EXHAUSTED = ("Your tool budget is exhausted. Call submit_proposals now with the best ranked "
                    "list you have.")


def format_known(names: list[str], limit: int = 40) -> str:
    if not names:
        return "none recorded"
    more = f" (+{len(names) - limit} more)" if len(names) > limit else ""
    return ", ".join(names[:limit]) + more
