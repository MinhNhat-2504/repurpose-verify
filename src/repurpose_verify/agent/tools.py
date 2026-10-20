"""Tools exposed to the repurposing agent. All of them only see pre-cutoff data."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import numpy as np

from ..data.kg import DISEASE, DRUG, KnowledgeGraph

ENTITY_TYPES = ["drug", "disease", "gene/protein", "pathway", "biological_process",
                "molecular_function", "effect/phenotype", "any"]

SUBMIT_TOOL = {
    "name": "submit_proposals",
    "description": (
        "Submit the final ranked list of repurposing candidates for the disease (best first). "
        "Each proposal needs a mechanism path: a chain of edges that starts at the drug, passes "
        "through at least one gene/protein, and ends at the disease. Use KG ids like 'kg:123' "
        "for every node you found with the tools. Only cite PMIDs returned by search_pubmed."),
    "input_schema": {
        "type": "object",
        "properties": {
            "proposals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "drug": {"type": "string", "description": "Drug name or kg id"},
                        "confidence": {"type": "number", "description": "0-1"},
                        "rationale": {"type": "string"},
                        "path": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "source": {"type": "string"},
                                    "relation": {"type": "string"},
                                    "target": {"type": "string"},
                                    "pmids": {"type": "array", "items": {"type": "string"}},
                                },
                                "required": ["source", "relation", "target"],
                            },
                        },
                    },
                    "required": ["drug", "path"],
                },
            }
        },
        "required": ["proposals"],
    },
}

KG_TOOLS = [
    {
        "name": "search_entities",
        "description": "Find knowledge-graph nodes by name, synonym or identifier (DrugBank, MONDO, MeSH).",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "entity_type": {"type": "string", "enum": ENTITY_TYPES},
                "limit": {"type": "integer", "minimum": 1, "maximum": 25},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_neighbors",
        "description": ("List knowledge-graph neighbours of a node, grouped by relation. Useful "
                        "relations: drug_protein (drug targets/enzymes/transporters), "
                        "disease_protein (disease-associated genes), protein_protein, "
                        "pathway_protein, indication, contraindication, off-label use."),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "kg id, e.g. kg:123"},
                "relation": {"type": "string"},
                "neighbor_type": {"type": "string", "enum": ENTITY_TYPES},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            "required": ["entity_id"],
        },
    },
    {
        "name": "find_mechanism_paths",
        "description": ("Enumerate drug -> gene -> (gene | pathway -> gene) -> disease paths in the "
                        "knowledge graph between a drug and a disease, hub-penalised, best first."),
        "input_schema": {
            "type": "object",
            "properties": {
                "drug_id": {"type": "string"},
                "disease_id": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["drug_id", "disease_id"],
        },
    },
]

CANDIDATE_TOOL = {
    "name": "rank_candidate_drugs",
    "description": ("Network-proximity screen: drugs whose targets hit (or are PPI neighbours of) "
                    "genes associated with the disease. Drugs already indicated for it are excluded."),
    "input_schema": {
        "type": "object",
        "properties": {
            "disease_id": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100},
        },
        "required": ["disease_id"],
    },
}

PUBMED_TOOL = {
    "name": "search_pubmed",
    "description": ("Search PubMed (records published on or before the knowledge cutoff only). "
                    "Returns PMID, title, date and abstract excerpt."),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "required": ["query"],
    },
}


class Toolbox:
    def __init__(self, kg: KnowledgeGraph, pubmed=None, use_kg: bool = True,
                 use_candidates: bool = True, use_pubmed: bool = True,
                 known_by_disease: dict[int, set[int]] | None = None, max_chars: int = 6000,
                 pubmed_max_results: int = 5):
        self.kg = kg
        self.pubmed = pubmed if use_pubmed else None
        self.use_kg = use_kg
        self.use_candidates = use_candidates and use_kg
        self.known = known_by_disease or {}
        self.max_chars = max_chars
        self.pubmed_max_results = pubmed_max_results
        self.seen_pmids: set[str] = set()

    def specs(self) -> list[dict]:
        tools: list[dict] = []
        if self.use_kg:
            tools += KG_TOOLS
        if self.use_candidates:
            tools.append(CANDIDATE_TOOL)
        if self.pubmed is not None:
            tools.append(PUBMED_TOOL)
        return tools + [SUBMIT_TOOL]

    # ------------------------------------------------------------------ helpers
    def _node_id(self, ref: Any, node_type: str | None = None) -> int:
        ref = str(ref).strip()
        hits = self.kg.lookup(ref, node_type)
        if not hits:
            hits = self.kg.search(ref, node_type, limit=1)
        if not hits:
            raise ValueError(f"No node matches '{ref}'")
        return hits[0]

    def _brief(self, i: int) -> dict:
        return {"id": f"kg:{int(i)}", "name": self.kg.names[i], "type": self.kg.types[i]}

    def _dump(self, obj: Any) -> str:
        text = json.dumps(obj, ensure_ascii=False)
        if len(text) > self.max_chars:
            text = text[:self.max_chars] + " ... [truncated]"
        return text

    # ------------------------------------------------------------------ dispatch
    def execute(self, name: str, args: dict) -> tuple[str, bool]:
        try:
            handler = getattr(self, f"_t_{name}")
        except AttributeError:
            return f"Unknown tool: {name}", True
        allowed = {t["name"] for t in self.specs()}
        if name not in allowed:
            return f"Tool {name} is disabled in this configuration", True
        try:
            return self._dump(handler(**args)), False
        except TypeError as err:
            return f"Bad arguments for {name}: {err}", True
        except ValueError as err:
            return str(err), True

    def _t_search_entities(self, query: str, entity_type: str = "any", limit: int = 10):
        t = None if entity_type in (None, "", "any") else entity_type
        return [{**self._brief(i), "degree": self.kg.degree(i)}
                for i in self.kg.search(query, t, limit=int(limit))]

    def _t_get_neighbors(self, entity_id: str, relation: str | None = None,
                         neighbor_type: str | None = None, limit: int = 30):
        i = self._node_id(entity_id)
        groups: dict[str, list] = defaultdict(list)
        for j, rel, disp in self.kg.neighbor_relations(i):
            if relation and rel != relation:
                continue
            if neighbor_type and neighbor_type != "any" and self.kg.types[j] != neighbor_type:
                continue
            groups[f"{rel}:{disp}" if disp and disp != rel else rel].append(j)
        out = {"node": self._brief(i), "relations": {}}
        for key, idxs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            idxs = sorted(set(idxs), key=lambda j: self.kg.degree(j))
            out["relations"][key] = {"count": len(idxs),
                                     "items": [self._brief(j) for j in idxs[:int(limit)]]}
        return out

    def _t_find_mechanism_paths(self, drug_id: str, disease_id: str, limit: int = 8):
        d = self._node_id(drug_id, DRUG)
        dis = self._node_id(disease_id, DISEASE)
        paths = self.kg.mechanism_paths(d, dis, limit=int(limit))
        return {"drug": self._brief(d), "disease": self._brief(dis),
                "paths": [{"score": p["score"], "edges": self.kg.describe_path(p["nodes"])}
                          for p in paths]}

    def _t_rank_candidate_drugs(self, disease_id: str, limit: int = 30):
        dis = self._node_id(disease_id, DISEASE)
        scores = self.kg.drug_scores(dis)
        for k in self.known.get(dis, set()):
            scores[k] = 0.0
        order = np.argsort(-scores)[:int(limit)]
        return [{**self._brief(int(i)), "score": round(float(scores[i]), 4)}
                for i in order if scores[i] > 0]

    def _t_search_pubmed(self, query: str, max_results: int | None = None):
        n = int(max_results or self.pubmed_max_results)
        arts = self.pubmed.search_articles(query, n)
        self.seen_pmids.update(a.pmid for a in arts)
        return [a.to_dict(max_chars=900) for a in arts]

    def _t_submit_proposals(self, proposals: list):
        return {"received": len(proposals)}
