"""Ranking, faithfulness and cost metrics with disease-level bootstrap confidence intervals.

Ranking metrics are *filtered*: drugs already known for the disease at the cutoff are removed
from each ranked list before scoring. A disease with an empty or missing ranking scores 0, so
systems that abstain are not rewarded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def per_disease_ranking(ranked: list[int], positives: set[int], known: set[int],
                        ks: Iterable[int]) -> dict[str, float]:
    filtered = [d for d in dict.fromkeys(ranked) if d not in known]
    first = next((i for i, d in enumerate(filtered, 1) if d in positives), None)
    out = {"mrr": 1.0 / first if first else 0.0}
    for k in ks:
        top = filtered[:k]
        hits = sum(d in positives for d in top)
        out[f"hit@{k}"] = float(hits > 0)
        out[f"recall@{k}"] = hits / len(positives) if positives else 0.0
        out[f"precision@{k}"] = hits / k
    out["n_hits_total"] = float(sum(d in positives for d in filtered))
    return out


def bootstrap_ci(values: np.ndarray, n: int = 1000, seed: int = 13,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    values = np.asarray(values, float)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), (n, len(values)))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 1000, seed: int = 13) -> dict:
    """Difference of means (a - b) over the same diseases; two-sided p-value."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), (n, len(diff)))
    boots = diff[idx].mean(axis=1)
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"diff": float(diff.mean()), "lo": float(np.quantile(boots, 0.025)),
            "hi": float(np.quantile(boots, 0.975)), "p": float(min(1.0, p))}


@dataclass
class Faithfulness:
    n_proposals: int = 0
    n_verified: int = 0
    n_edges: int = 0
    n_kg: int = 0
    n_literature: int = 0
    n_opentargets: int = 0
    n_unsupported: int = 0
    n_ungrounded: int = 0
    n_circular: int = 0
    n_structure_bad: int = 0
    n_drug_ungrounded: int = 0
    n_citations: int = 0
    n_citations_invalid: int = 0

    def add_proposal(self, ver: dict) -> None:
        self.n_proposals += 1
        self.n_verified += bool(ver.get("verified"))
        self.n_structure_bad += not ver.get("structure_ok", False)
        self.n_drug_ungrounded += ver.get("drug_idx") is None
        for e in ver.get("edges", []):
            self.n_edges += 1
            st = e["status"]
            self.n_kg += st == "kg"
            self.n_literature += st == "literature"
            self.n_opentargets += st == "opentargets"
            self.n_unsupported += st == "unsupported"
            self.n_ungrounded += st == "ungrounded"
            self.n_circular += st == "circular"
            self.n_citations += len(e.get("pmids_admissible", [])) + len(e.get("pmids_invalid", []))
            self.n_citations_invalid += len(e.get("pmids_invalid", []))

    def rates(self) -> dict[str, float]:
        def r(a, b):
            return a / b if b else float("nan")
        fabricated = self.n_unsupported + self.n_ungrounded + self.n_circular
        return {
            "proposals": self.n_proposals,
            "verified_rate": r(self.n_verified, self.n_proposals),
            "fabricated_edge_rate": r(fabricated, self.n_edges),
            "unsupported_edge_rate": r(self.n_unsupported, self.n_edges),
            "ungrounded_edge_rate": r(self.n_ungrounded, self.n_edges),
            "circular_edge_rate": r(self.n_circular, self.n_edges),
            "kg_edge_rate": r(self.n_kg, self.n_edges),
            "literature_edge_rate": r(self.n_literature, self.n_edges),
            "malformed_path_rate": r(self.n_structure_bad, self.n_proposals),
            "ungrounded_drug_rate": r(self.n_drug_ungrounded, self.n_proposals),
            "invalid_citation_rate": r(self.n_citations_invalid, self.n_citations),
            "edges": self.n_edges,
        }
