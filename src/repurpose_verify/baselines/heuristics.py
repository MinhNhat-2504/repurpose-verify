"""Non-learned KG baselines: network proximity and indication popularity."""

from __future__ import annotations

import numpy as np

from ..data.kg import DRUG, KNOWN_USE_RELS, KnowledgeGraph


def ranking_record(kg: KnowledgeGraph, disease_idx: int, scores: np.ndarray, system: str,
                   exclude: set[int], top_n: int = 200) -> dict:
    s = scores.astype(np.float64).copy()
    s[kg.types != DRUG] = -np.inf
    if exclude:
        s[list(exclude)] = -np.inf
    order = np.argsort(-s, kind="stable")[:top_n]
    ranked = [{"drug_idx": int(i), "drug": kg.names[i], "score": round(float(s[i]), 6)}
              for i in order if np.isfinite(s[i])]
    return {"disease_idx": int(disease_idx), "disease_name": kg.names[disease_idx],
            "system": system, "model": system, "ranked": ranked, "proposals": [],
            "usage": {}, "trace": {}}


def proximity_scores(kg: KnowledgeGraph, disease_idx: int, ppi_weight: float = 0.5) -> np.ndarray:
    return kg.drug_scores(disease_idx, ppi_weight)


def popularity_scores(kg: KnowledgeGraph) -> np.ndarray:
    """Number of known indications / off-label uses per drug (ties broken by KG degree)."""
    counts = np.zeros(kg.n)
    for d, _ in kg.drug_disease_pairs(KNOWN_USE_RELS):
        counts[d] += 1
    deg = np.array([kg.degree(i) for i in range(kg.n)], dtype=float)
    return counts + deg / (deg.max() + 1.0)
