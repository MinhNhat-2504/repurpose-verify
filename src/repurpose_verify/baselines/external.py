"""Import rankings produced outside this package (e.g. TxGNN) into the common run format.

CSV columns: disease (kg id, MONDO id or name), drug (kg id, DrugBank id or name), score.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from ..data.kg import DISEASE, DRUG, KnowledgeGraph


def import_rankings(kg: KnowledgeGraph, csv_path: Path, system: str,
                    exclude: dict[int, set[int]] | None = None) -> tuple[list[dict], dict]:
    df = pd.read_csv(csv_path, dtype={"disease": str, "drug": str})
    exclude = exclude or {}
    per: dict[int, list[tuple[float, int]]] = defaultdict(list)
    unmapped = {"disease": 0, "drug": 0}
    dis_cache: dict[str, int | None] = {}
    drug_cache: dict[str, int | None] = {}
    for dis, drug, score in df[["disease", "drug", "score"]].itertuples(index=False):
        if dis not in dis_cache:
            dis_cache[dis] = kg.resolve(dis, DISEASE)
        if drug not in drug_cache:
            drug_cache[drug] = kg.resolve(drug, DRUG)
        di, dr = dis_cache[dis], drug_cache[drug]
        if di is None:
            unmapped["disease"] += 1
            continue
        if dr is None:
            unmapped["drug"] += 1
            continue
        if dr not in exclude.get(di, set()):
            per[di].append((float(score), dr))
    records = []
    for di, items in per.items():
        items.sort(key=lambda x: -x[0])
        seen, ranked = set(), []
        for s, dr in items:
            if dr not in seen:
                seen.add(dr)
                ranked.append({"drug_idx": dr, "drug": kg.names[dr], "score": s})
        records.append({"disease_idx": di, "disease_name": kg.names[di], "system": system,
                        "model": system, "ranked": ranked, "proposals": [], "usage": {},
                        "trace": {}})
    return records, unmapped
