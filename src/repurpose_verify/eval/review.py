"""Blinded expert review: export an anonymised rating sheet, then score two raters' sheets."""

from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..verify.schema import Proposal
from .evaluate import load_run

RATING_COLUMNS = ["plausibility_1to5", "mechanism_validity_1to5"]
_KG_REF = re.compile(r"\[\s*kg:(\d+)\s*\]|kg:(\d+)", re.I)


def name_refs(text: str, kg) -> str:
    """Replace bare ``kg:123`` references with node names so a rater can read the item.

    Agents routinely answer entirely in graph ids - a reviewer cannot rate "kg:16492".
    """
    if kg is None or not text:
        return text

    def sub(m: re.Match) -> str:
        i = int(m.group(1) or m.group(2))
        return str(kg.names[i]) if 0 <= i < kg.n else m.group(0)

    out = _KG_REF.sub(sub, text).strip()
    return re.sub(r"\s{2,}", " ", out)


def export_review(run_dirs: list[Path], out_dir: Path, n_items: int = 50, top_per_disease: int = 3,
                  seed: int = 13, kg=None) -> tuple[Path, Path]:
    """Sample proposals evenly across systems, strip system identity, shuffle."""
    rng = random.Random(seed)
    pools: dict[str, list[dict]] = {}
    for rd in run_dirs:
        name, _, recs = load_run(rd)
        items = []
        for rec in recs.values():
            for rank, p in enumerate(rec.get("proposals", [])[:top_per_disease], 1):
                prop = Proposal.from_any(p)
                drug = name_refs(prop.drug, kg)
                idx = (p.get("verification") or {}).get("drug_idx")
                if kg is not None and idx is not None and not drug:
                    drug = str(kg.names[int(idx)])
                items.append({"system": name, "disease": rec["disease_name"], "drug": drug,
                              "rank": rank, "mechanism": name_refs(prop.path_text(), kg),
                              "rationale": name_refs(prop.rationale, kg)})
        rng.shuffle(items)
        pools[name] = items
    per_system = max(1, n_items // max(1, len(pools)))
    chosen = [it for items in pools.values() for it in items[:per_system]]
    rng.shuffle(chosen)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet = pd.DataFrame([{"item_id": f"R{i:03d}", "disease": it["disease"], "drug": it["drug"],
                           "mechanism": it["mechanism"], "rationale": it["rationale"],
                           **{c: "" for c in RATING_COLUMNS}, "comments": ""}
                          for i, it in enumerate(chosen, 1)])
    key = pd.DataFrame([{"item_id": f"R{i:03d}", "system": it["system"], "rank": it["rank"]}
                        for i, it in enumerate(chosen, 1)])
    sheet_path, key_path = out_dir / "review_sheet.csv", out_dir / "review_key.csv"
    sheet.to_csv(sheet_path, index=False, encoding="utf-8-sig")
    key.to_csv(key_path, index=False)
    return sheet_path, key_path


def weighted_kappa(a: np.ndarray, b: np.ndarray, categories: list[int], weights: str = "quadratic") -> float:
    k = len(categories)
    pos = {c: i for i, c in enumerate(categories)}
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[pos[x], pos[y]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    i, j = np.meshgrid(range(k), range(k), indexing="ij")
    w = ((i - j) ** 2 if weights == "quadratic" else np.abs(i - j)) / (k - 1) ** (2 if weights == "quadratic" else 1)
    denom = (w * exp).sum()
    return float(1 - (w * obs).sum() / denom) if denom else float("nan")


def score_review(sheet_a: Path, sheet_b: Path, key_path: Path) -> dict:
    a = pd.read_csv(sheet_a, encoding="utf-8-sig").set_index("item_id")
    b = pd.read_csv(sheet_b, encoding="utf-8-sig").set_index("item_id")
    key = pd.read_csv(key_path).set_index("item_id")
    out = {"agreement": {}, "by_system": {}}
    merged = key.copy()
    for col in RATING_COLUMNS:
        both = pd.concat([a[col], b[col]], axis=1, keys=["a", "b"]).dropna().astype(int)
        out["agreement"][col] = {"n": len(both),
                                 "quadratic_kappa": weighted_kappa(both.a.to_numpy(), both.b.to_numpy(),
                                                                   [1, 2, 3, 4, 5])}
        merged[col] = (a[col].astype(float) + b[col].astype(float)) / 2
    for system, grp in merged.groupby("system"):
        out["by_system"][system] = {col: {"mean": float(grp[col].mean()), "n": int(grp[col].notna().sum())}
                                    for col in RATING_COLUMNS}
    return out
