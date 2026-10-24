"""Measure what the entity-grounding correction of Section 7.7 did to the ground truth.

    python scripts/measure_grounding_correction.py

Builds the ground truth twice from the same inputs - once with the current grounding rules and
once with the rules as they stood before the correction, in which every bracketed fragment was
indexed as a name of its own - and writes the difference.

This is a measurement script, not part of the pipeline: the old rule is reinstated here by
monkey-patching so that the number reported in the paper can be re-derived, not so that it stays
available as an option. Outputs:

    data/processed/groundtruth/grounding_correction.csv    every pair added or removed
    data/processed/groundtruth/grounding_correction.json   the counts quoted in Section 7.7
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from repurpose_verify.config import load_config  # noqa: E402
from repurpose_verify.context import Context  # noqa: E402
from repurpose_verify.data import normalize  # noqa: E402

OUT_CSV = ROOT / "data" / "processed" / "groundtruth" / "grounding_correction.csv"
OUT_JSON = ROOT / "data" / "processed" / "groundtruth" / "grounding_correction.json"


def build(cfg, kg, pre, post) -> pd.DataFrame:
    from repurpose_verify.data.groundtruth import build_groundtruth

    g = cfg["groundtruth"]
    gt, _ = build_groundtruth(
        kg, pre, post, cfg.cutoff, cfg.eval_end, single_drug=g["single_drug_trials"],
        post_phases=g["post_phases"], manual_path=cfg.path("groundtruth", "manual_approvals"),
        fuzzy=g.get("fuzzy_threshold", 0), n_eval=g.get("n_eval_diseases", 0),
        min_pos=g.get("min_positives_per_disease", 1),
        min_disease_genes=g.get("min_disease_genes", 0),
        eval_split_path=cfg.path("groundtruth", "eval_split"), seed=g.get("seed", 13))
    return gt.positives


def main() -> None:
    from repurpose_verify.data.ctgov import ClinicalTrialsClient

    cfg = load_config(str(ROOT / "configs" / "default.toml"))
    ctx = Context(cfg)
    ct = ClinicalTrialsClient(ctx.http("ctgov", 0.2))
    pre = list(ct.iter_years(2000, cfg.cutoff))
    post = list(ct.iter_years(cfg.cutoff.year, cfg.eval_end, cfg["groundtruth"]["post_phases"],
                              first_date=date.fromordinal(cfg.cutoff.toordinal() + 1)))

    now = build(cfg, ctx.kg, pre, post)

    # The rule as it stood: any bracketed fragment is a lookup key, however short and whether or
    # not it reads as an acronym. The graph's name index is rebuilt under it.
    normalize.MIN_ABBREVIATION_LEN = 0
    normalize.is_acronym = lambda text: True
    kg_old = Context(cfg).kg
    before = build(cfg, kg_old, pre, post)

    k_now = set(zip(now.drug_idx, now.disease_idx))
    k_before = set(zip(before.drug_idx, before.disease_idx))
    removed = before.set_index(["drug_idx", "disease_idx"]).loc[sorted(k_before - k_now)]
    added = now.set_index(["drug_idx", "disease_idx"]).loc[sorted(k_now - k_before)]
    rows = pd.concat([removed.assign(change="removed"), added.assign(change="added")])
    rows.reset_index().to_csv(OUT_CSV, index=False, encoding="utf-8")

    bracketed = removed.drug_name.str.contains(r"[\(\[]", regex=True)
    summary = {
        "positives_before": int(len(before)),
        "positives_after": int(len(now)),
        "removed": int(len(removed)),
        "added": int(len(added)),
        "removed_with_bracketed_drug_name": int(bracketed.sum()),
        "removed_drug_is_fdg": int((removed.drug_name == "Fludeoxyglucose (18F)").sum()),
        "removed_distinct_drugs": int(removed.drug_name.nunique()),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\n{OUT_CSV}\n{OUT_JSON}")


if __name__ == "__main__":
    main()
