"""Score run directories against the time-split ground truth and write a report."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.groundtruth import GroundTruth
from ..utils import read_jsonl
from .metrics import Faithfulness, bootstrap_ci, paired_bootstrap, per_disease_ranking

log = logging.getLogger(__name__)


def load_run(run_dir: Path) -> tuple[str, dict, dict[int, dict]]:
    run_dir = Path(run_dir)
    meta = {}
    if (run_dir / "meta.json").exists():
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    records = {int(r["disease_idx"]): r for r in read_jsonl(run_dir / "rankings.jsonl")}
    name = meta.get("run_name") or run_dir.name
    return name, meta, records


def load_probe_mask(path: Path) -> set[tuple[int, int]]:
    """Pairs the probed model does NOT already know about (kept in the masked evaluation)."""
    keep = set()
    for r in read_jsonl(Path(path)):
        if r.get("kind") == "target" and not r.get("model_knows", False):
            keep.add((int(r["drug_idx"]), int(r["disease_idx"])))
    return keep


def evaluate_runs(gt: GroundTruth, run_dirs: list[Path], ks: list[int], n_boot: int = 1000,
                  seed: int = 13, min_tier: str = "phase2", probe_keep: set | None = None,
                  reference: str | None = None, limit: int | None = None) -> dict:
    positives = gt.positives_by_disease(min_tier)
    if probe_keep is not None:
        positives = {d: {x for x in s if (x, d) in probe_keep} for d, s in positives.items()}
    known = gt.known_by_disease()
    ids = [int(d) for d in gt.eval_diseases["disease_idx"]]
    if limit:
        # Same prefix the agent's --limit takes, so a pilot is scored only on what it ran.
        ids = ids[:limit]
    diseases = [d for d in ids if positives.get(d)]
    summary, per_rows, metric_arrays = [], [], {}
    for run_dir in run_dirs:
        name, meta, recs = load_run(run_dir)
        rows, faith = [], Faithfulness()
        tokens_in = tokens_out = 0
        for d in diseases:
            rec = recs.get(d)
            ranked = [r["drug_idx"] for r in rec["ranked"]] if rec else []
            m = per_disease_ranking(ranked, positives[d], known.get(d, set()), ks)
            m.update(disease_idx=d, system=name, answered=float(bool(rec and ranked)))
            rows.append(m)
            if rec:
                for p in rec.get("proposals", []):
                    faith.add_proposal(p.get("verification", {}))
                u = rec.get("usage", {})
                tokens_in += u.get("input_tokens", 0)
                tokens_out += u.get("output_tokens", 0)
        df = pd.DataFrame(rows)
        per_rows.append(df)
        metric_arrays[name] = df
        out = {"system": name, "model": meta.get("model", ""),
               "knowledge_cutoff": meta.get("knowledge_cutoff", ""), "n_diseases": len(df),
               "coverage": float(df["answered"].mean()) if len(df) else 0.0}
        for col in ["mrr"] + [f"hit@{k}" for k in ks] + [f"recall@{k}" for k in ks]:
            mean, lo, hi = bootstrap_ci(df[col].to_numpy(), n_boot, seed)
            out[col], out[f"{col}_lo"], out[f"{col}_hi"] = mean, lo, hi
        out.update(faith.rates())
        price_in = float(meta.get("price_in_per_mtok", 0) or 0)
        price_out = float(meta.get("price_out_per_mtok", 0) or 0)
        cost = (tokens_in * price_in + tokens_out * price_out) / 1e6
        kmax = max(ks)
        hits = float(df[f"recall@{kmax}"].mul([len(positives[d]) for d in df.disease_idx]).sum()) if len(df) else 0.0
        out.update(input_tokens=tokens_in, output_tokens=tokens_out, cost_usd=cost,
                   correct_hits=hits,
                   tokens_per_correct_hit=(tokens_in + tokens_out) / hits if hits else float("nan"),
                   cost_per_correct_hit=cost / hits if hits else float("nan"))
        summary.append(out)
    comparisons = []
    if reference and reference in metric_arrays:
        ref = metric_arrays[reference]
        for name, df in metric_arrays.items():
            if name == reference:
                continue
            for col in ["mrr", f"hit@{max(ks)}"]:
                comparisons.append({"system": name, "vs": reference, "metric": col,
                                    **paired_bootstrap(ref[col].to_numpy(), df[col].to_numpy(),
                                                       n_boot, seed)})
    return {"summary": pd.DataFrame(summary),
            "per_disease": pd.concat(per_rows) if per_rows else pd.DataFrame(),
            "comparisons": pd.DataFrame(comparisons), "n_diseases": len(diseases),
            "n_positive_pairs": int(sum(len(positives[d]) for d in diseases))}


def _fmt(v, digits=3):
    if isinstance(v, float):
        return "–" if np.isnan(v) else f"{v:.{digits}f}"
    return str(v)


def write_report(results: dict, out_dir: Path, ks: list[int], title: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s = results["summary"]
    s.to_csv(out_dir / "summary.csv", index=False)
    results["per_disease"].to_csv(out_dir / "per_disease.csv", index=False)
    if len(results["comparisons"]):
        results["comparisons"].to_csv(out_dir / "comparisons.csv", index=False)
    kmax = max(ks)
    lines = [f"# {title}", "",
             f"Diseases: {results['n_diseases']} · positive pairs: {results['n_positive_pairs']}",
             "", "## Ranking (filtered, 95% bootstrap CI)", "",
             "| System | Model (cutoff) | MRR | " + " | ".join(f"Hit@{k}" for k in ks) + " | Coverage |",
             "|---|---|---|" + "---|" * len(ks) + "---|"]
    for r in s.to_dict("records"):
        cells = [f"{_fmt(r['mrr'])} [{_fmt(r['mrr_lo'])}, {_fmt(r['mrr_hi'])}]"]
        cells += [f"{_fmt(r[f'hit@{k}'])} [{_fmt(r[f'hit@{k}_lo'])}, {_fmt(r[f'hit@{k}_hi'])}]" for k in ks]
        lines.append(f"| {r['system']} | {r['model']} ({r['knowledge_cutoff'] or '?'}) | "
                     + " | ".join(cells) + f" | {_fmt(r['coverage'], 2)} |")
    lines += ["", "## Mechanism faithfulness", "",
              "**Fabricated** = both endpoints resolve to real entities but no admissible "
              "evidence links them (unsupported), or the only link is the drug-disease label "
              "itself (circular): a checkable claim the evidence base contradicts. "
              "**Unresolvable** = the endpoint does not pin down one node of an acceptable type "
              "(ungrounded, or ambiguous across nodes) - usually a process written as prose, so "
              "the claim is not checkable rather than false. The two are reported apart, since "
              "merging them overstates fabrication.", "",
              "| System | Proposals | Verified | Fabricated | Unsupported | Circular | "
              "Unresolvable | Ungrounded | Ambiguous | Unverified | Invalid citations |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in s.to_dict("records"):
        if not r.get("proposals"):
            continue
        lines.append(f"| {r['system']} | {r['proposals']} | {_fmt(r['verified_rate'])} | "
                     f"**{_fmt(r['fabricated_edge_rate'])}** | {_fmt(r['unsupported_edge_rate'])} | "
                     f"{_fmt(r['circular_edge_rate'])} | {_fmt(r['unresolvable_edge_rate'])} | "
                     f"{_fmt(r['ungrounded_edge_rate'])} | {_fmt(r['ambiguous_edge_rate'])} | "
                     f"{_fmt(r['unverified_edge_rate'])} | {_fmt(r['invalid_citation_rate'])} |")
    lines += ["", f"## Cost (correct hits counted in top-{kmax})", "",
              "| System | Input tok | Output tok | USD | Tokens / correct hit | USD / correct hit |",
              "|---|---|---|---|---|---|"]
    for r in s.to_dict("records"):
        if not (r["input_tokens"] or r["output_tokens"]):
            continue
        lines.append(f"| {r['system']} | {r['input_tokens']:,} | {r['output_tokens']:,} | "
                     f"{_fmt(r['cost_usd'], 2)} | {_fmt(r['tokens_per_correct_hit'], 0)} | "
                     f"{_fmt(r['cost_per_correct_hit'], 3)} |")
    c = results["comparisons"]
    if len(c):
        lines += ["", "## Paired bootstrap vs reference", "",
                  "| System | Metric | Reference − system | 95% CI | p |", "|---|---|---|---|---|"]
        for r in c.to_dict("records"):
            lines.append(f"| {r['system']} | {r['metric']} | {_fmt(r['diff'])} | "
                         f"[{_fmt(r['lo'])}, {_fmt(r['hi'])}] | {_fmt(r['p'])} |")
    path = out_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
