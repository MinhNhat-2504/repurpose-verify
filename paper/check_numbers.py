"""Check that every headline number in draft.md matches the generated artefacts.

A paper that reports fabricated citations as a finding cannot carry numbers nobody re-derived.
Run this before every commit that touches the draft:

    python paper/check_numbers.py

Each claim names the file it comes from, so a mismatch says which to trust: if the draft is
stale, fix the draft; if a run was re-done, re-read the artefact and update the draft.
Exits non-zero on any mismatch or any claim whose supporting file is missing.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "paper" / "draft.md"
GT_REPORT = ROOT / "data" / "processed" / "groundtruth" / "report.json"
PROBE = ROOT / "runs" / "probes" / "qwen25-7b.jsonl"
SUMMARIES = {
    "baselines": ROOT / "runs" / "report_baselines" / "summary.csv",
    "full": ROOT / "runs" / "report_full" / "summary.csv",
    "masked": ROOT / "runs" / "report_masked" / "summary.csv",
    "phase3": ROOT / "runs" / "report_full_phase3" / "summary.csv",
}
COMPARISONS = ROOT / "runs" / "report_full" / "comparisons.csv"
STRICT = ROOT / "runs" / "report_strict" / "summary.csv"
EXTERNAL = [ROOT / "runs" / "report_external" / "moa.json",
            ROOT / "runs" / "report_external" / "literature.json"]
SYSTEMS = ("qwen25-7b__closed_book", "qwen25-7b__rag_verify-none",
           "qwen25-7b__rag_verify-revise")


def _fmt(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def collect_claims() -> list[tuple[str, str, str]]:
    """(label, expected string, source) triples that must appear verbatim in the draft."""
    claims: list[tuple[str, str, str]] = []

    gt = json.loads(GT_REPORT.read_text(encoding="utf-8"))
    src = GT_REPORT.name
    m = gt["mapping"]
    drug_cov = m["trials_with_drug"] / (m["trials_with_drug"] + m["trials_without_drug"])
    dis_cov = m["trials_with_disease"] / (m["trials_with_disease"] + m["trials_without_disease"])
    discarded = 1 - gt["n_positives_novel"] / gt["n_post_cutoff_pairs"]
    claims += [
        ("pre-cutoff trials", f"{gt['n_pre_trials']:,}", src),
        ("post-cutoff trials", f"{gt['n_post_trials']:,}", src),
        ("kg known pairs", f"{gt['n_kg_known_pairs']:,}", src),
        ("pre-cutoff trial pairs", f"{gt['n_pre_cutoff_trial_pairs']:,}", src),
        ("post-cutoff pairs", f"{gt['n_post_cutoff_pairs']:,}", src),
        ("novel positives", f"{gt['n_positives_novel']:,}", src),
        ("phase 2 positives", f"{gt['positives_by_tier']['phase2']:,}", src),
        ("phase 3 positives", str(gt["positives_by_tier"]["phase3"]), src),
        ("phase 4 positives", str(gt["positives_by_tier"]["phase4"]), src),
        ("diseases with a positive", f"{gt['n_diseases_with_positive']:,}", src),
        ("diseases after gene filter", str(gt["n_diseases_after_gene_filter"]), src),
        ("diseases dropped, no genes", str(gt["n_diseases_dropped_no_genes"]), src),
        ("eval diseases", str(gt["n_eval_diseases"]), src),
        ("positives in eval", str(gt["ceiling"]["positives_in_eval"]), src),
        ("mechanism-reachable", str(gt["ceiling"]["mechanism_reachable"]), src),
        ("reachable rate", f"{gt['ceiling']['mechanism_reachable_rate'] * 100:.1f}%", src),
        ("targetless positive drugs", str(gt["ceiling"]["drug_has_no_target"]), src),
        ("drug mapping coverage", f"{drug_cov * 100:.1f}%", src),
        ("disease mapping coverage", f"{dis_cov * 100:.1f}%", src),
        ("share of post-cutoff pairs discarded", f"{discarded * 100:.0f}%", src),
    ]
    if gt["leakage_check_positives_in_kg"] != 0:
        raise SystemExit("leakage audit is non-zero; the draft's claim of 0 is wrong")

    base = pd.read_csv(SUMMARIES["baselines"]).set_index("system")
    for system in ("popularity", "proximity", "complex"):
        row = base.loc[system]
        claims.append((f"{system} MRR", _fmt(row["mrr"], 3), "report_baselines/summary.csv"))
        for k in (1, 5, 10, 20):
            claims.append((f"{system} Hit@{k}", _fmt(row[f"hit@{k}"], 3),
                           "report_baselines/summary.csv"))

    full = pd.read_csv(SUMMARIES["full"]).set_index("system")
    src = "report_full/summary.csv"
    for system in SYSTEMS:
        row = full.loc[system]
        claims += [
            (f"{system} proposals", f"{int(row['proposals']):,}", src),
            (f"{system} verified", _fmt(row["verified_rate"], 3), src),
            (f"{system} fabricated", _fmt(row["fabricated_edge_rate"], 3), src),
            (f"{system} unresolvable", _fmt(row["unresolvable_edge_rate"], 3), src),
            (f"{system} malformed", _fmt(row["malformed_path_rate"], 3), src),
            (f"{system} MRR", _fmt(row["mrr"], 3), src),
            (f"{system} Hit@20", _fmt(row["hit@20"], 3), src),
            (f"{system} coverage", _fmt(row["coverage"], 3), src),
            (f"{system} correct hits", str(int(row["correct_hits"])), src),
            (f"{system} input tokens", f"{int(row['input_tokens']):,}", src),
            (f"{system} output tokens", f"{int(row['output_tokens']):,}", src),
            (f"{system} tokens per hit", f"{int(round(row['tokens_per_correct_hit'])):,}", src),
        ]
    rev, nov, cb = (full.loc[SYSTEMS[2]], full.loc[SYSTEMS[1]], full.loc[SYSTEMS[0]])
    claims += [
        ("total proposals", f"{int(full.loc[list(SYSTEMS), 'proposals'].sum()):,}", src),
        ("fabricated reduction factor",
         f"{nov['fabricated_edge_rate'] / rev['fabricated_edge_rate']:.1f}-fold", src),
        ("verification token overhead",
         f"{(rev['input_tokens'] / nov['input_tokens'] - 1) * 100:.0f}%", src),
        ("tool-call input blow-up",
         f"{nov['input_tokens'] / cb['input_tokens']:.0f} times", src),
        ("closed book cost advantage per hit",
         f"{rev['tokens_per_correct_hit'] / cb['tokens_per_correct_hit']:.0f} times", src),
    ]

    cmp_df = pd.read_csv(COMPARISONS)
    pairs = {(r["system"], r["metric"]): r for _, r in cmp_df.iterrows()}
    nov_mrr = pairs[(SYSTEMS[1], "mrr")]
    pop_mrr = pairs[("popularity", "mrr")]
    pop_h20 = pairs[("popularity", "hit@20")]
    cb_mrr = pairs[(SYSTEMS[0], "mrr")]
    src = "report_full/comparisons.csv"
    claims += [
        ("verification effect on MRR", _fmt(nov_mrr["diff"], 4), src),
        ("verification p-value", _fmt(nov_mrr["p"], 2), src),
        ("popularity vs verified, MRR p", _fmt(pop_mrr["p"], 3), src),
        ("popularity vs verified, Hit@20 p", _fmt(pop_h20["p"], 3), src),
        ("closed book vs verified, MRR p", _fmt(cb_mrr["p"], 2), src),
    ]

    p3 = pd.read_csv(SUMMARIES["phase3"]).set_index("system")
    claims += [
        ("phase3 diseases", str(int(p3["n_diseases"].iloc[0])), "report_full_phase3/summary.csv"),
        ("phase3 popularity hits", str(int(p3.loc["popularity", "correct_hits"])),
         "report_full_phase3/summary.csv"),
    ]

    masked = pd.read_csv(SUMMARIES["masked"]).set_index("system")
    worst = max(abs(masked.loc[k, "hit@20"] - base.loc[k, "hit@20"]) for k in base.index)
    claims.append(("largest Hit@20 shift under masking", _fmt(worst, 3),
                   "report_masked/summary.csv vs report_baselines/summary.csv"))
    claims.append(("masked diseases", str(int(masked["n_diseases"].iloc[0])),
                   "report_masked/summary.csv"))

    # Section 7.6: role sensitivity of "verified", re-scored without any model inference.
    # The default-verifier figures are read from report_full rather than from report_strict,
    # which may hold only the re-scored runs: requiring both in one file made this whole block
    # silently emit nothing when the strict report was regenerated with the _strict runs alone.
    if STRICT.exists():
        strict = pd.read_csv(STRICT).set_index("system")
        src = "report_strict/summary.csv + report_full/summary.csv"
        for base in (SYSTEMS[2], SYSTEMS[1]):
            name = f"{base}_strict"
            if name not in strict.index:
                continue
            defaults = strict if base in strict.index else full
            claims += [
                (f"{base} verified, default", _fmt(defaults.loc[base, "verified_rate"], 3), src),
                (f"{base} verified, target-only", _fmt(strict.loc[name, "verified_rate"], 3), src),
                (f"{base} malformed, default",
                 _fmt(defaults.loc[base, "malformed_path_rate"], 3), src),
                (f"{base} malformed, target-only",
                 _fmt(strict.loc[name, "malformed_path_rate"], 3), src),
            ]

    # Section 7.6: independent confirmation against ChEMBL mechanism of action.
    ext: dict = {}
    for path in EXTERNAL:
        if path.exists():
            for system, res in json.loads(path.read_text(encoding="utf-8")).items():
                ext.setdefault(system, {}).update({k: v for k, v in res.items() if v})
    if ext:
        src = "report_external/*.json"
        # The draft quotes the full ChEMBL breakdown for the primary system and only the
        # contrasting figures for the others. The checker's job is that the numbers the draft
        # states are right - not that every value in an artefact appears in the text.
        for system, res in ext.items():
            m = res.get("moa")
            if not m:
                continue
            primary = system == SYSTEMS[2]
            claims += [
                (f"{system} MoA edges checked", f"{m['edges_checked']:,}", src),
                (f"{system} confirmation rate", f"{m['independent_confirmation_rate'] * 100:.1f}%",
                 src),
            ] if primary else [
                (f"{system} MoA edges checked", f"{m['edges_checked']:,}", src),
            ]
            if primary:
                claims += [
                    (f"{system} MoA drugs", f"{m['drugs_checked']:,}", src),
                    (f"{system} ChEMBL coverage", f"{m['drug_chembl_coverage'] * 100:.1f}%", src),
                    (f"{system} MoA coverage", f"{m['drug_moa_coverage'] * 100:.1f}%", src),
                    (f"{system} edges with an opinion", f"{m['edges_with_opinion']:,}", src),
                ]
                for role, n in (m.get("edges_by_role") or {}).items():
                    claims.append((f"{system} {role} edges", f"{n:,}", src))
            for role, rate in (m.get("confirmation_by_role") or {}).items():
                if rate == rate and (primary or role == "target"):
                    claims.append((f"{system} {role} confirmed", _fmt(rate, 3), src))
            lit = res.get("literature")
            if lit:
                claims += [
                    (f"{system} co-mention rate", _fmt(lit["system_comention_rate"], 3), src),
                    (f"{system} random control", _fmt(lit["control_comention_rate"], 3), src),
                    (f"{system} co-mention pairs", f"{lit['system_pairs']:,}", src),
                ]
                if "matched_control_comention_rate" in lit:
                    claims += [
                        (f"{system} matched control",
                         _fmt(lit["matched_control_comention_rate"], 3), src),
                        (f"{system} lift vs matched",
                         f"+{lit['lift_over_matched_control']:.3f}", src),
                    ]

    rows = [json.loads(line) for line in PROBE.read_text(encoding="utf-8").splitlines() if line]
    kinds = {k: [r for r in rows if r["kind"] == k] for k in
             ("target", "positive_control", "negative_control")}
    src = PROBE.name
    for kind, rs in kinds.items():
        claims += [
            (f"probe {kind} n", str(len(rs)), src),
            (f"probe {kind} said yes", _fmt(sum(r["answer"] == "yes" for r in rs) / len(rs), 3), src),
            (f"probe {kind} flagged", _fmt(sum(r["model_knows"] for r in rs) / len(rs), 3), src),
        ]
    yes = [r for r in kinds["target"] if r["answer"] == "yes"]
    gap = (sum(r["answer"] == "yes" for r in kinds["positive_control"]) / len(kinds["positive_control"])
           - sum(r["answer"] == "yes" for r in kinds["negative_control"]) / len(kinds["negative_control"]))
    claims += [
        ("probe items", str(len(rows)), src),
        ("probe sensitivity gap", _fmt(gap, 3), src),
        ("positives claimed yes", str(len(yes)), src),
        ("claimed with pre-cutoff year", str(sum(1 for r in yes if r["year"] and r["year"] <= 2021)), src),
        ("claimed with no year", str(sum(1 for r in yes if not r["year"])), src),
        ("positive-control recall",
         _fmt(sum(r["answer"] == "yes" for r in kinds["positive_control"])
              / len(kinds["positive_control"]) * 100, 1) + "%", src),
    ]
    if any(r["year"] and r["year"] > 2021 for r in yes):
        raise SystemExit("a positive was claimed with a post-cutoff year; the draft says none was")
    return claims


def _artefact_present(name: str) -> bool:
    """Does a report directory of this name hold anything the checker could read?"""
    base = ROOT / "runs"
    if name == "report_external":
        return any(p.exists() for p in EXTERNAL)
    return (base / name / "summary.csv").exists()


def main() -> int:
    required = [DRAFT, GT_REPORT, PROBE, COMPARISONS, *SUMMARIES.values()]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("Cannot verify the draft - these artefacts are missing:")
        for p in missing:
            print(f"  {p.relative_to(ROOT)}")
        return 2
    text = DRAFT.read_text(encoding="utf-8")
    # Numbers are written with thousands separators in prose and without in tables.
    haystack = text + "\n" + re.sub(r"(?<=\d),(?=\d)", "", text)
    claims = collect_claims()
    failures = [(label, expected, source) for label, expected, source in claims
                if expected not in haystack]
    print(f"{len(claims) - len(failures)} claims verified against the artefacts.")

    # A block that stops producing claims is worse than one that fails: the checker keeps
    # reporting success while guarding less. Each artefact that exists is expected to
    # contribute, so silence from one is reported loudly.
    contributing = {source for _, _, source in claims}
    silent = [name for name, token in (("report_strict", "report_strict"),
                                       ("report_external", "report_external"),
                                       ("report_full_phase3", "report_full_phase3"),
                                       ("report_masked", "report_masked"))
              if _artefact_present(name) and not any(token in s for s in contributing)]
    if silent:
        print("\nArtefacts present but contributing no claims - the guard has weakened:")
        for name in silent:
            print(f"  {name}")
        return 1

    if failures:
        print(f"\n{len(failures)} claim(s) in draft.md are not supported:")
        for label, expected, source in failures:
            print(f"  {label}: expected '{expected}' (from {source})")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
