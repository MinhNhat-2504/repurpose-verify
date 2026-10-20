"""Weight-level leakage probe.

An LLM may already "know" post-cutoff trials and approvals from pre-training. For every
ground-truth pair we ask the model, closed-book, whether the drug has been trialled/approved for
the disease and since when. Pairs it claims to know with a post-cutoff year (or without a year)
are flagged; the masked evaluation keeps only the unflagged pairs. Positive controls (pairs
known at the cutoff) and negative controls (random unrelated pairs) calibrate the probe.
"""

from __future__ import annotations

import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..agent.llm import LLMClient
from ..data.groundtruth import GroundTruth
from ..data.kg import DRUG, KnowledgeGraph
from ..utils import append_jsonl, extract_json, read_jsonl

PROBE_SYSTEM = "You are a pharmacology expert. Answer from your own knowledge only, without tools."
PROBE_USER = """Has {drug} been evaluated in a phase 2 or later clinical trial, or approved, for \
{disease}? If yes, give the earliest year you know of.
Reply with JSON only: {{"answer": "yes" | "no" | "unsure", "year": <number or null>}}"""


def build_probe_items(gt: GroundTruth, kg: KnowledgeGraph, n_controls: int = 200,
                      seed: int = 13, eval_only: bool = True) -> list[dict]:
    """Probe items: the positives, plus controls that calibrate the model's "yes".

    Only positives inside the evaluation diseases are probed by default - the masked
    evaluation can only use those, and probing all of them costs hours of inference for
    answers nothing reads.
    """
    rng = random.Random(seed)
    positives = gt.positives
    if eval_only:
        keep = set(int(d) for d in gt.eval_diseases["disease_idx"])
        positives = positives[positives.disease_idx.isin(keep)]
    items = [{"drug_idx": int(r.drug_idx), "disease_idx": int(r.disease_idx), "kind": "target"}
             for r in positives.itertuples()]
    known = [tuple(map(int, x)) for x in gt.known[["drug_idx", "disease_idx"]].itertuples(index=False)]
    for d, s in rng.sample(known, min(n_controls, len(known))):
        items.append({"drug_idx": d, "disease_idx": s, "kind": "positive_control"})
    known_set = set(known) | {(i["drug_idx"], i["disease_idx"]) for i in items}
    drugs = [int(i) for i in kg.nodes_of_type(DRUG)]
    diseases = sorted({i["disease_idx"] for i in items})
    added = 0
    while added < n_controls and drugs and diseases:
        pair = (rng.choice(drugs), rng.choice(diseases))
        if pair not in known_set:
            known_set.add(pair)
            items.append({"drug_idx": pair[0], "disease_idx": pair[1], "kind": "negative_control"})
            added += 1
    # Interleave the kinds: the controls are what show the probe can say "yes" at all, so a
    # truncated run (--limit) must still contain them rather than only targets.
    rng.shuffle(items)
    return items


def run_probe(llm: LLMClient, kg: KnowledgeGraph, items: list[dict], cutoff_year: int,
              out_path: Path, workers: int = 4) -> Path:
    out_path = Path(out_path)
    done = set()
    if out_path.exists():
        done = {(r["drug_idx"], r["disease_idx"], r["kind"]) for r in read_jsonl(out_path)}
    todo = [i for i in items if (i["drug_idx"], i["disease_idx"], i["kind"]) not in done]
    lock = threading.Lock()

    def ask(item: dict) -> None:
        prompt = PROBE_USER.format(drug=kg.names[item["drug_idx"]],
                                   disease=kg.names[item["disease_idx"]])
        try:
            turn = llm.chat(PROBE_SYSTEM, [{"role": "user", "content": prompt}])
            text = turn.text
            try:
                obj = extract_json(text)
            except ValueError:
                obj = {}
            answer = str(obj.get("answer", "unsure")).lower() if isinstance(obj, dict) else "unsure"
            year = obj.get("year") if isinstance(obj, dict) else None
            year = int(year) if isinstance(year, (int, float)) or (isinstance(year, str) and year.isdigit()) else None
            usage = turn.usage.to_dict()
        except Exception as err:  # record and continue
            text, answer, year, usage = f"ERROR {err}", "error", None, {}
        knows = answer == "yes" and (year is None or year > cutoff_year)
        append_jsonl(out_path, {**item, "drug": kg.names[item["drug_idx"]],
                                "disease": kg.names[item["disease_idx"]], "answer": answer,
                                "year": year, "model_knows": knows, "raw": text[:500],
                                "usage": usage}, lock)

    if workers <= 1:
        for it in todo:
            ask(it)
    else:
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(ask, todo))
    return out_path


def load_probe_mask_items(path: Path) -> set[tuple[int, int]]:
    """Positives the probed model does NOT claim to know: what the masked evaluation keeps."""
    return {(int(r["drug_idx"]), int(r["disease_idx"])) for r in read_jsonl(Path(path))
            if r.get("kind") == "target" and not r.get("model_knows", False)}


def summarize_probe(path: Path) -> dict:
    rows = list(read_jsonl(Path(path)))
    out: dict = {}
    for kind in ("target", "positive_control", "negative_control"):
        rs = [r for r in rows if r["kind"] == kind]
        if not rs:
            continue
        out[kind] = {"n": len(rs),
                     "said_yes": sum(r["answer"] == "yes" for r in rs) / len(rs),
                     "unsure": sum(r["answer"] == "unsure" for r in rs) / len(rs),
                     "errors": sum(r["answer"] == "error" for r in rs),
                     "flagged_post_cutoff": sum(r["model_knows"] for r in rs) / len(rs)}
    pos, neg = out.get("positive_control"), out.get("negative_control")
    if pos and neg:
        # Without this gap the probe has no discriminating power and a "no leakage found"
        # result would only mean the model declines to answer.
        out["sensitivity_check"] = {
            "positive_minus_negative_yes_rate": round(pos["said_yes"] - neg["said_yes"], 4),
            "usable": bool(pos["said_yes"] - neg["said_yes"] >= 0.2),
            "note": ("A model that answers 'no' to everything would show zero leakage; the gap "
                     "between pairs known at the cutoff and random pairs is what makes the "
                     "masked evaluation meaningful."),
        }
    return out


def write_probe_report(path: Path, cutoff_year: int, out_path: Path | None = None) -> Path:
    """Markdown table of the probe, for the paper's leakage section."""
    s = summarize_probe(path)
    rows = list(read_jsonl(Path(path)))
    flagged = [r for r in rows if r["kind"] == "target" and r["model_knows"]]
    lines = [f"# Weight-level leakage probe (knowledge cutoff {cutoff_year})", "",
             "The model is asked, closed book, whether each pair has reached a phase 2+ trial "
             "or approval, and since when. A pair it claims with a post-cutoff year (or with no "
             "year) is flagged; `evaluate --probe` then scores only the unflagged pairs.", "",
             "| Item | n | Said yes | Unsure | Flagged as post-cutoff knowledge |",
             "|---|---|---|---|---|"]
    labels = {"target": "ground-truth positives", "positive_control": "pairs known at the cutoff",
              "negative_control": "random unrelated pairs"}
    for kind, label in labels.items():
        r = s.get(kind)
        if r:
            lines.append(f"| {label} | {r['n']} | {r['said_yes']:.3f} | {r['unsure']:.3f} | "
                         f"{r['flagged_post_cutoff']:.3f} |")
    check = s.get("sensitivity_check")
    if check:
        verdict = "usable" if check["usable"] else "NOT usable - the probe cannot say yes"
        lines += ["", f"**Sensitivity:** yes-rate gap between known pairs and random pairs = "
                      f"{check['positive_minus_negative_yes_rate']:.3f} ({verdict}). "
                      f"{check['note']}"]
    tgt, pos, neg = s.get("target"), s.get("positive_control"), s.get("negative_control")
    if tgt and pos and neg:
        # Controls were already known at the cutoff, so any post-cutoff year they attract is the
        # probe mistaking confabulated dates for knowledge. That is the floor to compare against.
        false_positive = max(pos["flagged_post_cutoff"], neg["flagged_post_cutoff"])
        lines += ["", "**How to read the flag rate.** Controls are pairs that were already known "
                      f"at the cutoff or unrelated, so their flag rate - {false_positive:.3f} - is "
                      "the probe mistaking a confabulated date for knowledge, not leakage.",
                  "", f"The positives' flag rate is {tgt['flagged_post_cutoff']:.3f}, "
                      + ("at or below that floor, so this probe finds no leakage signal above "
                         "its own noise." if tgt["flagged_post_cutoff"] <= false_positive else
                         "above that floor; the excess is the part worth treating as leakage."),
                  "", f"The probe is a weak detector either way: it recalls only "
                      f"{pos['said_yes']:.1%} of pairs that were genuinely known at the cutoff, so "
                      "the flag rate is a lower bound on leakage and the masked evaluation is a "
                      "sanity check, not a guarantee."]
    if flagged:
        lines += ["", f"## {len(flagged)} flagged pairs (excluded from the masked evaluation)", "",
                  "| Drug | Disease | Claimed year |", "|---|---|---|"]
        for r in flagged[:50]:
            lines.append(f"| {r['drug']} | {r['disease']} | {r['year'] if r['year'] else '-'} |")
        if len(flagged) > 50:
            lines.append(f"| ... | ... | {len(flagged) - 50} more |")
    else:
        lines += ["", "No ground-truth positive was claimed with post-cutoff knowledge, so the "
                      "masked evaluation keeps every pair."]
    out_path = Path(out_path) if out_path else Path(path).with_suffix(".md")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


if __name__ == "__main__":  # pragma: no cover
    import sys

    print(json.dumps(summarize_probe(Path(sys.argv[1])), indent=2))
