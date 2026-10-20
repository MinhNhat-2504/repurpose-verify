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
                      seed: int = 13) -> list[dict]:
    rng = random.Random(seed)
    items = [{"drug_idx": int(r.drug_idx), "disease_idx": int(r.disease_idx), "kind": "target"}
             for r in gt.positives.itertuples()]
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


def summarize_probe(path: Path) -> dict:
    rows = list(read_jsonl(Path(path)))
    out: dict = {}
    for kind in ("target", "positive_control", "negative_control"):
        rs = [r for r in rows if r["kind"] == kind]
        if not rs:
            continue
        out[kind] = {"n": len(rs),
                     "said_yes": sum(r["answer"] == "yes" for r in rs) / len(rs),
                     "flagged_post_cutoff": sum(r["model_knows"] for r in rs) / len(rs)}
    return out


if __name__ == "__main__":  # pragma: no cover
    import sys

    print(json.dumps(summarize_probe(Path(sys.argv[1])), indent=2))
