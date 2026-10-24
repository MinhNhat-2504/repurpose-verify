"""Time-split ground truth: drug-disease pairs that first reach the clinic after the cutoff.

A pair (drug, disease) is a *positive* when
  * a phase 2+ interventional trial testing the drug for the disease started in (cutoff, eval_end],
    or a curated approval of that new indication falls in the same window, and
  * the pair is *not known* at the cutoff: no indication / off-label edge in the KG snapshot and
    no trial of the pair (any phase) that started on or before the cutoff.
Everything known at the cutoff is also written out so that rankings can be *filtered*
(known uses removed before computing metrics), as in TxGNN's disease-centric evaluation.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .ctgov import PHASE_RANK, Trial
from .kg import DISEASE, DRUG, KNOWN_USE_RELS, KnowledgeGraph
from .normalize import is_generic_intervention

log = logging.getLogger(__name__)

TIER_RANK = {"phase2": 2, "phase3": 3, "phase4": 4, "approved": 5}
_SPLIT = (" + ", "+", " / ", "/", " plus ", " and ", ";", " with ")


@dataclass
class PairEvidence:
    first_date: date
    max_phase: int = -1
    nct_ids: list[str] = field(default_factory=list)
    sources: set[str] = field(default_factory=set)

    def add(self, when: date, phase: int, ref: str, source: str) -> None:
        self.first_date = min(self.first_date, when)
        self.max_phase = max(self.max_phase, phase)
        if ref and ref not in self.nct_ids:
            self.nct_ids.append(ref)
        self.sources.add(source)


class TrialMapper:
    """Maps CT.gov intervention / condition mentions onto KG drug / disease nodes."""

    def __init__(self, kg: KnowledgeGraph, fuzzy_threshold: float = 0):
        self.kg = kg
        self.fuzzy = fuzzy_threshold
        self.stats: dict[str, int] = defaultdict(int)
        self._drug = lru_cache(maxsize=None)(self._resolve_drug)
        self._disease = lru_cache(maxsize=None)(self._resolve_disease)

    def _resolve_drug(self, mention: str) -> int | None:
        return self.kg.resolve(mention, DRUG, self.fuzzy)

    def _resolve_disease(self, mention: str) -> tuple[int, ...]:
        hit = self.kg.resolve(mention, DISEASE, self.fuzzy)
        return (hit,) if hit is not None else ()

    def drugs(self, trial: Trial) -> set[int]:
        found: set[int] = set()
        for iv in trial.interventions:
            if iv.get("type") not in ("DRUG", "BIOLOGICAL"):
                continue
            names = [iv.get("name", "")] + list(iv.get("otherNames", []) or [])
            for name in names:
                parts = [name]
                for sep in _SPLIT:
                    parts = [q for p in parts for q in p.split(sep)]
                for part in parts:
                    part = part.strip()
                    if len(part) < 3 or is_generic_intervention(part):
                        continue
                    hit = self._drug(part)
                    if hit is not None:
                        found.add(hit)
        for mesh in trial.intervention_mesh:
            term = mesh.get("term", "")
            if term and not is_generic_intervention(term):
                hit = self._drug(term)
                if hit is not None:
                    found.add(hit)
        self.stats["trials_with_drug" if found else "trials_without_drug"] += 1
        return found

    def diseases(self, trial: Trial) -> set[int]:
        found: set[int] = set()
        for mesh in trial.condition_mesh:
            found.update(self.kg.disease_for_mesh(mesh.get("id", "")))
        if not found:
            for cond in trial.conditions + [m.get("term", "") for m in trial.condition_mesh]:
                found.update(self._disease(cond))
        self.stats["trials_with_disease" if found else "trials_without_disease"] += 1
        return found


def collect_pairs(trials: Iterable[Trial], mapper: TrialMapper, single_drug: bool,
                  exclude_status: tuple[str, ...] = ("WITHDRAWN",)) -> dict[tuple[int, int], PairEvidence]:
    pairs: dict[tuple[int, int], PairEvidence] = {}
    for t in trials:
        if t.start_date is None or t.status in exclude_status:
            continue
        drugs = mapper.drugs(t)
        if not drugs or (single_drug and len(drugs) != 1):
            continue
        diseases = mapper.diseases(t)
        for d in drugs:
            for dis in diseases:
                ev = pairs.get((d, dis))
                if ev is None:
                    ev = pairs[(d, dis)] = PairEvidence(first_date=t.start_date)
                ev.add(t.start_date, t.max_phase, t.nct_id, "ctgov")
    return pairs


def load_manual_approvals(path: Path, kg: KnowledgeGraph, cutoff: date, eval_end: date,
                          fuzzy: float = 0) -> tuple[dict[tuple[int, int], PairEvidence], list[dict]]:
    """CSV columns: drug, disease, date (YYYY-MM-DD), source, [note]."""
    pairs: dict[tuple[int, int], PairEvidence] = {}
    unmapped: list[dict] = []
    if not Path(path).exists():
        return pairs, unmapped
    df = pd.read_csv(path, dtype=str).fillna("")
    for row in df.to_dict("records"):
        if not row.get("date"):
            continue
        when = date.fromisoformat(row["date"].strip())
        if not (cutoff < when <= eval_end):
            continue
        drug = kg.resolve(row["drug"], DRUG, fuzzy)
        dis = kg.resolve(row["disease"], DISEASE, fuzzy)
        if drug is None or dis is None:
            unmapped.append(row)
            continue
        ev = pairs.setdefault((drug, dis), PairEvidence(first_date=when))
        ev.add(when, 5, row.get("source", ""), "approval")
    return pairs, unmapped


def tier_of(ev: PairEvidence) -> str:
    if "approval" in ev.sources:
        return "approved"
    return {4: "phase4", 3: "phase3"}.get(ev.max_phase, "phase2")


@dataclass
class GroundTruth:
    positives: pd.DataFrame        # drug_idx, disease_idx, tier, first_date, ...
    known: pd.DataFrame            # drug_idx, disease_idx, source
    eval_diseases: pd.DataFrame    # disease_idx, disease_name, n_positives

    def save(self, out_dir: Path, report: dict) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.positives.to_csv(out_dir / "positives.csv", index=False)
        self.known.to_csv(out_dir / "known_pairs.csv.gz", index=False)
        self.eval_diseases.to_csv(out_dir / "eval_diseases.csv", index=False)
        (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str),
                                             encoding="utf-8")

    @classmethod
    def load(cls, out_dir: Path) -> "GroundTruth":
        out_dir = Path(out_dir)
        return cls(pd.read_csv(out_dir / "positives.csv"),
                   pd.read_csv(out_dir / "known_pairs.csv.gz"),
                   pd.read_csv(out_dir / "eval_diseases.csv"))

    def positives_by_disease(self, min_tier: str = "phase2") -> dict[int, set[int]]:
        floor = TIER_RANK[min_tier]
        out: dict[int, set[int]] = defaultdict(set)
        for d, dis, tier in self.positives[["drug_idx", "disease_idx", "tier"]].itertuples(index=False):
            if TIER_RANK[tier] >= floor:
                out[int(dis)].add(int(d))
        return out

    def known_by_disease(self) -> dict[int, set[int]]:
        out: dict[int, set[int]] = defaultdict(set)
        for d, dis in self.known[["drug_idx", "disease_idx"]].itertuples(index=False):
            out[int(dis)].add(int(d))
        return out


def audit_mapping(kg: KnowledgeGraph, trials: Iterable[Trial], mapper: TrialMapper,
                  n: int = 200, seed: int = 13) -> pd.DataFrame:
    """Sample trials and show how their interventions / conditions were mapped, for manual audit."""
    import random

    rng = random.Random(seed)
    pool = [t for t in trials if t.start_date is not None]
    sample = rng.sample(pool, min(n, len(pool)))
    rows = []
    for t in sample:
        drugs = mapper.drugs(t)
        diseases = mapper.diseases(t)
        rows.append({
            "nct_id": t.nct_id, "link": f"https://clinicaltrials.gov/study/{t.nct_id}",
            "start_date": t.start_date.isoformat(), "phases": ";".join(t.phases),
            "interventions": " | ".join(i.get("name", "") for i in t.interventions
                                        if i.get("type") in ("DRUG", "BIOLOGICAL")),
            # The mapper reads otherNames too, so a sheet without them hides the evidence for
            # its own decisions: an arm called "Chemo-immunotherapy" whose other name is
            # "carboplatin/etoposide/durvalumab" maps to three drugs that otherwise look invented.
            "intervention_other_names": " | ".join(
                n for i in t.interventions if i.get("type") in ("DRUG", "BIOLOGICAL")
                for n in (i.get("otherNames") or [])),
            "intervention_mesh": " | ".join(m.get("term", "") for m in t.intervention_mesh),
            "mapped_drugs": " | ".join(f"{kg.names[d]} [kg:{d}]" for d in sorted(drugs)),
            "conditions": " | ".join(t.conditions),
            "condition_mesh": " | ".join(f"{m.get('term', '')} ({m.get('id', '')})"
                                         for m in t.condition_mesh),
            "mapped_diseases": " | ".join(f"{kg.names[d]} [kg:{d}]" for d in sorted(diseases)),
            "drug_correct_yn": "", "disease_correct_yn": "", "comment": "",
        })
    return pd.DataFrame(rows)


def score_mapping_audit(path: Path, n_boot: int = 2000, seed: int = 13) -> dict:
    """Mapping precision from a filled audit sheet, with a bootstrap interval.

    Only rows where a mapping was actually produced can be judged: a trial with no mapped drug
    is a recall miss, already reported as coverage, and scoring it as a precision error would
    count the same gap twice.
    """
    import numpy as np

    df = pd.read_csv(path, encoding="utf-8-sig")
    rng = np.random.default_rng(seed)
    out: dict = {"rows": len(df)}
    for kind, mapped_col, judged_col in (("drug", "mapped_drugs", "drug_correct_yn"),
                                         ("disease", "mapped_diseases", "disease_correct_yn")):
        if judged_col not in df.columns:
            continue
        mask = df[mapped_col].astype(str).str.strip().ne("") & df[mapped_col].notna()
        judged = df.loc[mask, judged_col].astype(str).str.strip().str.lower()
        verdicts = judged[judged.isin(["y", "yes", "n", "no", "1", "0", "true", "false"])]
        correct = verdicts.isin(["y", "yes", "1", "true"]).to_numpy()
        entry = {"mapped": int(mask.sum()), "judged": int(len(verdicts)),
                 "unjudged": int(mask.sum() - len(verdicts))}
        if len(correct):
            boots = correct[rng.integers(0, len(correct), (n_boot, len(correct)))].mean(axis=1)
            entry.update(precision=float(correct.mean()),
                         ci_low=float(np.quantile(boots, 0.025)),
                         ci_high=float(np.quantile(boots, 0.975)))
        out[kind] = entry
    return out


def stable_sample(keys: Iterable[int], n: int, seed: int) -> list[int]:
    """Positions of the `n` keys with the lowest hash, as a split that barely moves.

    `DataFrame.sample` draws by row position, so it depends on how many diseases are eligible
    and in what order. Re-deriving the ground truth after a mapping fix changed the pool by 18
    diseases out of 1149 and that reshuffled 129 of the 200 evaluation diseases - which would
    have invalidated every model run. Hashing each disease's own id instead makes membership a
    property of the disease, so a pool change only moves the split at its margin.
    """
    keys = list(keys)
    order = sorted(range(len(keys)),
                   key=lambda i: hashlib.sha1(f"{seed}:{keys[i]}".encode()).digest())
    return sorted(order[:n])


def load_eval_split(path: Path | None) -> tuple[set[int] | None, list[int]]:
    """The frozen evaluation split, or (None, []) when the benchmark has not fixed one yet."""
    if path is None or not Path(path).exists():
        return None, []
    df = pd.read_csv(path)
    return set(df.disease_idx.astype(int)), []


def build_groundtruth(kg: KnowledgeGraph, pre_trials: Iterable[Trial], post_trials: Iterable[Trial],
                      cutoff: date, eval_end: date, *, single_drug: bool = True,
                      post_phases: Iterable[str] = ("PHASE2", "PHASE3", "PHASE4"),
                      manual_path: Path | None = None, fuzzy: float = 0, n_eval: int = 0,
                      min_pos: int = 1, min_disease_genes: int = 0,
                      eval_split_path: Path | None = None,
                      seed: int = 13) -> tuple[GroundTruth, dict]:
    mapper = TrialMapper(kg, fuzzy)
    min_phase = min(PHASE_RANK[p] for p in post_phases)

    # Known at the cutoff: KG uses + any trial of the pair that started on/before the cutoff.
    kg_known = kg.drug_disease_pairs(KNOWN_USE_RELS)
    pre_pairs = collect_pairs((t for t in pre_trials if t.start_date and t.start_date <= cutoff),
                              mapper, single_drug=False)
    post_pairs = collect_pairs(
        (t for t in post_trials if t.start_date and cutoff < t.start_date <= eval_end
         and t.max_phase >= min_phase), mapper, single_drug=single_drug)
    manual_pairs, unmapped_manual = (load_manual_approvals(manual_path, kg, cutoff, eval_end, fuzzy)
                                     if manual_path else ({}, []))
    for key, ev in manual_pairs.items():
        if key in post_pairs:
            post_pairs[key].add(ev.first_date, 5, ev.nct_ids[0] if ev.nct_ids else "", "approval")
        else:
            post_pairs[key] = ev

    known_keys = set(kg_known) | set(pre_pairs)
    rows = []
    for (drug, dis), ev in post_pairs.items():
        if (drug, dis) in known_keys:
            continue
        rows.append({"drug_idx": drug, "drug_name": kg.names[drug], "disease_idx": dis,
                     "disease_name": kg.names[dis], "tier": tier_of(ev),
                     "first_date": ev.first_date.isoformat(), "max_phase": ev.max_phase,
                     "n_trials": len([r for r in ev.nct_ids if r.startswith("NCT")]),
                     "refs": ";".join(ev.nct_ids[:20]), "sources": ";".join(sorted(ev.sources))})
    positives = pd.DataFrame(rows, columns=["drug_idx", "drug_name", "disease_idx", "disease_name",
                                            "tier", "first_date", "max_phase", "n_trials", "refs",
                                            "sources"])
    positives = positives.sort_values(["disease_idx", "first_date"]).reset_index(drop=True)
    # A positive is "mechanism-reachable" when the graph admits at least one drug -> gene ...
    # -> disease path for it. Positives that are not reachable bound what any verified system
    # can score, so the ceiling is reported next to the results.
    positives["n_targets"] = [len(kg.neighbors(int(d), ["drug_protein"]))
                              for d in positives.drug_idx]
    positives["mechanism_reachable"] = [
        bool(kg.mechanism_paths(int(d), int(s), limit=1))
        for d, s in zip(positives.drug_idx, positives.disease_idx)]

    known_rows = [{"drug_idx": d, "disease_idx": s, "source": "kg"} for d, s in kg_known]
    known_rows += [{"drug_idx": d, "disease_idx": s, "source": "pre_cutoff_trial"}
                   for (d, s) in pre_pairs if (d, s) not in kg_known]
    known = pd.DataFrame(known_rows, columns=["drug_idx", "disease_idx", "source"])

    counts = positives.groupby("disease_idx").size()
    eligible = counts[counts >= min_pos]
    ev_df = pd.DataFrame({"disease_idx": eligible.index.astype(int),
                          "disease_name": [kg.names[i] for i in eligible.index],
                          "n_positives": eligible.values})
    ev_df["n_genes"] = [len(kg.neighbors(int(i), ["disease_protein"])) for i in ev_df.disease_idx]
    n_eligible = len(ev_df)
    if min_disease_genes > 0:
        ev_df = ev_df[ev_df.n_genes >= min_disease_genes]
    n_with_genes = len(ev_df)
    pinned, pin_dropped = load_eval_split(eval_split_path)
    if pinned is not None:
        ev_df = ev_df[ev_df.disease_idx.isin(pinned)]
        pin_dropped = sorted(pinned - set(ev_df.disease_idx))
        if pin_dropped:
            # Scores stay comparable across runs only while the split holds, so a disease that
            # falls out of eligibility is reported rather than quietly dropped.
            log.warning("%d pinned evaluation diseases are no longer eligible: %s",
                        len(pin_dropped), pin_dropped[:20])
    elif n_eval and len(ev_df) > n_eval:
        ev_df = ev_df.loc[stable_sample(ev_df.disease_idx, n_eval, seed)]
    ev_df = ev_df.sort_values("disease_idx").reset_index(drop=True)
    in_eval = positives.disease_idx.isin(set(ev_df.disease_idx))

    overlap = sum(1 for k in zip(positives.drug_idx, positives.disease_idx) if k in kg_known)
    report = {
        "cutoff": cutoff, "eval_end": eval_end, "single_drug_trials": single_drug,
        "mapping": dict(mapper.stats),
        "n_pre_cutoff_trial_pairs": len(pre_pairs), "n_kg_known_pairs": len(kg_known),
        "n_post_cutoff_pairs": len(post_pairs), "n_positives_novel": len(positives),
        "positives_by_tier": positives["tier"].value_counts().to_dict(),
        "n_diseases_with_positive": int(len(counts)),
        "min_disease_genes": min_disease_genes,
        "n_diseases_eligible": n_eligible,
        "n_diseases_dropped_no_genes": n_eligible - n_with_genes,
        "n_diseases_after_gene_filter": n_with_genes,
        "n_eval_diseases": int(len(ev_df)),
        "eval_split_pinned": pinned is not None,
        "eval_split_pinned_missing": pin_dropped,
        "unmapped_manual_approvals": unmapped_manual,
        "leakage_check_positives_in_kg": overlap,
        "ceiling": {
            "positives_in_eval": int(in_eval.sum()),
            "drug_has_no_target": int((positives.n_targets[in_eval] == 0).sum()),
            "mechanism_reachable": int(positives.mechanism_reachable[in_eval].sum()),
            "mechanism_reachable_rate": round(float(positives.mechanism_reachable[in_eval].mean()), 4)
            if in_eval.any() else None,
        },
    }
    return GroundTruth(positives, known, ev_df), report
