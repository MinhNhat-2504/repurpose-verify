"""Validation that needs no expert rater, and answers a question an expert panel would not.

Two checks, both automated:

``moa``
    The verifier accepts a drug -> gene edge because the snapshot contains it - and the agent
    read that same snapshot. For the first edge of every path that objection is fair, so the
    drug -> target claims are re-checked against ChEMBL's mechanism-of-action curation, which
    the agent never saw. Drugs are mapped DrugBank -> ChEMBL through UniChem by structure
    (InChI), never by name.

``literature``
    Did the literature published *after* the cutoff come to mention the pair at all? This is an
    outcome signal independent of ClinicalTrials.gov. Co-mention is weak evidence on its own,
    so each system is scored against a control: random drugs drawn for the same diseases. A
    system that merely names famous drugs scores well on co-mention and no better than its
    control, which is exactly what the control is there to reveal.
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

from ..data.kg import DRUG, GENE, KnowledgeGraph
from ..utils import CachedHTTP

log = logging.getLogger(__name__)

UNICHEM = "https://www.ebi.ac.uk/unichem/api/v1/compounds"
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
UNICHEM_DRUGBANK_SOURCE = 2


class ChemblMoA:
    """DrugBank id -> ChEMBL mechanism-of-action target gene symbols."""

    def __init__(self, http: CachedHTTP):
        self.http = http
        self._chembl: dict[str, str | None] = {}
        self._genes: dict[str, set[str] | None] = {}
        self._targets: dict[str, set[str]] = {}

    def chembl_id(self, drugbank_id: str) -> str | None:
        if drugbank_id not in self._chembl:
            self._chembl[drugbank_id] = self._lookup_chembl(drugbank_id)
        return self._chembl[drugbank_id]

    def _lookup_chembl(self, drugbank_id: str) -> str | None:
        try:
            data = self.http.post(UNICHEM, {"type": "sourceID", "compound": drugbank_id,
                                            "sourceID": UNICHEM_DRUGBANK_SOURCE})
        except Exception as err:  # an unmapped drug is data, not a failure
            log.debug("UniChem lookup failed for %s: %s", drugbank_id, err)
            return None
        for compound in data.get("compounds", []) or []:
            for source in compound.get("sources", []) or []:
                if str(source.get("shortName", "")).lower() == "chembl":
                    return source.get("compoundId")
        return None

    def target_genes(self, target_chembl_id: str) -> set[str]:
        if target_chembl_id not in self._targets:
            data = self.http.get(f"{CHEMBL}/target/{target_chembl_id}.json")
            genes: set[str] = set()
            for comp in data.get("target_components", []) or []:
                for syn in comp.get("target_component_synonyms", []) or []:
                    if syn.get("syn_type") == "GENE_SYMBOL":
                        genes.add(str(syn["component_synonym"]).upper())
            self._targets[target_chembl_id] = genes
        return self._targets[target_chembl_id]

    def moa_genes(self, drugbank_id: str) -> set[str] | None:
        """Gene symbols ChEMBL records as this drug's mechanism targets.

        ``None`` means ChEMBL has no mechanism entry for the drug - the pipeline has no opinion,
        which must be reported separately from a disagreement.
        """
        if drugbank_id in self._genes:
            return self._genes[drugbank_id]
        chembl = self.chembl_id(drugbank_id)
        result: set[str] | None = None
        if chembl:
            data = self.http.get(f"{CHEMBL}/mechanism.json",
                                 params={"molecule_chembl_id": chembl, "limit": 100})
            mechanisms = data.get("mechanisms", []) or []
            if mechanisms:
                result = set()
                for m in mechanisms:
                    target = m.get("target_chembl_id")
                    if target:
                        result |= self.target_genes(target)
        self._genes[drugbank_id] = result
        return result


@dataclass
class MoAResult:
    edges_checked: int = 0
    drugs_checked: int = 0
    drugs_unmapped: int = 0            # no ChEMBL id at all
    drugs_without_moa: int = 0         # mapped, but ChEMBL records no mechanism
    edges_with_opinion: int = 0        # ChEMBL has a mechanism for the drug
    edges_confirmed: int = 0           # the agent's gene is among ChEMBL's targets
    # PrimeKG's drug_protein edges carry a role: target, enzyme, transporter or carrier. Only
    # the first is a candidate mechanism of action; a path through a metabolising enzyme is
    # pharmacokinetics, not a mechanism. Confirmation is therefore broken out by role.
    by_role: Counter = field(default_factory=Counter)          # role -> edges seen
    confirmed_by_role: Counter = field(default_factory=Counter)
    opinion_by_role: Counter = field(default_factory=Counter)
    examples_confirmed: list[str] = field(default_factory=list)
    examples_contradicted: list[str] = field(default_factory=list)

    def rates(self) -> dict:
        def r(a, b):
            return a / b if b else float("nan")
        out = {
            "edges_checked": self.edges_checked,
            "drugs_checked": self.drugs_checked,
            "drug_chembl_coverage": r(self.drugs_checked - self.drugs_unmapped, self.drugs_checked),
            "drug_moa_coverage": r(
                self.drugs_checked - self.drugs_unmapped - self.drugs_without_moa,
                self.drugs_checked),
            "edges_with_opinion": self.edges_with_opinion,
            "independent_confirmation_rate": r(self.edges_confirmed, self.edges_with_opinion),
            "edges_by_role": dict(self.by_role),
            "non_target_role_share": r(
                sum(n for role, n in self.by_role.items() if role != "target"),
                sum(self.by_role.values())),
        }
        out["confirmation_by_role"] = {
            role: r(self.confirmed_by_role[role], self.opinion_by_role[role])
            for role in sorted(self.opinion_by_role)
        }
        return out


def drug_gene_edges(records: Iterable[dict], kg: KnowledgeGraph,
                    verified_only: bool = True) -> list[tuple[int, int]]:
    """Distinct (drug, gene) pairs asserted as the first edge of a proposal's path."""
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for rec in records:
        for prop in rec.get("proposals", []):
            ver = prop.get("verification", {})
            if verified_only and not ver.get("verified"):
                continue
            edges = ver.get("edges", [])
            if not edges:
                continue
            first = edges[0]
            u, v = first.get("source_idx"), first.get("target_idx")
            if u is None or v is None:
                continue
            if kg.types[u] == DRUG and kg.types[v] == GENE and (u, v) not in seen:
                seen.add((u, v))
                out.append((u, v))
    return out


def validate_moa(kg: KnowledgeGraph, records: Iterable[dict], moa: ChemblMoA,
                 verified_only: bool = True, max_edges: int = 0) -> MoAResult:
    pairs = drug_gene_edges(records, kg, verified_only)
    if max_edges:
        pairs = pairs[:max_edges]
    res = MoAResult(edges_checked=len(pairs))
    per_drug: dict[int, set[str] | None] = {}
    for drug, gene in pairs:
        if drug not in per_drug:
            drugbank = str(kg.nodes.iloc[drug]["id"]).strip()
            per_drug[drug] = moa.moa_genes(drugbank)
            res.drugs_checked += 1
            if moa.chembl_id(drugbank) is None:
                res.drugs_unmapped += 1
            elif per_drug[drug] is None:
                res.drugs_without_moa += 1
        roles = {disp for rel, disp in kg.edge_relations(drug, gene) if rel == "drug_protein"}
        role = "target" if "target" in roles else (sorted(roles)[0] if roles else "none")
        res.by_role[role] += 1
        genes = per_drug[drug]
        if genes is None:
            continue
        res.edges_with_opinion += 1
        res.opinion_by_role[role] += 1
        claim = f"{kg.names[drug]} -[{role}]-> {kg.names[gene]}"
        if str(kg.names[gene]).upper() in genes:
            res.edges_confirmed += 1
            res.confirmed_by_role[role] += 1
            if len(res.examples_confirmed) < 10:
                res.examples_confirmed.append(claim)
        elif len(res.examples_contradicted) < 10:
            res.examples_contradicted.append(f"{claim} (ChEMBL: {sorted(genes)[:4]})")
    return res


class PostCutoffLiterature:
    """Counts PubMed records published after the cutoff that co-mention a drug and a disease."""

    def __init__(self, http: CachedHTTP, cutoff: date, api_key_env: str = "NCBI_API_KEY",
                 email: str = ""):
        import os

        self.http = http
        self.mindate = (cutoff + timedelta(days=1)).strftime("%Y/%m/%d")
        self.base = {"db": "pubmed", "retmode": "json", "retmax": 0, "datetype": "pdat",
                     "mindate": self.mindate, "maxdate": "3000/12/31", "tool": "repurpose-verify"}
        if email:
            self.base["email"] = email
        key = os.environ.get(api_key_env, "")
        if key:
            self.base["api_key"] = key
        self.http.min_interval = max(self.http.min_interval, 0.12 if key else 0.35)

    def count(self, drug: str, disease: str) -> int:
        term = f'"{drug}" AND "{disease}"'
        data = self.http.get(f"{EUTILS}/esearch.fcgi", params={**self.base, "term": term})
        try:
            return int(data["esearchresult"]["count"])
        except (KeyError, TypeError, ValueError):
            return 0


def validate_literature(kg: KnowledgeGraph, records: list[dict], lit: PostCutoffLiterature,
                        top_k: int = 5, controls_per_disease: int = 5, seed: int = 13,
                        known_by_disease: dict[int, set[int]] | None = None) -> dict:
    """Post-cutoff co-mention rate for a system's top proposals, against a random-drug control."""
    rng = random.Random(seed)
    known = known_by_disease or {}
    drugs = [int(i) for i in kg.nodes_of_type(DRUG)]
    hits = Counter()
    totals = Counter()
    examples: list[str] = []
    for rec in records:
        disease_idx = int(rec["disease_idx"])
        disease = str(kg.names[disease_idx])
        proposed = [r["drug_idx"] for r in rec.get("ranked", [])[:top_k]]
        for drug_idx in proposed:
            n = lit.count(str(kg.names[drug_idx]), disease)
            totals["system"] += 1
            if n > 0:
                hits["system"] += 1
                if len(examples) < 10:
                    examples.append(f"{kg.names[drug_idx]} / {disease}: {n} post-cutoff records")
        pool = [d for d in drugs if d not in set(proposed) | known.get(disease_idx, set())]
        for drug_idx in rng.sample(pool, min(controls_per_disease, len(pool))):
            n = lit.count(str(kg.names[drug_idx]), disease)
            totals["control"] += 1
            if n > 0:
                hits["control"] += 1
    out = {
        "top_k": top_k,
        "system_pairs": totals["system"],
        "system_comention_rate": hits["system"] / totals["system"] if totals["system"] else float("nan"),
        "control_pairs": totals["control"],
        "control_comention_rate": hits["control"] / totals["control"] if totals["control"] else float("nan"),
        "examples": examples,
    }
    if totals["system"] and totals["control"]:
        out["lift_over_control"] = out["system_comention_rate"] - out["control_comention_rate"]
    return out


def write_external_report(results: dict[str, dict], out_path: Path, cutoff: date) -> Path:
    """results: {system -> {"moa": rates dict, "literature": dict}}"""
    lines = ["# External validation (no expert rater required)", "",
             "## Independent confirmation of drug-target claims (ChEMBL mechanism of action)", "",
             "The verifier accepts a drug-gene edge from the same snapshot the agent reads. These",
             "edges are re-checked against ChEMBL's independently curated mechanism of action,",
             "with drugs mapped DrugBank -> ChEMBL through UniChem by structure, never by name.",
             "ChEMBL has no mechanism entry for many drugs; that is reported as coverage rather",
             "than counted as a disagreement.", "",
             "| System | Edges checked | Drugs | ChEMBL id found | MoA recorded | Edges with an opinion | **Confirmed** |",
             "|---|---|---|---|---|---|---|"]
    for system, res in results.items():
        m = res.get("moa")
        if not m:
            continue
        lines.append(
            f"| {system} | {m['edges_checked']} | {m['drugs_checked']} | "
            f"{m['drug_chembl_coverage']:.3f} | {m['drug_moa_coverage']:.3f} | "
            f"{m['edges_with_opinion']} | **{m['independent_confirmation_rate']:.3f}** |")
    if any(r.get("moa", {}).get("edges_by_role") for r in results.values()):
        lines += ["", "### By the role PrimeKG assigns the drug-protein edge", "",
                  "Only `target` can be a mechanism of action; `enzyme`, `transporter` and",
                  "`carrier` describe pharmacokinetics. A confirmation rate of zero for those",
                  "roles is the expected result and is independent evidence that the",
                  "distinction matters.", "",
                  "| System | Role | Edges | Confirmed |", "|---|---|---|---|"]
        for system, res in results.items():
            m = res.get("moa")
            if not m or not m.get("edges_by_role"):
                continue
            by_role = m["edges_by_role"]
            conf = m.get("confirmation_by_role", {})
            for role in sorted(by_role, key=lambda k: -by_role[k]):
                rate = conf.get(role)
                shown = f"{rate:.3f}" if rate is not None and rate == rate else "no opinion"
                lines.append(f"| {system} | {role} | {by_role[role]} | {shown} |")
            lines.append(f"| {system} | *non-target share* | "
                         f"{m['non_target_role_share']:.3f} | |")
    lines += ["", f"## Post-cutoff literature corroboration (published after {cutoff})", "",
              "Share of a system's top proposals for which at least one PubMed record published",
              "after the cutoff co-mentions the drug and the disease, against random drugs drawn",
              "for the same diseases. Co-mention is weak evidence; the control is what makes the",
              "comparison meaningful, since naming famous drugs scores well on co-mention alone.",
              "",
              "| System | Top-K | Pairs | **Co-mention** | Control pairs | Control | Lift |",
              "|---|---|---|---|---|---|---|"]
    for system, res in results.items():
        lit = res.get("literature")
        if not lit:
            continue
        lift = lit.get("lift_over_control")
        lines.append(
            f"| {system} | {lit['top_k']} | {lit['system_pairs']} | "
            f"**{lit['system_comention_rate']:.3f}** | {lit['control_pairs']} | "
            f"{lit['control_comention_rate']:.3f} | "
            f"{('%+.3f' % lift) if lift is not None else '-'} |")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Structured copy so the manuscript's number checker reads values, not a rendered table.
    json_path = out_path.with_suffix(".json")
    existing: dict = {}
    if json_path.exists():
        try:
            existing = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    for system, res in results.items():
        merged = {**existing.get(system, {}), **{k: v for k, v in res.items() if v}}
        existing[system] = merged
    json_path.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")
    return out_path
