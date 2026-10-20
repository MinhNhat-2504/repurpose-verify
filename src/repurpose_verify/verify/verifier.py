"""Mechanism verifier: checks every edge of a proposed drug -> ... -> disease path.

Edge statuses
  kg            the two grounded endpoints are connected in the cutoff KG by a biological relation
  literature    a cited PubMed record published before the cutoff co-mentions both endpoints
  opentargets   (optional) Open Targets gene-disease association score above threshold
  circular      the only KG link is a drug-disease relation (indication/off-label/contraindication);
                that is the label being predicted, so it is not mechanistic evidence
  unsupported   both endpoints grounded, no admissible evidence  -> fabricated edge
  ungrounded    an endpoint does not exist in the KG             -> fabricated entity
A proposal is *verified* when the drug is grounded, the path is a well-formed chain from the drug
to the queried disease through at least one gene/protein, and every edge is supported.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date

from ..data.kg import DISEASE, DRUG, DRUG_DISEASE_RELS, GENE, KnowledgeGraph
from ..data.normalize import drug_variants, name_variants, norm
from .schema import Proposal

_KG_REF = re.compile(r"kg:(\d+)", re.I)
SUPPORTED = ("kg", "literature", "opentargets")
FABRICATED = ("unsupported", "ungrounded", "circular")
MIDDLE_TYPES = (GENE, "pathway", "biological_process", "molecular_function", "effect/phenotype",
                "cellular_component", "anatomy", "exposure", DISEASE, DRUG)


@dataclass
class EdgeCheck:
    source: str
    relation: str
    target: str
    source_idx: int | None
    target_idx: int | None
    status: str
    kg_relations: list[str] = field(default_factory=list)
    pmids_admissible: list[str] = field(default_factory=list)
    pmids_invalid: list[str] = field(default_factory=list)
    detail: str = ""


@dataclass
class Verification:
    drug_idx: int | None
    disease_idx: int
    structure_ok: bool
    issues: list[str]
    edges: list[EdgeCheck]
    verified: bool

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def n_supported(self) -> int:
        return sum(e.status in SUPPORTED for e in self.edges)

    @property
    def n_fabricated(self) -> int:
        return sum(e.status in FABRICATED for e in self.edges)

    def summary(self) -> str:
        if self.verified:
            return "verified"
        bad = [f"{e.source} -> {e.target}: {e.status}" for e in self.edges if e.status not in SUPPORTED]
        return "; ".join(self.issues + bad) or "unverified"

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(n_edges=self.n_edges, n_supported=self.n_supported, n_fabricated=self.n_fabricated)
        return d


class MechanismVerifier:
    def __init__(self, kg: KnowledgeGraph, cutoff: date, pubmed=None, opentargets=None,
                 accept_literature: bool = True, use_opentargets: bool = False,
                 opentargets_min_score: float = 0.1, min_edges: int = 2, max_edges: int = 4,
                 fuzzy_threshold: float = 0):
        self.kg = kg
        self.cutoff = cutoff
        self.pubmed = pubmed
        self.opentargets = opentargets
        self.accept_literature = accept_literature and pubmed is not None
        self.use_opentargets = use_opentargets and opentargets is not None
        self.ot_min = opentargets_min_score
        self.min_edges = min_edges
        self.max_edges = max_edges
        self.fuzzy = fuzzy_threshold

    # ------------------------------------------------------------------ grounding
    def ground(self, mention: str, preferred: tuple[str, ...]) -> int | None:
        m = _KG_REF.search(mention)
        if m:
            i = int(m.group(1))
            if 0 <= i < self.kg.n:
                # A kg id paired with a name that belongs to another node is a fabrication,
                # not a lookup: models cite plausible ids for drugs they invented.
                return i if self._name_agrees(i, mention) else None
        clean = re.sub(r"\[.*?\]", "", mention).strip() or mention
        for t in preferred:
            hit = self.kg.resolve(clean, t, self.fuzzy)
            if hit is not None:
                return hit
        return None

    def _kg_forms(self, idx: int) -> set[str]:
        name = self.kg.names[idx]
        forms = set(drug_variants(name) if self.kg.types[idx] == DRUG else name_variants(name))
        for syn in self.kg._synonyms.get(idx, []):
            forms.update(name_variants(syn))
        return {f for f in forms if len(f) >= 2}

    def _name_agrees(self, idx: int, mention: str) -> bool:
        text = norm(re.sub(r"\[?\s*kg:\d+\s*\]?", " ", mention))
        if not text:
            return True
        forms = self._kg_forms(idx)
        if any(text == f or text in f or f in text for f in forms):
            return True
        from rapidfuzz import fuzz

        return any(fuzz.ratio(text, f) >= 85 for f in forms)

    def _surface_forms(self, idx: int, mention: str) -> set[str]:
        forms = set(name_variants(re.sub(r"\[.*?\]", "", mention)))
        forms.update(self._kg_forms(idx))
        return {f for f in forms if len(f) >= 2}

    # ------------------------------------------------------------------ evidence
    def _literature(self, edge, u: int, v: int, check: EdgeCheck) -> bool:
        if not edge.pmids or self.pubmed is None:
            return False
        arts = self.pubmed.fetch(edge.pmids)
        for pmid in edge.pmids:
            art = arts.get(pmid)
            if art is None or not self.pubmed.is_admissible(art):
                check.pmids_invalid.append(pmid)
            else:
                check.pmids_admissible.append(pmid)
        if not self.accept_literature:
            return False
        fu, fv = self._surface_forms(u, edge.source), self._surface_forms(v, edge.target)
        for pmid in check.pmids_admissible:
            text = f" {norm(arts[pmid].title + ' ' + arts[pmid].abstract)} "
            if any(f" {f} " in text for f in fu) and any(f" {f} " in text for f in fv):
                check.detail = f"co-mention in PMID {pmid}"
                return True
        return False

    def _opentargets(self, u: int, v: int, check: EdgeCheck) -> bool:
        if not self.use_opentargets:
            return False
        types = {self.kg.types[u], self.kg.types[v]}
        if types != {GENE, DISEASE}:
            return False
        gene, dis = (u, v) if self.kg.types[u] == GENE else (v, u)
        score = self.opentargets.association_score(self.kg.names[gene], self.kg.names[dis])
        if score is not None and score >= self.ot_min:
            check.detail = f"Open Targets score {score:.3f}"
            return True
        return False

    def check_edge(self, edge, u: int | None, v: int | None) -> EdgeCheck:
        c = EdgeCheck(edge.source, edge.relation, edge.target, u, v, "unsupported")
        if u is None or v is None:
            c.status = "ungrounded"
            c.detail = "unknown entity: " + (edge.source if u is None else edge.target)
            if edge.pmids and self.pubmed is not None:
                arts = self.pubmed.fetch(edge.pmids)
                for pmid in edge.pmids:
                    art = arts.get(pmid)
                    ok = art is not None and self.pubmed.is_admissible(art)
                    (c.pmids_admissible if ok else c.pmids_invalid).append(pmid)
            return c
        rels = self.kg.edge_relations(u, v)
        bio = [f"{r}:{d}" if d else r for r, d in rels if r not in DRUG_DISEASE_RELS]
        c.kg_relations = bio or [r for r, _ in rels]
        literature = self._literature(edge, u, v, c)
        if bio:
            c.status = "kg"
        elif literature:
            c.status = "literature"
        elif self._opentargets(u, v, c):
            c.status = "opentargets"
        elif rels:
            c.status = "circular"
        return c

    # ------------------------------------------------------------------ proposal
    def verify(self, proposal: Proposal, disease_idx: int) -> Verification:
        issues: list[str] = []
        drug_idx = self.ground(proposal.drug, (DRUG,))
        if drug_idx is None:
            issues.append(f"drug not in KG: {proposal.drug}")
        edges = proposal.path
        if not edges:
            return Verification(drug_idx, disease_idx, False, issues + ["no mechanism path"], [], False)
        if not (self.min_edges <= len(edges) <= self.max_edges):
            issues.append(f"path length {len(edges)} outside [{self.min_edges}, {self.max_edges}]")
        checks: list[EdgeCheck] = []
        nodes: list[int | None] = []
        for k, e in enumerate(edges):
            src_types = (DRUG,) + MIDDLE_TYPES if k == 0 else MIDDLE_TYPES
            tgt_types = (DISEASE,) + MIDDLE_TYPES if k == len(edges) - 1 else MIDDLE_TYPES
            u = self.ground(e.source, src_types)
            v = self.ground(e.target, tgt_types)
            if k > 0 and nodes and nodes[-1] is not None and u is not None and nodes[-1] != u:
                issues.append(f"broken chain at edge {k + 1}: {edges[k - 1].target} != {e.source}")
            nodes.extend([u, v] if k == 0 else [v])
            checks.append(self.check_edge(e, u, v))
        first, last = nodes[0], nodes[-1]
        if first is None or drug_idx is None or first != drug_idx:
            issues.append("path does not start at the proposed drug")
        if last != disease_idx:
            related = last is not None and disease_idx in set(
                self.kg.neighbors(last, ["disease_disease"]).tolist())
            if not related:
                issues.append("path does not end at the queried disease")
        middle = [n for n in nodes[1:-1] if n is not None]
        if not any(self.kg.types[n] == GENE for n in middle):
            issues.append("no gene/protein on the path")
        structure_ok = not issues
        verified = structure_ok and all(c.status in SUPPORTED for c in checks)
        return Verification(drug_idx, disease_idx, structure_ok, issues, checks, verified)
