"""Lazily constructed shared resources (KG, ground truth, API clients, verifier)."""

from __future__ import annotations

import logging
from functools import cached_property

from .config import Config
from .data.groundtruth import GroundTruth
from .data.kg import KnowledgeGraph
from .data.mondo import parse_mondo_obo
from .data.opentargets import OpenTargetsClient
from .data.pubmed import PubMedClient
from .utils import CachedHTTP
from .verify.verifier import MechanismVerifier

log = logging.getLogger(__name__)


class Context:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    @cached_property
    def kg(self) -> KnowledgeGraph:
        kg = KnowledgeGraph.load(self.cfg.path("kg", "processed_dir"))
        mondo = self.cfg.path("kg", "mondo_file")
        if mondo.exists():
            kg.attach_mondo(parse_mondo_obo(mondo))
        else:
            log.warning("MONDO file missing (%s): disease grounding uses KG names only", mondo)
        return kg

    @cached_property
    def gt(self) -> GroundTruth:
        return GroundTruth.load(self.cfg.path("groundtruth", "out_dir"))

    def masked_kg(self) -> KnowledgeGraph:
        """KG with every held-out pair's drug-disease edges removed (defence in depth)."""
        kg = self.kg
        if not getattr(kg, "_masked", False):
            removed = kg.mask_pairs(zip(self.gt.positives.drug_idx, self.gt.positives.disease_idx))
            log.info("Masked %d drug-disease edges of held-out pairs", removed)
            kg._masked = True
        return kg

    def http(self, name: str, min_interval: float = 0.0) -> CachedHTTP:
        return CachedHTTP(self.cfg.cache_dir / name, min_interval=min_interval)

    @cached_property
    def pubmed(self) -> PubMedClient:
        pm = self.cfg["pubmed"]
        return PubMedClient(self.http("pubmed"), self.cfg.cutoff, pm.get("email", ""),
                            pm.get("api_key_env", "NCBI_API_KEY"))

    @cached_property
    def opentargets(self) -> OpenTargetsClient:
        return OpenTargetsClient(self.http("opentargets", 0.1))

    def verifier(self, **overrides) -> MechanismVerifier:
        v = {**self.cfg["verifier"], **overrides}
        return MechanismVerifier(
            self.masked_kg(), self.cfg.cutoff, pubmed=self.pubmed,
            opentargets=self.opentargets if v.get("use_opentargets") else None,
            accept_literature=v.get("accept_literature", True),
            use_opentargets=v.get("use_opentargets", False),
            opentargets_min_score=v.get("opentargets_min_score", 0.1),
            min_edges=v.get("min_path_edges", 2), max_edges=v.get("max_path_edges", 4),
            fuzzy_threshold=self.cfg["groundtruth"].get("fuzzy_threshold", 0))
