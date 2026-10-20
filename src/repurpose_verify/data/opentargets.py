"""Open Targets Platform GraphQL client (optional external evidence for gene-disease edges).

The live platform aggregates evidence published after the benchmark cutoff, so it is disabled
by default and, when enabled, its verifications are reported separately from KG evidence.
"""

from __future__ import annotations

from functools import lru_cache

from ..utils import CachedHTTP

API = "https://api.platform.opentargets.org/api/v4/graphql"


class OpenTargetsClient:
    def __init__(self, http: CachedHTTP):
        self.http = http
        self.map_target = lru_cache(maxsize=None)(self._map_target)
        self.map_disease = lru_cache(maxsize=None)(self._map_disease)

    def _query(self, query: str, variables: dict) -> dict:
        data = self.http.post(API, {"query": query, "variables": variables})
        return data.get("data") or {}

    def _map(self, term: str, entity: str) -> str | None:
        q = ("query($t:[String!]!,$e:[String!]){ mapIds(queryTerms:$t, entityNames:$e)"
             "{ mappings { term hits { id name } } } }")
        data = self._query(q, {"t": [term], "e": [entity]})
        for m in data.get("mapIds", {}).get("mappings", []):
            if m.get("hits"):
                return m["hits"][0]["id"]
        return None

    def _map_target(self, symbol: str) -> str | None:
        return self._map(symbol, "target")

    def _map_disease(self, name: str) -> str | None:
        return self._map(name, "disease")

    def association_score(self, gene_symbol: str, disease_name: str) -> float | None:
        """Overall association score in [0, 1], or None when either entity cannot be mapped."""
        tid, did = self.map_target(gene_symbol), self.map_disease(disease_name)
        if not tid or not did:
            return None
        q = ("query($t:String!,$d:[String!]){ target(ensemblId:$t){ associatedDiseases(Bs:$d)"
             "{ rows { disease { id } score } } } }")
        data = self._query(q, {"t": tid, "d": [did]})
        rows = (data.get("target") or {}).get("associatedDiseases", {}).get("rows", [])
        return float(rows[0]["score"]) if rows else 0.0
