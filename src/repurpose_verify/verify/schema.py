"""Proposal schema shared by agents, baselines and the verifier."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PathEdge:
    source: str
    relation: str
    target: str
    pmids: list[str] = field(default_factory=list)

    @classmethod
    def from_any(cls, obj: Any) -> "PathEdge":
        if isinstance(obj, dict):
            pm = obj.get("pmids") or obj.get("evidence") or []
            if isinstance(pm, (str, int)):
                pm = [pm]
            return cls(str(obj.get("source", obj.get("head", ""))),
                       str(obj.get("relation", obj.get("rel", ""))),
                       str(obj.get("target", obj.get("tail", ""))),
                       [str(p).replace("PMID:", "").strip() for p in pm])
        if isinstance(obj, (list, tuple)) and len(obj) == 3:
            return cls(str(obj[0]), str(obj[1]), str(obj[2]))
        raise ValueError(f"Cannot parse path edge: {obj!r}")


@dataclass
class Proposal:
    drug: str
    path: list[PathEdge] = field(default_factory=list)
    rationale: str = ""
    confidence: float | None = None

    @classmethod
    def from_any(cls, obj: Any) -> "Proposal":
        if isinstance(obj, str):
            return cls(drug=obj)
        raw_path = obj.get("path") or obj.get("mechanism") or []
        edges: list[PathEdge] = []
        if raw_path and all(isinstance(x, str) for x in raw_path):
            # Node list form: [drug, target, ..., disease]
            edges = [PathEdge(a, "", b) for a, b in zip(raw_path[:-1], raw_path[1:])]
        else:
            for e in raw_path:
                try:
                    edges.append(PathEdge.from_any(e))
                except ValueError:
                    continue
        conf = obj.get("confidence")
        try:
            conf = float(conf) if conf is not None else None
        except (TypeError, ValueError):
            conf = None
        return cls(drug=str(obj.get("drug", obj.get("name", ""))), path=edges,
                   rationale=str(obj.get("rationale", "")), confidence=conf)

    def to_dict(self) -> dict:
        return asdict(self)

    def path_text(self) -> str:
        if not self.path:
            return ""
        parts = [self.path[0].source]
        for e in self.path:
            parts.append(f"-[{e.relation or '?'}]-> {e.target}")
        return " ".join(parts)
