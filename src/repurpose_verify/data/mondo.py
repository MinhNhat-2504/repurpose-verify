"""Minimal MONDO .obo parser: names, exact synonyms and MeSH cross-references."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_SYN = re.compile(r'^synonym:\s+"(.*?)"\s+(EXACT|NARROW|BROAD|RELATED)')


@dataclass
class MondoTerm:
    mondo_id: int
    name: str
    synonyms: list[str] = field(default_factory=list)
    mesh_ids: list[str] = field(default_factory=list)


def parse_mondo_obo(path: Path, synonym_scopes: tuple[str, ...] = ("EXACT",)) -> dict[int, MondoTerm]:
    terms: dict[int, MondoTerm] = {}
    cur: dict | None = None

    def flush() -> None:
        if cur and cur.get("id", "").startswith("MONDO:") and not cur.get("obsolete"):
            mid = int(cur["id"].split(":")[1])
            terms[mid] = MondoTerm(mid, cur.get("name", ""), cur.get("syn", []), cur.get("mesh", []))

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("["):
                flush()
                cur = {} if line == "[Term]" else None
                continue
            if cur is None or ":" not in line:
                continue
            tag, _, value = line.partition(": ")
            if tag == "id":
                cur["id"] = value.strip()
            elif tag == "name":
                cur["name"] = value.strip()
            elif tag == "is_obsolete" and value.strip() == "true":
                cur["obsolete"] = True
            elif tag == "synonym":
                m = _SYN.match(line)
                if m and m.group(2) in synonym_scopes:
                    cur.setdefault("syn", []).append(m.group(1))
            elif tag == "xref":
                ref = value.split()[0]
                if ref.upper().startswith("MESH:"):
                    cur.setdefault("mesh", []).append(ref.split(":", 1)[1])
    flush()
    return terms
