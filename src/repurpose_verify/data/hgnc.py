"""HGNC gene symbol aliases, joined onto KG gene nodes by NCBI gene id.

Without them an agent writing "COX-2" looks like it invented an entity, when the graph holds
the same protein as PTGS2. Guessing from the name instead is worse than useless here: a
substring search for "COX-2" returns COX2, cytochrome c oxidase subunit II - a different gene.
The join is therefore on identifiers only, and the alias strings are used purely as extra
surface forms for that node.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

log = logging.getLogger(__name__)

HGNC_URL = ("https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/"
            "hgnc_complete_set.txt")
ALIAS_FIELDS = ("alias_symbol", "prev_symbol", "alias_name", "prev_name")


def parse_hgnc(path: Path, with_names: bool = False) -> dict[str, list[str]]:
    """Map NCBI gene id -> alias symbols (and optionally alias names) for approved genes."""
    fields = ALIAS_FIELDS if with_names else ALIAS_FIELDS[:2]
    out: dict[str, list[str]] = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            entrez = (row.get("entrez_id") or "").strip()
            if not entrez or row.get("status") != "Approved":
                continue
            aliases: list[str] = []
            for field in fields:
                raw = (row.get(field) or "").strip().strip('"')
                aliases += [a.strip() for a in raw.split("|") if a.strip()]
            symbol = (row.get("symbol") or "").strip()
            if symbol:
                aliases.append(symbol)
            if aliases:
                out[entrez] = list(dict.fromkeys(aliases))
    log.info("HGNC: aliases for %d genes", len(out))
    return out
