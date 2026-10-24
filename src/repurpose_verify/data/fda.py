"""Candidate new-indication approvals from the Drugs@FDA bulk data.

The ground truth's strongest label is a regulatory approval of a *new indication for an existing
drug*. Drugs@FDA publishes the approvals as data, but not what the new indication was: a
submission is tagged "Efficacy-New Indication" without naming the disease, which is only in the
approval letter or the revised label. This module therefore does the half that can be automated
- find every such approval in the window and match the drug to the knowledge graph - and leaves
a CSV whose `disease` column a person fills from the FDA record, with the link to open.

Filtering on the action type matters. The EFFICACY submission class also covers new dosing,
new populations, paediatric studies and labelling changes with clinical data; only action type
"Efficacy-New Indication" is the thing wanted here.
"""

from __future__ import annotations

import csv
import io
import logging
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .kg import DRUG, KnowledgeGraph

log = logging.getLogger(__name__)

BULK_URL = "https://www.fda.gov/media/89850/download"
NEW_INDICATION = "Efficacy-New Indication"
APPROVED = "AP"
APPLICATION_URL = ("https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm"
                   "?event=overview.process&ApplNo={appl}")


def _table(archive: zipfile.ZipFile, name: str):
    with archive.open(name) as f:
        yield from csv.DictReader(io.TextIOWrapper(f, "utf-8", errors="replace"), delimiter="\t")


@dataclass
class ApprovalCandidate:
    appl_no: str
    submission_no: str
    approval_date: date
    drug_name: str
    active_ingredient: str
    kg_drug: int | None
    kg_name: str


def find_new_indication_approvals(zip_path: Path, kg: KnowledgeGraph, start: date, end: date,
                                  fuzzy: float = 0) -> list[ApprovalCandidate]:
    z = zipfile.ZipFile(zip_path)

    action_ids = {r["ActionTypes_LookupID"] for r in _table(z, "ActionTypes_Lookup.txt")
                  if (r.get("ActionTypes_LookupDescription") or "").strip() == NEW_INDICATION}
    if not action_ids:
        raise ValueError(f"no action type named {NEW_INDICATION!r} in this release")

    wanted: set[tuple[str, str]] = set()
    for r in _table(z, "Join_Submission_ActionTypes_Lookup.txt"):
        if r["ActionTypes_LookupID"] in action_ids:
            wanted.add((r["ApplNo"].strip(), r["SubmissionNo"].strip()))

    products: dict[str, tuple[str, str]] = {}
    for r in _table(z, "Products.txt"):
        products.setdefault(r["ApplNo"].strip(),
                            (r.get("DrugName", "").strip(), r.get("ActiveIngredient", "").strip()))

    out: list[ApprovalCandidate] = []
    seen: set[tuple[str, str]] = set()
    for r in _table(z, "Submissions.txt"):
        key = (r["ApplNo"].strip(), r["SubmissionNo"].strip())
        if key not in wanted or r.get("SubmissionStatus", "").strip() != APPROVED:
            continue
        raw = (r.get("SubmissionStatusDate") or "")[:10]
        try:
            when = date.fromisoformat(raw)
        except ValueError:
            continue
        if not (start < when <= end) or key in seen:
            continue
        seen.add(key)
        drug_name, ingredient = products.get(key[0], ("", ""))
        idx = None
        for candidate in (ingredient, drug_name):
            if candidate:
                idx = kg.resolve(candidate.split(";")[0], DRUG, fuzzy)
                if idx is not None:
                    break
        out.append(ApprovalCandidate(key[0], key[1], when, drug_name, ingredient, idx,
                                     kg.names[idx] if idx is not None else ""))
    out.sort(key=lambda c: (c.kg_drug is None, c.approval_date))
    log.info("%d new-indication approvals in window, %d matched to a KG drug",
             len(out), sum(c.kg_drug is not None for c in out))
    return out


def write_candidates(candidates: list[ApprovalCandidate], path: Path,
                     matched_only: bool = True) -> Path:
    """Write the worksheet. `disease` is the one column a person has to fill."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [c for c in candidates if c.kg_drug is not None] if matched_only else candidates
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["drug", "disease", "date", "source", "note",
                    "kg_id", "fda_drug_name", "active_ingredient", "fda_link"])
        for c in rows:
            w.writerow([c.kg_name or c.active_ingredient, "", c.approval_date.isoformat(),
                        f"FDA efficacy supplement {c.appl_no}/{c.submission_no}", "",
                        f"kg:{c.kg_drug}" if c.kg_drug is not None else "",
                        c.drug_name, c.active_ingredient,
                        APPLICATION_URL.format(appl=c.appl_no)])
    return path
