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
import re
import subprocess
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .kg import DISEASE, DRUG, KnowledgeGraph
from .normalize import _PAREN

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


LETTER_URL = ("https://www.accessdata.fda.gov/drugsatfda_docs/appletter/{year}/"
              "{appl}Orig{orig}s{sub:03d}ltr.pdf")
# The operative sentence of an approval letter names the new indication, e.g. "This Prior
# Approval supplemental new drug application provides for treatment of adults with giant cell
# arteritis." Everything before "provides for" is boilerplate that varies by application type.
_PROVIDES = re.compile(r"provides?\s+for\s+(?:the\s+)?(.{10,400}?)(?=\.\s|\sAPPROVAL|$)",
                       re.I | re.S)


def _pdf_to_text(pdf: Path) -> str | None:
    """Extract text with pdftotext, which is the only PDF reader we require."""
    txt = pdf.with_suffix(".txt")
    if not txt.exists():
        try:
            subprocess.run(["pdftotext", "-layout", str(pdf), str(txt)],
                           check=True, capture_output=True, timeout=120)
        except (OSError, subprocess.SubprocessError):
            return None
    try:
        return txt.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def fetch_letter(appl_no: str, submission_no: str, year: int, cache_dir: Path,
                 timeout: float = 60.0) -> tuple[str | None, str | None]:
    """Download an approval letter and return (text, url). Tries the usual Orig numbers."""
    import requests

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        sub = int(submission_no)
    except ValueError:
        return None, None
    for orig in (1, 2, 3):
        url = LETTER_URL.format(year=year, appl=appl_no.zfill(6), orig=orig, sub=sub)
        pdf = cache_dir / f"{appl_no}_s{sub:03d}_o{orig}.pdf"
        if not pdf.exists():
            try:
                r = requests.get(url, timeout=timeout,
                                 headers={"User-Agent": "repurpose-verify/0.1 (research)"})
            except requests.RequestException:
                continue
            if r.status_code != 200 or not r.content.startswith(b"%PDF"):
                continue
            pdf.write_bytes(r.content)
        text = _pdf_to_text(pdf)
        if text:
            return text, url
    return None, None


def extract_indication(letter_text: str) -> str | None:
    """The clause naming what the supplement provides for, or None if the letter does not say."""
    flat = re.sub(r"\s+", " ", letter_text)
    m = _PROVIDES.search(flat)
    if not m:
        return None
    clause = m.group(1).strip(" .,;")
    # Letters sometimes run on into the next heading; keep the first sentence.
    return re.split(r"(?<=[a-z])\.\s+[A-Z]{2,}", clause)[0].strip()


def write_candidates(candidates: list[ApprovalCandidate], path: Path, matched_only: bool = True,
                     letters: dict[tuple[str, str], tuple[str, str]] | None = None,
                     kg: KnowledgeGraph | None = None, fuzzy: float = 0) -> Path:
    """Write the worksheet. `disease` is the one column a person has to fill.

    When approval-letter text is supplied, `letter_indication` carries the clause in which the
    letter says what the supplement provides for. That is the evidence for the disease, quoted
    rather than interpreted: turning "treatment of adults with giant cell arteritis" into an
    ontology term is a judgement, and it stays with the person making it.

    Given the graph, two further columns carry a *proposed* ontology term for that clause and the
    span it came from, so the judgement is a confirmation rather than a search. `disease` is still
    left empty: a proposal that nobody checked is not evidence.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    letters = letters or {}
    rows = [c for c in candidates if c.kg_drug is not None] if matched_only else candidates
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["drug", "disease", "date", "source", "note", "kg_id", "letter_indication",
                    "proposed_disease", "proposed_kg_id", "proposed_from_span",
                    "fda_drug_name", "active_ingredient", "letter_link", "fda_link"])
        for c in rows:
            indication, letter_url = letters.get((c.appl_no, c.submission_no), ("", ""))
            span, idx = propose_disease(indication, kg, fuzzy) if kg and indication else ("", None)
            w.writerow([c.kg_name or c.active_ingredient, "", c.approval_date.isoformat(),
                        f"FDA efficacy supplement {c.appl_no}/{c.submission_no}", "",
                        f"kg:{c.kg_drug}" if c.kg_drug is not None else "",
                        indication,
                        kg.names[idx] if idx is not None else "",
                        f"kg:{idx}" if idx is not None else "", span,
                        c.drug_name, c.active_ingredient, letter_url,
                        APPLICATION_URL.format(appl=c.appl_no)])
    return path


# Where a letter stops being boilerplate and starts naming the disease.
_ANCHORS = re.compile(r"\b(?:for the treatment of|for treatment of|treatment of|indicated (?:in|for)"
                      r"|indication (?:of|for)|management of|to treat)\b", re.I)
# Population and eligibility wording that surrounds the disease but is not part of its name.
_POPULATION = re.compile(
    r"\b(?:adults?|adult|pediatric|paediatric|children|patients?|subjects?|aged?|years?|months?"
    r"|of age|and older|older|weighing|at least|\d+\s*(?:kg|mg|cm))\b", re.I)
_TRAILING_CLAUSE = re.compile(r"\b(?:who|whose|that|which|after|following|based on|as determined"
                              r"|in combination with|with no|not adequately|requiring)\b.*$", re.I)


def propose_disease(letter_indication: str, kg: KnowledgeGraph, fuzzy: float = 0
                    ) -> tuple[str, int | None]:
    """Propose the KG disease a letter's indication clause names, as the longest span that resolves.

    The clause carries qualifiers that no ontology label contains - "newly diagnosed acute myeloid
    leukemia", "unresectable or metastatic ... carcinoma", "seizures associated with Lennox-Gastaut
    syndrome" - so an exact lookup of the whole phrase fails. Rather than maintain a list of
    qualifiers to strip, every contiguous span of the cleaned phrase is tried and the longest one
    that resolves wins: the qualifiers fall away because they are not part of any disease name.

    The result is a *proposal*. Turning a letter's words into an ontology term is a judgement, so
    the pair is written to a worksheet for a person to confirm, never straight into the ground truth.
    """
    text = letter_indication or ""
    parts = _ANCHORS.split(text)
    phrase = parts[-1] if len(parts) > 1 else text
    phrase = _TRAILING_CLAUSE.sub(" ", phrase)
    phrase = _PAREN.sub(" ", phrase)
    phrase = _POPULATION.sub(" ", phrase)
    phrase = re.sub(r"[^A-Za-z0-9\-/' ]+", " ", phrase)
    tokens = phrase.split()
    best: tuple[str, int] | None = None
    for i in range(len(tokens)):
        for j in range(len(tokens), i, -1):
            if best is not None and j - i <= len(best[0].split()):
                break
            span = " ".join(tokens[i:j])
            # A three-letter span is an acronym, and an ontology indexes acronyms as synonyms:
            # "osteoporosis in men" proposed multiple endocrine neoplasia from the word "men".
            if len(span) < 4:
                continue
            idx = kg.resolve(span, DISEASE, fuzzy)
            if idx is not None:
                best = (span, idx)
                break
    return best if best is not None else (" ".join(tokens), None)
