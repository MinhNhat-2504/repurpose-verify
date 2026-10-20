"""ClinicalTrials.gov API v2 client (drug/biological interventional studies by start date)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Iterator

from ..utils import CachedHTTP

log = logging.getLogger(__name__)

API = "https://clinicaltrials.gov/api/v2/studies"
FIELDS = ",".join([
    "NCTId", "StartDate", "Phase", "OverallStatus", "Condition", "ConditionMeshTerm",
    "ConditionMeshId", "InterventionName", "InterventionType", "InterventionOtherName",
    "InterventionMeshTerm", "InterventionMeshId",
])
PHASE_RANK = {"EARLY_PHASE1": 0, "PHASE1": 1, "PHASE2": 2, "PHASE3": 3, "PHASE4": 4}


@dataclass
class Trial:
    nct_id: str
    start_date: date | None
    phases: list[str]
    status: str
    conditions: list[str]
    condition_mesh: list[dict] = field(default_factory=list)
    interventions: list[dict] = field(default_factory=list)
    intervention_mesh: list[dict] = field(default_factory=list)

    @property
    def max_phase(self) -> int:
        return max((PHASE_RANK.get(p, -1) for p in self.phases), default=-1)


def parse_date(text: str | None) -> date | None:
    if not text:
        return None
    parts = text.split("-")
    try:
        y = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 else 1
        d = int(parts[2]) if len(parts) > 2 else 1
        return date(y, m, d)
    except (ValueError, IndexError):
        return None


def parse_study(study: dict) -> Trial:
    ps = study.get("protocolSection", {})
    ds = study.get("derivedSection", {})
    return Trial(
        nct_id=ps.get("identificationModule", {}).get("nctId", ""),
        start_date=parse_date(ps.get("statusModule", {}).get("startDateStruct", {}).get("date")),
        phases=ps.get("designModule", {}).get("phases", []) or [],
        status=ps.get("statusModule", {}).get("overallStatus", ""),
        conditions=ps.get("conditionsModule", {}).get("conditions", []) or [],
        condition_mesh=ds.get("conditionBrowseModule", {}).get("meshes", []) or [],
        interventions=ps.get("armsInterventionsModule", {}).get("interventions", []) or [],
        intervention_mesh=ds.get("interventionBrowseModule", {}).get("meshes", []) or [],
    )


class ClinicalTrialsClient:
    def __init__(self, http: CachedHTTP, page_size: int = 1000):
        self.http = http
        self.page_size = page_size

    def iter_studies(self, start: date | None, end: date | None,
                     phases: list[str] | None = None) -> Iterator[Trial]:
        lo = start.isoformat() if start else "MIN"
        hi = end.isoformat() if end else "MAX"
        expr = (f"AREA[InterventionType](DRUG OR BIOLOGICAL) AND "
                f"AREA[StartDate]RANGE[{lo},{hi}]")
        if phases:
            expr += " AND AREA[Phase](" + " OR ".join(phases) + ")"
        params = {"fields": FIELDS, "pageSize": self.page_size, "filter.advanced": expr,
                  "format": "json"}
        token = None
        while True:
            if token:
                params["pageToken"] = token
            page = self.http.get(API, params=dict(params))
            for s in page.get("studies", []):
                yield parse_study(s)
            token = page.get("nextPageToken")
            if not token:
                break

    def iter_years(self, first_year: int, last_date: date, phases: list[str] | None = None,
                   first_date: date | None = None) -> Iterator[Trial]:
        """Query year by year so each cached page set stays small and resumable."""
        if first_date is None:
            yield from self.iter_studies(None, date(first_year - 1, 12, 31), phases)
            first_date = date(first_year, 1, 1)
        for year in range(first_date.year, last_date.year + 1):
            lo = max(first_date, date(year, 1, 1))
            hi = min(last_date, date(year, 12, 31))
            if lo > hi:
                continue
            log.info("CT.gov %s..%s", lo, hi)
            yield from self.iter_studies(lo, hi, phases)
