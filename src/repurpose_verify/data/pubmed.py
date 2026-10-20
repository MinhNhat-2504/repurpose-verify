"""Date-capped PubMed access via NCBI E-utilities.

Every search is issued with ``datetype=pdat&maxdate=<cutoff>`` and every fetched record is
re-checked against the cutoff, so the agent can only read literature published before it.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import date

from ..utils import CachedHTTP

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


@dataclass
class Article:
    pmid: str
    title: str
    abstract: str
    pub_date: date | None
    journal: str = ""

    def to_dict(self, max_chars: int = 1200) -> dict:
        d = asdict(self)
        d["pub_date"] = self.pub_date.isoformat() if self.pub_date else None
        if len(d["abstract"]) > max_chars:
            d["abstract"] = d["abstract"][:max_chars] + " ..."
        return d


def _parse_pubdate(node: ET.Element | None) -> date | None:
    if node is None:
        return None
    year = node.findtext("Year")
    if year is None:
        medline = node.findtext("MedlineDate") or ""
        year = medline[:4]
    if not (year and year.isdigit()):
        return None
    month_txt = (node.findtext("Month") or "1").strip().lower()[:3]
    month = int(month_txt) if month_txt.isdigit() else _MONTHS.get(month_txt, 1)
    day_txt = node.findtext("Day") or "1"
    try:
        return date(int(year), max(1, min(month, 12)), int(day_txt) if day_txt.isdigit() else 1)
    except ValueError:
        return date(int(year), max(1, min(month, 12)), 1)


def parse_pubmed_xml(xml_text: str) -> dict[str, Article]:
    out: dict[str, Article] = {}
    root = ET.fromstring(xml_text)
    for art in root.iter("PubmedArticle"):
        pmid = art.findtext(".//MedlineCitation/PMID") or ""
        a = art.find(".//MedlineCitation/Article")
        if a is None or not pmid:
            continue
        title = "".join(a.find("ArticleTitle").itertext()) if a.find("ArticleTitle") is not None else ""
        abstract = " ".join("".join(t.itertext()) for t in a.findall("Abstract/AbstractText"))
        pub = _parse_pubdate(a.find("Journal/JournalIssue/PubDate"))
        journal = a.findtext("Journal/Title") or ""
        out[pmid] = Article(pmid, title.strip(), abstract.strip(), pub, journal)
    return out


class PubMedClient:
    def __init__(self, http: CachedHTTP, cutoff: date, email: str = "",
                 api_key_env: str = "NCBI_API_KEY"):
        self.http = http
        self.cutoff = cutoff
        self.base = {"tool": "repurpose-verify"}
        if email:
            self.base["email"] = email
        key = os.environ.get(api_key_env, "")
        if key:
            self.base["api_key"] = key
        # NCBI policy: <=3 req/s without a key, <=10 with one.
        self.http.min_interval = max(self.http.min_interval, 0.12 if key else 0.35)

    def search(self, query: str, max_results: int = 5) -> list[str]:
        params = {**self.base, "db": "pubmed", "term": query, "retmode": "json",
                  "retmax": max_results, "sort": "relevance", "datetype": "pdat",
                  "mindate": "1800/01/01", "maxdate": self.cutoff.strftime("%Y/%m/%d")}
        data = self.http.get(f"{EUTILS}/esearch.fcgi", params=params)
        return data.get("esearchresult", {}).get("idlist", [])

    def fetch(self, pmids: list[str]) -> dict[str, Article]:
        pmids = [p for p in dict.fromkeys(str(p).strip() for p in pmids) if p.isdigit()]
        if not pmids:
            return {}
        out: dict[str, Article] = {}
        for i in range(0, len(pmids), 100):
            batch = pmids[i:i + 100]
            params = {**self.base, "db": "pubmed", "id": ",".join(batch), "retmode": "xml"}
            xml = self.http.get(f"{EUTILS}/efetch.fcgi", params=params, as_json=False)
            out.update(parse_pubmed_xml(xml))
        return out

    def is_admissible(self, article: Article) -> bool:
        return article.pub_date is not None and article.pub_date <= self.cutoff

    def search_articles(self, query: str, max_results: int = 5) -> list[Article]:
        arts = self.fetch(self.search(query, max_results))
        return [a for a in arts.values() if self.is_admissible(a)]
