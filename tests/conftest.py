"""Tiny PrimeKG-format graph used by every test.

idx  type          name                          edges
0    drug          Metformin (DB00331)           target PRKAA1; indication T2D
1    drug          Celecoxib (DB00482)           target PTGS2
2    drug          Imatinib (DB00619)            target ABL1; indication CML
3    gene/protein  PRKAA1                        PPI TP53; in AMPK pathway; T2D gene
4    gene/protein  PTGS2                         PPI TP53; colorectal cancer gene
5    gene/protein  ABL1                          CML gene
6    gene/protein  TP53                          in AMPK pathway; colorectal cancer gene
7    pathway       AMPK signaling
8    disease       type 2 diabetes mellitus      MONDO:0005148
9    disease       colorectal cancer             MONDO:0005575
10   disease       chronic myeloid leukemia      MONDO:0011996
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from repurpose_verify.data.kg import KnowledgeGraph, build_from_primekg
from repurpose_verify.data.mondo import parse_mondo_obo

NODES = [
    (0, "DB00331", "drug", "Metformin", "DrugBank"),
    (1, "DB00482", "drug", "Celecoxib", "DrugBank"),
    (2, "DB00619", "drug", "Imatinib", "DrugBank"),
    (3, "5562", "gene/protein", "PRKAA1", "NCBI"),
    (4, "5743", "gene/protein", "PTGS2", "NCBI"),
    (5, "25", "gene/protein", "ABL1", "NCBI"),
    (6, "7157", "gene/protein", "TP53", "NCBI"),
    (7, "R-HSA-380972", "pathway", "AMPK signaling", "REACTOME"),
    (8, "5148", "disease", "type 2 diabetes mellitus", "MONDO"),
    (9, "5575", "disease", "colorectal cancer", "MONDO"),
    (10, "11996", "disease", "chronic myeloid leukemia", "MONDO"),
]
EDGES = [
    ("drug_protein", "target", 0, 3), ("drug_protein", "target", 1, 4),
    ("drug_protein", "target", 2, 5),
    ("protein_protein", "ppi", 3, 6), ("protein_protein", "ppi", 4, 6),
    ("disease_protein", "associated with", 8, 3), ("disease_protein", "associated with", 9, 6),
    ("disease_protein", "associated with", 9, 4), ("disease_protein", "associated with", 10, 5),
    ("pathway_protein", "interacts with", 7, 3), ("pathway_protein", "interacts with", 7, 6),
    ("indication", "indication", 0, 8), ("indication", "indication", 2, 10),
]

MONDO_OBO = """format-version: 1.2

[Term]
id: MONDO:0005575
name: colorectal cancer
synonym: "colorectal carcinoma" EXACT []
xref: MESH:D015179

[Term]
id: MONDO:0005148
name: type 2 diabetes mellitus
synonym: "T2DM" EXACT []
xref: MESH:D003924

[Term]
id: MONDO:0011996
name: chronic myeloid leukemia
xref: MESH:D015464
"""


def write_primekg_csv(path: Path) -> None:
    by = {n[0]: n for n in NODES}
    rows = []
    for rel, disp, a, b in EDGES:
        for x, y in ((a, b), (b, a)):
            nx, ny = by[x], by[y]
            rows.append({"relation": rel, "display_relation": disp,
                         "x_index": x, "x_id": nx[1], "x_type": nx[2], "x_name": nx[3], "x_source": nx[4],
                         "y_index": y, "y_id": ny[1], "y_type": ny[2], "y_name": ny[3], "y_source": ny[4]})
    pd.DataFrame(rows).to_csv(path, index=False)


def write_parts_csv(nodes_path: Path, edges_path: Path) -> None:
    """PrimeKG's nodes.tab + edges.csv layout."""
    pd.DataFrame(NODES, columns=["node_index", "node_id", "node_type", "node_name",
                                 "node_source"]).to_csv(nodes_path, index=False)
    rows = [{"relation": rel, "display_relation": disp, "x_index": a, "y_index": b}
            for rel, disp, a, b in EDGES]
    pd.DataFrame(rows).to_csv(edges_path, index=False)


@pytest.fixture()
def kg_dir(tmp_path) -> Path:
    csv = tmp_path / "kg.csv"
    write_primekg_csv(csv)
    out = tmp_path / "kg"
    build_from_primekg(csv, out)
    obo = tmp_path / "mondo.obo"
    obo.write_text(MONDO_OBO, encoding="utf-8")
    return tmp_path


@pytest.fixture()
def kg(kg_dir) -> KnowledgeGraph:
    g = KnowledgeGraph.load(kg_dir / "kg")
    g.attach_mondo(parse_mondo_obo(kg_dir / "mondo.obo"))
    return g


class FakeArticle:
    def __init__(self, pmid, title, abstract, year):
        from datetime import date

        self.pmid, self.title, self.abstract = pmid, title, abstract
        self.pub_date = date(year, 1, 1)
        self.journal = ""

    def to_dict(self, max_chars=1200):
        return {"pmid": self.pmid, "title": self.title, "abstract": self.abstract,
                "pub_date": self.pub_date.isoformat()}


class FakePubMed:
    """In-memory stand-in for PubMedClient."""

    def __init__(self, articles, cutoff_year=2021):
        self.articles = {a.pmid: a for a in articles}
        self.cutoff_year = cutoff_year

    def fetch(self, pmids):
        return {p: self.articles[p] for p in pmids if p in self.articles}

    def is_admissible(self, art):
        return art.pub_date.year <= self.cutoff_year

    def search_articles(self, query, max_results=5):
        q = query.lower()
        hits = [a for a in self.articles.values()
                if any(w in (a.title + a.abstract).lower() for w in q.split()) and self.is_admissible(a)]
        return hits[:max_results]


@pytest.fixture()
def pubmed():
    return FakePubMed([
        FakeArticle("111", "Metformin activates AMPK", "Metformin activates PRKAA1 (AMPK).", 2015),
        FakeArticle("222", "TP53 and colorectal cancer", "TP53 mutations drive colorectal cancer.", 2010),
        FakeArticle("333", "Celecoxib and TP53", "Celecoxib modulates TP53 signalling.", 2012),
        FakeArticle("999", "Future paper", "Metformin trial in colorectal cancer.", 2024),
    ])
