from datetime import date

import pytest

from repurpose_verify.eval.external_validation import (ChemblMoA, PostCutoffLiterature,
                                                       drug_gene_edges, validate_literature,
                                                       validate_moa, write_external_report)

CUTOFF = date(2021, 12, 31)


class FakeHTTP:
    """Stands in for CachedHTTP: canned UniChem / ChEMBL / E-utilities responses."""

    def __init__(self, chembl_for=None, mechanisms=None, targets=None, counts=None):
        self.chembl_for = chembl_for or {}
        self.mechanisms = mechanisms or {}
        self.targets = targets or {}
        self.counts = counts or {}
        self.min_interval = 0.0
        self.calls = []

    def post(self, url, body, **kw):
        self.calls.append(("post", url, body))
        chembl = self.chembl_for.get(body["compound"])
        if chembl is None:
            return {"compounds": []}
        return {"compounds": [{"sources": [{"shortName": "chembl", "compoundId": chembl}]}]}

    def get(self, url, params=None, **kw):
        self.calls.append(("get", url, params))
        if "mechanism.json" in url:
            mid = params["molecule_chembl_id"]
            return {"mechanisms": [{"target_chembl_id": t}
                                   for t in self.mechanisms.get(mid, [])]}
        if "/target/" in url:
            tid = url.rsplit("/", 1)[-1].replace(".json", "")
            return {"target_components": [
                {"target_component_synonyms": [{"syn_type": "GENE_SYMBOL",
                                                "component_synonym": g}]}
                for g in self.targets.get(tid, [])]}
        if "esearch" in url:
            return {"esearchresult": {"count": str(self.counts.get(params["term"], 0))}}
        raise AssertionError(f"unexpected url {url}")


def _record(disease_idx, proposals, ranked=None):
    return {"disease_idx": disease_idx, "disease_name": "d", "ranked": ranked or [],
            "proposals": proposals}


def _proposal(drug_idx, gene_idx, verified=True):
    return {"drug": "x", "verification": {
        "verified": verified, "drug_idx": drug_idx,
        "edges": [{"source_idx": drug_idx, "target_idx": gene_idx, "status": "kg"},
                  {"source_idx": gene_idx, "target_idx": 9, "status": "kg"}]}}


def test_drug_gene_edges_filters(kg):
    recs = [_record(9, [_proposal(0, 3), _proposal(1, 4, verified=False), _proposal(0, 3)])]
    assert drug_gene_edges(recs, kg) == [(0, 3)]                       # deduped, verified only
    assert drug_gene_edges(recs, kg, verified_only=False) == [(0, 3), (1, 4)]


def test_moa_confirms_and_contradicts(kg):
    # Metformin (DB00331) -> PRKAA1 is confirmed; Celecoxib (DB00482) -> PTGS2 is contradicted
    # because ChEMBL records a different target for it here.
    http = FakeHTTP(chembl_for={"DB00331": "CHEMBL1431", "DB00482": "CHEMBL118"},
                    mechanisms={"CHEMBL1431": ["T1"], "CHEMBL118": ["T2"]},
                    targets={"T1": ["PRKAA1", "PRKAA2"], "T2": ["PTGS1"]})
    recs = [_record(9, [_proposal(0, 3), _proposal(1, 4)])]
    res = validate_moa(kg, recs, ChemblMoA(http))
    r = res.rates()
    assert r["edges_checked"] == 2 and r["edges_with_opinion"] == 2
    assert r["independent_confirmation_rate"] == 0.5
    assert r["drug_chembl_coverage"] == 1.0 and r["drug_moa_coverage"] == 1.0
    assert res.examples_confirmed == ["Metformin -[target]-> PRKAA1"]
    assert "Celecoxib -[target]-> PTGS2" in res.examples_contradicted[0]
    # The KG role of each edge is reported, since only "target" can be a mechanism of action.
    assert r["edges_by_role"] == {"target": 2} and r["non_target_role_share"] == 0.0
    assert r["confirmation_by_role"] == {"target": 0.5}


def test_moa_reports_coverage_not_disagreement(kg):
    """A drug ChEMBL has no opinion about must not count as a contradiction."""
    http = FakeHTTP(chembl_for={"DB00331": "CHEMBL1431"}, mechanisms={})   # mapped, no MoA
    res = validate_moa(kg, [_record(9, [_proposal(0, 3), _proposal(1, 4)])], ChemblMoA(http))
    r = res.rates()
    assert r["edges_with_opinion"] == 0
    assert r["drug_chembl_coverage"] == 0.5          # DB00482 unmapped
    assert r["drug_moa_coverage"] == 0.0             # DB00331 mapped but no mechanism
    assert r["independent_confirmation_rate"] != r["independent_confirmation_rate"]   # NaN


def test_literature_uses_a_control(kg):
    counts = {'"Metformin" AND "colorectal cancer"': 12,
              '"Celecoxib" AND "colorectal cancer"': 0}
    http = FakeHTTP(counts=counts)
    lit = PostCutoffLiterature(http, CUTOFF)
    assert lit.mindate == "2022/01/01"
    recs = [_record(9, [], ranked=[{"drug_idx": 0}, {"drug_idx": 1}])]
    out = validate_literature(kg, recs, lit, top_k=2, controls_per_disease=1, seed=1)
    assert out["system_pairs"] == 2 and out["system_comention_rate"] == 0.5
    assert out["control_pairs"] == 1
    assert out["lift_over_control"] == pytest.approx(0.5 - out["control_comention_rate"])
    assert "12 post-cutoff records" in out["examples"][0]


def test_external_report(tmp_path, kg):
    http = FakeHTTP(chembl_for={"DB00331": "CHEMBL1431"}, mechanisms={"CHEMBL1431": ["T1"]},
                    targets={"T1": ["PRKAA1"]})
    moa = validate_moa(kg, [_record(9, [_proposal(0, 3)])], ChemblMoA(http)).rates()
    lit = {"top_k": 5, "system_pairs": 10, "system_comention_rate": 0.4, "control_pairs": 10,
           "control_comention_rate": 0.1, "lift_over_control": 0.3, "examples": []}
    text = write_external_report({"sys": {"moa": moa, "literature": lit}},
                                 tmp_path / "r.md", CUTOFF).read_text(encoding="utf-8")
    assert "ChEMBL mechanism of action" in text and "+0.300" in text
    assert "1.000" in text
