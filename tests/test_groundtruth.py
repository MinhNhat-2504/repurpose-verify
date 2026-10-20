from datetime import date

from repurpose_verify.data.ctgov import Trial, parse_study
from repurpose_verify.data.groundtruth import build_groundtruth

CUTOFF, END = date(2021, 12, 31), date(2026, 12, 31)


def trial(nct, when, phases, drugs, conds, mesh=None, status="COMPLETED"):
    return Trial(nct, when, phases, status, conds,
                 condition_mesh=[{"id": m, "term": ""} for m in (mesh or [])],
                 interventions=[{"type": "DRUG", "name": d} for d in drugs])


def test_parse_study():
    s = {"protocolSection": {
        "identificationModule": {"nctId": "NCT1"},
        "statusModule": {"overallStatus": "RECRUITING", "startDateStruct": {"date": "2023-05"}},
        "conditionsModule": {"conditions": ["Colorectal Cancer"]},
        "designModule": {"phases": ["PHASE2", "PHASE3"]},
        "armsInterventionsModule": {"interventions": [{"type": "DRUG", "name": "Metformin"}]}},
        "derivedSection": {"conditionBrowseModule": {"meshes": [{"id": "D015179", "term": "Colorectal Neoplasms"}]}}}
    t = parse_study(s)
    assert t.start_date == date(2023, 5, 1) and t.max_phase == 3
    assert t.condition_mesh[0]["id"] == "D015179"


def test_time_split(kg):
    pre = [trial("NCT_pre", date(2019, 3, 1), ["PHASE1"], ["Celecoxib"], ["colorectal cancer"])]
    post = [
        # novel pair -> positive (condition mapped through the MeSH xref)
        trial("NCT_a", date(2023, 1, 5), ["PHASE3"], ["Metformin hydrochloride", "Placebo"],
              ["Colorectal Neoplasms"], mesh=["D015179"]),
        # already an indication in the KG -> not novel
        trial("NCT_b", date(2022, 6, 1), ["PHASE3"], ["Metformin"], ["Type 2 diabetes"],
              mesh=["D003924"]),
        # already trialled before the cutoff -> not novel
        trial("NCT_c", date(2022, 2, 1), ["PHASE2"], ["Celecoxib"], ["colorectal cancer"]),
        # combination trial -> dropped by the single-drug rule
        trial("NCT_d", date(2024, 1, 1), ["PHASE2"], ["Imatinib + Celecoxib"], ["type 2 diabetes mellitus"]),
        # phase 1 -> below the post-cutoff phase floor
        trial("NCT_e", date(2024, 1, 1), ["PHASE1"], ["Imatinib"], ["colorectal cancer"]),
        # withdrawn -> ignored
        trial("NCT_f", date(2024, 1, 1), ["PHASE2"], ["Imatinib"], ["colorectal cancer"], status="WITHDRAWN"),
        # before the cutoff -> not in the post window
        trial("NCT_g", date(2021, 6, 1), ["PHASE3"], ["Imatinib"], ["type 2 diabetes mellitus"]),
    ]
    gt, report = build_groundtruth(kg, pre, post, CUTOFF, END)
    pos = set(zip(gt.positives.drug_idx, gt.positives.disease_idx))
    assert pos == {(0, 9)}
    row = gt.positives.iloc[0]
    assert row.tier == "phase3" and row.refs == "NCT_a"
    known = set(zip(gt.known.drug_idx, gt.known.disease_idx))
    assert {(0, 8), (2, 10), (1, 9)} <= known
    assert list(gt.eval_diseases.disease_idx) == [9]
    assert report["leakage_check_positives_in_kg"] == 0
    assert gt.positives_by_disease("phase3") == {9: {0}}
    assert gt.positives_by_disease("approved") == {}


def test_audit_mapping(kg):
    from repurpose_verify.data.groundtruth import TrialMapper, audit_mapping

    trials = [trial("NCT_a", date(2023, 1, 5), ["PHASE3"], ["Metformin"], ["Colorectal Neoplasms"],
                    mesh=["D015179"]),
              trial("NCT_x", date(2023, 1, 5), ["PHASE2"], ["Unobtainium"], ["Sniffles"])]
    df = audit_mapping(kg, trials, TrialMapper(kg), n=10)
    row = df.set_index("nct_id").loc["NCT_a"]
    assert row.mapped_drugs == "Metformin [kg:0]"
    assert row.mapped_diseases == "colorectal cancer [kg:9]"
    assert df.set_index("nct_id").loc["NCT_x"].mapped_drugs == ""
    assert "drug_correct_yn" in df.columns


def test_manual_approvals(kg, tmp_path):
    csv = tmp_path / "approvals.csv"
    csv.write_text("drug,disease,date,source\nImatinib,colorectal cancer,2024-03-01,FDA\n"
                   "Imatinib,colorectal cancer,2020-03-01,FDA\n", encoding="utf-8")
    gt, _ = build_groundtruth(kg, [], [], CUTOFF, END, manual_path=csv)
    assert gt.positives.tier.tolist() == ["approved"]
