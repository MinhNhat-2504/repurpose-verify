import csv
import io
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd

from repurpose_verify.data.fda import find_new_indication_approvals, write_candidates
from repurpose_verify.data.groundtruth import score_mapping_audit

CUTOFF, END = date(2021, 12, 31), date(2026, 12, 31)


def _tsv(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()), delimiter="\t",
                       lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def _archive(path: Path) -> Path:
    """A miniature Drugs@FDA release covering every case the filter must get right."""
    actions = [
        {"ActionTypes_LookupID": "8", "ActionTypes_LookupDescription": "Efficacy-New Indication",
         "SupplCategoryLevel1Code": "EFFICACY", "SupplCategoryLevel2Code": "INDICATION"},
        {"ActionTypes_LookupID": "7", "ActionTypes_LookupDescription": "Efficacy-New Dosing Regimen",
         "SupplCategoryLevel1Code": "EFFICACY", "SupplCategoryLevel2Code": "DOSING"},
    ]
    joins = [
        {"SubmissionType": "SUPPL", "j_submissionActionTypeID": "1", "ApplNo": "001",
         "SubmissionNo": "5", "ActionTypes_LookupID": "8"},     # wanted
        {"SubmissionType": "SUPPL", "j_submissionActionTypeID": "2", "ApplNo": "002",
         "SubmissionNo": "1", "ActionTypes_LookupID": "7"},     # dosing, not an indication
        {"SubmissionType": "SUPPL", "j_submissionActionTypeID": "3", "ApplNo": "003",
         "SubmissionNo": "2", "ActionTypes_LookupID": "8"},     # before the cutoff
        {"SubmissionType": "SUPPL", "j_submissionActionTypeID": "4", "ApplNo": "004",
         "SubmissionNo": "9", "ActionTypes_LookupID": "8"},     # not approved
        {"SubmissionType": "SUPPL", "j_submissionActionTypeID": "5", "ApplNo": "005",
         "SubmissionNo": "3", "ActionTypes_LookupID": "8"},     # drug not in the graph
    ]
    subs = [
        {"ApplNo": "001", "SubmissionClassCodeID": "2", "SubmissionType": "SUPPL",
         "SubmissionNo": "5", "SubmissionStatus": "AP",
         "SubmissionStatusDate": "2023-04-01 00:00:00", "SubmissionsPublicNotes": "",
         "ReviewPriority": "P"},
        {"ApplNo": "002", "SubmissionClassCodeID": "2", "SubmissionType": "SUPPL",
         "SubmissionNo": "1", "SubmissionStatus": "AP",
         "SubmissionStatusDate": "2023-05-01 00:00:00", "SubmissionsPublicNotes": "",
         "ReviewPriority": "S"},
        {"ApplNo": "003", "SubmissionClassCodeID": "2", "SubmissionType": "SUPPL",
         "SubmissionNo": "2", "SubmissionStatus": "AP",
         "SubmissionStatusDate": "2019-02-01 00:00:00", "SubmissionsPublicNotes": "",
         "ReviewPriority": "S"},
        {"ApplNo": "004", "SubmissionClassCodeID": "2", "SubmissionType": "SUPPL",
         "SubmissionNo": "9", "SubmissionStatus": "TA",
         "SubmissionStatusDate": "2023-06-01 00:00:00", "SubmissionsPublicNotes": "",
         "ReviewPriority": "S"},
        {"ApplNo": "005", "SubmissionClassCodeID": "2", "SubmissionType": "SUPPL",
         "SubmissionNo": "3", "SubmissionStatus": "AP",
         "SubmissionStatusDate": "2024-01-01 00:00:00", "SubmissionsPublicNotes": "",
         "ReviewPriority": "S"},
    ]
    products = [
        {"ApplNo": "001", "ProductNo": "1", "Form": "TABLET", "Strength": "1",
         "ReferenceDrug": "0", "DrugName": "GLUCOPHAGE", "ActiveIngredient": "METFORMIN",
         "ReferenceStandard": "0"},
        {"ApplNo": "002", "ProductNo": "1", "Form": "TABLET", "Strength": "1",
         "ReferenceDrug": "0", "DrugName": "CELEBREX", "ActiveIngredient": "CELECOXIB",
         "ReferenceStandard": "0"},
        {"ApplNo": "003", "ProductNo": "1", "Form": "TABLET", "Strength": "1",
         "ReferenceDrug": "0", "DrugName": "GLEEVEC", "ActiveIngredient": "IMATINIB",
         "ReferenceStandard": "0"},
        {"ApplNo": "004", "ProductNo": "1", "Form": "TABLET", "Strength": "1",
         "ReferenceDrug": "0", "DrugName": "GLEEVEC", "ActiveIngredient": "IMATINIB",
         "ReferenceStandard": "0"},
        {"ApplNo": "005", "ProductNo": "1", "Form": "TABLET", "Strength": "1",
         "ReferenceDrug": "0", "DrugName": "UNOBTANIUM", "ActiveIngredient": "UNOBTAINIUM",
         "ReferenceStandard": "0"},
    ]
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("ActionTypes_Lookup.txt", _tsv(actions))
        z.writestr("Join_Submission_ActionTypes_Lookup.txt", _tsv(joins))
        z.writestr("Submissions.txt", _tsv(subs))
        z.writestr("Products.txt", _tsv(products))
    return path


def test_only_new_indication_approvals_in_window(tmp_path, kg):
    found = find_new_indication_approvals(_archive(tmp_path / "daf.zip"), kg, CUTOFF, END)
    by_appl = {c.appl_no: c for c in found}
    # A dosing supplement, a pre-cutoff approval and an unapproved submission are all excluded.
    assert set(by_appl) == {"001", "005"}
    assert by_appl["001"].kg_name == "Metformin"
    assert by_appl["001"].approval_date == date(2023, 4, 1)
    assert by_appl["005"].kg_drug is None, "a drug absent from the graph has no match"


def test_worksheet_is_ready_to_fill(tmp_path, kg):
    found = find_new_indication_approvals(_archive(tmp_path / "daf.zip"), kg, CUTOFF, END)
    out = write_candidates(found, tmp_path / "cands.csv")
    df = pd.read_csv(out, encoding="utf-8-sig")
    assert list(df.columns)[:5] == ["drug", "disease", "date", "source", "note"], \
        "the first five columns paste straight into the curated approvals file"
    assert len(df) == 1 and df.drug.iloc[0] == "Metformin"      # unmatched drug dropped
    assert df.disease.isna().all(), "disease is what a person fills in"
    assert "accessdata.fda.gov" in df.fda_link.iloc[0]
    assert len(pd.read_csv(write_candidates(found, tmp_path / "all.csv", matched_only=False),
                           encoding="utf-8-sig")) == 2


def test_extract_indication_from_letter_text():
    """The operative clause names the indication; the boilerplate around it varies."""
    from repurpose_verify.data.fda import extract_indication

    real = ("Please refer to your Supplemental New Drug Application (sNDA) dated ... "
            "RINVOQ (upadacitinib) extended-release tablets. This Prior Approval supplemental "
            "new drug application provides for treatment of adults with giant cell arteritis. "
            "APPROVAL & LABELING We have completed our review of this application, as amended.")
    assert extract_indication(real) == "treatment of adults with giant cell arteritis"

    wrapped = ("This supplemental new drug application provides for the addition of a new "
               "indication for the treatment of adult patients with moderately to severely "
               "active ulcerative colitis. CONTENT OF LABELING ...")
    got = extract_indication(wrapped)
    assert got is not None and "ulcerative colitis" in got

    # A letter that never says what it provides for yields nothing rather than a guess.
    assert extract_indication("We have completed our review of this application.") is None


def test_worksheet_carries_the_letter_quote(tmp_path, kg):
    from repurpose_verify.data.fda import find_new_indication_approvals, write_candidates

    found = find_new_indication_approvals(_archive(tmp_path / "daf.zip"), kg, CUTOFF, END)
    letters = {("001", "5"): ("treatment of adults with colorectal cancer",
                              "https://example.invalid/letter.pdf")}
    out = write_candidates(found, tmp_path / "c.csv", letters=letters)
    df = pd.read_csv(out, encoding="utf-8-sig")
    row = df.iloc[0]
    assert row.letter_indication == "treatment of adults with colorectal cancer"
    assert row.letter_link == "https://example.invalid/letter.pdf"
    # The quote is evidence, not a decision: the disease column stays empty.
    assert pd.isna(row.disease)


def test_score_mapping_audit(tmp_path):
    path = tmp_path / "audit.csv"
    pd.DataFrame([
        {"mapped_drugs": "A [kg:1]", "mapped_diseases": "X [kg:2]",
         "drug_correct_yn": "y", "disease_correct_yn": "y"},
        {"mapped_drugs": "B [kg:3]", "mapped_diseases": "Y [kg:4]",
         "drug_correct_yn": "n", "disease_correct_yn": "y"},
        {"mapped_drugs": "C [kg:5]", "mapped_diseases": "", "drug_correct_yn": "Y",
         "disease_correct_yn": ""},
        # Nothing mapped: a recall miss, already counted as coverage, so not a precision error.
        {"mapped_drugs": "", "mapped_diseases": "", "drug_correct_yn": "",
         "disease_correct_yn": ""},
    ]).to_csv(path, index=False, encoding="utf-8-sig")
    out = score_mapping_audit(path)
    assert out["drug"] == {"mapped": 3, "judged": 3, "unjudged": 0, **out["drug"]}
    assert out["drug"]["precision"] == 2 / 3
    assert out["drug"]["ci_low"] <= 2 / 3 <= out["drug"]["ci_high"]
    assert out["disease"]["mapped"] == 2 and out["disease"]["precision"] == 1.0


def test_stable_sample_barely_moves_when_the_pool_changes():
    """Positional sampling reshuffled 129 of 200 evaluation diseases over an 18-disease change."""
    from repurpose_verify.data.groundtruth import stable_sample

    pool = list(range(1000, 2149))
    pick = lambda keys: {keys[i] for i in stable_sample(keys, 200, 13)}
    before = pick(pool)
    # Drop 18 diseases from the middle of the pool, as a mapping fix does.
    after = pick([k for k in pool if k not in set(range(1400, 1418))])
    assert len(before) == len(after) == 200
    # Only the dropped members and their replacements may move.
    assert len(before - after) <= 18, f"{len(before - after)} diseases changed"


def test_pinned_split_wins_over_sampling(tmp_path):
    from repurpose_verify.data.groundtruth import load_eval_split

    assert load_eval_split(None) == (None, [])
    assert load_eval_split(tmp_path / "absent.csv") == (None, [])
    path = tmp_path / "split.csv"
    path.write_text("disease_idx\n7\n9\n", encoding="utf-8")
    assert load_eval_split(path)[0] == {7, 9}


def test_propose_disease_strips_what_no_ontology_label_carries(kg):
    """The letter's clause is a sentence; the ontology term is the longest span inside it."""
    from repurpose_verify.data.fda import propose_disease

    # Qualifiers no label contains: "newly diagnosed", "metastatic", "seizures associated with".
    span, idx = propose_disease(
        "for the treatment of adult patients with metastatic colorectal cancer (CRC) who have "
        "progressed following prior therapy", kg)
    assert span == "colorectal cancer" and idx == 9
    # A clause that names no disease yields no proposal rather than a stray word: "clinical"
    # used to resolve to a plasmacytoma, so a junk span could look like a real approval.
    assert propose_disease("addition of new clinical study results", kg)[1] is None
    assert propose_disease("", kg)[1] is None
