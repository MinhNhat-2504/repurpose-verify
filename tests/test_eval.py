import json

import numpy as np
import pandas as pd
import pytest

from repurpose_verify.data.groundtruth import GroundTruth
from repurpose_verify.data.pubmed import parse_pubmed_xml
from repurpose_verify.eval.evaluate import evaluate_runs, write_report
from repurpose_verify.eval.metrics import bootstrap_ci, paired_bootstrap, per_disease_ranking
from repurpose_verify.eval.review import export_review, score_review, weighted_kappa


def test_ranking_metrics():
    m = per_disease_ranking([5, 1, 7, 3], positives={3, 9}, known={1}, ks=[1, 3])
    # filtered list: [5, 7, 3] -> first hit at rank 3
    assert m["mrr"] == pytest.approx(1 / 3)
    assert m["hit@1"] == 0 and m["hit@3"] == 1 and m["recall@3"] == 0.5
    assert per_disease_ranking([], {1}, set(), [1])["hit@1"] == 0


def test_bootstrap():
    mean, lo, hi = bootstrap_ci(np.array([0, 1] * 50), n=500)
    assert mean == 0.5 and lo < 0.5 < hi
    res = paired_bootstrap(np.ones(30), np.zeros(30), n=200)
    assert res["diff"] == 1 and res["p"] == 0


def _gt():
    pos = pd.DataFrame({"drug_idx": [0, 1], "disease_idx": [9, 10], "tier": ["phase3", "phase2"]})
    known = pd.DataFrame({"drug_idx": [2], "disease_idx": [9], "source": ["kg"]})
    ev = pd.DataFrame({"disease_idx": [9, 10], "disease_name": ["crc", "cml"], "n_positives": [1, 1]})
    return GroundTruth(pos, known, ev)


def _write_run(path, name, ranked, usage=None, meta=None):
    path.mkdir(parents=True)
    with open(path / "rankings.jsonl", "w") as f:
        for d, drugs in ranked.items():
            f.write(json.dumps({"disease_idx": d, "disease_name": str(d), "ranked": [
                {"drug_idx": x} for x in drugs], "proposals": [
                {"drug": "x", "path": [{"source": "a", "relation": "r", "target": "b"}],
                 "verification": {"verified": False, "structure_ok": True, "drug_idx": 1,
                                  "edges": [{"status": "unsupported", "pmids_invalid": ["1"]}]}}],
                "usage": usage or {"input_tokens": 1000, "output_tokens": 100}}) + "\n")
    (path / "meta.json").write_text(json.dumps({"run_name": name, **(meta or {})}))


def test_evaluate_and_report(tmp_path):
    gt = _gt()
    _write_run(tmp_path / "a", "a", {9: [2, 0], 10: [1]}, meta={"price_in_per_mtok": 1.0})
    _write_run(tmp_path / "b", "b", {9: [5]})
    res = evaluate_runs(gt, [tmp_path / "a", tmp_path / "b"], ks=[1, 5], n_boot=200, reference="a")
    s = res["summary"].set_index("system")
    assert s.loc["a", "hit@1"] == 1.0          # known drug 2 filtered out, drug 0 is rank 1
    assert s.loc["b", "hit@5"] == 0.0 and s.loc["b", "coverage"] == 0.5
    assert s.loc["a", "fabricated_edge_rate"] == 1.0
    assert s.loc["a", "cost_per_correct_hit"] == pytest.approx(0.002 / 2)
    res3 = evaluate_runs(gt, [tmp_path / "a"], ks=[1], n_boot=50, min_tier="phase3")
    assert res3["n_diseases"] == 1
    masked = evaluate_runs(gt, [tmp_path / "a"], ks=[1], n_boot=50, probe_keep={(1, 10)})
    assert masked["n_positive_pairs"] == 1
    report = write_report(res, tmp_path / "rep", [1, 5], "t").read_text(encoding="utf-8")
    assert "Hit@5" in report and "Paired bootstrap" in report


def test_review_roundtrip(tmp_path, kg):
    _write_run(tmp_path / "a", "a", {9: [0]})
    _write_run(tmp_path / "b", "b", {9: [1]})
    sheet, key = export_review([tmp_path / "a", tmp_path / "b"], tmp_path / "rv", n_items=2, kg=kg)
    df = pd.read_csv(sheet, encoding="utf-8-sig")
    assert len(df) == 2 and "system" not in df.columns
    for name, vals in (("ra.csv", [4, 2]), ("rb.csv", [5, 2])):
        d = df.copy()
        d["plausibility_1to5"] = vals
        d["mechanism_validity_1to5"] = vals
        d.to_csv(tmp_path / name, index=False, encoding="utf-8-sig")
    out = score_review(tmp_path / "ra.csv", tmp_path / "rb.csv", key)
    assert set(out["by_system"]) == {"a", "b"}
    from repurpose_verify.eval.review import name_refs

    assert name_refs("kg:0 inhibits [kg:4]", kg) == "Metformin inhibits PTGS2"
    assert name_refs("kg:99999 stays", kg) == "kg:99999 stays"
    assert weighted_kappa(np.array([1, 2, 3]), np.array([1, 2, 3]), [1, 2, 3, 4, 5]) == pytest.approx(1.0)


def test_pubmed_xml():
    xml = """<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>42</PMID><Article>
    <Journal><JournalIssue><PubDate><Year>2020</Year><Month>Mar</Month></PubDate></JournalIssue>
    <Title>J</Title></Journal><ArticleTitle>Metformin <i>and</i> cancer</ArticleTitle>
    <Abstract><AbstractText>Part one.</AbstractText><AbstractText>Part two.</AbstractText></Abstract>
    </Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"""
    art = parse_pubmed_xml(xml)["42"]
    assert art.title == "Metformin and cancer" and art.abstract == "Part one. Part two."
    assert art.pub_date.isoformat() == "2020-03-01"
