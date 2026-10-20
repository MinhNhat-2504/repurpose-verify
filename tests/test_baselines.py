import pytest

from repurpose_verify.baselines.heuristics import popularity_scores, proximity_scores, ranking_record


def test_heuristics(kg):
    rec = ranking_record(kg, 9, proximity_scores(kg, 9), "prox", exclude={1})
    assert [r["drug_idx"] for r in rec["ranked"]][0] == 0
    assert 1 not in [r["drug_idx"] for r in rec["ranked"]]
    assert all(kg.types[r["drug_idx"]] == "drug" for r in rec["ranked"])
    pop = popularity_scores(kg)
    assert pop[0] >= 1 and pop[2] >= 1 and pop[1] < 1


def test_kge_trains(kg):
    torch = pytest.importorskip("torch")
    from repurpose_verify.baselines.kge import KGEBaseline, KGEConfig

    for name in ("transe", "distmult", "complex"):
        model = KGEBaseline(kg, KGEConfig(model=name, dim=8, epochs=3, batch_size=8, negatives=4,
                                          device="cpu", lr=0.05))
        losses = model.train()
        assert len(losses) == 3 and all(l == l for l in losses)
        s = model.scores(9)
        assert s.shape == (kg.n,)
