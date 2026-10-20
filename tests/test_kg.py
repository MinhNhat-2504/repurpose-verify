import numpy as np
from conftest import write_parts_csv

from repurpose_verify.data.kg import DISEASE, DRUG, GENE, KnowledgeGraph, build_from_parts
from repurpose_verify.data.normalize import drug_variants, name_variants, uninvert


def test_build_from_parts_matches_kg_csv(tmp_path, kg):
    """nodes.tab + edges.csv must yield the same graph as the monolithic kg.csv."""
    write_parts_csv(tmp_path / "nodes.csv", tmp_path / "edges.csv")
    build_from_parts(tmp_path / "nodes.csv", tmp_path / "edges.csv", tmp_path / "out")
    other = KnowledgeGraph.load(tmp_path / "out")
    assert other.n == kg.n
    assert other.names.tolist() == kg.names.tolist()
    assert other.drug_disease_pairs() == kg.drug_disease_pairs()
    for i in range(kg.n):
        assert sorted(other.neighbor_relations(i)) == sorted(kg.neighbor_relations(i))


def test_normalize():
    assert uninvert("Diabetes Mellitus, Type 2") == "Type 2 Diabetes Mellitus"
    assert "type 2 diabetes mellitus" in name_variants("Diabetes Mellitus, Type 2")
    assert "metformin" in drug_variants("Metformin Hydrochloride 500 mg")


def test_structure(kg):
    assert kg.n == 11
    assert set(kg.neighbors(0, ["drug_protein"]).tolist()) == {3}
    assert ("protein_protein", "ppi") in kg.edge_relations(3, 6)
    assert kg.edge_relations(6, 3) == kg.edge_relations(3, 6)
    assert kg.edge_relations(0, 4) == []
    assert kg.drug_disease_pairs() == {(0, 8), (2, 10)}


def test_lookup(kg):
    assert kg.resolve("metformin hydrochloride", DRUG) == 0
    assert kg.resolve("DB00619", DRUG) == 2
    assert kg.resolve("kg:6") == 6
    assert kg.resolve("TP53", GENE) == 6
    assert kg.resolve("Diabetes Mellitus, Type 2", DISEASE) == 8
    assert kg.resolve("colorectal carcinoma", DISEASE) == 9      # MONDO synonym
    assert kg.disease_for_mesh("D015179") == [9]                   # MeSH xref
    assert kg.resolve("MONDO:0011996", DISEASE) == 10
    assert kg.resolve("colorectl cancer", DISEASE, fuzzy_threshold=90) == 9
    assert kg.resolve("metrormin", DRUG, fuzzy_threshold=85) == 0      # typo inside the prefix
    assert kg.resolve("unobtainium", DRUG) is None
    assert kg.resolve("unobtainium", DRUG, fuzzy_threshold=85) is None


def test_mechanism_paths(kg):
    paths = [tuple(p["nodes"]) for p in kg.mechanism_paths(0, 9)]
    assert (0, 3, 6, 9) in paths            # drug -> target -PPI-> disease gene
    assert (0, 3, 7, 6, 9) in paths         # drug -> target -> pathway -> gene -> disease
    assert [tuple(p["nodes"]) for p in kg.mechanism_paths(1, 9)][0] == (1, 4, 9)
    assert kg.mechanism_paths(2, 9) == []


def test_drug_scores(kg):
    s = kg.drug_scores(9)
    assert s[1] > 0 and s[0] > 0
    assert s[1] > s[2]
    assert np.all(s[[3, 4, 5, 6, 7, 8, 9, 10]] == 0)


def test_mask_pairs(kg):
    assert kg.mask_pairs([(0, 8)]) == 1
    assert kg.edge_relations(0, 8) == []
    assert kg.drug_disease_pairs() == {(2, 10)}
    assert set(kg.neighbors(0).tolist()) == {3}
