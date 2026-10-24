from pathlib import Path

import numpy as np
from conftest import write_parts_csv

from repurpose_verify.data.kg import DISEASE, DRUG, GENE, KnowledgeGraph, build_from_parts
from repurpose_verify.data.normalize import (drug_variants, is_acronym, name_variants,
                                              uninvert)


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


def test_short_parentheticals_are_not_lookup_keys():
    """Abbreviations collide across specialties, and the expansion is in the same string."""
    # "(AD)" once resolved atopic dermatitis to Alzheimer disease, and "(CLL)" a chronic
    # leukaemia to an acute one.
    assert "ad" not in name_variants("mild to moderate atopic dermatitis (AD)")
    assert "cll" not in name_variants("chronic lymphocytic leukemia (CLL)")
    assert "mild to moderate atopic dermatitis" in name_variants(
        "mild to moderate atopic dermatitis (AD)")
    # Longer acronyms stay: they disambiguate rather than collide.
    assert "nsclc" in name_variants("non-small cell lung cancer (NSCLC)")
    assert "hfpef" in name_variants("heart failure (HFpEF)")


def test_bracketed_qualifiers_are_not_lookup_keys():
    """Ontologies and chemical names bracket qualifiers, and indexing those mis-grounds words."""
    # MONDO tags its labels, which made the bare word "clinical" resolve to a plasmacytoma.
    assert "clinical" not in name_variants("extramedullary plasmacytoma (clinical)")
    assert "disease" not in name_variants("glutaric aciduria (disease)")
    assert "acute" not in name_variants("gonococcal cervicitis (acute)")
    # Chemical nomenclature brackets isotopes, stereo descriptors and oxidation states. "(18F)"
    # in the KG's own name for FDG made every 18F PET tracer resolve to it: 33 of the 85
    # spurious ground-truth positives this class of key produced.
    assert "18f" not in name_variants("Fludeoxyglucose (18F)")
    assert "iii" not in name_variants("Porphyrin Fe(III)")
    assert not any(v in ("3r", "1ah") for v in name_variants("(3R)-3-(aminomethyl)-1(1aH)-one"))
    # A trial or study name in brackets is not an alias for the drug either.
    assert "emperor preserved" not in name_variants("empagliflozin (EMPEROR-Preserved)")
    assert is_acronym("NSCLC") and is_acronym("HFpEF")
    assert not is_acronym("clinical") and not is_acronym("EMPEROR-Preserved")


def test_category_nouns_never_resolve(kg):
    """MONDO lists the bare category noun as a synonym of its grouping terms."""
    for word in ("disease", "disorder", "syndrome", "clinical", "acute"):
        assert kg.resolve(word, DISEASE) is None, word
        assert kg.resolve(word, DRUG) is None, word


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


def test_fuzzy_candidates_do_not_depend_on_set_iteration_order(kg, monkeypatch):
    """The candidate pool must be a function of the query, not of a set's iteration order."""
    kg._ensure_index()
    query = "colorectl cancer"
    first = kg._fuzzy_candidates(DISEASE, query)

    # Shuffling every posting list and the trigram set leaves the pool unchanged.
    real = KnowledgeGraph._trigrams_of
    monkeypatch.setattr(KnowledgeGraph, "_trigrams_of",
                        staticmethod(lambda text: set(reversed(sorted(real(text))))))
    for names in kg._trigrams[DISEASE].values():
        names.reverse()
    assert kg._fuzzy_candidates(DISEASE, query) == first


def test_resolve_is_stable_across_processes():
    """Two interpreters with different string hash seeds must ground a mention identically.

    Set iteration order varies with PYTHONHASHSEED, so a tie broken by it is reproducible
    within one process and not between two. That is how a non-deterministic ground truth hides:
    two builds of this benchmark from identical inputs disagreed on three positives.
    """
    import os
    import subprocess
    import sys

    tests_dir = Path(__file__).parent
    child = tests_dir / "ground_once.py"
    env_path = os.pathsep.join([str(tests_dir), str(tests_dir.parent / "src")])
    seen = set()
    for seed in ("0", "1", "12345"):
        r = subprocess.run([sys.executable, str(child)], capture_output=True, text=True,
                           env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": env_path})
        assert r.returncode == 0, r.stderr
        seen.add(r.stdout.strip())
    assert len(seen) == 1, f"grounding varied with the hash seed: {seen}"
