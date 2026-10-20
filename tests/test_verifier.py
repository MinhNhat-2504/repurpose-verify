from datetime import date

from repurpose_verify.verify.schema import Proposal
from repurpose_verify.data.kg import GENE
from repurpose_verify.verify.verifier import MechanismVerifier

CUTOFF = date(2021, 12, 31)


def prop(drug, *edges):
    return Proposal.from_any({"drug": drug, "path": [
        {"source": s, "relation": r, "target": t, **({"pmids": p} if p else {})}
        for s, r, t, *rest in edges for p in [rest[0] if rest else None]]})


def test_verified_path(kg):
    v = MechanismVerifier(kg, CUTOFF)
    res = v.verify(prop("metformin", ("kg:0", "activates", "kg:3"), ("PRKAA1", "interacts", "TP53"),
                        ("TP53", "associated with", "colorectal cancer")), 9)
    assert res.structure_ok, res.issues
    assert [e.status for e in res.edges] == ["kg", "kg", "kg"]
    assert res.verified


def test_unsupported_and_ungrounded(kg):
    v = MechanismVerifier(kg, CUTOFF)
    res = v.verify(prop("metformin", ("metformin", "inhibits", "PTGS2"),
                        ("PTGS2", "associated with", "colorectal cancer")), 9)
    assert [e.status for e in res.edges] == ["unsupported", "kg"]
    assert not res.verified and res.n_fabricated == 1
    res = v.verify(prop("metformin", ("metformin", "inhibits", "FAKEGENE9"),
                        ("FAKEGENE9", "drives", "colorectal cancer")), 9)
    assert [e.status for e in res.edges] == ["ungrounded", "ungrounded"]


def test_unresolvable_is_not_fabrication(kg):
    """Prose endpoints ("inflammation") are unresolvable, not fabricated claims."""
    v = MechanismVerifier(kg, CUTOFF)
    res = v.verify(prop("celecoxib", ("Celecoxib", "reduces", "inflammation"),
                        ("inflammation", "drives", "colorectal cancer")), 9)
    assert [e.status for e in res.edges] == ["ungrounded", "ungrounded"]
    assert res.n_fabricated == 0 and res.n_unresolvable == 2
    # Whereas a link between two real nodes with no evidence is a fabricated claim.
    res = v.verify(prop("metformin", ("Metformin", "inhibits", "PTGS2"),
                        ("PTGS2", "associated with", "colorectal cancer")), 9)
    assert res.n_fabricated == 1 and res.n_unresolvable == 0


def test_circular_and_structure(kg):
    v = MechanismVerifier(kg, CUTOFF)
    res = v.verify(prop("metformin", ("metformin", "treats", "type 2 diabetes mellitus"),
                        ("type 2 diabetes mellitus", "related", "colorectal cancer")), 9)
    assert res.edges[0].status == "circular"
    assert "no gene/protein on the path" in res.issues
    res = v.verify(prop("metformin", ("metformin", "activates", "PRKAA1"),
                        ("PRKAA1", "associated", "type 2 diabetes mellitus")), 9)
    assert "path does not end at the queried disease" in res.issues
    assert not v.verify(Proposal("metformin"), 9).verified


def test_literature_evidence(kg, pubmed):
    v = MechanismVerifier(kg, CUTOFF, pubmed=pubmed, accept_literature=True)
    res = v.verify(prop("celecoxib", ("celecoxib", "modulates", "TP53", ["333"]),
                        ("TP53", "associated with", "colorectal cancer")), 9)
    assert res.edges[0].status == "literature"
    assert res.verified
    # Post-cutoff and non-existent PMIDs are invalid citations and give no support.
    res = v.verify(prop("metformin", ("metformin", "inhibits", "PTGS2", ["999", "12345"]),
                        ("PTGS2", "associated with", "colorectal cancer")), 9)
    assert res.edges[0].status == "unsupported"
    assert sorted(res.edges[0].pmids_invalid) == ["12345", "999"]
    off = MechanismVerifier(kg, CUTOFF, pubmed=pubmed, accept_literature=False)
    res = off.verify(prop("celecoxib", ("celecoxib", "modulates", "TP53", ["333"]),
                          ("TP53", "associated with", "colorectal cancer")), 9)
    assert res.edges[0].status == "unsupported"


def test_name_id_mismatch(kg):
    """A kg id attached to a different drug's name must not launder a fabricated entity."""
    v = MechanismVerifier(kg, CUTOFF)
    res = v.verify(prop("Rasagiline [kg:1]", ("Rasagiline [kg:1]", "inhibits", "kg:4"),
                        ("kg:4", "associated with", "kg:9")), 9)
    assert res.drug_idx is None
    assert res.edges[0].status == "ungrounded"
    assert "drug not in KG: Rasagiline [kg:1]" in res.issues
    # The same path with the node's real name verifies, as do harmless name variants.
    assert v.verify(prop("Celecoxib [kg:1]", ("Celecoxib [kg:1]", "inhibits", "kg:4"),
                         ("PTGS2 (COX-2) [kg:4]", "associated with", "colorectal cancer [kg:9]")),
                    9).verified


def test_ambiguous_endpoint_is_not_fabricated(kg):
    """A name matching two nodes is reported as ambiguous, never as an invented entity."""
    kg._ensure_index()
    # Give PTGS2 and ABL1 the same alias, as HGNC does for several historical symbols.
    kg._synonyms[4].append("DUPSYM")
    kg._synonyms[5].append("DUPSYM")
    kg._name_index = None
    v = MechanismVerifier(kg, CUTOFF)
    res = v.verify(prop("celecoxib", ("Celecoxib", "inhibits", "DUPSYM"),
                        ("DUPSYM", "associated with", "colorectal cancer")), 9)
    assert [e.status for e in res.edges] == ["ambiguous", "ambiguous"]
    assert res.n_fabricated == 0 and not res.verified
    assert "several nodes" in res.edges[0].detail
    # An exact primary-name match still wins over a shared alias.
    assert v.ground("PTGS2", (GENE,)) == 4


def test_curated_override_pins_a_mention(kg):
    kg.attach_overrides({"COX-2": 4})
    v = MechanismVerifier(kg, CUTOFF)
    assert v.ground("COX-2", (GENE,)) == 4
    assert v.verify(prop("celecoxib", ("Celecoxib", "inhibits", "COX-2"),
                         ("COX-2", "associated with", "colorectal cancer")), 9).verified


def test_node_list_paths(kg):
    p = Proposal.from_any({"drug": "Celecoxib", "path": ["Celecoxib", "PTGS2", "colorectal cancer"]})
    assert MechanismVerifier(kg, CUTOFF).verify(p, 9).verified
