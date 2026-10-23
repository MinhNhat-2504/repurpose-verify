import json
from datetime import date

import pytest

from repurpose_verify.agent.agent import AgentSettings, RepurposingAgent
from repurpose_verify.agent.llm import AssistantTurn, MockClient, ToolCall
from repurpose_verify.agent.tools import Toolbox
from repurpose_verify.utils import extract_json
from repurpose_verify.verify.verifier import MechanismVerifier

CUTOFF = date(2021, 12, 31)
GOOD = {"drug": "kg:0", "confidence": 0.8, "rationale": "AMPK-TP53 axis",
        "path": [{"source": "kg:0", "relation": "activates", "target": "kg:3"},
                 {"source": "kg:3", "relation": "ppi", "target": "kg:6"},
                 {"source": "kg:6", "relation": "associated with", "target": "kg:9"}]}
BAD = {"drug": "Imatinib", "path": [{"source": "Imatinib", "relation": "inhibits", "target": "TP53"},
                                     {"source": "TP53", "relation": "drives", "target": "colorectal cancer"}]}
FIXED = {"drug": "Celecoxib", "path": [{"source": "kg:1", "relation": "inhibits", "target": "kg:4"},
                                        {"source": "kg:4", "relation": "associated", "target": "kg:9"}]}


def call(name, **args):
    return AssistantTurn("", [ToolCall(f"id_{name}", name, args)])


def make_agent(kg, script, **settings):
    llm = MockClient({"model": "mock", **settings.pop("profile", {})}, script=script)
    return RepurposingAgent(llm, kg, MechanismVerifier(kg, CUTOFF), AgentSettings(**settings),
                            str(CUTOFF), known_by_disease={9: set()})


def test_toolbox(kg, pubmed):
    tb = Toolbox(kg, pubmed, known_by_disease={9: {1}})
    out, err = tb.execute("search_entities", {"query": "metformin", "entity_type": "drug"})
    assert not err and json.loads(out)[0]["id"] == "kg:0"
    out, _ = tb.execute("find_mechanism_paths", {"drug_id": "kg:0", "disease_id": "colorectal cancer"})
    assert json.loads(out)["paths"]
    out, _ = tb.execute("rank_candidate_drugs", {"disease_id": "kg:9"})
    assert [r["id"] for r in json.loads(out)] == ["kg:0"]          # celecoxib excluded as known
    out, _ = tb.execute("search_pubmed", {"query": "metformin"})
    assert {a["pmid"] for a in json.loads(out)} == {"111"}         # 2024 paper filtered out
    _, err = tb.execute("get_neighbors", {"entity_id": "nonexistent thing"})
    assert err
    tb2 = Toolbox(kg, None, use_kg=False, use_pubmed=False)
    assert [t["name"] for t in tb2.specs()] == ["submit_proposals"]
    assert tb2.execute("search_entities", {"query": "x"})[1]


def test_native_tool_loop(kg):
    script = [call("find_mechanism_paths", drug_id="kg:0", disease_id="kg:9"),
              call("submit_proposals", proposals=[GOOD, BAD])]
    agent = make_agent(kg, script, mode="rag", verifier="filter", unverified_policy="drop")
    rec = agent.run_disease(9)
    assert rec["trace"]["tool_calls"] == {"find_mechanism_paths": 1, "submit_proposals": 1}
    assert [r["drug_idx"] for r in rec["ranked"]] == [0]
    assert [p["verification"]["verified"] for p in rec["proposals"]] == [True, False]
    # verifier=none keeps the unverified proposal in place
    agent = make_agent(kg, [call("submit_proposals", proposals=[BAD, GOOD])], mode="rag", verifier="none")
    assert [r["drug_idx"] for r in agent.run_disease(9)["ranked"]] == [2, 0]


def test_revise_round(kg):
    def script(system, messages, tools):
        revising = any(m["role"] == "user" and "mechanism verifier" in m.get("content", "")
                       for m in messages)
        props = [GOOD, FIXED] if revising else [BAD, GOOD]
        return call("submit_proposals", proposals=props)

    agent = make_agent(kg, script, mode="rag", verifier="revise")
    rec = agent.run_disease(9)
    assert rec["trace"]["rounds"] == 1
    assert [r["drug_idx"] for r in rec["ranked"]] == [0, 1]
    assert all(r["verified"] for r in rec["ranked"])


def test_demote_policy(kg):
    agent = make_agent(kg, [call("submit_proposals", proposals=[BAD, GOOD])], mode="rag",
                       verifier="filter", unverified_policy="demote")
    assert [r["drug_idx"] for r in agent.run_disease(9)["ranked"]] == [0, 2]


def test_react_mode(kg):
    script = ['Let me look.\n<tool_call>{"name": "search_entities", "arguments": {"query": "TP53"}}</tool_call>',
              "<tool_call>" + json.dumps({"name": "submit_proposals", "arguments": {"proposals": [GOOD]}})
              + "</tool_call>"]
    agent = make_agent(kg, script, mode="rag", verifier="filter", tool_mode="react",
                       profile={"native_tools": False})
    rec = agent.run_disease(9)
    assert rec["trace"]["tool_calls"] == {"search_entities": 1, "submit_proposals": 1}
    assert [r["drug_idx"] for r in rec["ranked"]] == [0]


def test_closed_book(kg):
    text = "Here you go:\n```json\n" + json.dumps({"proposals": [BAD, GOOD]}) + "\n```"
    agent = make_agent(kg, [text], mode="closed_book")
    rec = agent.run_disease(9)
    assert rec["trace"]["steps"] == 1
    assert [r["drug_idx"] for r in rec["ranked"]] == [2, 0]   # closed-book is never filtered
    assert rec["proposals"][0]["verification"]["verified"] is False


def test_nudge_then_budget(kg):
    agent = make_agent(kg, ["I think metformin.", "still no tool call"], mode="rag", max_steps=3)
    rec = agent.run_disease(9)
    assert rec["ranked"] == [] and rec["trace"]["steps"] == 2


def test_run_agent_parallel_and_resumable(kg, tmp_path):
    from repurpose_verify.agent.agent import run_agent
    from repurpose_verify.utils import read_jsonl

    out = tmp_path / "run" / "rankings.jsonl"   # the run directory must be created for us
    factory = lambda: make_agent(kg, lambda *a: call("submit_proposals", proposals=[GOOD]),  # noqa: E731
                                 mode="rag", verifier="filter")
    run_agent(factory, [9, 10, 8], out, workers=3, meta={"run_name": "t"})
    assert {r["disease_idx"] for r in read_jsonl(out)} == {8, 9, 10}
    assert json.loads((tmp_path / "run" / "meta.json").read_text())["run_name"] == "t"

    # A second call must skip the diseases already on disk and only add the new one.
    def boom():
        raise AssertionError("should not re-run a finished disease")

    run_agent(lambda: boom(), [9, 10, 8], out, workers=2)
    assert len(list(read_jsonl(out))) == 3
    run_agent(factory, [9, 10, 8, 0], out, workers=2)
    assert len(list(read_jsonl(out))) == 4


def test_server_failures_are_not_results(kg, tmp_path):
    """A dead server must not fill the file with empty records that resume treats as done."""
    from repurpose_verify.agent.agent import ServerUnreachable, run_agent
    from repurpose_verify.utils import read_jsonl, write_jsonl

    out = tmp_path / "run" / "rankings.jsonl"

    def dead_server(*a):
        raise ConnectionError("connection refused")

    factory = lambda: make_agent(kg, dead_server, mode="rag", verifier="filter")  # noqa: E731
    with pytest.raises(ServerUnreachable):
        run_agent(factory, [8, 9, 10, 0, 1, 2, 3], out, workers=1, max_consecutive_failures=3)
    # Nothing was written: an unreachable server produced no results, not empty ones.
    assert not out.exists() or list(read_jsonl(out)) == []

    # A record stored from an earlier crash is re-run rather than counted as done.
    write_jsonl(out, [
        {"disease_idx": 9, "ranked": [], "proposals": [],
         "trace": {"errors": ["APIConnectionError: Connection error."]}},
        {"disease_idx": 10, "ranked": [{"drug_idx": 2}], "proposals": [],
         "trace": {"errors": []}},
    ])
    good = lambda *a: call("submit_proposals", proposals=[GOOD])  # noqa: E731
    run_agent(lambda: make_agent(kg, good, mode="rag", verifier="filter"), [9, 10], out, workers=1)
    rows = {r["disease_idx"]: r for r in read_jsonl(out)}
    assert set(rows) == {9, 10}
    assert rows[9]["ranked"] == [{"drug_idx": 0, "drug": "Metformin", "verified": True}]
    assert rows[10]["ranked"] == [{"drug_idx": 2}]      # the clean record was left alone


def test_extract_json():
    assert extract_json('noise {"a": [1, {"b": "}"}]} tail') == {"a": [1, {"b": "}"}]}
    assert extract_json("```json\n[1,2]\n```") == [1, 2]
