"""Are the verified proposals repurposing candidates at all?

    python scripts/measure_candidate_eligibility.py

Mechanism verification establishes that the edges a proposal asserts exist. It says nothing about
whether the compound named has ever been given to a patient, and repurposing means finding a new
indication for an *existing* drug. This measures that directly rather than leaving it to a
reader's impression: every distinct drug named by a verified proposal is mapped DrugBank ->
ChEMBL through UniChem, and ChEMBL's highest development phase is read for it.

ChEMBL records max_phase 4 for an approved drug, 1 to 3 for one that reached a clinical trial,
and 0 or null for a compound that did not. A proposal naming a compound below phase 4 is not a
repurposing candidate in the sense the task requires, however well its mechanism checks out.

Writes runs/report_external/candidate_eligibility.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from repurpose_verify.config import load_config  # noqa: E402
from repurpose_verify.context import Context  # noqa: E402
from repurpose_verify.eval.external_validation import CHEMBL, ChemblMoA  # noqa: E402

RUNS = ["runs/qwen25-7b__rag_verify-revise", "runs/llama31-8b__rag_verify-revise"]
OUT = ROOT / "runs" / "report_external" / "candidate_eligibility.json"
BATCH = 20


def verified_proposals(run_dirs: list[str]) -> list[tuple[str, int]]:
    """(run, drug_idx) for every verified proposal."""
    out = []
    for rd in run_dirs:
        path = ROOT / rd / "rankings.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            for p in rec.get("proposals", []):
                v = p.get("verification", {})
                if v.get("verified") and v.get("drug_idx") is not None:
                    out.append((Path(rd).name, int(v["drug_idx"])))
    return out


def max_phases(http, chembl_ids: list[str]) -> dict[str, float | None]:
    """ChEMBL's highest development phase per molecule, fetched in batches."""
    phases: dict[str, float | None] = {}
    for i in range(0, len(chembl_ids), BATCH):
        chunk = [c for c in chembl_ids[i:i + BATCH] if c]
        if not chunk:
            continue
        data = http.get(f"{CHEMBL}/molecule.json",
                        params={"molecule_chembl_id__in": ",".join(chunk),
                                "only": "molecule_chembl_id,max_phase", "limit": BATCH})
        for mol in data.get("molecules", []) or []:
            phases[mol["molecule_chembl_id"]] = mol.get("max_phase")
        print(f"  {min(i + BATCH, len(chembl_ids))}/{len(chembl_ids)} molecules", end="\r")
    print()
    return phases


def main() -> None:
    cfg = load_config(str(ROOT / "configs" / "default.toml"))
    ctx = Context(cfg)
    kg = ctx.kg
    http = ctx.http("chembl", 0.2)
    moa = ChemblMoA(http)

    proposals = verified_proposals(RUNS)
    drug_idx = sorted({d for _, d in proposals})
    print(f"{len(proposals)} verified proposals over {len(drug_idx)} distinct drugs")

    drugbank = {i: str(kg.nodes.iloc[i]["id"]) for i in drug_idx}
    chembl = {i: moa.chembl_id(db) for i, db in drugbank.items()}
    mapped = [c for c in chembl.values() if c]
    print(f"{len(mapped)} map to ChEMBL")

    phases = max_phases(http, sorted(set(mapped)))

    def bucket(i: int) -> str:
        c = chembl.get(i)
        if not c:
            return "unmapped"
        p = phases.get(c)
        if p is None:
            return "no phase recorded"
        p = float(p)
        return "approved" if p >= 4 else ("reached a clinical trial" if p >= 1 else "preclinical")

    by_drug: dict[str, int] = {}
    for i in drug_idx:
        by_drug[bucket(i)] = by_drug.get(bucket(i), 0) + 1
    by_proposal: dict[str, int] = {}
    for _, i in proposals:
        by_proposal[bucket(i)] = by_proposal.get(bucket(i), 0) + 1

    n_d, n_p = len(drug_idx), len(proposals)
    summary = {
        "verified_proposals": n_p,
        "distinct_drugs": n_d,
        "drugs_by_status": by_drug,
        "proposals_by_status": by_proposal,
        "approved_drug_rate": round(by_drug.get("approved", 0) / n_d, 4) if n_d else None,
        "approved_proposal_rate": round(by_proposal.get("approved", 0) / n_p, 4) if n_p else None,
        "examples_not_approved": [
            kg.names[i] for i in drug_idx if bucket(i) in ("preclinical", "no phase recorded")
        ][:12],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "examples_not_approved"}, indent=2))
    print(f"\n  {OUT}")


if __name__ == "__main__":
    main()
