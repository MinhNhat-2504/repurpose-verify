"""Command-line entry point: ``python -m repurpose_verify <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

from .config import load_config
from .context import Context

log = logging.getLogger("repurpose_verify")


# ---------------------------------------------------------------------- data
def cmd_download(ctx: Context, args) -> None:
    from .utils import download

    cfg = ctx.cfg
    if args.what in ("kg", "all"):
        if args.kg_csv:
            download(cfg["kg"]["url"], cfg.path("kg", "raw_file"))
        else:
            download(cfg["kg"]["nodes_url"], cfg.path("kg", "nodes_file"))
            download(cfg["kg"]["edges_url"], cfg.path("kg", "edges_file"))
    if args.what in ("mondo", "all"):
        download(cfg["kg"]["mondo_url"], cfg.path("kg", "mondo_file"))
    if args.what in ("hgnc", "all"):
        download(cfg["kg"]["hgnc_url"], cfg.path("kg", "hgnc_file"))


def cmd_build_kg(ctx: Context, args) -> None:
    from .data.kg import build_from_parts, build_from_primekg

    cfg = ctx.cfg
    out = cfg.path("kg", "processed_dir")
    nodes, edges, kg_csv = (cfg.path("kg", "nodes_file"), cfg.path("kg", "edges_file"),
                            cfg.path("kg", "raw_file"))
    if nodes.exists() and edges.exists() and not args.from_kg_csv:
        build_from_parts(nodes, edges, out)
    elif kg_csv.exists():
        build_from_primekg(kg_csv, out)
    else:
        sys.exit(f"No PrimeKG input found. Run 'download kg' first (looked for {nodes}, {kg_csv})")


def cmd_build_gt(ctx: Context, args) -> None:
    from .data.ctgov import ClinicalTrialsClient
    from .data.groundtruth import build_groundtruth

    cfg, g = ctx.cfg, ctx.cfg["groundtruth"]
    ct = ClinicalTrialsClient(ctx.http("ctgov", 0.2))
    log.info("Fetching pre-cutoff trials (any phase) ...")
    pre = list(ct.iter_years(args.first_year, cfg.cutoff))
    log.info("Fetching post-cutoff trials %s ...", g["post_phases"])
    post = list(ct.iter_years(cfg.cutoff.year, cfg.eval_end, g["post_phases"],
                              first_date=date.fromordinal(cfg.cutoff.toordinal() + 1)))
    log.info("%d pre-cutoff and %d post-cutoff trials", len(pre), len(post))
    gt, report = build_groundtruth(
        ctx.kg, pre, post, cfg.cutoff, cfg.eval_end, single_drug=g["single_drug_trials"],
        post_phases=g["post_phases"], manual_path=cfg.path("groundtruth", "manual_approvals"),
        fuzzy=g.get("fuzzy_threshold", 0), n_eval=g.get("n_eval_diseases", 0),
        min_pos=g.get("min_positives_per_disease", 1),
        min_disease_genes=g.get("min_disease_genes", 0), seed=g.get("seed", 13))
    report.update(n_pre_trials=len(pre), n_post_trials=len(post))
    gt.save(cfg.path("groundtruth", "out_dir"), report)
    print(json.dumps({k: v for k, v in report.items() if k != "unmapped_manual_approvals"},
                     indent=2, default=str))


def cmd_audit_mapping(ctx: Context, args) -> None:
    from .data.ctgov import ClinicalTrialsClient
    from .data.groundtruth import TrialMapper, audit_mapping

    cfg, g = ctx.cfg, ctx.cfg["groundtruth"]
    ct = ClinicalTrialsClient(ctx.http("ctgov", 0.2))
    trials = list(ct.iter_years(cfg.cutoff.year, cfg.eval_end, g["post_phases"],
                                first_date=date.fromordinal(cfg.cutoff.toordinal() + 1)))
    df = audit_mapping(ctx.kg, trials, TrialMapper(ctx.kg, g.get("fuzzy_threshold", 0)),
                       n=args.n, seed=g.get("seed", 13))
    out = Path(args.out or cfg.path("groundtruth", "out_dir") / "mapping_audit.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    mapped = (df.mapped_drugs != "") & (df.mapped_diseases != "")
    print(f"Wrote {len(df)} sampled trials to {out}\n"
          f"both endpoints mapped: {mapped.mean():.1%} "
          f"(drug {(df.mapped_drugs != '').mean():.1%}, disease {(df.mapped_diseases != '').mean():.1%})\n"
          "Fill drug_correct_yn / disease_correct_yn by hand and report the precision.")


# ---------------------------------------------------------------------- systems
def _eval_diseases(ctx: Context, limit: int | None) -> list[int]:
    ids = [int(x) for x in ctx.gt.eval_diseases["disease_idx"]]
    return ids[:limit] if limit else ids


def _write_run(out_dir: Path, records: list[dict], meta: dict) -> None:
    from .utils import write_jsonl

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "rankings.jsonl", records)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def cmd_baseline(ctx: Context, args) -> None:
    from .baselines.heuristics import popularity_scores, proximity_scores, ranking_record

    kg = ctx.masked_kg()
    known = ctx.gt.known_by_disease()
    diseases = _eval_diseases(ctx, args.limit)
    name = args.run_name or args.method
    records = []
    if args.method == "proximity":
        records = [ranking_record(kg, d, proximity_scores(kg, d), name, known.get(d, set()))
                   for d in diseases]
    elif args.method == "popularity":
        pop = popularity_scores(kg)
        records = [ranking_record(kg, d, pop, name, known.get(d, set())) for d in diseases]
    elif args.method in ("transe", "distmult", "complex"):
        from .baselines.kge import KGEBaseline, KGEConfig

        kcfg = KGEConfig(model=args.method, dim=args.dim, epochs=args.epochs,
                         batch_size=args.batch_size, negatives=args.negatives, lr=args.lr,
                         max_triples=args.max_triples, device=args.device)
        model = KGEBaseline(kg, kcfg)
        ckpt = ctx.cfg.runs_dir / name / "model.pt"
        if ckpt.exists() and not args.retrain:
            model.load(ckpt)
        else:
            model.train()
            model.save(ckpt)
        records = [ranking_record(kg, d, model.scores(d), name, known.get(d, set()))
                   for d in diseases]
    _write_run(ctx.cfg.runs_dir / name, records, {"run_name": name, "model": args.method,
                                                  "knowledge_cutoff": str(ctx.cfg.cutoff)})
    print(f"Wrote {len(records)} rankings to {ctx.cfg.runs_dir / name}")


def cmd_import(ctx: Context, args) -> None:
    from .baselines.external import import_rankings

    records, unmapped = import_rankings(ctx.kg, Path(args.csv), args.run_name,
                                        ctx.gt.known_by_disease())
    _write_run(ctx.cfg.runs_dir / args.run_name, records,
               {"run_name": args.run_name, "model": args.run_name, "source_csv": args.csv,
                "knowledge_cutoff": args.knowledge_cutoff})
    print(f"Imported {len(records)} diseases; unmapped rows: {unmapped}")


def cmd_agent(ctx: Context, args) -> None:
    from .agent.agent import AgentSettings, RepurposingAgent, run_agent
    from .agent.llm import make_client

    cfg = ctx.cfg
    profile = cfg.model(args.model)
    a = cfg["agent"]
    settings = AgentSettings(
        mode=args.mode, verifier=args.verifier,
        unverified_policy=args.policy or a.get("unverified_policy", "demote"),
        top_k=args.top_k or a.get("top_k", 20), max_steps=args.max_steps or a.get("max_steps", 16),
        revise_rounds=a.get("revise_rounds", 1),
        tool_mode=args.tool_mode or profile.get("tool_mode", "native"),
        use_kg=not args.no_kg, use_candidates=not args.no_candidates,
        use_pubmed=not args.no_pubmed)
    kg = ctx.masked_kg()
    known = ctx.gt.known_by_disease()
    verifier = ctx.verifier()
    pubmed = ctx.pubmed if settings.use_pubmed else None
    run_name = args.run_name or f"{args.model}__{settings.system_name}"

    def factory():
        return RepurposingAgent(make_client(profile), kg, verifier, AgentSettings(**settings.__dict__),
                                str(cfg.cutoff), pubmed=pubmed, known_by_disease=known,
                                pubmed_max_results=cfg["pubmed"].get("max_results", 5))

    meta = {"run_name": run_name, "model": profile["model"], "profile": args.model,
            "backend": profile.get("backend"), "knowledge_cutoff": profile.get("knowledge_cutoff", ""),
            "price_in_per_mtok": profile.get("price_in_per_mtok", 0),
            "price_out_per_mtok": profile.get("price_out_per_mtok", 0),
            "settings": settings.__dict__, "system": settings.system_name,
            "cutoff": str(cfg.cutoff)}
    out = cfg.runs_dir / run_name / "rankings.jsonl"
    run_agent(factory, _eval_diseases(ctx, args.limit), out,
              workers=args.workers or a.get("workers", 1), meta=meta)
    print(f"Run written to {out}")


def cmd_reverify(ctx: Context, args) -> None:
    """Re-score saved proposals with the current verifier, leaving the LLM calls untouched."""
    from .agent.agent import AgentSettings, RepurposingAgent
    from .utils import read_jsonl, write_jsonl
    from .verify.schema import Proposal

    ver = ctx.verifier()
    kg = ctx.masked_kg()
    for run_dir in (Path(r) for r in args.runs):
        path = run_dir / "rankings.jsonl"
        records = list(read_jsonl(path))
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8")) \
            if (run_dir / "meta.json").exists() else {}
        settings = AgentSettings(**meta.get("settings", {})) if meta.get("settings") \
            else AgentSettings(mode="rag", verifier="none")
        agent = RepurposingAgent.__new__(RepurposingAgent)   # ranking logic only, no LLM
        agent.kg, agent.s = kg, settings
        before = after = 0
        for rec in records:
            props = [Proposal.from_any(p) for p in rec.get("proposals", [])]
            checks = [ver.verify(p, int(rec["disease_idx"])) for p in props]
            before += sum(bool(p.get("verification", {}).get("verified"))
                          for p in rec.get("proposals", []))
            after += sum(c.verified for c in checks)
            rec["proposals"] = [{**p.to_dict(), "verification": c.to_dict()}
                                for p, c in zip(props, checks)]
            rec["ranked"] = agent._rank(props, checks)
        if args.in_place:
            write_jsonl(path, records)
            out = run_dir
        else:
            out = run_dir.parent / f"{run_dir.name}{args.suffix}"
            write_jsonl(out / "rankings.jsonl", records)
            (out / "meta.json").write_text(json.dumps(
                {**meta, "run_name": out.name, "reverified_from": run_dir.name}, indent=2),
                encoding="utf-8")
        print(f"{run_dir.name}: verified proposals {before} -> {after}  (written to {out})")


def cmd_probe(ctx: Context, args) -> None:
    from .agent.llm import make_client
    from .eval.leakage_probe import (build_probe_items, run_probe, summarize_probe,
                                     write_probe_report)

    profile = ctx.cfg.model(args.model)
    items = build_probe_items(ctx.gt, ctx.kg, n_controls=args.controls,
                              eval_only=not args.all_positives)
    if args.limit:
        items = items[:args.limit]
    out = ctx.cfg.runs_dir / "probes" / f"{args.model}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    run_probe(make_client(profile), ctx.kg, items, ctx.cfg.cutoff.year, out, workers=args.workers)
    print(json.dumps(summarize_probe(out), indent=2))
    print(f"Report: {write_probe_report(out, ctx.cfg.cutoff.year)}")


# ---------------------------------------------------------------------- evaluation
def cmd_evaluate(ctx: Context, args) -> None:
    from .eval.evaluate import evaluate_runs, load_probe_mask, write_report

    e = ctx.cfg["eval"]
    ks = [int(k) for k in e.get("ks", [1, 5, 10, 20])]
    run_dirs = [Path(r) for r in args.runs] if args.runs else sorted(
        p.parent for p in ctx.cfg.runs_dir.glob("*/rankings.jsonl"))
    probe_keep = load_probe_mask(Path(args.probe)) if args.probe else None
    res = evaluate_runs(ctx.gt, run_dirs, ks, e.get("bootstrap", 1000), e.get("seed", 13),
                        min_tier=args.min_tier, probe_keep=probe_keep, reference=args.reference,
                        limit=args.limit)
    title = f"Time-split repurposing benchmark (cutoff {ctx.cfg.cutoff}, tier >= {args.min_tier}"
    if args.limit:
        title += f", first {args.limit} eval diseases"
    title += ", probe-masked)" if probe_keep is not None else ")"
    out = write_report(res, Path(args.out) if args.out else ctx.cfg.runs_dir / "report", ks, title)
    print(out.read_text(encoding="utf-8"))


def cmd_validate(ctx: Context, args) -> None:
    """Independent validation that needs no expert rater (ChEMBL MoA, post-cutoff literature)."""
    from .eval.evaluate import load_run
    from .eval.external_validation import (ChemblMoA, PostCutoffLiterature, validate_literature,
                                           validate_moa, write_external_report)

    cfg = ctx.cfg
    results: dict[str, dict] = {}
    moa = ChemblMoA(ctx.http("chembl", 0.15)) if args.what in ("moa", "all") else None
    lit = None
    if args.what in ("literature", "all"):
        pm = cfg["pubmed"]
        lit = PostCutoffLiterature(ctx.http("pubmed_post"), cfg.cutoff,
                                   pm.get("api_key_env", "NCBI_API_KEY"), pm.get("email", ""))
    known = ctx.gt.known_by_disease()
    for run_dir in (Path(r) for r in args.runs):
        name, _, recs = load_run(run_dir)
        records = list(recs.values())
        if args.limit:
            records = records[:args.limit]
        entry: dict = {}
        if moa is not None:
            log.info("%s: checking drug-target claims against ChEMBL", name)
            entry["moa"] = validate_moa(ctx.kg, records, moa,
                                        verified_only=not args.all_proposals,
                                        max_edges=args.max_edges).rates()
        if lit is not None:
            log.info("%s: checking post-cutoff literature co-mention", name)
            entry["literature"] = validate_literature(ctx.kg, records, lit, top_k=args.top_k,
                                                      controls_per_disease=args.controls,
                                                      known_by_disease=known)
        results[name] = entry
    out = Path(args.out) if args.out else cfg.runs_dir / "report_external" / "report.md"
    path = write_external_report(results, out, cfg.cutoff)
    print(json.dumps(results, indent=2, default=str))
    print(f"Report: {path}")


def cmd_review(ctx: Context, args) -> None:
    from .eval.review import export_review, score_review

    if args.action == "export":
        sheet, key = export_review([Path(r) for r in args.runs], Path(args.out),
                                   n_items=args.n_items, kg=ctx.kg)
        print(f"Sheet for raters: {sheet}\nKey (keep hidden): {key}")
    else:
        print(json.dumps(score_review(Path(args.sheet_a), Path(args.sheet_b), Path(args.key)),
                         indent=2))


def cmd_verify(ctx: Context, args) -> None:
    from .data.kg import DISEASE
    from .verify.schema import Proposal

    data = json.loads(Path(args.proposals).read_text(encoding="utf-8"))
    disease = ctx.kg.resolve(args.disease, DISEASE)
    if disease is None:
        sys.exit(f"Disease not found: {args.disease}")
    ver = ctx.verifier()
    for item in data if isinstance(data, list) else data.get("proposals", []):
        res = ver.verify(Proposal.from_any(item), disease)
        print(json.dumps({"drug": item.get("drug"), **res.to_dict()}, indent=1, default=str))


# ---------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="repurpose-verify", description=__doc__)
    p.add_argument("--config", default=None, help="TOML file merged over configs/default.toml")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("download", help="download PrimeKG and/or MONDO")
    s.add_argument("what", choices=["kg", "mondo", "hgnc", "all"], nargs="?", default="all")
    s.add_argument("--kg-csv", action="store_true",
                   help="fetch the 982 MB kg.csv instead of nodes.tab + edges.csv (395 MB)")
    s.set_defaults(fn=cmd_download)

    s = sub.add_parser("build-kg", help="convert the PrimeKG download into compact files")
    s.add_argument("--from-kg-csv", action="store_true", help="force building from kg.csv")
    s.set_defaults(fn=cmd_build_kg)

    s = sub.add_parser("build-gt", help="build the time-split ground truth from ClinicalTrials.gov")
    s.add_argument("--first-year", type=int, default=2000,
                   help="trials before this year are fetched in a single open-ended query")
    s.set_defaults(fn=cmd_build_gt)

    s = sub.add_parser("audit-mapping", help="sample trials to hand-check entity mapping")
    s.add_argument("-n", type=int, default=200)
    s.add_argument("--out")
    s.set_defaults(fn=cmd_audit_mapping)

    s = sub.add_parser("baseline", help="run a KG baseline")
    s.add_argument("method", choices=["proximity", "popularity", "transe", "distmult", "complex"])
    s.add_argument("--run-name")
    s.add_argument("--limit", type=int)
    s.add_argument("--dim", type=int, default=128)
    s.add_argument("--epochs", type=int, default=5)
    s.add_argument("--batch-size", type=int, default=8192)
    s.add_argument("--negatives", type=int, default=32)
    s.add_argument("--lr", type=float, default=1e-3)
    s.add_argument("--max-triples", type=int, default=0)
    s.add_argument("--device", default="cuda")
    s.add_argument("--retrain", action="store_true")
    s.set_defaults(fn=cmd_baseline)

    s = sub.add_parser("import-rankings", help="import an external ranking CSV (e.g. TxGNN)")
    s.add_argument("csv")
    s.add_argument("--run-name", required=True)
    s.add_argument("--knowledge-cutoff", default="")
    s.set_defaults(fn=cmd_import)

    s = sub.add_parser("agent", help="run the LLM agent / LLM baselines")
    s.add_argument("--model", required=True, help="model profile name from the config")
    s.add_argument("--mode", choices=["closed_book", "rag"], default="rag")
    s.add_argument("--verifier", choices=["none", "filter", "revise"], default="revise")
    s.add_argument("--policy", choices=["drop", "demote"])
    s.add_argument("--tool-mode", choices=["native", "react"])
    s.add_argument("--no-kg", action="store_true")
    s.add_argument("--no-candidates", action="store_true")
    s.add_argument("--no-pubmed", action="store_true")
    s.add_argument("--top-k", type=int)
    s.add_argument("--max-steps", type=int)
    s.add_argument("--limit", type=int)
    s.add_argument("--workers", type=int)
    s.add_argument("--run-name")
    s.set_defaults(fn=cmd_agent)

    s = sub.add_parser("reverify", help="re-score saved runs with the current verifier")
    s.add_argument("runs", nargs="+")
    s.add_argument("--in-place", action="store_true")
    s.add_argument("--suffix", default="_rv")
    s.set_defaults(fn=cmd_reverify)

    s = sub.add_parser("probe", help="weight-level knowledge leakage probe")
    s.add_argument("--model", required=True)
    s.add_argument("--controls", type=int, default=200)
    s.add_argument("--limit", type=int)
    s.add_argument("--all-positives", action="store_true",
                   help="probe every positive, not only those in the evaluation diseases")
    s.add_argument("--workers", type=int, default=1,
                   help="raise only against a server that serves requests in parallel")
    s.set_defaults(fn=cmd_probe)

    s = sub.add_parser("evaluate", help="score runs and write the report")
    s.add_argument("--runs", nargs="*")
    s.add_argument("--out")
    s.add_argument("--min-tier", choices=["phase2", "phase3", "phase4", "approved"], default="phase2")
    s.add_argument("--probe", help="probe jsonl: evaluate only pairs the model does not know")
    s.add_argument("--reference", help="system name for paired bootstrap comparisons")
    s.add_argument("--limit", type=int,
                   help="score only the first N evaluation diseases (matches agent --limit)")
    s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("validate", help="independent validation without an expert rater")
    s.add_argument("what", choices=["moa", "literature", "all"], nargs="?", default="all")
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--top-k", type=int, default=5, help="proposals per disease for co-mention")
    s.add_argument("--controls", type=int, default=5, help="random control drugs per disease")
    s.add_argument("--limit", type=int, help="first N diseases only")
    s.add_argument("--max-edges", type=int, default=0, help="cap drug-target edges checked")
    s.add_argument("--all-proposals", action="store_true",
                   help="check unverified proposals too, not only verified ones")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_validate)

    s = sub.add_parser("review", help="blinded pharmacist review")
    s.add_argument("action", choices=["export", "score"])
    s.add_argument("--runs", nargs="*", default=[])
    s.add_argument("--out", default="runs/review")
    s.add_argument("--n-items", type=int, default=50)
    s.add_argument("--sheet-a")
    s.add_argument("--sheet-b")
    s.add_argument("--key")
    s.set_defaults(fn=cmd_review)

    s = sub.add_parser("verify", help="verify a JSON file of proposals for one disease")
    s.add_argument("proposals")
    s.add_argument("--disease", required=True)
    s.set_defaults(fn=cmd_verify)
    return p


def main(argv: list[str] | None = None) -> None:
    # Reports carry disease and drug names from MONDO and DrugBank; a console in a legacy
    # code page must not abort a finished run over a character it cannot render.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "urllib3", "openai", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    ctx = Context(load_config(args.config))
    args.fn(ctx, args)


if __name__ == "__main__":
    main()
