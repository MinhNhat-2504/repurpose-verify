"""Assemble the data deposit: the artefacts that are ours to release.

    python scripts/make_data_deposit.py [--out dist/deposit] [--licence "CC BY 4.0"]

A repository licence is only ours to grant over data we created, so this packages the curated
inputs, the derived ground truth, the model runs and the reports, plus the two caches that are
works of the United States government, and leaves out the third-party sources - each identified
by version and access date in the manuscript instead, and refetchable with `download all`.

Writes one archive per logical group, a MANIFEST.tsv carrying every member file with its size and
SHA-256, and a README stating what is included and what is not.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (archive name, source directory, description for the manifest, excluded path fragments)
GROUPS: list[tuple[str, str, str, tuple[str, ...]]] = [
    ("curated_inputs", "data/manual",
     "Hand-curated inputs: the frozen evaluation split, 114 judged FDA approvals and the "
     "worksheet they came from, the 200-trial entity-mapping audit with its judgements, and the "
     "alias overrides.", ()),
    ("ground_truth", "data/processed/groundtruth",
     "Derived time-split ground truth: novel positives with tier, date and trial identifiers; "
     "pairs known at the cutoff; the evaluation-disease table; the build report; and the "
     "grounding-correction difference.", ()),
    ("model_runs", "runs",
     "Every agent run: proposals with their full verification record, token accounting and "
     "rankings, the leakage probes, the knowledge-graph baselines, and all generated reports. "
     "This is the part of the work that cannot be regenerated - the agents sample, and the same "
     "prompt returns different proposals on a second pass.",
     ("__smoke", "/logs/", "runs/_", "complex/model.pt")),
    ("figures", "paper/figures", "Manuscript figures as PNG and PDF.", ()),
    ("cache_clinicaltrials", "data/cache/ctgov",
     "Cached ClinicalTrials.gov API v2 responses, queried 2026-10-20. A work of the United "
     "States government. Included because the registry is revised in place: a record refetched "
     "later is not the record that produced these labels.", ()),
    ("cache_fda_letters", "data/cache/fda_letters",
     "Cached Drugs@FDA approval letters (PDF and extracted text), retrieved 2026-10-24. A work "
     "of the United States government, and the evidence for every curated approval.", ()),
]

# Deliberately excluded, with the reason the curators will ask for.
EXCLUDED = [
    ("data/raw/primekg", "PrimeKG, refetchable from doi:10.7910/DVN/IXA7BM under its own terms"),
    ("data/raw/mondo", "Mondo Disease Ontology, distributed under CC BY 4.0"),
    ("data/raw/hgnc", "HGNC complete set, refetchable from genenames.org"),
    ("data/raw/drugsatfda", "Drugs@FDA bulk release, refetchable from fda.gov"),
    ("data/processed/kg", "a re-encoding of PrimeKG, so a derivative of it; `build-kg` rebuilds "
                          "it in minutes from the source above"),
    ("data/cache/chembl", "ChEMBL, distributed under CC BY-SA 3.0, whose share-alike terms we "
                          "cannot relicense"),
    ("data/cache/pubmed", "PubMed records whose abstracts carry publisher copyright"),
    ("data/cache/pubmed_post", "as above, the post-cutoff literature queries"),
    ("runs/complex/model.pt", "a 132 MB PyTorch checkpoint, which is a Python pickle rather than "
                              "an open format; training is seeded, so "
                              "`baseline complex --epochs 5` regenerates it, and the rankings it "
                              "produced are deposited"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="dist/deposit")
    ap.add_argument("--licence", default="CC BY 4.0",
                    help="the licence the deposit is released under; GigaDB requires CC0")
    args = ap.parse_args()
    out = (ROOT / args.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    manifest: list[tuple[str, str, int, str]] = []
    summary: list[tuple[str, int, int, str]] = []
    for name, rel, description, skip in GROUPS:
        src = ROOT / rel
        if not src.exists():
            print(f"  ! {rel} is missing, skipping {name}")
            continue
        members = [f for f in sorted(src.rglob("*")) if f.is_file()
                   and not any(frag in f.as_posix() for frag in skip)]
        archive = out / f"{name}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for f in members:
                tar.add(f, arcname=f"{name}/{f.relative_to(src).as_posix()}")
                manifest.append((name, f.relative_to(ROOT).as_posix(), f.stat().st_size, ""))
        summary.append((name, len(members), archive.stat().st_size, description))
        print(f"  {archive.name:28s} {len(members):6d} files  "
              f"{archive.stat().st_size / 1e6:9.1f} MB")

    lines = ["archive\tpath_in_repository\tbytes\tsha256_of_archive"]
    digests = {name: sha256(out / f"{name}.tar.gz") for name, _, _, _ in summary}
    for group, path, size, _ in manifest:
        lines.append(f"{group}\t{path}\t{size}\t{digests.get(group, '')}")
    (out / "MANIFEST.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    readme = ["# Supporting data for \"Can an LLM agent justify what it recommends?\"",
              "",
              "A mechanism-verified, time-split benchmark for drug repurposing.",
              "Code: https://github.com/MinhNhat-2504/repurpose-verify (MIT).",
              "",
              "## Archives", ""]
    for name, n, size, description in summary:
        readme += [f"### `{name}.tar.gz`  ({n} files, {size / 1e6:.1f} MB)", "", description, ""]
    readme += [f"Everything here is released under {args.licence}.", "",
               "## Not included, and why", "",
               f"{args.licence} applies only to data we created. These sources are therefore "
               "identified by version and access date in the manuscript rather than "
               "redistributed here; `python -m repurpose_verify download all` refetches them "
               "into the same layout.",
               ""]
    for rel, why in EXCLUDED:
        readme += [f"* `{rel}` — {why}"]
    readme += ["", "## Integrity", "",
               "`MANIFEST.tsv` lists every file in every archive with its size, and the SHA-256 "
               "of the archive that contains it:", ""]
    for name in digests:
        readme += [f"* `{name}.tar.gz`  `{digests[name]}`"]
    readme += [""]
    (out / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    total = sum(size for _, _, size, _ in summary)
    print(f"\n  {len(manifest)} files in {len(summary)} archives, {total / 1e6:.1f} MB total")
    print(f"  {out}")


if __name__ == "__main__":
    main()
