"""Ground a few mentions in a fresh interpreter, for the cross-process determinism test."""
from __future__ import annotations

import pathlib
import tempfile

from conftest import MONDO_OBO, write_primekg_csv

from repurpose_verify.data.kg import DISEASE, DRUG, KnowledgeGraph, build_from_primekg
from repurpose_verify.data.mondo import parse_mondo_obo

d = pathlib.Path(tempfile.mkdtemp())
write_primekg_csv(d / "kg.csv")
build_from_primekg(d / "kg.csv", d / "kg")
(d / "mondo.obo").write_text(MONDO_OBO, encoding="utf-8")
kg = KnowledgeGraph.load(d / "kg")
kg.attach_mondo(parse_mondo_obo(d / "mondo.obo"))

print([kg.resolve(q, DISEASE, 90) for q in ("colorectl cancer", "diabetes melitus")],
      [kg.resolve(q, DRUG, 85) for q in ("metrormin", "asprin")])
