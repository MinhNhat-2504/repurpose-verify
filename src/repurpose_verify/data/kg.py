"""Knowledge-graph store (PrimeKG format) with grounding, edge lookup and mechanism paths.

The graph is held as a symmetric CSR matrix whose rows are sorted, so an edge lookup is a
binary search and neighbour queries are slices. Relation and display-relation codes are kept
per stored edge so the verifier can report *what kind* of evidence supports an edge.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .normalize import drug_variants, name_variants, norm

log = logging.getLogger(__name__)

DRUG, DISEASE, GENE, PATHWAY = "drug", "disease", "gene/protein", "pathway"
# Direct drug-disease relations. They are the prediction target, so they are never accepted as
# mechanistic evidence and are the edges masked for held-out pairs.
DRUG_DISEASE_RELS = frozenset({"indication", "contraindication", "off-label use"})
KNOWN_USE_RELS = frozenset({"indication", "off-label use"})

PRIMEKG_COLS = ["relation", "display_relation", "x_index", "x_id", "x_type", "x_name",
                "x_source", "y_index", "y_id", "y_type", "y_name", "y_source"]
NODE_COLS = {"node_index": "idx", "node_id": "id", "node_type": "type", "node_name": "name",
             "node_source": "source"}


def _write_kg(nodes: pd.DataFrame, edges: pd.DataFrame, rel_vocab: dict[str, int],
              disp_vocab: dict[str, int], out_dir: Path) -> None:
    if not np.array_equal(nodes["idx"].to_numpy(), np.arange(len(nodes))):
        raise ValueError("Node indices are not contiguous 0..N-1")
    edges = edges[edges.src != edges.dst].drop_duplicates()
    nodes.to_csv(out_dir / "nodes.csv.gz", index=False)
    np.savez_compressed(out_dir / "edges.npz", src=edges.src.to_numpy(np.int32),
                        dst=edges.dst.to_numpy(np.int32), rel=edges.rel.to_numpy(np.int16),
                        disp=edges.disp.to_numpy(np.int16))
    vocab = {"relations": sorted(rel_vocab, key=rel_vocab.get),
             "display_relations": sorted(disp_vocab, key=disp_vocab.get)}
    (out_dir / "vocab.json").write_text(json.dumps(vocab, indent=1), encoding="utf-8")
    log.info("KG built: %d nodes, %d undirected edges", len(nodes), len(edges))


def build_from_parts(nodes_path: Path, edges_path: Path, out_dir: Path,
                     chunksize: int = 2_000_000) -> None:
    """Convert PrimeKG ``nodes.tab`` + ``edges.csv`` into compact node/edge files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes = pd.read_csv(nodes_path, dtype=str)
    nodes = nodes.rename(columns=NODE_COLS)[list(NODE_COLS.values())]
    nodes["idx"] = nodes["idx"].astype(np.int64)
    nodes = nodes.drop_duplicates("idx").sort_values("idx").reset_index(drop=True)
    src, dst, rel, disp = [], [], [], []
    rel_vocab: dict[str, int] = {}
    disp_vocab: dict[str, int] = {}
    for i, chunk in enumerate(pd.read_csv(edges_path, dtype=str, chunksize=chunksize)):
        log.info("edges chunk %d (%d rows)", i, len(chunk))
        x = chunk["x_index"].astype(np.int64).to_numpy()
        y = chunk["y_index"].astype(np.int64).to_numpy()
        src.append(np.minimum(x, y))
        dst.append(np.maximum(x, y))
        rel.append(chunk["relation"].map(
            lambda v: rel_vocab.setdefault(v, len(rel_vocab))).to_numpy(np.int16))
        disp.append(chunk["display_relation"].fillna("").map(
            lambda v: disp_vocab.setdefault(v, len(disp_vocab))).to_numpy(np.int16))
    edges = pd.DataFrame({"src": np.concatenate(src), "dst": np.concatenate(dst),
                          "rel": np.concatenate(rel), "disp": np.concatenate(disp)})
    _write_kg(nodes, edges, rel_vocab, disp_vocab, out_dir)


def build_from_primekg(csv_path: Path, out_dir: Path, chunksize: int = 1_000_000) -> None:
    """Convert PrimeKG ``kg.csv`` into compact node/edge files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src, dst, rel, disp = [], [], [], []
    node_rows: dict[int, tuple] = {}
    rel_vocab: dict[str, int] = {}
    disp_vocab: dict[str, int] = {}
    reader = pd.read_csv(csv_path, usecols=PRIMEKG_COLS, dtype=str, chunksize=chunksize,
                         low_memory=False)
    for i, chunk in enumerate(reader):
        log.info("PrimeKG chunk %d (%d rows)", i, len(chunk))
        for side in ("x", "y"):
            part = chunk[[f"{side}_index", f"{side}_id", f"{side}_type", f"{side}_name",
                          f"{side}_source"]].drop_duplicates(f"{side}_index")
            for idx, sid, ntype, name, source in part.itertuples(index=False):
                node_rows.setdefault(int(idx), (sid, ntype, name, source))
        x = chunk["x_index"].astype(np.int64).to_numpy()
        y = chunk["y_index"].astype(np.int64).to_numpy()
        r = chunk["relation"].map(lambda v: rel_vocab.setdefault(v, len(rel_vocab)))
        d = chunk["display_relation"].fillna("").map(
            lambda v: disp_vocab.setdefault(v, len(disp_vocab)))
        src.append(np.minimum(x, y))
        dst.append(np.maximum(x, y))
        rel.append(r.to_numpy(np.int16))
        disp.append(d.to_numpy(np.int16))
    nodes = pd.DataFrame([(idx, *vals) for idx, vals in sorted(node_rows.items())],
                         columns=["idx", "id", "type", "name", "source"])
    edges = pd.DataFrame({"src": np.concatenate(src), "dst": np.concatenate(dst),
                          "rel": np.concatenate(rel), "disp": np.concatenate(disp)})
    _write_kg(nodes, edges, rel_vocab, disp_vocab, out_dir)


def _source_ids(row_id: str, source: str) -> list[str]:
    """IDs under which a node can be referenced (MONDO groups are '_'-joined)."""
    sid = str(row_id)
    if source.upper().startswith("MONDO"):
        return [f"MONDO:{int(p):07d}" for p in sid.split("_") if p.isdigit()]
    if source.upper() == "NCBI":
        return [f"NCBIGENE:{sid}"]
    return [sid.upper()]


class KnowledgeGraph:
    def __init__(self, nodes: pd.DataFrame, src: np.ndarray, dst: np.ndarray, rel: np.ndarray,
                 disp: np.ndarray, relations: Sequence[str], display_relations: Sequence[str]):
        self.nodes = nodes.reset_index(drop=True)
        self.relations = list(relations)
        self.display_relations = list(display_relations)
        self.rel_code = {r: i for i, r in enumerate(self.relations)}
        self.n = len(self.nodes)
        self.types = self.nodes["type"].to_numpy()
        self.names = self.nodes["name"].astype(str).to_numpy()
        self._edges = (np.asarray(src, np.int32), np.asarray(dst, np.int32),
                       np.asarray(rel, np.int16), np.asarray(disp, np.int16))
        self._build_csr(*self._edges)
        self._name_index: dict[str, dict[str, list[int]]] | None = None
        self._id_index: dict[str, list[int]] | None = None
        self._mesh_index: dict[str, list[int]] = {}
        self._synonyms: dict[int, list[str]] = defaultdict(list)
        self._sparse: dict[str, object] = {}
        self._trigrams: dict[str, dict[str, list[str]]] = {}
        self._index_lock = threading.Lock()

    # ------------------------------------------------------------------ construction
    @classmethod
    def load(cls, processed_dir: Path) -> "KnowledgeGraph":
        processed_dir = Path(processed_dir)
        nodes = pd.read_csv(processed_dir / "nodes.csv.gz", dtype={"id": str, "name": str})
        e = np.load(processed_dir / "edges.npz")
        vocab = json.loads((processed_dir / "vocab.json").read_text(encoding="utf-8"))
        return cls(nodes, e["src"], e["dst"], e["rel"], e["disp"], vocab["relations"],
                   vocab["display_relations"])

    def _build_csr(self, src, dst, rel, disp) -> None:
        rows = np.concatenate([src, dst])
        cols = np.concatenate([dst, src])
        rels = np.concatenate([rel, rel])
        disps = np.concatenate([disp, disp])
        order = np.lexsort((cols, rows))
        self._cols = cols[order]
        self._rels = rels[order]
        self._disps = disps[order]
        counts = np.bincount(rows, minlength=self.n)
        self._indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    def mask_pairs(self, pairs: Iterable[tuple[int, int]],
                   relations: Iterable[str] = DRUG_DISEASE_RELS) -> int:
        """Remove drug-disease edges for held-out pairs. Returns number of edges removed."""
        keys = {(min(a, b), max(a, b)) for a, b in pairs}
        codes = {self.rel_code[r] for r in relations if r in self.rel_code}
        if not keys or not codes:
            return 0
        src, dst, rel, disp = self._edges
        hit = np.isin(rel, list(codes))
        drop = np.zeros(len(src), bool)
        for i in np.flatnonzero(hit):
            if (int(src[i]), int(dst[i])) in keys:
                drop[i] = True
        if drop.any():
            keep = ~drop
            self._edges = (src[keep], dst[keep], rel[keep], disp[keep])
            self._build_csr(*self._edges)
            self._sparse.clear()
        return int(drop.sum())

    # ------------------------------------------------------------------ basic queries
    def node(self, i: int) -> dict:
        r = self.nodes.iloc[int(i)]
        return {"kg_id": f"kg:{int(i)}", "name": str(r["name"]), "type": r["type"],
                "source": r["source"], "source_id": str(r["id"])}

    def degree(self, i: int) -> int:
        return int(self._indptr[i + 1] - self._indptr[i])

    def nodes_of_type(self, node_type: str) -> np.ndarray:
        return np.flatnonzero(self.types == node_type)

    def neighbors(self, i: int, relations: Iterable[str] | None = None,
                  node_type: str | None = None) -> np.ndarray:
        a, b = self._indptr[i], self._indptr[i + 1]
        cols = self._cols[a:b]
        mask = np.ones(len(cols), bool)
        if relations is not None:
            codes = [self.rel_code[r] for r in relations if r in self.rel_code]
            mask &= np.isin(self._rels[a:b], codes)
        if node_type is not None:
            mask &= self.types[cols] == node_type
        return np.unique(cols[mask])

    def neighbor_relations(self, i: int) -> list[tuple[int, str, str]]:
        a, b = self._indptr[i], self._indptr[i + 1]
        return [(int(c), self.relations[r], self.display_relations[d])
                for c, r, d in zip(self._cols[a:b], self._rels[a:b], self._disps[a:b])]

    def edge_relations(self, u: int, v: int) -> list[tuple[str, str]]:
        """All (relation, display_relation) pairs stored between u and v."""
        a, b = self._indptr[u], self._indptr[u + 1]
        lo = a + np.searchsorted(self._cols[a:b], v, side="left")
        hi = a + np.searchsorted(self._cols[a:b], v, side="right")
        return [(self.relations[self._rels[k]], self.display_relations[self._disps[k]])
                for k in range(lo, hi)]

    def drug_disease_pairs(self, relations: Iterable[str] = KNOWN_USE_RELS) -> set[tuple[int, int]]:
        """(drug, disease) pairs connected by the given relations."""
        codes = [self.rel_code[r] for r in relations if r in self.rel_code]
        src, dst, rel, _ = self._edges
        sel = np.isin(rel, codes)
        out = set()
        for s, d in zip(src[sel], dst[sel]):
            s, d = int(s), int(d)
            if self.types[s] == DRUG and self.types[d] == DISEASE:
                out.add((s, d))
            elif self.types[d] == DRUG and self.types[s] == DISEASE:
                out.add((d, s))
        return out

    # ------------------------------------------------------------------ grounding
    def attach_mondo(self, mondo_terms: dict) -> None:
        """Add MONDO synonyms and MeSH xrefs to disease nodes."""
        by_mondo: dict[int, list[int]] = defaultdict(list)
        for i in self.nodes_of_type(DISEASE):
            row = self.nodes.iloc[i]
            for sid in _source_ids(row["id"], str(row["source"])):
                by_mondo[int(sid.split(":")[1])].append(int(i))
        for mid, term in mondo_terms.items():
            for i in by_mondo.get(mid, []):
                self._synonyms[i].extend([term.name, *term.synonyms])
                for mesh in term.mesh_ids:
                    self._mesh_index.setdefault(mesh.upper(), []).append(i)
        self._name_index = None

    def _ensure_index(self) -> None:
        # The agent runs several diseases in parallel against one shared graph, so the index is
        # built under a lock and published last: _name_index is the flag the fast path checks.
        if self._name_index is not None:
            return
        with self._index_lock:
            if self._name_index is None:
                self._build_index()

    def _build_index(self) -> None:
        name_index: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        id_index: dict[str, list[int]] = defaultdict(list)
        for i, (name, ntype, sid, source) in enumerate(zip(
                self.names, self.types, self.nodes["id"].astype(str), self.nodes["source"].astype(str))):
            variants = drug_variants(name) if ntype == DRUG else name_variants(name)
            for syn in self._synonyms.get(i, []):
                variants += name_variants(syn)
            for v in dict.fromkeys(variants):
                name_index[ntype][v].append(i)
            for key in _source_ids(sid, source):
                id_index[key].append(i)
        self._id_index = dict(id_index)
        self._trigrams = {}
        self._name_index = {t: dict(d) for t, d in name_index.items()}

    # Blocking for fuzzy matching: candidates must share a character trigram with the query.
    # Without it every unmatched mention would be scored against every name of its type, which
    # is far too slow for the hundreds of thousands of trial mentions in the ground truth.
    MAX_POSTINGS = 2000      # ignore trigrams this common - they carry no selectivity
    FUZZY_CANDIDATES = 200

    @staticmethod
    def _trigrams_of(text: str) -> set[str]:
        padded = f"  {text} "
        return {padded[i:i + 3] for i in range(len(padded) - 2)}

    def _trigram_index(self, node_type: str) -> dict[str, list[str]]:
        index = self._trigrams.get(node_type)
        if index is None:
            postings: dict[str, list[str]] = defaultdict(list)
            for name in self._name_index.get(node_type, {}):
                for tri in self._trigrams_of(name):
                    postings[tri].append(name)
            index = self._trigrams[node_type] = dict(postings)
        return index

    def _fuzzy_candidates(self, node_type: str, query: str) -> list[str]:
        index = self._trigram_index(node_type)
        counts: Counter[str] = Counter()
        for tri in self._trigrams_of(query):
            names = index.get(tri)
            if names and len(names) <= self.MAX_POSTINGS:
                counts.update(names)
        return [name for name, _ in counts.most_common(self.FUZZY_CANDIDATES)]

    def lookup(self, mention: str, node_type: str | None = None, fuzzy_threshold: float = 0,
               limit: int = 10) -> list[int]:
        """Ground a mention (KG id, source id, MeSH id or name) to candidate node indices."""
        self._ensure_index()
        m = mention.strip()
        if not m:
            return []
        types = [node_type] if node_type else list(self._name_index)
        if m.lower().startswith("kg:") and m[3:].isdigit():
            i = int(m[3:])
            return [i] if 0 <= i < self.n and (not node_type or self.types[i] == node_type) else []
        key = m.upper()
        if key.startswith("MESH:"):
            key = key[5:]
        hits = [i for i in self._mesh_index.get(key, []) if not node_type or self.types[i] == node_type]
        hits += [i for i in self._id_index.get(key, []) if not node_type or self.types[i] == node_type]
        if hits:
            return list(dict.fromkeys(hits))[:limit]
        variants = drug_variants(m) if node_type == DRUG else name_variants(m)
        for v in variants:
            for t in types:
                found = self._name_index.get(t, {}).get(v)
                if found:
                    hits.extend(found)
            if hits:
                return list(dict.fromkeys(hits))[:limit]
        if fuzzy_threshold > 0:
            from rapidfuzz import fuzz, process

            query = variants[0] if variants else norm(m)
            for t in types:
                choices = self._name_index.get(t, {})
                candidates = self._fuzzy_candidates(t, query)
                if not candidates:
                    continue
                best = process.extract(query, candidates, scorer=fuzz.ratio, limit=3,
                                       score_cutoff=fuzzy_threshold)
                for name, _, _ in best:
                    hits.extend(choices[name])
        return list(dict.fromkeys(hits))[:limit]

    def resolve(self, mention: str, node_type: str | None = None,
                fuzzy_threshold: float = 0) -> int | None:
        hits = self.lookup(mention, node_type, fuzzy_threshold, limit=50)
        if not hits:
            return None
        # Prefer the most connected node when a name is ambiguous.
        return max(hits, key=self.degree)

    def search(self, query: str, node_type: str | None = None, limit: int = 10) -> list[int]:
        """Lenient search for agent tools: exact, then substring, then fuzzy."""
        hits = self.lookup(query, node_type, fuzzy_threshold=0, limit=limit)
        if len(hits) >= limit:
            return hits
        q = norm(query)
        if not q:
            return hits
        self._ensure_index()
        types = [node_type] if node_type else list(self._name_index)
        for t in types:
            for name, idxs in self._name_index.get(t, {}).items():
                if q in name:
                    hits.extend(idxs)
                if len(hits) >= 5 * limit:
                    break
        if not hits:
            hits = self.lookup(query, node_type, fuzzy_threshold=80, limit=limit)
        hits = list(dict.fromkeys(hits))
        hits.sort(key=lambda i: (len(self.names[i]), -self.degree(i)))
        return hits[:limit]

    def disease_for_mesh(self, mesh_id: str) -> list[int]:
        return list(dict.fromkeys(self._mesh_index.get(mesh_id.upper(), [])))

    # ------------------------------------------------------------------ mechanisms
    def _hub_weight(self, i: int) -> float:
        return 1.0 / math.log2(2 + self.degree(i))

    def mechanism_paths(self, drug: int, disease: int, max_edges: int = 4,
                        limit: int = 10) -> list[dict]:
        """Enumerate drug -> gene ... -> disease paths following biological templates:

        2 edges: drug -target-> gene -associated-> disease
        3 edges: drug -> gene -PPI-> gene -> disease
        4 edges: drug -> gene -> pathway -> gene -> disease
        Paths through hub nodes are down-weighted (1/log2(2+degree) per intermediate node).
        """
        targets = self.neighbors(drug, ["drug_protein"])
        dgenes = self.neighbors(disease, ["disease_protein"])
        if len(targets) == 0 or len(dgenes) == 0:
            return []
        dset = set(dgenes.tolist())
        paths: list[tuple[float, list[int]]] = []
        for g in np.intersect1d(targets, dgenes):
            paths.append((self._hub_weight(g), [drug, int(g), disease]))
        if max_edges >= 3:
            for t in targets:
                for g in np.intersect1d(self.neighbors(int(t), ["protein_protein"]), dgenes):
                    if g != t:
                        paths.append((self._hub_weight(t) * self._hub_weight(g),
                                      [drug, int(t), int(g), disease]))
        if max_edges >= 4:
            for t in targets:
                for p in self.neighbors(int(t), ["pathway_protein"]):
                    members = self.neighbors(int(p), ["pathway_protein"])
                    for g in members:
                        if int(g) in dset and g != t:
                            w = self._hub_weight(t) * self._hub_weight(p) * self._hub_weight(g)
                            paths.append((w, [drug, int(t), int(p), int(g), disease]))
        paths.sort(key=lambda x: -x[0])
        out, seen = [], set()
        for w, nodes in paths:
            key = tuple(nodes)
            if key in seen:
                continue
            seen.add(key)
            out.append({"score": round(w, 4), "nodes": nodes})
            if len(out) >= limit:
                break
        return out

    def describe_path(self, nodes: Sequence[int]) -> list[dict]:
        edges = []
        for u, v in zip(nodes[:-1], nodes[1:]):
            rels = self.edge_relations(int(u), int(v))
            label = (rels[0][1] or rels[0][0]) if rels else "?"
            edges.append({"source": f"{self.names[u]} [kg:{u}]", "relation": label,
                          "target": f"{self.names[v]} [kg:{v}]"})
        return edges

    # ------------------------------------------------------------------ propagation scores
    def _matrices(self):
        if self._sparse:
            return self._sparse
        import scipy.sparse as sp

        src, dst, rel, _ = self._edges

        def adj(rel_name: str):
            code = self.rel_code.get(rel_name, -1)
            sel = rel == code
            s, d = src[sel], dst[sel]
            data = np.ones(2 * len(s), np.float32)
            m = sp.csr_matrix((data, (np.concatenate([s, d]), np.concatenate([d, s]))),
                              shape=(self.n, self.n))
            m.data[:] = 1.0
            return m

        self._sparse["drug_protein"] = adj("drug_protein")
        self._sparse["protein_protein"] = adj("protein_protein")
        self._sparse["disease_protein"] = adj("disease_protein")
        deg = np.asarray(self._sparse["protein_protein"].sum(axis=1)).ravel()
        self._sparse["ppi_norm"] = np.where(deg > 0, 1.0 / np.sqrt(deg + 1.0), 0.0).astype(np.float32)
        return self._sparse

    def drug_scores(self, disease: int, ppi_weight: float = 0.5) -> np.ndarray:
        """Network-proximity style score of every node as a drug candidate for ``disease``.

        score(drug) = sum_{g in targets(drug)} [ v(g) + ppi_weight * (A_ppi_norm v)(g) ] / sqrt(|targets|)
        where v marks disease-associated genes. Non-drug nodes score 0.
        """
        m = self._matrices()
        v =np.asarray(m["disease_protein"][disease].todense()).ravel().astype(np.float32)
        norm_vec = m["ppi_norm"]
        spread = norm_vec * (m["protein_protein"] @ (norm_vec * v))
        signal = v + ppi_weight * spread
        dp = m["drug_protein"]
        raw = dp @ signal
        ntargets = np.asarray(dp.sum(axis=1)).ravel()
        score = np.where(ntargets > 0, raw / np.sqrt(np.maximum(ntargets, 1)), 0.0)
        score[self.types != DRUG] = 0.0
        return score
