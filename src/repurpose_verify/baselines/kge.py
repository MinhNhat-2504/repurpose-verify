"""Knowledge-graph embedding baselines (TransE, DistMult, ComplEx) trained on the cutoff KG.

Drug-disease scores are read off the ``indication`` relation. Held-out (post-cutoff) pairs are
absent from the KG by construction, and ``mask_pairs`` is applied before training anyway.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..data.kg import KnowledgeGraph

log = logging.getLogger(__name__)


@dataclass
class KGEConfig:
    model: str = "complex"        # transe | distmult | complex
    dim: int = 128
    epochs: int = 5
    batch_size: int = 8192
    negatives: int = 32
    lr: float = 1e-3
    margin: float = 6.0           # TransE only
    l2: float = 1e-6
    seed: int = 13
    device: str = "cuda"
    max_triples: int = 0          # subsample for quick runs (0 = all)


def _model(torch, n_ent: int, n_rel: int, cfg: KGEConfig):
    nn = torch.nn

    class KGE(nn.Module):
        def __init__(self):
            super().__init__()
            width = 2 * cfg.dim if cfg.model == "complex" else cfg.dim
            self.ent = nn.Embedding(n_ent, width)
            self.rel = nn.Embedding(n_rel, width)
            nn.init.xavier_uniform_(self.ent.weight)
            nn.init.xavier_uniform_(self.rel.weight)

        def score(self, h, r, t):
            """h, r, t: (..., width) tensors -> plausibility (higher = more plausible)."""
            if cfg.model == "transe":
                return cfg.margin - (h + r - t).norm(p=1, dim=-1)
            if cfg.model == "distmult":
                return (h * r * t).sum(-1)
            hr, hi = h.chunk(2, -1)
            rr, ri = r.chunk(2, -1)
            tr, ti = t.chunk(2, -1)
            return (hr * rr * tr + hi * rr * ti + hr * ri * ti - hi * ri * tr).sum(-1)

        def forward(self, hi, ri, ti):
            return self.score(self.ent(hi), self.rel(ri), self.ent(ti))

    return KGE()


class KGEBaseline:
    def __init__(self, kg: KnowledgeGraph, cfg: KGEConfig):
        self.kg = kg
        self.cfg = cfg
        self.model = None

    def train(self) -> list[float]:
        import torch

        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        device = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
        src, dst, rel, _ = self.kg._edges
        # PrimeKG relations are symmetric: train on both directions.
        h = np.concatenate([src, dst]).astype(np.int64)
        t = np.concatenate([dst, src]).astype(np.int64)
        r = np.concatenate([rel, rel]).astype(np.int64)
        if cfg.max_triples and len(h) > cfg.max_triples:
            sel = rng.choice(len(h), cfg.max_triples, replace=False)
            h, r, t = h[sel], r[sel], t[sel]
        types = self.kg.types
        # Type-constrained negatives: corrupt the tail with a node of the same type.
        type_codes, type_index = np.unique(types, return_inverse=True)
        pools = [np.flatnonzero(type_index == k) for k in range(len(type_codes))]
        tail_type = type_index[t]

        self.model = _model(torch, self.kg.n, len(self.kg.relations), cfg).to(device)
        opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        losses = []
        n = len(h)
        for epoch in range(cfg.epochs):
            perm = rng.permutation(n)
            total, batches = 0.0, 0
            for start in range(0, n, cfg.batch_size):
                idx = perm[start:start + cfg.batch_size]
                bh, br, bt = h[idx], r[idx], t[idx]
                neg = np.empty((len(idx), cfg.negatives), np.int64)
                for k, pool in enumerate(pools):
                    rows = np.flatnonzero(tail_type[idx] == k)
                    if len(rows):
                        neg[rows] = pool[rng.integers(0, len(pool), (len(rows), cfg.negatives))]
                th = torch.from_numpy(bh).to(device)
                tr = torch.from_numpy(br).to(device)
                tt = torch.from_numpy(bt).to(device)
                tn = torch.from_numpy(neg).to(device)
                eh, er = self.model.ent(th), self.model.rel(tr)
                pos = self.model.score(eh, er, self.model.ent(tt))
                negs = self.model.score(eh.unsqueeze(1), er.unsqueeze(1), self.model.ent(tn))
                if cfg.model == "transe":
                    loss = torch.relu(1.0 - pos.unsqueeze(1) + negs).mean()
                else:
                    sp = torch.nn.functional.softplus
                    loss = sp(-pos).mean() + sp(negs).mean()
                loss = loss + cfg.l2 * (eh.pow(2).mean() + er.pow(2).mean())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += float(loss)
                batches += 1
            losses.append(total / max(batches, 1))
            log.info("KGE %s epoch %d loss %.4f", cfg.model, epoch + 1, losses[-1])
        self.model.eval()
        return losses

    def scores(self, disease_idx: int, relation: str = "indication") -> np.ndarray:
        import torch

        if self.model is None:
            raise RuntimeError("train() or load() first")
        device = next(self.model.parameters()).device
        with torch.no_grad():
            ent = self.model.ent.weight
            r = self.model.rel.weight[self.kg.rel_code[relation]]
            d = ent[disease_idx]
            # Symmetric training: average both directions.
            s = 0.5 * (self.model.score(ent, r.expand_as(ent), d.expand_as(ent))
                       + self.model.score(d.expand_as(ent), r.expand_as(ent), ent))
        return s.float().cpu().numpy()

    def save(self, path: Path) -> None:
        import torch

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": self.cfg.__dict__, "state": self.model.state_dict()}, path)

    def load(self, path: Path) -> None:
        import torch

        ckpt = torch.load(path, map_location="cpu")
        self.cfg = KGEConfig(**ckpt["cfg"])
        device = self.cfg.device if (self.cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
        self.model = _model(torch, self.kg.n, len(self.kg.relations), self.cfg)
        self.model.load_state_dict(ckpt["state"])
        self.model.to(device).eval()
