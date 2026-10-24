"""Generate the manuscript's figures from the stored artefacts.

    python paper/make_figures.py

Reads runs/report_full/summary.csv and runs/report_external/*.json; writes PNG and PDF into
paper/figures/. Every value plotted comes from a file check_numbers.py also verifies, so a
figure cannot drift from the text.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
FULL = ROOT / "runs" / "report_full" / "summary.csv"
SECOND = ROOT / "runs" / "report_llama31-8b" / "summary.csv"
EXTERNAL = [ROOT / "runs" / "report_external" / "moa.json",
            ROOT / "runs" / "report_external" / "literature.json"]

LABELS = {
    "qwen25-7b__closed_book": "qwen closed book",
    "qwen25-7b__rag_verify-none": "qwen RAG",
    "qwen25-7b__rag_verify-revise": "qwen RAG + verify",
    "llama31-8b__closed_book": "llama closed book",
    "llama31-8b__rag_verify-none": "llama RAG",
    "llama31-8b__rag_verify-revise": "llama RAG + verify",
    "popularity": "popularity",
    "proximity": "proximity",
    "complex": "ComplEx",
}
# One accent for the systems the paper contrasts, grey for the baselines that need no tokens.
COLOURS = {
    "qwen25-7b__closed_book": "#c2410c",
    "qwen25-7b__rag_verify-none": "#b45309",
    "qwen25-7b__rag_verify-revise": "#15803d",
    "llama31-8b__closed_book": "#9a3412",
    "llama31-8b__rag_verify-none": "#7c2d12",
    "llama31-8b__rag_verify-revise": "#166534",
    "popularity": "#525252",
    "proximity": "#737373",
    "complex": "#a3a3a3",
}
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    # Journals ask for 600 dpi for line art (Briefings in Bioinformatics among them); 200 was
    # below every target venue's floor.
    "figure.dpi": 600, "savefig.bbox": "tight",
})


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote paper/figures/{name}.png (+pdf)")


# Label placement is set per point: the systems sit close enough together that automatic
# offsets collide, and an overlapping label is worse than no figure.
LABEL_OFFSETS = {
    "qwen25-7b__closed_book": (-10, 10, "right"),
    "qwen25-7b__rag_verify-none": (9, -13, "left"),
    "qwen25-7b__rag_verify-revise": (9, 7, "left"),
    "llama31-8b__closed_book": (-10, 8, "right"),
    "llama31-8b__rag_verify-none": (-9, -14, "right"),
    "llama31-8b__rag_verify-revise": (-9, 6, "right"),
    "popularity": (9, 0, "left"),
    "proximity": (9, 2, "left"),
    "complex": (9, -3, "left"),
}


def fig_faithfulness_vs_accuracy(full: pd.DataFrame) -> None:
    """The paper's argument in one picture: the two axes do not move together."""
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    # The baselines emit no mechanism at all, so they have no fabrication rate. They are drawn
    # in a banded strip left of the axis break, never at x = 0, which would read as "never
    # fabricates" rather than "makes no checkable claim".
    band = -0.048
    ax.axvspan(band - 0.022, band + 0.022, color="#f5f5f4", zorder=0)
    ax.axvline(band + 0.028, color="#d4d4d4", linestyle=(0, (3, 3)), linewidth=0.9, zorder=1)
    ax.text(band, 0.0012, "no mechanism\nemitted", ha="center", va="bottom", fontsize=7.5,
            color="#8a8a8a", linespacing=1.2)
    for system, row in full.iterrows():
        fab = row["fabricated_edge_rate"]
        has_claims = fab == fab
        x = fab if has_claims else band
        dx, dy, ha = LABEL_OFFSETS.get(system, (9, 3, "left"))
        colour = COLOURS.get(system, "#525252")
        ax.scatter(x, row["mrr"], s=95, marker="o" if has_claims else "D",
                   facecolor=colour if has_claims else "none", edgecolor=colour,
                   linewidth=1.6, zorder=3)
        ax.annotate(LABELS.get(system, system), (x, row["mrr"]), textcoords="offset points",
                    xytext=(dx, dy), fontsize=8, color=colour, ha=ha)
    ax.set_xlabel("fabricated mechanism edges  (lower is better)")
    ax.set_ylabel("MRR  (higher is better)")
    ax.set_title("Faithfulness and accuracy do not move together")
    ax.set_ylim(0, 0.035)
    ax.set_xlim(-0.08, 1.02)
    ax.set_ylim(0, 0.035)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.6, zorder=0)
    save(fig, "fig1_faithfulness_vs_accuracy")


def fig_verification_effect(full: pd.DataFrame) -> None:
    """What grounding and verification change, for each model, and what they leave alone."""
    groups = [("qwen2.5-7b", ["qwen25-7b__closed_book", "qwen25-7b__rag_verify-none",
                              "qwen25-7b__rag_verify-revise"]),
              ("llama3.1-8b", ["llama31-8b__closed_book", "llama31-8b__rag_verify-none",
                               "llama31-8b__rag_verify-revise"])]
    groups = [(m, [s for s in ss if s in full.index]) for m, ss in groups]
    groups = [(m, ss) for m, ss in groups if ss]
    panels = [("fabricated_edge_rate", "fabricated edges"),
              ("unresolvable_edge_rate", "unresolvable endpoints"),
              ("hit@20", "Hit@20")]
    short = {"closed_book": "closed\nbook", "rag_verify-none": "RAG", "rag_verify-revise": "RAG\n+ver"}
    # The popularity baseline belongs on the ranking panel, and the panel has to be scaled to
    # show it: drawn above an axis limit set from the systems alone, the single most important
    # comparison in the paper would sit outside the frame.
    reference = full.loc["popularity", "hit@20"] if "popularity" in full.index else None
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.0))
    for ax, (col, title) in zip(axes, panels):
        xs, values, colours, ticks = [], [], [], []
        x = 0.0
        for gi, (model, systems) in enumerate(groups):
            for s in systems:
                xs.append(x)
                values.append(full.loc[s, col])
                colours.append(COLOURS[s])
                ticks.append(short.get(s.split("__")[1], s))
                x += 1
            if gi == 0:
                x += 0.8          # a gap, so the two models read as two groups
        ceiling = max(values + ([reference] if col == "hit@20" and reference else []))
        ax.bar(xs, values, color=colours, width=0.72, zorder=3)
        for xv, v in zip(xs, values):
            ax.text(xv, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(xs)
        ax.set_xticklabels(ticks, fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, ceiling * 1.34 if ceiling > 0 else 1)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.6, zorder=0)
        if len(groups) == 2:
            mid = (xs[2] + xs[3]) / 2
            ax.axvline(mid, color="#e5e5e5", linewidth=0.8, zorder=1)
            ax.text(xs[1], -ceiling * 0.30, groups[0][0], ha="center", fontsize=7.5,
                    color="#525252")
            ax.text(xs[4], -ceiling * 0.30, groups[1][0], ha="center", fontsize=7.5,
                    color="#525252")
    if reference:
        axes[2].axhline(reference, color="#525252", linestyle="--", linewidth=1.2, zorder=4)
        axes[2].text(xs[-1] + 0.4, reference * 1.03, f"popularity {reference:.3f}",
                     va="bottom", ha="right", fontsize=7.5, color="#525252")
    fig.suptitle("Grounding moves fabrication in opposite directions; ranking moves in neither",
                 y=1.03, fontsize=10)
    save(fig, "fig2_verification_effect")


def fig_moa_by_role(ext: dict) -> None:
    """Independent confirmation of drug-target claims, by the role the graph assigns the edge."""
    primary = "qwen25-7b__rag_verify-revise"
    moa = (ext.get(primary) or {}).get("moa")
    if not moa or not moa.get("edges_by_role"):
        print("  skipping fig3: no MoA results")
        return
    roles = ["target", "enzyme", "transporter", "carrier"]
    roles = [r for r in roles if r in moa["edges_by_role"]]
    conf = moa.get("confirmation_by_role", {})
    counts = [moa["edges_by_role"][r] for r in roles]
    rates = [conf.get(r, float("nan")) for r in roles]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.9),
                                   gridspec_kw={"width_ratios": [1, 1.1]})
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", labelsize=8)
    ax1.bar(roles, counts, color=["#15803d"] + ["#a3a3a3"] * (len(roles) - 1),
            width=0.62, zorder=3)
    for i, c in enumerate(counts):
        ax1.text(i, c, f"{c:,}", ha="center", va="bottom", fontsize=8)
    ax1.set_title("edges asserted, by graph role")
    ax1.set_ylim(0, max(counts) * 1.25)
    ax1.grid(axis="y", color="#e5e5e5", linewidth=0.6, zorder=0)

    ax2.bar(roles, [0 if r != r else r for r in rates],
            color=["#15803d"] + ["#c2410c"] * (len(roles) - 1), width=0.62, zorder=3)
    for i, r in enumerate(rates):
        ax2.text(i, 0 if r != r else r, f"{r:.3f}" if r == r else "n/a",
                 ha="center", va="bottom", fontsize=8)
    ax2.set_title("confirmed by ChEMBL mechanism of action")
    ax2.set_ylim(0, 0.68)
    ax2.grid(axis="y", color="#e5e5e5", linewidth=0.6, zorder=0)
    fig.suptitle("Only target-role edges are mechanisms of action", y=1.05, fontsize=10)
    save(fig, "fig3_moa_by_role")


def main() -> int:
    if not FULL.exists():
        print(f"missing {FULL.relative_to(ROOT)} - run the evaluation first")
        return 2
    full = pd.read_csv(FULL).set_index("system")
    if SECOND.exists():
        second = pd.read_csv(SECOND).set_index("system")
        full = pd.concat([full, second[~second.index.isin(full.index)]])
    ext: dict = {}
    for path in EXTERNAL:
        if path.exists():
            for system, res in json.loads(path.read_text(encoding="utf-8")).items():
                ext.setdefault(system, {}).update({k: v for k, v in res.items() if v})
    print("Figures:")
    fig_faithfulness_vs_accuracy(full)
    fig_verification_effect(full)
    fig_moa_by_role(ext)
    return 0


if __name__ == "__main__":
    sys.exit(main())
