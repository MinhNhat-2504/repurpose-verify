"""Build the submission package: the manuscript as .docx, plus figures as separate files.

    python scripts/make_submission.py [--out dist/submission]

GigaScience accepts the manuscript as DOC, DOCX, RTF or LaTeX - not Markdown and not PDF - and
asks that figures be supplied as separate files rather than embedded in the manuscript. So the
image references are stripped on the way through and the figure legends stay where they are, each
numbered figure going out as its own file beside the document.

Styling comes from paper/reference.docx, built by make_reference_docx.py. Run that first, or the
document inherits Word's stock theme.

Needs pandoc, which pypandoc will fetch with `pypandoc.download_pandoc()` if it is missing.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "paper" / "draft.md"
COVER = ROOT / "paper" / "cover_letter.md"
FIGURES = ROOT / "paper" / "figures"

# An embedded image on its own line: dropped, so the legend below it stands alone.
IMAGE_LINE = re.compile(r"^!\[[^\]]*\]\([^)]*\)\s*$", re.M)


def literal_reference_numbers(markdown: str) -> str:
    """Give each reference its own paragraph, numbered as text rather than by Word.

    An in-text citation is literal - "[13]" - while a Word numbered list generates its own
    numbers from the list formatting. Those two can part company: renumber or split the list in
    production and [13] no longer points at entry 13. So the period is escaped, which stops
    pandoc reading the entry as a list item.

    Escaping alone is not enough. The entries are separated by a single newline and wrap onto
    indented continuation lines, so once they are no longer list items pandoc reads the whole
    block as one lazy-continued paragraph: all seventeen references arrived in the document as a
    single run of text. Each entry is therefore rebuilt as one line and separated by a blank one.
    """
    start = markdown.index("## References")
    # The section ends at the next top-level heading; the appendices after it must be left alone.
    nxt = markdown.index("\n## ", start + 1)
    head, refs, tail = markdown[:start], markdown[start:nxt], markdown[nxt:]
    heading, _, rest = refs.partition("\n")

    # Split on a line that opens with a reference number, keeping the number.
    parts = re.split(r"(?m)^(\d{1,2})\.[ \t]+", rest)
    entries = parts[1:]
    out = [heading]
    for number, text in zip(entries[::2], entries[1::2]):
        out.append(f"{number}\\. {' '.join(text.split())}")
    # `tail` opens with the newline before the next heading, so one more keeps the blank line
    # that separates it; without it pandoc reads "## Appendix A" as more of the last reference.
    return head + "\n\n".join(out) + "\n" + tail


REFERENCE = ROOT / "paper" / "reference.docx"


def to_docx(markdown: str, out: Path, title: str) -> None:
    import pypandoc

    args = ["--standalone", f"--metadata=title:{title}"]
    if REFERENCE.exists():
        # Without this, pandoc applies Word's stock theme: Calibri body text and blue headings.
        args.append(f"--reference-doc={REFERENCE}")
    pypandoc.convert_text(markdown, "docx", format="markdown", outputfile=str(out),
                          extra_args=args)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="dist/submission")
    args = ap.parse_args()
    out = (ROOT / args.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    (out / "figures").mkdir(parents=True)

    text = DRAFT.read_text(encoding="utf-8")
    dropped = len(IMAGE_LINE.findall(text))
    body = IMAGE_LINE.sub("", text)
    # Collapse the blank-line pairs the removed images leave behind.
    body = re.sub(r"\n{3,}", "\n\n", body)
    body = literal_reference_numbers(body)

    title = re.match(r"#\s*(.+)", body).group(1).strip()
    # --metadata=title already renders the title, so the heading would appear a second time.
    body = re.sub(r"\A#[^\n]*\n+", "", body, count=1)
    to_docx(body, out / "manuscript.docx", title)
    to_docx(COVER.read_text(encoding="utf-8"), out / "cover_letter.docx", "Cover letter")

    figs = sorted(FIGURES.glob("fig*"))
    for f in figs:
        shutil.copy2(f, out / "figures" / f.name)

    print(f"  manuscript.docx     {(out / 'manuscript.docx').stat().st_size / 1024:7.0f} KB"
          f"   ({len(body.split()):,} words, {dropped} embedded images removed)")
    print(f"  cover_letter.docx   {(out / 'cover_letter.docx').stat().st_size / 1024:7.0f} KB")
    print(f"  figures/            {len(figs)} files")
    print(f"\n  {out}")
    print("\n  Figure legends stay in the manuscript; the images go up as the separate files in"
          "\n  figures/. Read the .docx before sending: pandoc carries tables and references over"
          "\n  faithfully, but column widths and page breaks are worth a look.")


if __name__ == "__main__":
    main()
