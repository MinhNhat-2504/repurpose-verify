"""Build the Word reference document that gives the manuscript a manuscript's look.

    python scripts/make_reference_docx.py

Pandoc's default reference document uses Word's stock theme: Calibri body text and blue headings
at 20, 16 and 14 points. That is a Word document, not a manuscript, and it is what a submitted
file looks like if nothing is done about it. Cell Press asks only that an initial submission be
clear and complete rather than formatted to a template, so this aims at the convention reviewers
expect instead: Times New Roman 12, double spaced, black headings, continuous line numbers.

It starts from pandoc's own default so that every style name pandoc maps to still exists, and
changes only the formatting. make_submission.py passes the result to pandoc with --reference-doc.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from docx.enum.section import WD_SECTION_START
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "reference.docx"

SERIF = "Times New Roman"
MONO = "Consolas"
BLACK = RGBColor(0, 0, 0)


def pandoc_default(path: Path) -> None:
    """Pandoc's own reference document, so no style it maps to goes missing."""
    import pypandoc

    exe = pypandoc.get_pandoc_path()
    with path.open("wb") as f:
        subprocess.run([exe, "--print-default-data-file", "reference.docx"],
                       stdout=f, check=True)


def set_font(style, *, name=SERIF, size=12, bold=None, italic=None, color=BLACK) -> None:
    f = style.font
    f.name = name
    f.size = Pt(size)
    if bold is not None:
        f.bold = bold
    if italic is not None:
        f.italic = italic
    if color is not None:
        f.color.rgb = color
    # Word keeps a separate east-asian/complex-script font; without this it can still
    # render headings in the theme font.
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(attr), name)
    # Drop the theme references that otherwise win over an explicit font.
    for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        if rfonts.get(qn(attr)) is not None:
            del rfonts.attrib[qn(attr)]


def space(style, *, before=0, after=0, spacing=2.0, keep_with_next=None) -> None:
    pf = style.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    if spacing == 1.0:
        pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
    else:
        pf.line_spacing = spacing
    if keep_with_next is not None:
        pf.keep_with_next = keep_with_next


def rule_table(style) -> None:
    """Horizontal rules only, and a bold header row.

    Pandoc's reference document leaves the table style without borders, so a ten-table paper
    arrives as columns of floating text. Scientific tables are ruled above and below and under
    the header, with no vertical lines.
    """
    el = style.element

    def borders(parent, spec: dict[str, str]) -> None:
        old = parent.find(qn("w:tblBorders"))
        if old is not None:
            parent.remove(old)
        tb = parent.makeelement(qn("w:tblBorders"), {})
        for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
            b = tb.makeelement(qn(f"w:{side}"), {})
            b.set(qn("w:val"), spec.get(side, "none"))
            b.set(qn("w:sz"), "8" if spec.get(side) == "single" else "0")
            b.set(qn("w:space"), "0")
            b.set(qn("w:color"), "000000")
            tb.append(b)
        parent.append(tb)

    tbl_pr = el.find(qn("w:tblPr"))
    if tbl_pr is None:
        tbl_pr = el.makeelement(qn("w:tblPr"), {})
        el.append(tbl_pr)
    borders(tbl_pr, {"top": "single", "bottom": "single"})

    # Conditional formatting for the header row: bold, with a rule beneath it.
    for old in el.findall(qn("w:tblStylePr")):
        if old.get(qn("w:type")) == "firstRow":
            el.remove(old)
    first = el.makeelement(qn("w:tblStylePr"), {})
    first.set(qn("w:type"), "firstRow")
    rpr = first.makeelement(qn("w:rPr"), {})
    b = rpr.makeelement(qn("w:b"), {})
    rpr.append(b)
    first.append(rpr)
    inner = first.makeelement(qn("w:tcPr"), {})
    borders(inner, {"bottom": "single"})
    first.append(inner)
    el.append(first)


def add_line_numbers(section) -> None:
    """Continuous line numbering, which reviewers reference and pandoc does not add."""
    sect_pr = section._sectPr
    for existing in sect_pr.findall(qn("w:lnNumType")):
        sect_pr.remove(existing)
    ln = sect_pr.makeelement(qn("w:lnNumType"), {})
    ln.set(qn("w:countBy"), "1")
    ln.set(qn("w:restart"), "continuous")
    ln.set(qn("w:distance"), "360")
    sect_pr.append(ln)


def main() -> None:
    import docx

    pandoc_default(OUT)
    d = docx.Document(str(OUT))
    st = d.styles

    set_font(st["Normal"])
    space(st["Normal"], after=0, spacing=2.0)
    st["Normal"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT

    # Headings: black, serif, and only as large as they need to be to rank.
    for name, size, italic in (("Title", 14, False), ("Heading 1", 13, False),
                               ("Heading 2", 12, False), ("Heading 3", 12, True),
                               ("Heading 4", 12, True), ("Heading 5", 12, True),
                               ("Heading 6", 12, True)):
        if name in [s.name for s in st]:
            set_font(st[name], size=size, bold=True, italic=italic)
            space(st[name], before=12, after=6, spacing=1.0, keep_with_next=True)
    st["Title"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if "Author" in [s.name for s in st]:
        set_font(st["Author"], size=12)
        space(st["Author"], after=6, spacing=1.0)
        st["Author"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER

    for name in ("Body Text", "First Paragraph", "Compact", "Block Text"):
        if name in [s.name for s in st]:
            set_font(st[name])
            space(st[name], spacing=2.0)

    # Code, tables and captions stay single spaced; double spacing makes them unreadable.
    if "Verbatim Char" in [s.name for s in st]:
        set_font(st["Verbatim Char"], name=MONO, size=9)
    # Pandoc styles code blocks with "Source Code", which its own reference document does not
    # define: left undefined, the style is created inheriting Normal, so the shell commands in
    # the supplemental data arrive in double-spaced Times New Roman.
    if "Source Code" not in [s.name for s in st]:
        from docx.enum.style import WD_STYLE_TYPE

        st.add_style("Source Code", WD_STYLE_TYPE.PARAGRAPH)
    set_font(st["Source Code"], name=MONO, size=9)
    space(st["Source Code"], before=6, after=6, spacing=1.0)
    for name in ("Table Caption", "Image Caption", "Caption"):
        if name in [s.name for s in st]:
            set_font(st[name], size=10, italic=False)
            space(st[name], before=6, after=6, spacing=1.0)
    if "Table" in [s.name for s in st]:
        set_font(st["Table"], size=10)
        space(st["Table"], spacing=1.0)
        rule_table(st["Table"])

    sec = d.sections[0]
    sec.start_type = WD_SECTION_START.NEW_PAGE
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)         # A4
    for side in ("left", "right", "top", "bottom"):
        setattr(sec, f"{side}_margin", Cm(2.54))
    add_line_numbers(sec)

    d.save(str(OUT))
    print(f"  {OUT}")
    print(f"  body {SERIF} 12 pt, double spaced; headings black serif; A4 with 2.54 cm margins;"
          f"\n  continuous line numbers; code and tables single spaced at 9-10 pt")


if __name__ == "__main__":
    main()
