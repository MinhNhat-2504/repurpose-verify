"""Name normalisation used to ground free-text entity mentions onto KG nodes."""

from __future__ import annotations

import re
import unicodedata

_PUNCT = re.compile(r"[^a-z0-9]+")
_DOSE = re.compile(r"\b\d+(\.\d+)?\s*(mg|mcg|ug|g|ml|iu|units?|%|mg/kg|mg/m2)\b")
_PAREN = re.compile(r"\(([^)]*)\)|\[([^\]]*)\]")

SALT_WORDS = {
    "hydrochloride", "dihydrochloride", "hcl", "sodium", "potassium", "calcium", "magnesium",
    "mesylate", "maleate", "sulfate", "sulphate", "citrate", "acetate", "tartrate", "phosphate",
    "besylate", "fumarate", "succinate", "bromide", "chloride", "hydrobromide", "monohydrate",
    "dihydrate", "trihydrate", "hydrate", "anhydrous", "tosylate", "lactate", "nitrate",
    "dimesylate", "disodium", "malate", "hyclate", "propionate", "dipropionate", "valerate",
    "oral", "tablets", "tablet", "capsule", "capsules", "injection", "injectable", "extended",
    "release", "er", "xr", "sr", "ir", "iv", "solution", "suspension", "cream", "gel",
}

GENERIC_INTERVENTIONS = {
    "placebo", "placebos", "saline", "normal saline", "standard of care", "standard care",
    "usual care", "best supportive care", "control", "vehicle", "sham", "no intervention",
    "chemotherapy", "radiotherapy", "radiation", "surgery", "observation", "sugar pill",
}


def norm(text: str) -> str:
    """Lower-case, strip accents and punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("'s ", " ")
    return _PUNCT.sub(" ", text).strip()


def uninvert(text: str) -> str:
    """MeSH-style inverted headings: 'Diabetes Mellitus, Type 2' -> 'Type 2 Diabetes Mellitus'."""
    parts = [p.strip() for p in text.split(",")]
    if len(parts) == 2 and parts[0] and parts[1]:
        return f"{parts[1]} {parts[0]}"
    return text


def name_variants(text: str) -> list[str]:
    """Normalised surface forms of a generic entity mention (order = preference)."""
    out: list[str] = []
    raw = text.strip()
    for cand in (raw, uninvert(raw), _PAREN.sub(" ", raw)):
        n = norm(cand)
        if n and n not in out:
            out.append(n)
    for m in _PAREN.finditer(raw):
        inner = norm(m.group(1) or m.group(2) or "")
        if inner and inner not in out:
            out.append(inner)
    return out


def drug_variants(text: str) -> list[str]:
    """Surface forms of a drug mention: strips doses, salts and formulations."""
    out = name_variants(text)
    extra: list[str] = []
    for v in out:
        v2 = _DOSE.sub(" ", v)
        tokens = [t for t in v2.split() if t not in SALT_WORDS]
        stripped = " ".join(tokens).strip()
        if stripped and stripped not in out and stripped not in extra:
            extra.append(stripped)
    return out + extra


def is_generic_intervention(text: str) -> bool:
    n = norm(text)
    return n in GENERIC_INTERVENTIONS or n.startswith("placebo")
