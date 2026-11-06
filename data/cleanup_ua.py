"""Detect and scrub Wikipedia image-markup fragments left by a broken
wikitext extractor.
 
The delimiters ([[ ]] : |) are already gone from the corpus, so the original
structure is unrecoverable -- this is glued-token repair, not parsing, and it
is lossy by construction. Run diagnose() first: the density distribution tells
you whether to scrub, to drop documents, or to go find a better-extracted
copy of the corpus.
"""
 
import re
from collections import Counter
import unicodedata
from data.scrub_ua import scrub 

# --------------------------------------------------------------------------
# stress marks
# --------------------------------------------------------------------------
# Ukrainian marks stress with an acute that is not part of standard
# orthography. Two complications:
#   1. й (U+0439) and ї (U+0457) decompose under NFD into base + combining
#      mark, so the usual "strip everything with a combining class" recipe
#      silently destroys them. Hence the final NFC and the narrow mark class.
#   2. Stress is sometimes typed as a precomposed Latin vowel (ó = U+00F3)
#      because it is reachable on a keyboard. NFD exposes the Latin base,
#      which we map to Cyrillic ONLY when it carries a combining mark and
#      sits in an otherwise-Cyrillic word.
 
_LAT_VOWEL = {
    "a": "а", "e": "е", "i": "і", "o": "о", "y": "у",
    "A": "А", "E": "Е", "I": "І", "O": "О", "Y": "У",
}
 
_WORD         = re.compile(r"[\w\u0300-\u036F]+")
_HAS_CYR      = re.compile(r"[\u0400-\u04FF]")
_LAT_STRESSED = re.compile(r"([aeioyAEIOY])(?=[\u0300\u0301])")
_STRESS       = re.compile(r"([\u0400-\u04FF\u0500-\u052F])[\u0300\u0301]+")
 
 
def _fix_word(m: "re.Match") -> str:
    w = m.group(0)
    if not _HAS_CYR.search(w):
        return w  # pure Latin: café, iPhone -- leave alone
    # Lookahead keeps the mark in place; _STRESS removes it on the next pass.
    return _LAT_STRESSED.sub(lambda k: _LAT_VOWEL[k.group(1)], w)
 
 
def strip_stress(s: str) -> str:
    s = unicodedata.normalize("NFD", s)   # splits ó, and precomposed ѐ/ѝ
    s = _WORD.sub(_fix_word, s)           # Latin stressed vowel -> Cyrillic
    s = _STRESS.sub(r"\1", s)             # drop acute/grave on Cyrillic bases
    return unicodedata.normalize("NFC", s)  # rebuild й, ї, ё, café
 
 
# --------------------------------------------------------------------------
# apostrophes
# --------------------------------------------------------------------------
# The apostrophe is orthographic in Ukrainian (п'ять, об'єкт) and appears as
# at least four code points in the wild. Each variant is a distinct BPE token.
 
_APOS = {ord(c): "'" for c in "\u2019\u2018\u02BC\u02B9\u0060\u00B4"}
 
 
# --------------------------------------------------------------------------
# whitespace and invisibles
# --------------------------------------------------------------------------
 
_DELETE = re.compile(                     # controls, soft hyphen, zero-width
    r"[\u0000-\u0008\u000E-\u001F\u007F-\u009F"
    r"\u00AD"
    r"\u200B-\u200F\u202A-\u202E\u2060-\u2064\u206A-\u206F\uFEFF]"
)
_NEWLINE = re.compile(r"\r\n|[\r\v\f\u2028\u2029]")
_SPACE   = re.compile(r"[\t\u00A0\u1680\u2000-\u200A\u202F\u205F\u3000]")
_RUN     = re.compile(r" {2,}")
_TRIM    = re.compile(r"^ +| +$", re.M)
_BLANKS  = re.compile(r"\n{3,}")
 
 
def clean_ws(s: str) -> str:
    s = _DELETE.sub("", s)
    s = _NEWLINE.sub("\n", s)
    s = _SPACE.sub(" ", s)
    s = _RUN.sub(" ", s)
    s = _TRIM.sub("", s)      # must precede _BLANKS: a space-only line
    s = _BLANKS.sub("\n\n", s)  # blocks the \n{3,} match otherwise
    return s.strip()
 
 
# --------------------------------------------------------------------------
 
def clean_ua_str(s: str) -> str:
    """strip_stress ends in NFC; _APOS and clean_ws only touch ASCII and
    invisibles, so the output is still NFC."""
    return clean_ws(strip_stress(scrub(s)).translate(_APOS))

# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
 
CASES = [
    # stress removal
    ("Со́нячна систе́ма",      "Сонячна система"),
    ("В\u00F3день",           "Водень"),   # precomposed Latin ó
    ("В\u043E\u0301день",     "Водень"),   # Cyrillic о + combining acute
    ("\u0450 \u045D",         "е и"),      # precomposed grave ѐ ѝ
 
    # letters that must survive: these decompose under NFD
    ("Київ",                  "Київ"),     # ї = і + diaeresis
    ("Йосип",                 "Йосип"),    # й = и + breve
    ("ёлка",                  "ёлка"),     # diaeresis is not stress
    ("ґанок",                 "ґанок"),    # distinct letter, not г
    ("Flёur",                 "Flёur"),    # mixed script, no stress mark
    ("і\u0301\u0308жак",      "їжак"),     # adversarial mark order
 
    # lost word boundaries from the corpus -- must NOT be "repaired"
    ("рокуRHL",               "рокуRHL"),
    ("Anczewskiбурмистр",     "Anczewskiбурмистр"),
    ("АнчевськийMartinus",    "АнчевськийMartinus"),
    ("Al2О3",                 "Al2О3"),
    ("яhttps",                "яhttps"),
    ("Петербург150px",        "Петербург"),
    ("café",                  "café"),
    ("iPhone",                "iPhone"),
 
    # apostrophes
    ("п\u2019ять",            "п'ять"),
    ("здоров\u02BCя",         "здоров'я"),
 
    # whitespace
    ("а  б",                  "а б"),
    ("а\u00A0б",              "а б"),      # NBSP
    ("м'я\u00ADкий",          "м'який"),   # soft hyphen
    ("\uFEFFтекст",           "текст"),    # BOM
    ("рядок   \n\n\n\n  рядок", "рядок\n\nрядок"),
    ("а \n \n б",             "а\n\nб"),   # space-only line
    ("а\r\nб",                "а\nб"),
]
 
def _run_tests() -> None:
    failed = 0
    for src, want in CASES:
        got = clean_ua_str(src)
        if got != want:
            failed += 1
            print(f"FAIL {src!r}\n  got  {got!r}\n  want {want!r}")
    for src, _ in CASES:
        out = clean_ua_str(src)
        assert out == unicodedata.normalize("NFC", out), f"not NFC: {src!r}"
    print(f"{len(CASES) - failed}/{len(CASES)} passed")
 
 
if __name__ == "__main__":
    _run_tests()