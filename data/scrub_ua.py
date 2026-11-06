r"""Corrected artifact scrubber.

Two bugs found by running diagnose() on the real corpus:

1. `{_LEAD}(?={_L})` -- "a namespace word followed by a letter" is exactly
   Ukrainian inflection. файли, файлів, файлова, зображеннями, медіапростір
   all matched. `файли cookie` became `и cookie`. Removed that direction:
   only a lead GLUED TO A PRECEDING LETTER is markup (pngФайл), which is the
   case it was added for in the first place.

2. `\.?{_EXT}` -- the optional dot let bare `oga`, `tif`, `ogg` match inside
   ordinary words. `Yoga` became `Y`. The dot is now required.

Also: _LEAD is case-sensitive now. Markup namespaces are capitalised
(`Файл:`), the common noun is not. Everything else keeps IGNORECASE via
inline (?i:...) groups.

Cost of the fix: a markup `Файл` at the very start of a run is no longer
caught, since nothing precedes it. That is a handful of fragments against
~11% of documents previously being corrupted.
"""

import re
from collections import Counter

_L    = r"[^\W\d_]"
_EXT  = r"(?i:png|jpe?g|gif|svg|webp|tiff?|ogg|ogv|oga|webm|djvu|pdf)"
_SIZE = r"\d+(?:\s?[xх×]\s?\d+)?(?i:пкс|px)"
_KW   = (r"(?i:міні|мініатюра|thumb|thumbnail|праворуч|ліворуч|справа|зліва"
         r"|right|left|center|центр|межа|border|безрамки|frameless|рамка"
         r"|frame|верх|низ|top|bottom|upright|альт|alt|посилання|link"
         r"|клас|class)")
# case-SENSITIVE, and the script must CHANGE across the boundary: the real
# case is a Latin extension glued to a Cyrillic namespace (pngФайл). Same-script
# gluing is ordinary CamelCase -- SocialMedia, ПриватБанк -- and must survive.
_LEAD_CYR = r"(?<=[A-Za-z0-9])(?:Файл|Зображення|Медіа)"
_LEAD_LAT = r"(?<=[\u0400-\u04FF])(?:File|Image|Media)"

# An anchor is something that is essentially never legitimate prose:
#   - a pixel size spec                       131x131пкс, 180px
#   - a DOTTED file extension after a letter  Водень.png
#   - a namespace lead glued ACROSS SCRIPTS   pngФайл
_ANCHOR = (rf"(?:{_SIZE}"
           rf"|(?<={_L})\.{_EXT}(?![a-zA-Z])"
           rf"|{_LEAD_CYR}|{_LEAD_LAT})")

ARTIFACT = re.compile(rf"{_ANCHOR}(?:{_KW})*")
CYR_GLUE = re.compile(r"[а-щьюяїієґ][А-ЩЬЮЯЇІЄҐ]")


def scrub(s: str) -> str:
    return re.sub(r" {2,}", " ", ARTIFACT.sub(" ", s)).strip()

def diagnose(docs, limit=50_000):
    """Per-document artifact density, so you can pick a strategy."""
    buckets = Counter()
    samples, examples = [], Counter()
    for i, doc in enumerate(docs):
        if i >= limit:
            break
        hits = ARTIFACT.findall(doc)
        #glue = len(CYR_GLUE.findall(doc))
        examples.update(m if isinstance(m, str) else m[0] for m in hits)
        density = (sum(len(h) for h in hits) / len(doc)) if doc else 0.0
        if not hits: #and not glue:
            buckets["clean"] += 1
        elif density < 0.005:
            buckets["<0.5%"] += 1
        elif density < 0.02:
            buckets["0.5-2%"] += 1
        elif density < 0.05:
            buckets["2-5%"] += 1
        else:
            buckets[">5%"] += 1
            if len(samples) < 5:
                samples.append(doc[:600])
 
    total = sum(buckets.values())
    for k in ["clean", "<0.5%", "0.5-2%", "2-5%", ">5%"]:
        n = buckets[k]
        print(f"{k:>8}  {n:>8,}  {n/total*100:5.2f}%")
    print("\ntop artifact strings:")
    for s, n in examples.most_common(25):
        print(f"  {n:>7,}  {s!r}")
    print("\n--- worst documents ---")
    for s in samples:
        print(s, "\n" + "-" * 60)
    return buckets

if __name__ == "__main__":
    MUST_SURVIVE = [
        "Ми використовуємо файли cookie на цьому веб-сайті",
        "Політиці щодо файлів cookie",
        "керування зображеннями у статті",
        "Медіацентр університету",
        "медіапростір України",
        "Yoga студія",
        "творчий мотив",
        "файлова система",
        "Зображення було намальоване у 1856 році",
        "Він відкрив файл і почав читати",
        "Межа між областями проходить по річці",
        "Територія заповідника — 200 гектарів",
        "Формат PNG підтримує прозорість",
        "«Файл» — це документ",
        "Медіа відіграє важливу роль",
        "завантажте зображення на сайт",
        "SocialMedia маркетинг",
        "BackgroundImage у CSS",
        "MediaMarkt магазин",
        "ПриватБанк відкрив рахунок",
        "потужність 500 кВт на годину",
        "стаття КУпАП про порушення",
        "рівень рН води у водоймі",
        "мРНК вакцина проти вірусу",
        "ТзОВ Агро та ПрАТ Метал",
        "НаУКМА і УжНУ підписали угоду",
    ]
    MUST_SCRUB = [
        ("131x131пкс", ""),
        ("180pxмежа", ""),
        ("Санкт-Петербург150px", "Санкт-Петербург"),
        ("Мапа 40x26пкстериторія області.", "Мапа територія області."),
        ("Водень.pngФайлГелій.jpg180pxмежаПерший елемент.",
         "Водень Гелій Перший елемент."),
    ]

    bad = 0
    for s in MUST_SURVIVE:
        got = scrub(s)
        if got != s:
            bad += 1
            print(f"FALSE POSITIVE {s!r}\n            -> {got!r}")
    for s, want in MUST_SCRUB:
        got = scrub(s)
        if got != want:
            bad += 1
            print(f"MISSED {s!r}\n    got  {got!r}\n    want {want!r}")

    n = len(MUST_SURVIVE) + len(MUST_SCRUB)
    print(f"{n - bad}/{n} passed "
          f"({len(MUST_SURVIVE)} false-positive guards, "
          f"{len(MUST_SCRUB)} true positives)")