import re
from collections import Counter
 
CYR_GLUE = re.compile(r"[а-щьюяїієґ][А-ЩЬЮЯЇІЄҐ]")
# capture a window around each hit so the cause is visible
GLUE_CTX = re.compile(r"(\w{0,14}[а-щьюяїієґ][А-ЩЬЮЯЇІЄҐ]\w{0,14})")
 
 
def glue_report(docs, limit=50_000, show=30):
    per_doc = Counter()
    examples = Counter()
    ndocs = 0
    for i, doc in enumerate(docs):
        if i >= limit:
            break
        ndocs += 1
        hits = GLUE_CTX.findall(doc)
        if hits:
            per_doc[min(len(hits), 20)] += 1
            examples.update(hits)
 
    with_glue = sum(per_doc.values())
    print(f"{ndocs:,} docs | {with_glue:,} contain glue "
          f"({with_glue/ndocs*100:.2f}%)")
    print(f"{sum(examples.values()):,} total instances\n")
 
    print("instances per affected document:")
    for k in sorted(per_doc):
        label = f"{k}+" if k == 20 else str(k)
        print(f"  {label:>4}: {per_doc[k]:>7,} docs")
 
    print(f"\nmost common glued strings:")
    for s, n in examples.most_common(show):
        print(f"  {n:>6,}  {s}")
    return per_doc, examples
 
 
if __name__ == "__main__":
    docs = [
        "Це реченняНаступне речення без пробілу.",
        "Звичайний текст без проблем узагалі.",
        "ЗаголовокПідзаголовокТекст статті тут.",
        "УРСР Київ столиця України та найбільше місто.",
        "Нормальний абзац із кількох слів і розділових знаків.",
    ] * 20
    per_doc, ex = glue_report(docs, show=5)
    assert sum(ex.values()) > 0, "detector found nothing"
    assert not CYR_GLUE.search("УРСР Київ"), "space should break the match"
    print("\ndetector ok")
