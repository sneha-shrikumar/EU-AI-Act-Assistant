"""Finds provisions a question names outright ("article 97", "Annex XI",
"recitals 24 and 25", "Chapter III Section 4", "the preamble") and matches
them to chunk ids via chunk METADATA, not embeddings.

bge-small is poor at matching bare numbers: "what does article 97 talk about?"
can retrieve no Article 97 chunk at all, even though every chunk carries an
exact type/number/chapter/section. query.py pins these metadata matches ahead
of the similarity results.

Run directly for the parser selftest:  python provision_refs.py
"""
import re

# Strictly valid roman numerals only, so ordinary words made of roman letters
# ("civil", "mix") after "annex"/"chapter" are never read as numbers.
_VALID_ROMAN_RE = re.compile(
    r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"
)
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}

_KINDS = {
    "article": "article", "articles": "article", "art": "article",
    "annex": "annex", "annexes": "annex",
    "recital": "recital", "recitals": "recital",
    "section": "section", "sections": "section",
    "chapter": "chapter", "chapters": "chapter",
}
_NUM = r"(?:\d+[a-z]?|[ivxlcdm]+)(?:\(\w+\))*"  # "10(2)" -> suffix ignored later
_SEP = r"\s*(?:,|and|&|or|to|-|–)\s*"
REF_RE = re.compile(
    rf"\b(articles?|art\.?|annex(?:es)?|recitals?|sections?|chapters?)\s+"
    rf"({_NUM}(?:{_SEP}{_NUM})*)\b",
    re.IGNORECASE,
)
PREAMBLE_RE = re.compile(r"\bpreamble\b", re.IGNORECASE)
_TOKEN_RE = re.compile(r"\d+[a-z]?|[ivxlcdm]+|to|-|–", re.IGNORECASE)


def _roman_to_int(s: str) -> int:
    total, prev = 0, 0
    for ch in reversed(s.upper()):
        val = _ROMAN_VALUES[ch]
        total = total - val if val < prev else total + val
        prev = max(prev, val)
    return total


def _int_to_roman(n: int) -> str:
    out = ""
    for value, sym in ((1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
                       (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
                       (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while n >= value:
            out, n = out + sym, n - value
    return out


def _numbers(kind: str, raw: str) -> list[str]:
    """Normalised identifiers for one matched list: arabic strings for
    article/recital/section, roman strings for annex/chapter (the form the
    metadata stores). "9 to 12" / "9-12" expand to a range."""
    raw = re.sub(r"\(\w+\)", "", raw)  # paragraph suffixes: article-level match
    values: list[str] = []
    pending_range = False
    for tok in _TOKEN_RE.findall(raw):
        if tok.lower() in ("to", "-", "–"):
            pending_range = bool(values)
            continue
        if tok[0].isdigit():
            num = int(re.match(r"\d+", tok).group())
            if kind in ("annex", "chapter"):
                value = _int_to_roman(num)
            else:
                value = tok.lower() if kind == "article" else str(num)
        else:
            if kind not in ("annex", "chapter") or not _VALID_ROMAN_RE.match(tok.upper()):
                pending_range = False
                continue
            value = tok.upper()
        if pending_range and values:
            lo, hi = values[-1], value
            if kind in ("annex", "chapter"):
                a, b = _roman_to_int(lo), _roman_to_int(hi)
                values.extend(_int_to_roman(n) for n in range(a + 1, b + 1))
            elif lo.isdigit() and hi.isdigit():
                values.extend(str(n) for n in range(int(lo) + 1, int(hi) + 1))
            else:
                values.append(value)
        else:
            values.append(value)
        pending_range = False
    return values


def parse_question(question: str) -> dict[str, set[str]]:
    """{"article": {"97"}, "annex": {"XI"}, "preamble": {"0"}, ...} for every
    provision the question names. Empty dict when it names none."""
    refs: dict[str, set[str]] = {}
    for m in REF_RE.finditer(question):
        kind = _KINDS[m.group(1).lower().rstrip(".")]
        for value in _numbers(kind, m.group(2)):
            refs.setdefault(kind, set()).add(value)
    if PREAMBLE_RE.search(question):
        refs["preamble"] = {"0"}
    return refs


def _expand(label: str) -> set[str]:
    """Packed recital number "12-15" -> {"12", ..., "15"}."""
    parts = label.split("-")
    if len(parts) == 2 and all(p.isdigit() for p in parts):
        return {str(n) for n in range(int(parts[0]), int(parts[1]) + 1)}
    return {label}


def _prefix(value: str) -> str:
    """"III - HIGH-RISK AI SYSTEMS" -> "III"; "" -> ""."""
    return (value or "").split(" - ", 1)[0].strip()


def match_chunk_ids(refs: dict[str, set[str]], ids: list[str],
                    metadatas: list[dict]) -> list[str]:
    """Chunk ids whose metadata matches any named provision, in corpus order."""
    if not refs:
        return []
    chapters = refs.get("chapter", set())
    matched = []
    for chunk_id, meta in zip(ids, metadatas):
        type_ = meta.get("type")
        number = str(meta.get("number") or "")
        chapter = _prefix(meta.get("chapter"))
        section = _prefix(meta.get("section"))
        hit = (
            (type_ == "Article" and number.lower() in refs.get("article", ()))
            or (type_ == "Annex" and number.upper() in refs.get("annex", ()))
            or (type_ == "Recital" and _expand(number) & refs.get("recital", set()))
            or (type_ == "Preamble" and "preamble" in refs)
            # Section numbers restart in every chapter, so a named chapter
            # narrows the section; a section alone matches it in all chapters
            # and similarity ranking picks among them.
            or (section and section in refs.get("section", ())
                and (not chapters or chapter in chapters))
            # A chapter named together with a section is a qualifier, not a
            # request for the whole chapter.
            or (chapter and chapter in chapters and "section" not in refs)
        )
        if hit:
            matched.append(chunk_id)
    return matched


def selftest() -> None:
    cases = [
        ("what does article 97 talk about?", {"article": {"97"}}),
        ("summarize Article 112", {"article": {"112"}}),
        ("what does Annex 11 talk about?:", {"annex": {"XI"}}),
        ("explain annex III and annex iv", {"annex": {"III", "IV"}}),
        ("recitals 24, 25 and article 2",
         {"recital": {"24", "25"}, "article": {"2"}}),
        ("articles 9 to 12", {"article": {"9", "10", "11", "12"}}),
        ("article 10(2) on bias", {"article": {"10"}}),
        ("art. 5 prohibitions", {"article": {"5"}}),
        ("Chapter III Section 4", {"chapter": {"III"}, "section": {"4"}}),
        ("what does the preamble say", {"preamble": {"0"}}),
        ("how many annexes do we have?", {}),
        ("what article talks about AI literacy?", {}),
        ("which section talks about notifying everyone?", {}),
        ("annex civil liability", {}),
        ("what is an AI system", {}),
    ]
    failed = False
    for question, want in cases:
        got = parse_question(question)
        ok = got == want
        failed |= not ok
        print(f"[{'ok ' if ok else 'FAIL'}] {question!r} -> {got}")

    ids = ["a3", "a30", "a36", "a50", "r24", "x11", "p0"]
    metas = [
        {"type": "Article", "number": "3", "chapter": "I - GENERAL", "section": ""},
        {"type": "Article", "number": "30", "chapter": "III - HIGH-RISK",
         "section": "4 - Notifying authorities"},
        {"type": "Article", "number": "36", "chapter": "III - HIGH-RISK",
         "section": "4 - Notifying authorities"},
        {"type": "Article", "number": "50", "chapter": "V - GPAI",
         "section": "4 - Codes of practice"},
        {"type": "Recital", "number": "22-25", "chapter": "", "section": ""},
        {"type": "Annex", "number": "XI", "chapter": "", "section": ""},
        {"type": "Preamble", "number": "0", "chapter": "", "section": ""},
    ]
    match_cases = [
        ("recital 24", ["r24"]),
        ("section 4", ["a30", "a36", "a50"]),
        ("chapter III section 4", ["a30", "a36"]),
        ("chapter 3", ["a30", "a36"]),
        ("annex 11 and the preamble", ["x11", "p0"]),
        ("article 3", ["a3"]),
    ]
    for question, want in match_cases:
        got = match_chunk_ids(parse_question(question), ids, metas)
        ok = got == want
        failed |= not ok
        print(f"[{'ok ' if ok else 'FAIL'}] match {question!r} -> {got}")

    if failed:
        raise SystemExit(1)
    print("All provision_refs checks passed.")


if __name__ == "__main__":
    selftest()
