"""Splits the raw extracted EU AI Act text into fixed-size token chunks:
500 tokens with a 50-token overlap, measured with the embedding model's own
tokenizer.

The strategy is a three-stage pipeline:

  1. PARSE   -- find the legal structure (Recital / Article / Annex, and the
                Chapter/Section each Article sits under) and emit one "unit"
                per finest-grained provision: a numbered paragraph or point
                where one exists, else the whole provision. This stage exists
                only to produce metadata -- it makes no size decisions.
  2. PACK    -- merge consecutive sibling units while they fit inside one
                500-token budget (config.CHUNK_PACKING = "size"). Packing never
                crosses an Article/Annex boundary (see _citation_root), so
                every chunk keeps a single correct title/number/citation.
  3. WINDOW  -- slice any unit still over budget into non-overlapping windows.
  4. OVERLAP -- prepend the last 50 tokens of the previous window, but only to
                window 2+ of a unit that step 3 cut mid-text
                (config.OVERLAP_SCOPE = "window"). A chunk that starts at a
                structural boundary starts with its own text, not a tail of an
                unrelated provision.

Why this strategy: analysis/chunking_variants_diagnostic.py scored four
whole-corpus variants on the golden set (expected sources in the dense top 8,
of 61). Results: size + document overlap (the old default) 37, size + window
overlap 38, one chunk per structural unit 39, semantic packing 39. The
"none" and "semantic" modes did not clear the pre-set +3 noise margin over the
old default, and they cost 2.5x / 1.7x the chunks, so size packing with
window-only overlap was kept as the lower-risk fix. Both other modes remain
selectable via config.CHUNK_PACKING. See plans/structural-chunking-plan.md.

Two properties of the embedder drive the details, and both are easy to get
wrong:

  * bge-small-en-v1.5 has a 512-token max sequence length. The Chapter/Section
    breadcrumb we prepend is part of the chunk, so it is charged against the
    500-token budget -- otherwise the embedder would silently truncate the tail
    of every long chunk.
  * its tokenizer is UNCASED. Decoding token ids back to text would lowercase
    the corpus and mangle "AI", "CE", "EU". So we window over character offsets
    (return_offsets_mapping) and slice the ORIGINAL string instead of decoding.

PDF text extraction is messy (running headers/footers, mid-word line breaks),
so parsing is best-effort: ingest.py prints a chunk-count summary so you can
sanity-check that the regexes actually found the right structure.
"""
import re
from functools import lru_cache

from transformers import AutoTokenizer

import config

# All header anchors tolerate leading indentation ([ \t]*). Article/Annex
# headers require the provision number to be ALONE on its line (canonical OJ
# format: "Article 6\nClassification rules ..."). This is what distinguishes a
# real header from the many mid-sentence cross-references ("... as set out in
# Article 6 ...") that wrap to a line start in justified body text -- matching
# those too would roughly quadruple the article count.
HAVE_ADOPTED_RE = re.compile(r"HAVE ADOPTED THIS REGULATION", re.IGNORECASE)
# The preamble's own clause starts: the enacting institutions, each citation
# ("Having regard to ...", "After ...", "Acting in accordance ...") and the
# closing "Whereas:". Everything before the first of these is the title block.
PREAMBLE_CLAUSE_RE = re.compile(
    r"^[ \t]*(?=THE EUROPEAN PARLIAMENT|Having regard to|After |"
    r"Acting in accordance|Whereas:)",
    re.MULTILINE,
)
RECITAL_HEADER_RE = re.compile(r"^[ \t]*\((\d{1,3})\)\s+", re.MULTILINE)
ARTICLE_HEADER_RE = re.compile(r"^[ \t]*Article\s+(\d+[a-z]?)[ \t]*$", re.MULTILINE)
ANNEX_HEADER_RE = re.compile(r"^[ \t]*ANNEX\s+([IVXLCDM]+)[ \t]*$", re.MULTILINE)
PARAGRAPH_MARKER_RE = re.compile(r"^[ \t]*(\d{1,2})\.\s+", re.MULTILINE)
# Fallback for provisions that number their top-level entries with a
# parenthetical marker instead ("(1) 'AI system' means ..." -- Article 3's
# definitions, Article 108's amendment list) rather than the period style
# above. Tried only when PARAGRAPH_MARKER_RE finds nothing in a given block,
# so it never overrides the far more common period style. Digits only (no
# letters) and anchored at line start, so it can't match lettered sub-points
# like "(a)", footnote markers like "(*)", or an inline cross-reference like
# "Regulation (EU) 2018/1139" (which isn't at a line start anyway).
PAREN_PARAGRAPH_MARKER_RE = re.compile(r"^[ \t]*\((\d{1,2})\)\s+", re.MULTILINE)
# Chapter/Section headers carry the hierarchy each Article sits under. Their
# number is ALONE on its line (canonical OJ format: "CHAPTER III\nHIGH-RISK AI
# SYSTEMS", "SECTION 1\nClassification ..."), with the title on the next
# non-empty line. Chapters use roman numerals; sections use arabic.
CHAPTER_HEADER_RE = re.compile(r"^[ \t]*CHAPTER\s+([IVXLCDM]+)[ \t]*$", re.MULTILINE)
SECTION_HEADER_RE = re.compile(r"^[ \t]*SECTION\s+(\d+)[ \t]*$", re.MULTILINE)

# Running OJ footer/header noise, e.g. "EN 43 EN" or a bare page number line.
BOILERPLATE_LINE_RE = re.compile(r"^\s*(EN\s+\d+\s+EN|\d{1,4})\s*$", re.MULTILINE)

# Masthead / running-header / footer lines that survive body clipping on the few
# pages whose furniture rules aren't detected (e.g. the title page). Matched as
# whole standalone lines so body prose is never touched. "OJ L, <date>" is the
# running header; it is distinct from footnote citations like "OJ C 517, ...".
PAGE_FURNITURE_RE = re.compile(
    r"^[ \t]*("
    r"EN|L series|"
    r"Official Journal( of the European Union)?|of the European Union|"
    r"OJ [LC], \d{1,2}\.\d{1,2}\.\d{4}|"
    r"ELI:\s*http\S*|"
    r"\d{4}/\d{3,4}|\d{1,2}\.\d{1,2}\.\d{4}|\d{1,3}/\d{2,4}"
    r")[ \t]*$",
    re.MULTILINE,
)

# Page-bottom footnotes are legal citations that ALSO begin with "(N)", so they
# masquerade as recitals. A real recital is prose; a footnote opens with one of
# these citation forms. Combined with the strict 1..N sequence check in
# parse_recitals, this cleanly removes them.
FOOTNOTE_OPENER_RE = re.compile(
    r"^(OJ [CL]|Regulation \(|Directive |"
    r"Council (Directive|Regulation|Decision|Framework)|Decision (\(|No)|"
    r"Position of the European Parliament|European Council|"
    r"European Parliament resolution|Judgment of|Commission [A-Z]|"
    r"Recommendation|Proposal for)"
)

# Ranges use a plain ASCII hyphen in BOTH the citation string and the
# number/paragraph metadata. They must be the SAME character: eval_retrieval's
# completeness check effectively tests `paragraph in citation`, and an en dash
# on one side with a hyphen on the other silently never matches, scoring a
# false 0. ASCII also stops grep treating chunks_export.txt as a binary file.
RANGE_DASH = "-"

# Marks the carry-over text at the head of a chunk (see _add_overlap) so a
# chunk that opens mid-sentence reads as a continuation rather than as the
# start of the provision.
OVERLAP_MARKER = "… "
# Tokens held back from the body budget to cover the newlines and the marker
# that join breadcrumb + overlap + body. Cheap insurance against landing on 501.
_JOIN_SLACK = 4


# --- Tokenizer -------------------------------------------------------------

@lru_cache(maxsize=1)
def _tokenizer():
    """The EMBEDDING model's own tokenizer, so a "token" here means exactly
    what it means to embeddings.py. Using a different tokenizer (tiktoken, a
    word count) lets chunks drift past the model's 512-token limit and be
    truncated without any error. Cached: loading it is slow and every chunk
    measurement calls this.

    model_max_length is raised because this instance only ever MEASURES text --
    we hand it whole 3000-token articles to find their window boundaries and
    never feed those ids to the model. Left at 512 it prints a scary (and here
    meaningless) "sequence length is longer than the maximum" warning for every
    long provision. The real 512 limit is enforced by the budget in _emit."""
    return AutoTokenizer.from_pretrained(
        config.EMBEDDING_MODEL_NAME, model_max_length=int(1e9)
    )


def count_tokens(text: str) -> int:
    """Token length of `text` as the embedder counts it, excluding the [CLS]/
    [SEP] specials (those are the 512-vs-500 headroom, not chunk content)."""
    return len(_tokenizer()(text, add_special_tokens=False)["input_ids"])


def _token_spans(text: str) -> list[tuple[int, int]]:
    """Character (start, end) offsets of each token in `text`.

    We window over these rather than over token ids because the bge tokenizer
    is uncased -- decoding ids back to text would lowercase everything and
    destroy "AI", "CE", "EU", and every proper noun in the Act. Slicing the
    original string by offsets is lossless."""
    encoded = _tokenizer()(
        text, add_special_tokens=False, return_offsets_mapping=True
    )
    # Some tokenizers emit (0, 0) spans for specials/empties; drop them so a
    # window boundary can never collapse to a zero-length slice.
    return [(s, e) for s, e in encoded["offset_mapping"] if e > s]


# --- Cleanup and structural parsing ----------------------------------------

def clean_pdf_text(raw_text: str) -> str:
    """Best-effort cleanup of the extracted text before structural splitting:
    strips repeated OJ boilerplate lines, rejoins words that were hyphen-broken
    across a line end, and collapses excess blank lines."""
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"-\n(?=[a-z])", "", text)  # de-hyphenate line-wrapped words
    text = PAGE_FURNITURE_RE.sub("", text)
    text = BOILERPLATE_LINE_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _title_after(text: str, pos: int) -> str:
    """The title of a Chapter/Section is the first non-empty line following its
    header (which carries only the number). Returns that line, stripped."""
    for line in text[pos:].split("\n"):
        if line.strip():
            return line.strip()
    return ""


def _split_by_header(zone_text: str, header_re: re.Pattern) -> list[tuple[str, str]]:
    """Returns [(header_number, block_text), ...] for a zone split on header_re."""
    matches = list(header_re.finditer(zone_text))
    results = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(zone_text)
        results.append((match.group(1), zone_text[start:end]))
    return results


def _unit(text, type_, number, paragraph, title, citation, **extra) -> dict:
    """One parsed provision, before any size decision is made. Every unit
    carries the full metadata contract so pack/window stages only ever have to
    combine or copy it, never reconstruct it."""
    unit = {
        "text": text.strip(),
        "type": type_,
        "number": number,
        "paragraph": paragraph,
        "title": title,
        "citation": citation,
        "chapter": "",
        "section": "",
        "breadcrumb": "",
    }
    unit.update(extra)
    return unit


def _provision_units(number: str, block: str, kind: str) -> list[dict]:
    """Split one Article/Annex block into its numbered paragraphs/points.

    Unlike the old strategy, this is NOT a size decision -- we always go down
    to paragraph level when the provision has internal numbering, because that
    is what yields the fine-grained "Article 6(2)" citations the eval harness
    ranks on. The packer immediately merges short paragraphs back together."""
    block = block.strip()
    lines = block.split("\n", 1)

    marker = PARAGRAPH_MARKER_RE.search(block)
    marker_re = PARAGRAPH_MARKER_RE
    if not marker:
        marker = PAREN_PARAGRAPH_MARKER_RE.search(block)
        marker_re = PAREN_PARAGRAPH_MARKER_RE

    if marker:
        title = block[: marker.start()].strip()
        body = block[marker.start():]
    else:
        title = lines[0].strip() if lines else ""
        body = lines[1] if len(lines) > 1 else ""

    label = "Article" if kind == "article" else "Annex"

    if not marker:
        whole = f"{title}\n{body}".strip()
        if not whole:
            return []
        return [_unit(whole, label, number, None, title, f"{label} {number}")]

    units = []
    for para_num, para_text in _split_by_header(body, marker_re):
        para_text = para_text.strip()
        if not para_text:
            continue
        units.append(_unit(
            f"{para_num}. {para_text}",
            label, number, para_num, title, f"{label} {number}({para_num})",
        ))
    if units:
        return units
    # Internal numbering matched but yielded nothing usable -- keep the block
    # whole rather than dropping the provision entirely.
    whole = f"{title}\n{body}".strip()
    return [_unit(whole, label, number, None, title, f"{label} {number}")] if whole else []


def parse_preamble(recitals_zone: str) -> list[dict]:
    """Capture the material before Recital (1): the regulation title and the
    'Having regard to ... Whereas:' opening. It carries the Act's short title
    and legal basis, so it is worth retrieving rather than being dropped.

    Parsed like every other zone: one unit per structural clause (title block,
    enacting institutions, each citation, "Whereas:"), and packing decides how
    they become chunks -- no preamble-specific size rule. Every clause keeps
    citation "Preamble"."""
    first_recital = RECITAL_HEADER_RE.search(recitals_zone)
    head = recitals_zone[: first_recital.start()] if first_recital else recitals_zone
    head = head.strip()
    if len(head.split()) < 20:  # nothing substantive captured
        return []
    starts = [0] + [m.start() for m in PREAMBLE_CLAUSE_RE.finditer(head) if m.start() > 0]
    clauses = [head[a:b].strip() for a, b in zip(starts, starts[1:] + [len(head)])]
    return [
        _unit(c, "Preamble", "0", None, "Title and recitations", "Preamble")
        for c in clauses if c
    ]


def parse_recitals(recitals_zone: str) -> list[dict]:
    units = []
    expected = 1  # recitals run strictly 1, 2, 3, ... in order
    for number, body in _split_by_header(recitals_zone, RECITAL_HEADER_RE):
        body = body.strip()
        if not body:
            continue
        # Skip page-bottom footnotes: a real recital both continues the
        # sequence and is prose (not a legal citation). Either test alone is
        # insufficient -- footnote numbering overlaps the recital range, and
        # a stray "(4)" citation can coincide with the next expected number.
        if int(number) != expected or FOOTNOTE_OPENER_RE.match(body):
            continue
        # Re-attach the "(N)" marker that _split_by_header consumed. Without it
        # the recital number exists only in metadata: the chunk text reads as
        # anonymous prose, the LLM cannot cite it from the text it was given,
        # and a packed run of recitals has no visible boundary between one
        # recital and the next. Articles get the same treatment via the
        # "N. " prefix in _provision_units.
        units.append(_unit(
            f"({number}) {body}", "Recital", number, None, None,
            f"Recital {number}",
        ))
        expected += 1
    return units


def parse_articles(articles_zone: str) -> list[dict]:
    """Single ordered pass over CHAPTER / SECTION / Article markers. We track
    the current chapter and section as state (a new chapter resets the section)
    and stamp each article's units with that context. Article blocks run from an
    Article header to the next structural marker of any kind, so chapter/section
    headers never leak into an article body."""
    markers = []
    for m in CHAPTER_HEADER_RE.finditer(articles_zone):
        markers.append((m.start(), m.end(), "chapter", m.group(1)))
    for m in SECTION_HEADER_RE.finditer(articles_zone):
        markers.append((m.start(), m.end(), "section", m.group(1)))
    for m in ARTICLE_HEADER_RE.finditer(articles_zone):
        markers.append((m.start(), m.end(), "article", m.group(1)))
    markers.sort(key=lambda x: x[0])

    units = []
    chapter = section = None  # each is (number, title) or None
    for i, (_start, end, kind, number) in enumerate(markers):
        next_start = markers[i + 1][0] if i + 1 < len(markers) else len(articles_zone)
        if kind == "chapter":
            chapter = (number, _title_after(articles_zone, end))
            section = None  # a new chapter starts its section numbering afresh
        elif kind == "section":
            section = (number, _title_after(articles_zone, end))
        else:  # article
            block = articles_zone[end:next_start]
            for unit in _provision_units(number, block, "article"):
                units.append(_apply_context(unit, chapter, section))
    return units


def parse_annexes(annexes_zone: str) -> list[dict]:
    """Annexes sit outside the Chapter/Section hierarchy, so they carry no
    chapter/section metadata -- but they still need the "Annex III: <title>"
    heading line, or the annex title reaches nothing but metadata."""
    units = []
    for number, block in _split_by_header(annexes_zone, ANNEX_HEADER_RE):
        for unit in _provision_units(number, block, "annex"):
            unit["breadcrumb"] = _provision_line("Annex", unit)
            units.append(unit)
    return units


def _provision_line(label: str, unit: dict) -> str:
    """The "Article 7: Amendments to Annex III" line heading every chunk.

    The TITLE is the point of this. It is usually the most retrieval-relevant
    text in a provision -- "Amendments to Annex III" is the only place those
    words appear anywhere in Article 7 -- and _provision_units keeps it in
    metadata only. Without this line the title reaches neither the embedding
    nor the LLM, and a question phrased in the heading's words cannot match."""
    title = (unit.get("title") or "").strip()
    head = f"{label} {unit['number']}"
    return f"{head}: {title}" if title else head


def _apply_context(unit: dict, chapter: tuple | None, section: tuple | None) -> dict:
    """Attach the Chapter/Section an article sits under, as metadata fields and
    as a breadcrumb that will be prepended to the chunk text, so both the
    embedding and the LLM see the hierarchy. Source casing is preserved
    (title-casing would mangle acronyms like 'AI').

    The breadcrumb is stored separately rather than prepended here: it is
    charged against the 500-token budget in _emit, and windowing a long article
    must repeat it on every window, not just the first."""
    lines = []
    if chapter:
        unit["chapter"] = f"{chapter[0]} - {chapter[1]}"
        lines.append(f"Chapter {chapter[0]}: {chapter[1]}")
    if section:
        unit["section"] = f"{section[0]} - {section[1]}"
        lines.append(f"Section {section[0]}: {section[1]}")
    lines.append(_provision_line("Article", unit))
    unit["breadcrumb"] = "\n".join(lines)
    return unit


# --- Stage 2: packing -------------------------------------------------------

def _citation_root(unit: dict) -> tuple:
    """The identity a chunk's metadata is single-valued over. Two consecutive
    units may be packed into one chunk only if these match.

    Recitals share a root across numbers (they are a flat list whose only
    distinguishing field is the number, so "Recitals 12-15" stays exact).
    Article and Annex paragraphs share a root only within the same provision --
    packing Article 5 together with Article 6 would leave `number`, `title` and
    `citation` with no single correct value, and metadata exactness is the
    point of this design. Preamble clauses share one root, exactly like
    recitals, and all carry the same "Preamble" citation."""
    if unit["type"] in ("Recital", "Preamble"):
        return (unit["type"],)
    return (unit["type"], unit["number"], unit["chapter"], unit["section"])


def _merge(group: list[dict]) -> dict:
    """Collapse a packed group of sibling units into one unit, widening the
    number/paragraph/citation into a range. A single-unit group is returned
    untouched, so the common case never grows a spurious range."""
    if len(group) == 1:
        return group[0]

    first, last = group[0], group[-1]
    merged = dict(first)
    if first["type"] == "Preamble":
        # One provision, cited as a whole; its clauses are single lines in the
        # source, so rejoin them as such.
        merged["text"] = "\n".join(u["text"] for u in group)
        return merged
    merged["text"] = "\n\n".join(u["text"] for u in group)

    if first["type"] == "Recital":
        merged["number"] = f"{first['number']}-{last['number']}"
        merged["citation"] = (
            f"Recitals {first['number']}{RANGE_DASH}{last['number']}"
        )
    else:
        # Same provision, consecutive paragraphs: only the paragraph varies.
        merged["paragraph"] = f"{first['paragraph']}-{last['paragraph']}"
        merged["citation"] = (
            f"{first['type']} {first['number']}"
            f"({first['paragraph']}{RANGE_DASH}{last['paragraph']})"
        )
    return merged


def _adjacent_similarity(units: list[dict]) -> tuple[list[float], float]:
    """Cosine similarity of each unit to the one before it (sims[i] pairs
    units[i-1] and units[i]; sims[0] is unused), plus the merge threshold: the
    SEMANTIC_PACK_PERCENTILE-th percentile over adjacent SAME-ROOT pairs only,
    since only those could ever merge. The threshold is a property of the
    corpus, fixed before any eval runs -- never tuned on the golden set.

    Unit text only, without the breadcrumb: siblings share an identical
    breadcrumb, which would inflate every same-provision similarity alike."""
    import numpy as np

    from embeddings import embed_passages

    vecs = np.asarray(embed_passages([u["text"] for u in units]))
    sims = [0.0] + [float(vecs[i - 1] @ vecs[i]) for i in range(1, len(units))]
    same_root = [
        sims[i] for i in range(1, len(units))
        if _citation_root(units[i]) == _citation_root(units[i - 1])
    ]
    threshold = float(np.percentile(same_root, config.SEMANTIC_PACK_PERCENTILE))
    return sims, threshold


def _pack(units: list[dict], budget_for: callable) -> list[dict]:
    """Turn parsed units into chunk-sized units according to CHUNK_PACKING.

    "size" greedily merges consecutive same-root units while the combined text
    fits the root's token budget. "semantic" adds one condition: the incoming
    unit must be at least threshold-similar to the unit before it (see
    _adjacent_similarity). "none" keeps every structural unit as its own chunk;
    over-budget units are still windowed later by _emit."""
    if config.CHUNK_PACKING == "none":
        return list(units)
    semantic = config.CHUNK_PACKING == "semantic"
    if semantic:
        sims, threshold = _adjacent_similarity(units)

    packed, group, group_tokens = [], [], 0
    for i, unit in enumerate(units):
        tokens = count_tokens(unit["text"])
        fits = (
            group
            and _citation_root(unit) == _citation_root(group[0])
            # +1 approximates the blank line joining the two texts.
            and group_tokens + tokens + 1 <= budget_for(unit)
            and (not semantic or sims[i] >= threshold)
        )
        if fits:
            group.append(unit)
            group_tokens += tokens + 1
        else:
            if group:
                packed.append(_merge(group))
            group, group_tokens = [unit], tokens
    if group:
        packed.append(_merge(group))
    return packed


# --- Stage 3: windowing -----------------------------------------------------

def _windows(text: str, budget: int) -> list[str]:
    """Slice `text` into consecutive, NON-overlapping <=budget-token windows.

    Overlap is deliberately not applied here. It is added once, globally, by
    _add_overlap -- if windowing overlapped too, chunks inside a long provision
    would carry 50 tokens of duplication and chunks at a provision boundary
    would carry none, which is exactly the inconsistency this replaces.

    Slicing is done on character offsets so casing and punctuation survive (see
    _token_spans)."""
    spans = _token_spans(text)
    if len(spans) <= budget:
        return [text]

    out = []
    for start in range(0, len(spans), budget):
        end = min(start + budget, len(spans))
        out.append(text[spans[start][0]:spans[end - 1][1]].strip())
    return [w for w in out if w]


def _tail(text: str, n_tokens: int) -> str:
    """The last n_tokens of `text`, sliced on character offsets."""
    spans = _token_spans(text)
    if len(spans) <= n_tokens:
        return text.strip()
    return text[spans[-n_tokens][0]:].strip()


def _emit(unit: dict) -> list[dict]:
    """Turn one packed unit into its final chunk dicts.

    Body budget reserves room for BOTH the breadcrumb and the 50-token overlap
    prefix that _add_overlap will prepend, because the embedder truncates
    silently at 512 tokens: breadcrumb + overlap + body must land under 500,
    not body alone."""
    breadcrumb = unit.get("breadcrumb", "")
    budget = config.CHUNK_TOKENS - config.CHUNK_OVERLAP_TOKENS - _JOIN_SLACK
    if breadcrumb:
        budget -= count_tokens(breadcrumb)
    budget = max(budget, config.CHUNK_OVERLAP_TOKENS)

    parts = _windows(unit["text"], budget)
    chunks = []
    for i, part in enumerate(parts, start=1):
        chunk = {k: v for k, v in unit.items() if k not in ("breadcrumb", "text")}
        chunk["_breadcrumb"] = breadcrumb
        chunk["_body"] = part
        chunk["part"] = i if len(parts) > 1 else None
        chunks.append(chunk)
    return chunks


def _add_overlap(chunks: list[dict]) -> list[dict]:
    """Prepend the previous chunk's last CHUNK_OVERLAP_TOKENS tokens, in
    document order, and assemble the final text.

    OVERLAP_SCOPE "document" does this for every chunk, across provisions, so
    a chunk ending mid-Article-71 and the chunk starting Article 72 share the
    boundary text. "window" does it only for part 2+ of a unit that _windows
    cut at an arbitrary token position -- the only place a sentence can be
    split -- so a chunk starting at a structural boundary starts with its own
    text. The overlap is taken
    from the previous chunk's own body, never from its inherited overlap, so
    duplication can't compound down a run of chunks.

    Layout is: breadcrumb, then the ellipsis-marked carry-over, then the body.
    The ellipsis matters -- without it a chunk opens on a sentence fragment and
    the LLM reads it as the start of the provision it is being asked to cite."""
    for i, chunk in enumerate(chunks):
        pieces = []
        if chunk["_breadcrumb"]:
            pieces.append(chunk["_breadcrumb"])
        carries = (
            (chunk["part"] or 0) >= 2 if config.OVERLAP_SCOPE == "window" else True
        )
        if i > 0 and carries:
            carry = _tail(chunks[i - 1]["_body"], config.CHUNK_OVERLAP_TOKENS)
            if carry:
                pieces.append(f"{OVERLAP_MARKER}{carry}")
        pieces.append(chunk["_body"])
        chunk["text"] = "\n".join(pieces)
    for chunk in chunks:
        del chunk["_breadcrumb"], chunk["_body"]
    return chunks


# --- Public API -------------------------------------------------------------

def chunk_document(raw_text: str) -> list[dict]:
    """Cleans and splits the full document text into a flat list of chunk
    dicts, each with: text, type, number, paragraph, title, chapter, section,
    citation, source, chunk_id.

    Every chunk is at most config.CHUNK_TOKENS tokens as the embedding model
    counts them, breadcrumb included, with config.CHUNK_OVERLAP_TOKENS tokens
    of overlap between consecutive windows of the same provision."""
    text = clean_pdf_text(raw_text)

    adopted_match = HAVE_ADOPTED_RE.search(text)
    recitals_zone = text[: adopted_match.start()] if adopted_match else ""
    rest = text[adopted_match.end():] if adopted_match else text

    first_annex_match = ANNEX_HEADER_RE.search(rest)
    if first_annex_match:
        articles_zone = rest[: first_annex_match.start()]
        annexes_zone = rest[first_annex_match.start():]
    else:
        articles_zone = rest
        annexes_zone = ""

    units = (
        parse_preamble(recitals_zone)
        + parse_recitals(recitals_zone)
        + parse_articles(articles_zone)
        + parse_annexes(annexes_zone)
    )

    def budget_for(unit: dict) -> int:
        # Must mirror _emit exactly, or packing fills a unit to a size that
        # _emit then has to window back apart.
        crumb = unit.get("breadcrumb", "")
        budget = config.CHUNK_TOKENS - config.CHUNK_OVERLAP_TOKENS - _JOIN_SLACK
        return budget - (count_tokens(crumb) if crumb else 0)

    all_chunks = []
    for unit in _pack(units, budget_for):
        all_chunks.extend(_emit(unit))
    _add_overlap(all_chunks)

    # Some annexes (e.g. VIII, XI) restart their point numbering within
    # sub-sections, so (type, number, paragraph) is not globally unique. Track
    # seen ids and disambiguate collisions -- Chroma requires unique ids.
    seen_ids: dict[str, int] = {}
    for chunk in all_chunks:
        chunk["source"] = config.SOURCE_LABEL
        chunk.setdefault("chapter", "")
        chunk.setdefault("section", "")
        suffix = f"-{chunk['paragraph']}" if chunk["paragraph"] else ""
        part = f"#{chunk['part']}" if chunk.get("part") else ""
        base_id = f"{chunk['type'].lower()}-{chunk['number']}{suffix}{part}"
        count = seen_ids.get(base_id, 0) + 1
        seen_ids[base_id] = count
        # Use a "__" separator for the collision counter so it can't be
        # confused with the single-"-" paragraph suffix (e.g. article-47
        # occurrence #2 becomes "article-47__2", never "article-47-2", which
        # is already Article 47 paragraph 2).
        chunk["chunk_id"] = base_id if count == 1 else f"{base_id}__{count}"

    return all_chunks
