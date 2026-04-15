import os
import re
import json
import pdfplumber
from typing import Dict, List, Optional, Tuple

# =========================
# CONFIG
# =========================
DATA_PATH   = r"data\pdf"
OUTPUT_FILE = r"output\chunks.json"
MAX_CHUNK_CHARS = 800   # soft ceiling per chunk

# ─────────────────────────────────────────────
# SECTION PATTERNS  (matched on normalised line)
# ─────────────────────────────────────────────

SECTION_PATTERNS: Dict[str, List[str]] = {
    "education": [
        r"^education$",
        r"^academic background$",
        r"^degrees?$",
        r"^educational qualifications?$",
        r"^academic qualifications?$",
    ],
    "experience": [
        r"^(work\s+)?experience$",
        r"^professional experience$",
        r"^employment( history)?$",
        r"^career( history)?$",
        r"^work history$",
    ],
    "skills": [
        r"^(technical\s+)?skills?$",
        r"^core competencies$",
        r"^technologies$",
        r"^areas of expertise$",
        r"^design platforms?",
        r"^programming languages?$",
    ],
    "projects": [
        r"^(funded\s+)?research projects?$",
        r"^projects?$",
        r"^personal projects?$",
        r"^notable projects?$",
    ],
    "publications": [
        r"^publications?$",
        r"^journal\s+publications?$",
        r"^conference\s+publications?$",
        r"^research papers?$",
        r"^research publications?",
    ],
    "achievements": [
        r"^(academic\s+)?achievements?$",
        r"^awards?(\s+and\s+honors?)?$",
        r"^honors?$",
        r"^certificates?$",
        r"^certifications?$",
        r"^awards and honors?$",
    ],
    "conferences": [
        r"^conferences?(\s+and\s+workshops?)?$",
        r"^conferences?\s+attended$",
        r"^workshops?\s+attended$",
    ],
    "fields_of_interest": [
        r"^fields\s+of\s+interest$",
        r"^research interests?$",
        r"^areas\s+of\s+interest$",
    ],
    "summary": [
        r"^(professional\s+)?summary$",
        r"^objective$",
        r"^profile$",
        r"^about(\s+me)?$",
        r"^career objective$",
    ],
    "languages": [
        r"^language proficiency$",
        r"^spoken languages?$",
        r"^human languages?$",
    ],
    "references": [
        r"^references?$",
        r"^referees?$",
    ],
    "contact": [
        r"^contact(\s+information)?$",
        r"^personal details?$",
        r"^personal information$",
    ],
    "courses": [
        r"^courses?$",
        r"^certifications? and courses?$",
        r"^online courses?$",
        r"^courses taught$",
    ],
}

# ─────────────────────────────────────────────
# STEP 1 — Multi-column-aware text extraction
# ─────────────────────────────────────────────

def _extract_page_lines_column_aware(page) -> List[str]:
    """
    For pages that look multi-column (common in CV PDFs), split
    the page vertically at the midpoint and read each column as a
    separate stream of lines, then interleave them top-to-bottom.
    Falls back to plain left-to-right extraction for single-column pages.
    """
    words = page.extract_words(x_tolerance=3, y_tolerance=3)
    if not words:
        return []

    page_width = page.width

    # Detect multi-column: are there clusters of words on both the
    # left half and right half, with a gap in the middle?
    left_words  = [w for w in words if w["x1"] < page_width * 0.52]
    right_words = [w for w in words if w["x0"] > page_width * 0.48]

    gap_ratio = (len(left_words) + len(right_words)) / max(len(words), 1)
    is_multicolumn = (
        len(left_words) > 10
        and len(right_words) > 10
        and gap_ratio > 0.85   # most words are in one of the two halves
    )

    if not is_multicolumn:
        text = page.extract_text()
        return [l.strip() for l in (text or "").split("\n") if l.strip()]

    # Build lines from each column independently
    def words_to_lines(word_list: list) -> List[str]:
        if not word_list:
            return []
        # Sort by top (y0), then left (x0)
        word_list = sorted(word_list, key=lambda w: (round(w["top"] / 5) * 5, w["x0"]))
        lines: List[str] = []
        current_top  = word_list[0]["top"]
        current_line: List[str] = []
        for w in word_list:
            if abs(w["top"] - current_top) > 5:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [w["text"]]
                current_top  = w["top"]
            else:
                current_line.append(w["text"])
        if current_line:
            lines.append(" ".join(current_line))
        return lines

    # Assign each word exclusively to left or right column
    midpoint = page_width / 2
    left_only  = [w for w in words if w["x1"] <= midpoint]
    right_only = [w for w in words if w["x0"] >= midpoint]

    left_lines  = words_to_lines(left_only)
    right_lines = words_to_lines(right_only)

    # Merge the two columns in reading order (left first, then right)
    return left_lines + right_lines


def extract_lines(file_path: str) -> List[Dict]:
    """Return every non-empty line as {'text': str, 'page': int}."""
    lines = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                for raw in _extract_page_lines_column_aware(page):
                    text = raw.strip()
                    if text:
                        lines.append({"text": text, "page": page_num})
    except Exception as exc:
        print(f"  [ERROR] Could not read {file_path}: {exc}")
    return lines


# ─────────────────────────────────────────────
# STEP 2 — Header detection
# ─────────────────────────────────────────────

def _normalize(line: str) -> str:
    """Strip numbering, trailing colon, collapse whitespace, lowercase."""
    line = re.sub(r"^\s*(\d+|[IVXivxA-Za-z])[.)]\s*", "", line)
    line = line.rstrip(":").strip()
    line = re.sub(r"\s+", " ", line)
    return line.lower()


def detect_section_header(line: str) -> Optional[str]:
    """
    Return the section key if line looks like a CV section header,
    otherwise return None.

    Header candidate rules:
      - length after stripping <= 60 chars
      - is ALL CAPS, Title Case, or ends with ':'
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 60:
        return None

    ends_with_colon = stripped.endswith(":")
    # Ignore lines that are just a single letter or number
    core = stripped.rstrip(":").strip()
    if len(core) < 3:
        return None

    is_all_caps   = core.replace(" ", "").isupper() and len(core) > 2
    is_title_case = core.istitle()

    if not (ends_with_colon or is_all_caps or is_title_case):
        return None

    normalised = _normalize(stripped)

    for section, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, normalised, re.IGNORECASE):
                return section

    return None


# ─────────────────────────────────────────────
# STEP 3 — Split lines into labelled sections
# ─────────────────────────────────────────────

def split_into_sections(lines: List[Dict]) -> Dict[str, List[str]]:
    """
    Walk lines; on a detected header switch to a new bucket.
    Pre-header lines go into 'header'.
    """
    sections: Dict[str, List[str]] = {"header": []}
    current_section = "header"

    for line_info in lines:
        line     = line_info["text"]
        detected = detect_section_header(line)

        if detected:
            current_section = detected
            if current_section not in sections:
                sections[current_section] = []
            sections[current_section].append(line)   # keep the header line
        else:
            if current_section not in sections:
                sections[current_section] = []
            sections[current_section].append(line)

    return sections


# ─────────────────────────────────────────────
# STEP 4 — Post-process: reclassify ambiguous sections
# ─────────────────────────────────────────────

def _reclassify_languages_section(sections: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    'Languages' in CV PDFs sometimes means programming languages (→ skills)
    and sometimes means spoken languages.  Classify by content:
      - if the section mostly contains human-language names → keep as 'languages'
      - if it mostly contains tech keywords → merge into 'skills'
    """
    if "languages" not in sections:
        return sections

    lang_lines = sections["languages"]
    text = " ".join(lang_lines).lower()

    # Tech keywords that indicate this is really a skills section
    tech_keywords = [
        "python", "java", "c++", "c#", "dart", "javascript", "flutter",
        "pandas", "tensorflow", "pytorch", "sql", "html", "css", "php",
        "ruby", "rust", "kotlin", "swift", "matlab", "r ", "scala",
        "nodejs", "react", "angular", "vue",
    ]
    # Human language keywords
    human_keywords = [
        "arabic", "english", "french", "spanish", "german", "chinese",
        "urdu", "portuguese", "italian", "native", "fluent", "proficient",
        "beginner", "intermediate", "advanced", "mother tongue",
    ]

    tech_hits  = sum(1 for k in tech_keywords  if k in text)
    human_hits = sum(1 for k in human_keywords if k in text)

    if tech_hits > human_hits:
        # Merge into skills
        if "skills" not in sections:
            sections["skills"] = []
        sections["skills"].extend(lang_lines)
        del sections["languages"]

    return sections


def post_process_sections(sections: Dict[str, List[str]]) -> Dict[str, List[str]]:
    sections = _reclassify_languages_section(sections)
    return sections


# ─────────────────────────────────────────────
# STEP 5 — Clean & chunk each section
# ─────────────────────────────────────────────

def clean_lines(lines: List[str]) -> str:
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"[•●▪◆]", "-", line)   # normalise bullets
        line = re.sub(r"\s{2,}", " ", line)     # collapse spaces
        cleaned.append(line)
    return "\n".join(cleaned)


def _split_into_logical_items(text: str) -> List[str]:
    """
    Split section text into logical items for chunking:
      1. Numbered list items (1. / [1] / (1))
      2. Lettered headers (JOURNAL PAPERS:)
      3. Paragraph breaks (double newline)
      4. Bullet groups
    """
    # Try numbered items first
    if re.search(r"\n[\[\(]?\d+[\]\).]", text):
        parts = re.split(r"\n(?=[\[\(]?\d+[\]\).])", text)
        if len(parts) > 1:
            return parts

    # Try double-newline paragraphs
    parts = re.split(r"\n{2,}", text)
    if len(parts) > 1:
        return parts

    return [text]


def chunk_text(section_name: str, text: str,
               max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    """
    Chunk a section text.  Publications and conferences get smaller chunks
    (one entry per chunk ideally) for better retrieval precision.
    """
    if len(text) <= max_chars:
        return [text]

    items = _split_into_logical_items(text)
    chunks: List[str] = []
    current: List[str] = []
    size = 0

    for item in items:
        item = item.strip()
        if not item:
            continue

        # For publication/project sections, prefer one entry per chunk
        # regardless of max_chars (unless a single entry exceeds it)
        is_list_section = section_name in ("publications", "projects",
                                           "conferences", "achievements")
        force_split = is_list_section and size > 0 and len(item) > 100

        if force_split or (size + len(item) > max_chars and current):
            chunks.append("\n\n".join(current))
            current, size = [item], len(item)
        else:
            current.append(item)
            size += len(item)

    if current:
        chunks.append("\n\n".join(current))

    return chunks


# ─────────────────────────────────────────────
# STEP 6 — Process a single PDF
# ─────────────────────────────────────────────

def process_pdf(file_path: str) -> List[Dict]:
    filename = os.path.basename(file_path)
    print(f"Processing: {filename}")

    lines = extract_lines(file_path)
    if not lines:
        print("  [WARN] No text extracted — is the PDF image-based?")
        return []

    sections = split_into_sections(lines)
    sections = post_process_sections(sections)

    all_chunks: List[Dict] = []
    for section_name, section_lines in sections.items():
        text = clean_lines(section_lines)
        if not text or len(text) < 20:
            continue

        sub_chunks = chunk_text(section_name, text)
        for idx, chunk in enumerate(sub_chunks):
            all_chunks.append({
                "file_name":  filename,
                "section":    section_name,
                "chunk_id":   idx,
                "text":       chunk,
                "char_count": len(chunk),
            })

    detected = sorted({c["section"] for c in all_chunks})
    print(f"  Sections  : {', '.join(detected)}")
    print(f"  Chunks    : {len(all_chunks)}")

    # Warn about suspiciously large chunks
    large = [c for c in all_chunks if c["char_count"] > 2000]
    if large:
        print(f"  [WARN] {len(large)} chunk(s) exceed 2000 chars "
              f"(likely multi-column scrambling or very long section)")
        for c in large[:3]:
            print(f"    - section={c['section']}  chars={c['char_count']}")

    return all_chunks


# ─────────────────────────────────────────────
# STEP 7 — Process all PDFs
# ─────────────────────────────────────────────

def process_all_pdfs() -> List[Dict]:
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Folder not found: {DATA_PATH}")
        return []

    pdf_files = [f for f in os.listdir(DATA_PATH) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"[ERROR] No PDF files in {DATA_PATH}")
        return []

    print(f"Found {len(pdf_files)} PDF(s)\n")
    all_chunks: List[Dict] = []
    for pdf_file in pdf_files:
        chunks = process_pdf(os.path.join(DATA_PATH, pdf_file))
        all_chunks.extend(chunks)
        print()

    return all_chunks


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    chunks = process_all_pdfs()

    if not chunks:
        print("\nNo chunks generated.  Things to check:")
        print("  1. Are the PDFs text-based (not scanned images)?")
        print("  2. Do section headers appear in ALL CAPS or Title Case?")
        print("  3. Add custom patterns to SECTION_PATTERNS if needed.")
        raise SystemExit(1)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved {len(chunks)} chunks → {OUTPUT_FILE}")

    # ── Summary ──
    from collections import Counter
    counts = Counter(c["section"] for c in chunks)
    print("\nChunks per section:")
    for section, n in sorted(counts.items()):
        print(f"  {section:<30} {n:>3} chunk(s)")

    # ── Size distribution ──
    sizes = [c["char_count"] for c in chunks]
    print(f"\nChunk size stats:")
    print(f"  min={min(sizes)}  max={max(sizes)}  "
          f"avg={sum(sizes)//len(sizes)}  total_chunks={len(chunks)}")

    over_limit = sum(1 for s in sizes if s > MAX_CHUNK_CHARS)
    print(f"  Chunks over {MAX_CHUNK_CHARS} chars: {over_limit} "
          f"({100*over_limit//len(sizes)}%) — often unavoidable for "
          f"single-entry publications that are themselves long")

    # ── Previews ──
    print("\n── Sample chunks ──")
    seen: set = set()
    for chunk in chunks:
        if chunk["section"] not in seen:
            preview = chunk["text"][:200].replace("\n", " | ")
            print(f"\n[{chunk['section'].upper()}]\n{preview}…")
            seen.add(chunk["section"])
        if len(seen) >= 6:
            break