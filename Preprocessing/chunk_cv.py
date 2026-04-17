import os
import re
import json
import pdfplumber
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import logging

# =========================
# LOGGING
# =========================
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("cv_pipeline")

# =========================
# CONFIG
# =========================
DATA_PATH = r"data\pdf"
OUTPUT_FILE = r"output\chunks.json"
MAX_CHUNK_CHARS = 500
SENTENCE_OVERLAP = 1   # Number of sentences to carry over for context (avoids mid-word cuts)

# =========================
# DATA STRUCTURES
# =========================

@dataclass
class ContactInfo:
    email: str = ""
    phone: str = ""
    linkedin: str = ""
    github: str = ""

@dataclass
class Chunk:
    file_name: str
    candidate_name: str
    section: str
    chunk_id: int           # Global unique ID across all files
    local_chunk_id: int     # Index within the section (useful for ordering)
    text: str
    char_count: int
    word_count: int
    email: str = ""
    phone: str = ""
    linkedin: str = ""
    github: str = ""
    embedding_text: str = ""

# =========================
# SECTION PATTERNS
# Ordered from most-specific to least-specific to avoid false matches.
# Each pattern is anchored to the full line (after stripping punctuation).
# =========================

SECTION_PATTERNS = [
    # (section_name, [regex patterns matched against the normalised line])
    ("summary",     [r"^(summary|objective|about me|profile|professional profile|career summary)$"]),
    ("education",   [r"^(education|academic background|qualifications|academic qualifications)$",
                     r"^(university|faculty of|college of)",
                     r"^academic$"]),
    ("experience",  [r"^(experience|work experience|work history|employment|employment history)$",
                     r"^(professional experience|career history|internship|internships)$"]),
    ("skills",      [r"^(skills|technical skills|key skills|core skills|competencies)$",
                     r"^(technologies|tools|technical proficiency|technical stack)$"]),
    ("projects",    [r"^(projects|personal projects|academic projects|notable projects)$"]),
    ("courses",     [r"^(courses|certifications|training|professional development)$",
                     r"^(certificates|licenses|continuing education)$"]),
    ("languages",   [r"^(languages|spoken languages|language proficiency)$"]),
    ("publications",[r"^(publications|research|research publications|papers|journal articles)$"]),
    ("awards",      [r"^(awards|honors|honours|achievements|recognitions|accomplishments)$"]),
    ("volunteering",[r"^(volunteering|volunteer|community|extracurricular|activities)$"]),
    ("references",  [r"^(references|referees)$"]),
]

# Lines that look like section headers but aren't — guard against false positives
FALSE_POSITIVE_GUARDS = [
    r"\d",         # Has a digit (likely a date or year)
    r"@",          # Email address
    r"http",       # URL
    r"\|",         # Separator character common in experience entries
]

# =========================
# TEXT CLEANING
# =========================

def normalize_text(text: str) -> str:
    """Clean a single line: bullets → dashes, collapse whitespace, strip hyphen-breaks."""
    text = re.sub(r"[•●▪◆►▶]", "-", text)
    text = re.sub(r"[ \t]+", " ", text)          # Collapse horizontal whitespace only
    text = re.sub(r"-\s+(?=[a-z])", "", text)    # Remove hyphenation breaks (only before lowercase)
    return text.strip()

def clean_extracted_text(raw: str) -> str:
    """Post-process full extracted page text before splitting to lines."""
    # Fix common PDF ligature issues
    raw = raw.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("ﬀ", "ff")
    raw = raw.replace("ﬃ", "ffi").replace("ﬄ", "ffl")
    # Remove control characters except newline
    raw = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", raw)
    return raw

# =========================
# PDF EXTRACTION
# =========================

def extract_lines(pdf_path: str) -> List[str]:
    """Extract all non-empty, normalised lines from a PDF."""
    lines = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                text = clean_extracted_text(text)
                for line in text.split("\n"):
                    line = normalize_text(line)
                    if line and len(line) > 1:   # Skip single-character noise
                        lines.append(line)
    except Exception as e:
        logger.error(f"Failed to read {pdf_path}: {e}")
    return lines

# =========================
# CONTACT EXTRACTION
# =========================

def extract_contact(lines: List[str]) -> ContactInfo:
    """
    Extract contact fields from the first ~40 lines.
    Uses targeted, unambiguous patterns to avoid false matches.
    """
    text = " ".join(lines[:40])

    # Email — standard pattern
    emails = re.findall(r"[\w.+\-]+@[\w\-]+\.[a-zA-Z]{2,}", text)

    # Phone — international or local, must be 7+ digits, avoids matching years (4 digits)
    # Matches: +92-333-1234567, (123) 456-7890, 01225796530, +123-456-7890
    phones = re.findall(
        r"(?<!\d)(?:\+?\d{1,3}[\s.\-]?)?(?:\(?\d{2,5}\)?[\s.\-]?)?\d{3,5}[\s.\-]?\d{4}(?!\d)",
        text
    )
    # Filter out short matches that are just years
    phones = [p.strip() for p in phones if len(re.sub(r"\D", "", p)) >= 7]

    linkedin = re.findall(r"linkedin\.com/in/[\w\-]+", text, re.IGNORECASE)
    github   = re.findall(r"github\.com/[\w\-]+", text, re.IGNORECASE)

    return ContactInfo(
        email    = emails[0]   if emails    else "",
        phone    = phones[0]   if phones    else "",
        linkedin = linkedin[0] if linkedin  else "",
        github   = github[0]   if github    else "",
    )

# =========================
# NAME EXTRACTION
# =========================

def extract_name(lines: List[str]) -> str:
    """
    Heuristic: the candidate name is usually one of the first few lines,
    contains only alphabetic characters and spaces, and is 2–5 words long.
    Handles both "First Last" and "FIRST LAST" formats.
    """
    for line in lines[:8]:
        line = line.strip()
        words = line.split()
        if (
            2 <= len(words) <= 5
            and all(re.match(r"^[A-Za-z\-']+$", w) for w in words)
            and not any(kw in line.lower() for kw in ["engineer", "developer", "manager",
                                                        "analyst", "designer", "scientist",
                                                        "student", "accountant", "chemist",
                                                        "officer", "assistant", "specialist"])
        ):
            # Normalise ALL-CAPS names to Title Case
            if line.isupper():
                return line.title()
            return line
    return "Unknown"

# =========================
# SECTION DETECTION
# =========================

def detect_section(line: str) -> Optional[str]:
    """
    Determine if a line is a section header.
    Returns the section name or None if it's regular content.
    """
    stripped = line.strip().rstrip(":").strip()

    # Section headers are short
    if len(stripped) > 60:
        return None

    # Guard against false positives
    for guard in FALSE_POSITIVE_GUARDS:
        if re.search(guard, stripped):
            return None

    normalised = stripped.lower()

    for section_name, patterns in SECTION_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, normalised):
                return section_name

    return None

def split_sections(lines: List[str]) -> Dict[str, List[str]]:
    """
    Walk through lines and assign each to a section.
    Preserves insertion order (Python 3.7+ dicts).
    """
    sections: Dict[str, List[str]] = defaultdict(list)
    current = "header"

    for line in lines:
        detected = detect_section(line)
        if detected:
            current = detected
        else:
            sections[current].append(line)

    return dict(sections)

# =========================
# SENTENCE-AWARE CHUNKING
# =========================

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentence-like units.
    Handles: periods, newlines, bullet dashes, and common abbreviations.
    """
    # Split on: period/exclamation/question followed by space+capital,
    # OR on a dash-bullet at the start of a segment,
    # OR on double+ spaces (common in PDF extraction).
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])|(?=\s-\s)|(?<=\n)", text)
    sentences = []
    for p in parts:
        p = p.strip().lstrip("- ").strip()
        if p:
            sentences.append(p)
    return sentences

def chunk_section(text: str, max_size: int = MAX_CHUNK_CHARS,
                  overlap_sentences: int = SENTENCE_OVERLAP) -> List[str]:
    """
    Split section text into overlapping chunks.
    Overlap is sentence-based (not raw character slicing) to avoid mid-word cuts.

    Strategy:
      - Accumulate sentences until the chunk would exceed max_size.
      - When flushing a chunk, carry over the last `overlap_sentences` sentences
        as context for the next chunk.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current_sentences: List[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence) + 1  # +1 for space

        if current_len + sentence_len > max_size and current_sentences:
            # Flush current chunk
            chunks.append(" ".join(current_sentences).strip())
            # Carry over the tail sentences for context
            current_sentences = current_sentences[-overlap_sentences:] if overlap_sentences else []
            current_len = sum(len(s) + 1 for s in current_sentences)

        current_sentences.append(sentence)
        current_len += sentence_len

    # Flush the final chunk
    if current_sentences:
        chunks.append(" ".join(current_sentences).strip())

    return chunks

# =========================
# EMBEDDING TEXT GENERATION
# =========================

def build_embedding_text(candidate_name: str, section: str, text: str) -> str:
    """
    Format the text for embedding in a way that gives the model rich context.
    Including the candidate name helps retrieval when searching across many CVs.
    """
    return f"Candidate: {candidate_name}\nSection: {section}\nContent: {text}"

# =========================
# PIPELINE
# =========================

class CVPipeline:

    def process_pdf(self, path: str, global_id_start: int = 0) -> List[Chunk]:
        filename = os.path.basename(path)
        lines = extract_lines(path)
        if not lines:
            logger.warning(f"No text extracted from {filename}")
            return []

        name    = extract_name(lines)
        contact = extract_contact(lines)
        sections = split_sections(lines)

        logger.info(f"  Candidate: {name} | Sections found: {list(sections.keys())}")

        processed_chunks: List[Chunk] = []
        global_chunk_id = global_id_start

        for section_name, content_lines in sections.items():
            # Join lines with a space, preserving natural sentence flow
            full_section_text = " ".join(content_lines)
            full_section_text = re.sub(r"\s+", " ", full_section_text).strip()

            # Skip noise — very short sections are usually stray text
            if len(full_section_text) < 20:
                continue

            section_parts = chunk_section(full_section_text)

            for local_id, part in enumerate(section_parts):
                part = part.strip()
                if not part:
                    continue

                processed_chunks.append(Chunk(
                    file_name      = filename,
                    candidate_name = name,
                    section        = section_name,
                    chunk_id       = global_chunk_id,
                    local_chunk_id = local_id,
                    text           = part,
                    char_count     = len(part),
                    word_count     = len(part.split()),
                    email          = contact.email,
                    phone          = contact.phone,
                    linkedin       = contact.linkedin,
                    github         = contact.github,
                    embedding_text = build_embedding_text(name, section_name, part),
                ))
                global_chunk_id += 1

        return processed_chunks

    def run(self):
        if not os.path.exists(DATA_PATH):
            logger.error(f"Data path not found: {DATA_PATH}")
            return

        pdf_files = sorted(f for f in os.listdir(DATA_PATH) if f.lower().endswith(".pdf"))
        if not pdf_files:
            logger.error("No PDF files found.")
            return

        all_chunks: List[Chunk] = []
        global_id = 0

        for filename in pdf_files:
            logger.info(f"Processing: {filename}")
            path = os.path.join(DATA_PATH, filename)
            file_chunks = self.process_pdf(path, global_id_start=global_id)
            all_chunks.extend(file_chunks)
            global_id += len(file_chunks)

        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in all_chunks], f, indent=2, ensure_ascii=False)

        # Summary statistics
        from collections import Counter
        section_counts = Counter(c.section for c in all_chunks)
        candidate_counts = Counter(c.candidate_name for c in all_chunks)

        logger.info(f"\n{'='*50}")
        logger.info(f"Saved {len(all_chunks)} chunks to {OUTPUT_FILE}")
        logger.info(f"Candidates processed: {len(candidate_counts)}")
        logger.info(f"Sections distribution: {dict(section_counts.most_common())}")
        logger.info(f"{'='*50}")


if __name__ == "__main__":
    pipeline = CVPipeline()
    pipeline.run()