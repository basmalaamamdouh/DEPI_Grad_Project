# DEPI_Grad_Project
data used https://drive.google.com/drive/folders/1Y6DXboh3euFfdYmJRBwUQJJpcXz0MAQ2?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto

DEPI GenAI CV Search Engine

An AI pipeline for processing resumes, extracting structured information, and enabling semantic search using embeddings.

Features
Dataset loading from Kaggle
Text extraction from PDFs using PyMuPDF
OCR fallback for scanned documents using Tesseract
Image text extraction support
Resume section-based chunking (skills, experience, education, projects)
Semantic embeddings using Sentence Transformers
Cosine similarity search over CV chunks
Embedding persistence using JSON storage
Interactive search interface
Pipeline
Dataset → Text Extraction → Chunking → Embedding Generation → Semantic Search
Installation

1. Install dependencies
pip install pytesseract pillow pymupdf sentence-transformers kagglehub numpy
2. Install Tesseract OCR (Required for scanned PDFs)

Download from:

https://github.com/UB-Mannheim/tesseract/wiki

During installation:

Enable “Add to PATH”
Install English language pack
4. Verify installation
tesseract --version
5. Configure Tesseract path in code

If not automatically detected:

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
Usage
Run with Kaggle dataset
python pipeline.py --kaggle --rebuild
Run with local dataset
python pipeline.py --dataset <folder_path> --rebuild
Interactive search

After processing(for now), use:

Search > python machine learning
Output

Processed embeddings are stored in:

cv_embeddings.json

Each entry contains:

file name
section type
extracted text
embedding vector
Troubleshooting
Tesseract not found

Ensure:

Tesseract is installed
It is added to system PATH
Or manually set path in code
Empty embeddings file

Possible causes:

OCR failure
Missing Tesseract installation
PDFs are image-based and not processed correctly

Fix:

python pipeline.py --kaggle --rebuild
No search results

Ensure embeddings exist and are not empty. If needed, rebuild dataset.

Tech Stack
Python
PyMuPDF
Tesseract OCR
SentenceTransformers (MiniLM)
NumPy
KaggleHub
Future Improvements

Candidate ranking system based on job descriptions
Vector database integration (FAISS or ChromaDB)
Resume scoring model

Use Cases


----------------------------------------------------
# HR Assistant — CV RAG Search System

A fully local, offline-capable system for indexing large CV datasets and searching them with natural language. Built on semantic vector search, BM25 keyword retrieval, and a CrossEncoder reranker. No cloud services required.

---

## Table of Contents

1. [What This System Does](#1-what-this-system-does)
2. [Architecture Overview](#2-architecture-overview)
3. [Project Structure](#3-project-structure)
4. [Installation](#4-installation)
5. [Running the Pipeline (Ingestion)](#5-running-the-pipeline-ingestion)
6. [Running the Web App (Search)](#6-running-the-web-app-search)
7. [Pipeline Code — Detailed Walkthrough](#7-pipeline-code--detailed-walkthrough)
   - [7.1 Configuration](#71-configuration)
   - [7.2 The Skip-List (Never Start Over)](#72-the-skip-list-never-start-over)
   - [7.3 Embedding Model](#73-embedding-model-singleton)
   - [7.4 Text Extraction](#74-text-extraction--the-ocr-fix)
   - [7.5 Contact Extraction](#75-contact-extraction)
   - [7.6 Section-Aware Chunking](#76-section-aware-chunking)
   - [7.7 ChromaDB — Vector Store](#77-chromadb--the-vector-store)
   - [7.8 BM25 — Keyword Index](#78-bm25--the-keyword-index)
   - [7.9 The Search Pipeline](#79-the-search-pipeline)
   - [7.10 Fit Percentage](#710-fit-percentage)
8. [How the Two-Phase Design Works](#8-how-the-two-phase-design-works)
9. [Search Quality Explained](#9-search-quality-explained)
10. [Tuning & Configuration Reference](#10-tuning--configuration-reference)
11. [Next Steps for the Team](#11-next-steps-for-the-team)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. What This System Does

This system lets an HR team index an entire dataset of CV/resume PDFs once, then search through all of them instantly using plain English queries like:

> *"Senior Python developer with Django and PostgreSQL, 5 years experience"*
> *"Junior data analyst proficient in Excel and Power BI, Cairo-based"*
> *"Machine learning engineer with NLP and TensorFlow background"*

Each result shows:
- A **fit percentage** (0–100%) representing how well the candidate matches the query
- A **match quality badge** (Strong / Good / Partial)
- Extracted contact info: name, email, phone, LinkedIn
- Which CV sections matched and how many query keywords were found
- A text preview from the most relevant parts of their CV

The system runs entirely on your local machine. No API keys, no internet connection required after setup.

---

## 2. Architecture Overview

The system is split into two phases that are completely independent of each other.

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1 — OFFLINE INGESTION  (run once, takes 30–90 min)       │
│                                                                  │
│  PDF files  →  Text extraction  →  Chunking  →  Embedding       │
│                    ↓                   ↓            ↓            │
│              (PyMuPDF first,    (section-aware)  (BGE-small)    │
│               OCR fallback)                                      │
│                                                    ↓            │
│                                         ┌──────────┴─────────┐  │
│                                         │   ChromaDB (disk)  │  │
│                                         │   BM25 index (.pkl)│  │
│                                         │   indexed_files.txt│  │
│                                         └────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2 — ONLINE SEARCH  (Gradio app, always fast)            │
│                                                                  │
│  Query  →  Embed query  →  ChromaDB dense retrieval             │
│                        →  BM25 sparse retrieval                 │
│                        →  RRF fusion                            │
│                        →  Deduplicate per candidate             │
│                        →  CrossEncoder reranker                 │
│                        →  Fit percentage calculation            │
│                        →  Result cards in UI                    │
└─────────────────────────────────────────────────────────────────┘
```

**The key design principle:** Phase 1 runs once. Phase 2 runs on every search. The web app (`app.py`) only ever does Phase 2 — it never re-reads or re-processes any PDF.

---

## 3. Project Structure

```
project/
│
├── pipeline.py          # All ingestion + search logic
├── app.py               # Gradio web UI
│
├── chroma_db/           # Vector database (created automatically)
├── bm25_index.pkl       # BM25 keyword index (created automatically)
├── indexed_files.txt    # Fast skip-list of processed filenames
├── uploads/             # Folder for individually uploaded CVs
│
└── README.md            # This file
```

---

## 4. Installation

### Requirements

- Python 3.10 or higher
- ~2 GB disk space for models (downloaded automatically on first run)
- Tesseract OCR (only needed for scanned/image-based PDFs)

### Install Python dependencies

```bash
pip install gradio sentence-transformers chromadb rank-bm25 pymupdf pillow
```

### Install Tesseract (optional — for scanned PDFs only)

Most CV datasets use native-text PDFs and do not need Tesseract. Install it only if your dataset contains scanned documents.

- **Windows:** Download from https://github.com/UB-Mannheim/tesseract/wiki
- **Linux:** `sudo apt install tesseract-ocr`
- **macOS:** `brew install tesseract`

After installing, update this line in `pipeline.py`:

```python
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Windows
# or
TESSERACT_CMD = "tesseract"  # Linux / macOS (if on PATH)
```

---

## 5. Running the Pipeline (Ingestion)

Run the pipeline once on your CV dataset folder. It will index every PDF it finds, skip any it has already processed, and build the search database.

```bash
# Index a full folder
python pipeline.py --dataset "D:/path/to/CVs"

# Index only the first 100 files (useful for testing)
python pipeline.py --dataset "D:/path/to/CVs" --limit 100

# Download and index the Kaggle resume dataset automatically
python pipeline.py --kaggle

# Wipe everything and start fresh
python pipeline.py --dataset "D:/path/to/CVs" --rebuild
```

### What happens during ingestion

For each PDF, the pipeline prints a log line like:

```
Processing: john_smith_cv.pdf
  native chars: 3847
  Extraction: native | 3847 chars
  Contact: John Smith <john@email.com>
  Chunks: 6
✅ john_smith_cv.pdf — 6 chunks [native]
```

If `native chars` is a large number (> 30), OCR was skipped entirely. If it shows a small number, the PDF is scanned and Tesseract will run on it.

### Resuming after an interruption

If ingestion is interrupted, simply run the same command again. The pipeline reads `indexed_files.txt` and skips every file it has already processed. It will continue from where it left off.

---

## 6. Running the Web App (Search)

```bash
python app.py
```

Open your browser at **http://localhost:7860**

The app has four tabs:
- **Search Candidates** — natural language search with fit percentage results
- **Upload & Ingest** — add individual CVs or point to a new folder
- **Send Emails** — send interview invitations to candidates from your last search
- **Setup** — quick-start guide

---

## 7. Pipeline Code — Detailed Walkthrough

This section explains every part of `pipeline.py` in enough detail for the team to understand, modify, and extend it.

---

### 7.1 Configuration

```python
CHROMA_DIR         = "./chroma_db"
BM25_FILE          = "./bm25_index.pkl"
INDEXED_FILE       = "./indexed_files.txt"
EMBED_MODEL        = "BAAI/bge-small-en-v1.5"
TESSERACT_CMD      = r"D:\tessert\tesseract.exe"
N_DENSE_FETCH      = 500
BM25_REBUILD_EVERY = 20
SCORE_SCALE        = 0.35
SCORE_SHIFT        = 2.0
```

These are the only values you should need to change between deployments. Everything else in the file is logic, not configuration.

| Variable | What it does |
|---|---|
| `CHROMA_DIR` | Folder where the vector database is stored on disk |
| `BM25_FILE` | Path to the pickled BM25 keyword index |
| `INDEXED_FILE` | Plain text skip-list of already-processed filenames |
| `EMBED_MODEL` | The sentence transformer model used for embeddings |
| `TESSERACT_CMD` | Full path to the Tesseract executable |
| `N_DENSE_FETCH` | How many chunks to pull from ChromaDB before reranking |
| `BM25_REBUILD_EVERY` | How often to rebuild BM25 during ingestion (every N files) |
| `SCORE_SCALE` | Controls how spread-out fit percentages are (higher = more spread) |
| `SCORE_SHIFT` | Controls where 50% lands on the raw score scale (higher = stricter) |

---

### 7.2 The Skip-List (Never Start Over)

```python
def load_indexed_set() -> set[str]:
    p = Path(INDEXED_FILE)
    if not p.exists():
        return set()
    return {ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()}

def mark_indexed(filename: str):
    with open(INDEXED_FILE, "a", encoding="utf-8") as f:
        f.write(filename + "\n")
```

**The problem it solves:** The old approach called `processed_files()` which queried ChromaDB for all stored metadata to get the list of indexed files. With thousands of CVs, this is slow. It also meant any crash or interruption forced re-checking the entire database.

**How it works:** `indexed_files.txt` is a plain text file with one filename per line. When ingestion starts, it reads this file into a Python `set` in milliseconds. After each CV is successfully processed, its filename is appended to the file immediately. If the process crashes on file 500 of 2000, restarting skips the first 499 instantly.

**Rebuilding:** Running with `--rebuild` deletes this file along with ChromaDB and BM25, so everything is re-indexed from scratch.

---

### 7.3 Embedding Model (Singleton)

```python
_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"  Loading embedding model: {EMBED_MODEL} (CPU) …")
        _embed_model = SentenceTransformer(EMBED_MODEL, device="cpu")
    return _embed_model
```

**What an embedding model does:** It converts any piece of text into a list of numbers (a vector) that represents the semantic meaning of that text. Two pieces of text with similar meanings will produce vectors that are close together in this mathematical space, even if they use completely different words. For example, "software engineer with Python experience" and "developer proficient in Python programming" will produce similar vectors.

**The model:** `BAAI/bge-small-en-v1.5` produces 384-dimensional vectors. It is specifically designed for retrieval tasks (matching queries to documents) and is one of the most accurate models available at its size. It runs well on CPU — no GPU required.

**The singleton pattern:** The `global _embed_model` with a `None` check means the model is loaded from disk exactly once, no matter how many times `get_embed_model()` is called. Loading takes 3–5 seconds. Subsequent calls return the already-loaded model instantly.

**The BGE query prefix:**

```python
BGE_PREFIX = "Represent this sentence for searching relevant passages: "

def embed_query(query: str) -> list[float]:
    vec = get_embed_model().encode([BGE_PREFIX + query], ...)
```

This prefix is specific to the BGE model family. It tells the model that this text is a search query that should be matched against documents, as opposed to a document being stored. Without it, query vectors are slightly less accurate. Document chunks are embedded without this prefix.

---

### 7.4 Text Extraction — The OCR Fix

```python
def extract_text(file_path: str) -> tuple[str, str]:
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        native = _native_text(file_path)
        print(f"    native chars: {len(native)}")

        if len(native) >= 30:
            return native, "native"   # ← early return — OCR never runs

        # Only reaches here if PDF has no embedded text (truly scanned)
        print(f"    → OCR fallback")
        ocr = _ocr_pdf(file_path)
        ...
```

**The original bug:** Most CV datasets contain PDFs with text already embedded — the kind you can select and copy in a PDF viewer. PyMuPDF (`fitz`) extracts this text in milliseconds. OCR (Tesseract) renders each page as an image and reads it visually, which takes 5–20 seconds per page. The previous version had the `>= 30` check but still allowed the code to fall through to OCR on some paths.

**The fix:** A hard `return` statement on the happy path makes it structurally impossible for OCR to run when native text was successfully extracted. The `print(f"native chars: {len(native)}")` line is intentional — it lets you verify in the ingestion logs that OCR is only running on the few genuinely scanned PDFs.

**Two extraction functions:**

`_native_text(path)` — opens the PDF with PyMuPDF and reads the embedded text layer from each page. Returns an empty string if the PDF has no text layer.

`_ocr_pdf(path)` — opens the PDF with PyMuPDF, renders each page as a PNG image at 200 DPI, passes each image to Tesseract, and concatenates the results. Only called when `_native_text` returns fewer than 30 characters.

---

### 7.5 Contact Extraction

```python
def extract_contact(text: str) -> dict:
    email_m  = re.search(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", text)
    phone    = re.search(r"\+?[\d][\d\s\-().]{8,14}[\d]", text[:600])
    linkedin = re.search(r"linkedin\.com/in/[\w\-]+", text, re.I)

    # Name: first short capitalised line that isn't metadata
    for line in lines[:10]:
        if re.search(r"[@|http|www|\d{4,}|CV|Resume]", line, re.I):
            continue
        if 2 <= len(words) <= 5 and len(line) < 55 and line[0].isupper():
            name = line; break
```

This uses regular expressions, not AI — it is fast and entirely predictable. Each pattern:

- **Email:** Standard email regex, searched across the entire text.
- **Phone:** Searched only in the first 600 characters (where contact info appears), matching 10–16 digit sequences with common separators.
- **LinkedIn:** Looks for the pattern `linkedin.com/in/username`.
- **Name:** Inspects the first 10 lines for the first line that is 2–5 words, starts with a capital letter, is under 55 characters, and contains no `@`, `http`, `www`, four-digit numbers, or the words "CV" or "Resume". This handles most standard resume formats but will miss names on CVs that open with a job title instead.

---

### 7.6 Section-Aware Chunking

```python
SECTION_KEYWORDS = {
    "skills":     r"\b(skills?|technologies|tools|competenc|tech stack|...)\b",
    "experience": r"\b(experience|work history|employment|career|...)\b",
    "education":  r"\b(education|university|college|degree|...)\b",
    "projects":   r"\b(projects?|portfolio|github|capstone|...)\b",
    ...
}
```

**Why not fixed-size chunks?** Most RAG tutorials split documents into fixed-size windows of, say, 500 characters. This is simple but problematic for CVs — a chunk might cut halfway through a job description, mixing experience text with education text in the same chunk. This confuses the embedding model and degrades search quality.

**How section chunking works:** The function `detect_section(line)` scans each line of the CV text looking for section headers. When it finds one (e.g. "SKILLS", "Work Experience", "Education"), it saves everything accumulated since the last header as one chunk labeled with that section name, then starts a new buffer. The result is chunks like:

```
Chunk(section="skills",     text="Python, Django, PostgreSQL, Docker, AWS...")
Chunk(section="experience", text="Software Engineer at XYZ Corp (2020–2023)...")
Chunk(section="education",  text="BSc Computer Science, Cairo University, 2019...")
```

**False positive filtering:** The regex `_FALSE_POS` prevents common phrases like "professional experience", "technical skills", and "university of" from being misidentified as section headers. These are content words that appear inside sections, not titles.

**Tiny chunk merging:** After chunking, any chunk with fewer than 60 real characters (controlled by `MIN_CHUNK_CHARS`) is merged into the previous chunk. This handles CVs where section headers are followed by only a line or two of content.

---

### 7.7 ChromaDB — The Vector Store

```python
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    _collection = client.get_or_create_collection(
        name="cvs",
        metadata={"hnsw:space": "cosine"},
    )
```

ChromaDB is the vector database that stores the embedded chunks. It persists everything to the `chroma_db/` folder on disk, so the data survives restarts and is only built once.

**What gets stored per chunk:**

| Field | Example |
|---|---|
| `id` | `john_smith.pdf::skills::a3f2b1c0` |
| `document` | `"Python, TensorFlow, Django, PostgreSQL..."` |
| `embedding` | `[0.023, -0.147, 0.891, ...]` (384 numbers) |
| `file` | `"john_smith.pdf"` |
| `section` | `"skills"` |
| `name` | `"John Smith"` |
| `email` | `"john@email.com"` |
| `phone` | `"+20 100 123 4567"` |
| `linkedin` | `"linkedin.com/in/johnsmith"` |

**`hnsw:space: cosine`** tells ChromaDB to use cosine similarity when comparing vectors. This is the correct metric for normalized embeddings — it measures the angle between two vectors, ignoring their magnitude. Two vectors pointing in the same direction have a cosine similarity of 1.0 (identical meaning); two vectors pointing in opposite directions have a similarity of -1.0.

**`upsert` instead of `insert`** means if a chunk with the same ID already exists in the database, it is updated in place rather than creating a duplicate. This makes ingestion safe to re-run on the same files.

---

### 7.8 BM25 — The Keyword Index

```python
def rebuild_bm25():
    corpus  = [document.lower().split() for document in all_chunks]
    bm25    = BM25Okapi(corpus)
    pickle.dump({"bm25": bm25, "ids": doc_ids}, open(BM25_FILE, "wb"))
```

BM25 (Best Match 25) is a classical information retrieval algorithm from the 1990s. It scores documents based on two factors:
- How many times a query word appears in the document (term frequency)
- How rare the query word is across all documents (inverse document frequency)

A word like "Python" that appears in half the CVs gets a lower IDF weight than a rare term like "Fortran". This means BM25 naturally rewards exact matches for rare, specific terms.

**Why use BM25 alongside vector search?** Vector search understands meaning but can miss exact keyword matches. BM25 finds exact matches but has no understanding of meaning. The combination is consistently more accurate than either alone — a well-known result in information retrieval research.

**When BM25 is rebuilt:** During ingestion it is rebuilt every `BM25_REBUILD_EVERY` new files (currently 20) and once at the very end. During a single-file upload through the web app it is rebuilt immediately after indexing. It is saved as a pickle file and loaded into memory at search time — no disk reads during a search.

---

### 7.9 The Search Pipeline

A query goes through five stages. Each stage progressively narrows the candidate pool while improving ranking quality.

#### Stage 1 — Dense retrieval (semantic search)

```python
qvec = embed_query(query)
r = col.query(
    query_embeddings=[qvec],
    n_results=500,
    include=["documents", "metadatas", "distances"],
)
```

The query is embedded into a 384-dimensional vector using the same BGE model used during ingestion. ChromaDB's HNSW index finds the 500 most similar chunk vectors using approximate nearest-neighbour search. This is the semantic understanding step — it finds candidates who match the meaning of the query even if they use different words.

#### Stage 2 — Sparse retrieval (keyword search)

```python
bm25_scores = bm25.get_scores(query.lower().split())
ranked = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
sparse_ids = [bm25_ids[i] for i in ranked if bm25_scores[i] > 0]
```

The query is split into words and scored against every chunk in the BM25 index. This produces a ranked list of chunks ordered by keyword relevance. It runs in milliseconds because BM25 is entirely in memory.

#### Stage 3 — RRF fusion

```python
def _rrf(dense_ids, sparse_ids, k=60):
    score = 1/(k + rank_in_dense) + 1/(k + rank_in_sparse)
```

Reciprocal Rank Fusion merges the two ranked lists into one combined ranking. The formula gives each chunk a score based on its position in each list — not the raw similarity score, just the rank. A chunk ranked #1 in both lists gets the highest possible combined score. A chunk ranked #50 in one list and not appearing at all in the other gets a much lower score.

The constant `k=60` is a smoothing factor that prevents the top-ranked position from having an overwhelming influence on the combined score.

#### Stage 4 — Deduplication and merging

After fusion, results are deduplicated per candidate. All chunks from the same CV file are merged into a single candidate entry, with all their text concatenated. This means the reranker in Stage 5 can see the full picture of each candidate, not just one chunk.

#### Stage 5 — CrossEncoder reranking

```python
reranker = get_reranker()   # singleton — loaded once
pairs    = [(query, candidate_text[:2000]) for c in candidates]
scores   = reranker.predict(pairs)
```

The CrossEncoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) is a fundamentally different type of model from the embedding model. Instead of encoding the query and document separately and comparing their vectors, it reads the query and the candidate text together as a single input and outputs a relevance score.

This is more accurate because the model can see the relationship between specific query words and specific candidate words. The trade-off is speed — the CrossEncoder must process each candidate individually, which is why it only runs on the already-filtered top candidates, not all 500 chunks from Stage 1.

The reranker is a singleton (like the embedding model), loaded once and kept in memory. The first search call loads it in ~5 seconds; every subsequent call uses the already-loaded model.

---

### 7.10 Fit Percentage

```python
SCORE_SCALE = 0.35
SCORE_SHIFT = 2.0

def _sigmoid_pct(raw_score: float) -> int:
    return round(100 / (1 + math.exp(-SCORE_SCALE * (raw_score - SCORE_SHIFT))))
```

The CrossEncoder outputs a raw number roughly in the range -10 to +10. This number is not a percentage and has no intuitive meaning on its own. The sigmoid function maps it smoothly to a 0–100% value:

| Raw score | Fit % (default settings) |
|---|---|
| -4 | ~5% |
| 0  | ~24% |
| 2  | ~50% |
| 4  | ~72% |
| 7  | ~93% |
| 10 | ~98% |

`SCORE_SHIFT` controls where 50% sits on the raw scale. With the default value of 2.0, a raw score of 2.0 maps to exactly 50%. Increasing `SCORE_SHIFT` moves the 50% point higher, making the system stricter. `SCORE_SCALE` controls the steepness of the curve — a larger value spreads scores further apart.

When the reranker is disabled, fit percentage falls back to keyword hit ratio × 100.

**Match quality thresholds:**

| Badge | Fit % | Meaning |
|---|---|---|
| Strong match | ≥ 70% | High semantic and keyword match |
| Good match | 45–69% | Moderate relevance |
| Partial | < 45% | Low match, shown for completeness |

---

## 8. How the Two-Phase Design Works

This design decision is what makes the system fast in production.

**Phase 1 (ingestion)** is slow by nature — it reads PDFs, runs models, and writes to disk. It runs once, unattended, typically overnight for a large dataset. A dataset of 2,000 CVs takes 30–90 minutes on CPU depending on how many are scanned.

**Phase 2 (search)** is fast by design — everything is precomputed and stored. When a search query arrives:

1. Embedding the query: ~50ms
2. ChromaDB vector search: ~100–200ms
3. BM25 scoring: ~20ms
4. RRF fusion + deduplication: ~10ms
5. CrossEncoder reranking (top 5–20 candidates): ~200–800ms
6. Fit percentage + enrichment: ~5ms

**Total: 0.5–1.5 seconds per search**, regardless of how many CVs are in the database.

Adding a new CV through the web app triggers the full ingestion pipeline for just that one file, then rebuilds BM25. This takes a few seconds and does not affect the database state for other users.

---

## 9. Search Quality Explained

The system uses three complementary signals to rank candidates:

**Semantic similarity (vector search)** captures meaning. "Experienced software developer" and "senior programmer" will match a query about "software engineer" even though the words are different. This is the dominant signal for general-purpose queries.

**Keyword matching (BM25)** captures specificity. If a query contains a rare technical term like "COBOL" or "Simulink", BM25 will surface candidates who use that exact term with high weight. Vector search might miss these if the model hasn't seen the term often enough during training.

**CrossEncoder reranking** is the final arbiter. It reads the query and each candidate's full merged text together and produces the most accurate relevance score of the three. It is slower than the other two signals, which is why it only runs on the shortlist produced by Steps 1–3.

The **fit percentage** is derived from the CrossEncoder score and represents the system's overall confidence that this candidate matches the query.

---

## 10. Tuning & Configuration Reference

### Improving recall (finding more relevant candidates)

Increase `N_DENSE_FETCH` from 500 to 1000. This means more chunks are considered before reranking, at the cost of slightly slower searches.

### Making fit percentages stricter

Increase `SCORE_SHIFT` from 2.0 to 3.5. This moves the 50% midpoint higher on the raw score scale, so fewer candidates will show high percentages.

### Spreading out fit percentages more

Increase `SCORE_SCALE` from 0.35 to 0.5. This steepens the sigmoid curve so scores are more spread between 0% and 100%.

### Faster ingestion

Increase the embedding batch size from 64 to 128 if your machine has enough RAM (> 8GB). Reduce `OCR_DPI` from 200 to 150 for faster but slightly less accurate OCR on scanned PDFs.

### GPU acceleration

Change `device="cpu"` to `device="cuda"` in `get_embed_model()`. This requires a CUDA-capable GPU with 4GB+ VRAM and reduces ingestion time by 10–20×.

---

## 11. Next Steps for the Team

### Priority 1 — Evaluate result quality

Before any further development, run 20–30 representative job description queries against the indexed dataset and manually assess whether the top 5 results are genuinely good candidates. This will:
- Reveal whether `SCORE_SHIFT` needs adjustment for your specific dataset
- Identify CV formats that the section chunker misparses
- Give you a baseline accuracy number to measure future improvements against

### Priority 2 — Job description matching

Currently users type a freeform query. A better workflow would accept a full job description — the system would automatically extract required skills, preferred experience, and seniority level, then construct a weighted query. The skills section would carry more weight than the summary section. This requires a small LLM call to parse the job description, then passes the extracted requirements into the existing search pipeline.

### Priority 3 — GPU deployment

For teams ingesting datasets continuously (new CVs arriving daily), GPU acceleration on the embedding model reduces per-CV processing time from ~2 seconds to ~0.1 seconds. The change is a single argument: `device="cuda"` in `get_embed_model()`. The CrossEncoder reranker benefits similarly.

### Priority 4 — Structured output and export

Add a results export button to the UI that produces a CSV or Excel file with all result fields (name, email, phone, LinkedIn, fit score, sections matched). This is useful when the hiring team wants to share a shortlist with someone who doesn't have access to the web app.

---

## 12. Troubleshooting

### OCR is running on every file (slow ingestion)

Check the ingestion log. If you see `native chars: 0` for files that should have extractable text, PyMuPDF may not be installed correctly. Run `pip install pymupdf` and try again.

If you see `native chars: 15` (a small but non-zero number), those PDFs have minimal embedded text (perhaps just page numbers). They are genuinely scanned documents and OCR is appropriate.

### "No candidates found" on every search

The BM25 index may be missing or empty. After ingestion completes, check that `bm25_index.pkl` exists in the project folder. If it doesn't, run:

```python
from pipeline import rebuild_bm25
rebuild_bm25()
```

### ChromaDB error on startup

If ChromaDB fails to open, the database may be corrupt (e.g. from a hard crash during a write). Run with `--rebuild` to start fresh. Your `indexed_files.txt` will also be deleted, so ingestion will re-process all files.

### Reranker not loading

The CrossEncoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) is downloaded automatically on first use from HuggingFace. This requires an internet connection on the first run. If the download fails, you can disable the reranker with the checkbox in the UI and results will still be ranked by RRF fusion alone.

### Fit percentages all clustering near the same value

Adjust `SCORE_SCALE` upward (e.g. from 0.35 to 0.55) to spread scores further apart. If all scores are very low (< 20%), also try reducing `SCORE_SHIFT` from 2.0 to 1.0.
Resume screening automation
Recruitment candidate matching
Semantic CV search systems
HR analytics tools
