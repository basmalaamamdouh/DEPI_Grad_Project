"""
HR RAG Pipeline
PDF → smart text extraction → chunking → embeddings → ChromaDB + BM25 → hybrid search

Key improvements:
- OCR only runs when native text extraction actually fails (< 30 chars)
- Fast file skip-list stored in indexed_files.txt (no ChromaDB query needed)
- Singleton CrossEncoder reranker (loaded once, reused across searches)
- Fit percentage score exposed on every result (0-100%)
- BM25 rebuilt incrementally every 20 new files, not every file
"""

import os, re, sys, uuid, pickle, subprocess, tempfile, math
from pathlib import Path
from dataclasses import dataclass, field

# ── CPU-only safety ────────────────────────────────────────────────────────────
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  — edit these to match your machine
# ══════════════════════════════════════════════════════════════════════════════

CHROMA_DIR      = "./chroma_db"
BM25_FILE       = "./bm25_index.pkl"
INDEXED_FILE    = "./indexed_files.txt"   # fast skip-list (one filename per line)
UPLOAD_DIR      = "./uploads"
COLLECTION      = "cvs"
EMBED_MODEL     = "BAAI/bge-small-en-v1.5"
TESSERACT_CMD   = r"D:\tessert\tesseract.exe"   # ← adjust to your install path
OCR_TIMEOUT     = 20       # seconds per page
OCR_DPI         = 200
MIN_CHUNK_CHARS = 60
N_DENSE_FETCH   = 500      # chunks pulled from ChromaDB before RRF
BM25_REBUILD_EVERY = 20    # rebuild BM25 every N new files (not every file)

# ── Fit-score calibration ──────────────────────────────────────────────────────
# Raw rerank scores from ms-marco-MiniLM are in roughly (-10, +10).
# We map them to 0-100% with a sigmoid so the UI always shows a meaningful number.
SCORE_SCALE     = 0.35     # sigmoid steepness — increase to spread scores out more
SCORE_SHIFT     = 0.5      # centre of the sigmoid — lowered from 2.0 so entry-level
                           # CVs (CrossEncoder score ~1-3) reach a fair 50%+ baseline

for d in (CHROMA_DIR, UPLOAD_DIR):
    Path(d).mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# FAST SKIP-LIST  (indexed_files.txt — one filename per line)
# ══════════════════════════════════════════════════════════════════════════════

def load_indexed_set() -> set[str]:
    p = Path(INDEXED_FILE)
    if not p.exists():
        return set()
    return {ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()}

def mark_indexed(filename: str):
    with open(INDEXED_FILE, "a", encoding="utf-8") as f:
        f.write(filename + "\n")

# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDING MODEL  (singleton, lazy-loaded)
# ══════════════════════════════════════════════════════════════════════════════

_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"  Loading embedding model: {EMBED_MODEL} (CPU) …")
        _embed_model = SentenceTransformer(EMBED_MODEL, device="cpu")
    return _embed_model

BGE_PREFIX = "Represent this sentence for searching relevant passages: "

def embed_texts(texts: list[str]) -> list[list[float]]:
    vecs = get_embed_model().encode(
        texts, normalize_embeddings=True,
        batch_size=64, show_progress_bar=len(texts) > 20,
    )
    return [v.tolist() for v in vecs]

def embed_query(query: str) -> list[float]:
    vec = get_embed_model().encode(
        [BGE_PREFIX + query], normalize_embeddings=True
    )[0]
    return vec.tolist()

# ══════════════════════════════════════════════════════════════════════════════
# RERANKER  (singleton — loaded once, reused for every search)
# ══════════════════════════════════════════════════════════════════════════════

_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        print("  Loading reranker model …")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker

def _sigmoid_pct(raw_score: float) -> int:
    """
    Map a raw CrossEncoder score → 0-100 integer percentage.

    ms-marco-MiniLM-L-6-v2 scores fresh-grad / short CVs in roughly the 0-4
    range even when they are clearly relevant. The original SCORE_SHIFT=2.0
    mapped a score of 2 → 50%, which unfairly penalised those candidates.

    New default (SCORE_SHIFT=0.5) centres the sigmoid so a score of ~1.5
    already reaches 50%, giving entry-level candidates a fairer baseline.
    Adjust SCORE_SHIFT in the CONFIG block at the top of this file if needed.
    """
    return round(100 / (1 + math.exp(-SCORE_SCALE * (raw_score - SCORE_SHIFT))))


# ── Skill concept groups for query-aware scoring ──────────────────────────────
# Each entry is (query_triggers, cv_evidence).
#   query_triggers : if ANY of these appear in the query → this skill is "requested"
#   cv_evidence    : if ANY of these appear in the CV text → the skill is "present"
#
# Using concept groups instead of word-splitting avoids double-counting
# ("machine learning deep learning" → 2 concepts, not 4 word hits) and
# correctly handles multi-word phrases and abbreviations.

_SKILL_GROUPS: list[tuple[list[str], list[str]]] = [
    # ── LLMs / Generative AI ──────────────────────────────────────────────────
    (
        ["large language model", "llm", "llms", "generative ai", "genai", "gpt", "chatgpt"],
        ["llm", "llms", "gpt", "chatgpt", "large language model", "generative ai", "genai",
         "foundation model", "gemini", "claude", "mistral", "llama"],
    ),
    # ── Machine Learning ──────────────────────────────────────────────────────
    (
        ["machine learning", " ml ", "sklearn", "scikit"],
        ["machine learning", " ml ", "sklearn", "scikit", "xgboost", "random forest",
         "gradient boosting", "supervised learning", "unsupervised learning",
         "classification", "regression", "clustering"],
    ),
    # ── Deep Learning ─────────────────────────────────────────────────────────
    (
        ["deep learning", " dl ", "neural network", "cnn", "rnn", "lstm", "transformer"],
        ["deep learning", " dl ", "neural network", "cnn", "rnn", "lstm", "transformer",
         "pytorch", "tensorflow", "keras", "backpropagation", "convolutional"],
    ),
    # ── NLP ───────────────────────────────────────────────────────────────────
    (
        ["natural language processing", "nlp", "bert", "text classification", "sentiment"],
        ["nlp", "natural language processing", "bert", "roberta", "transformers", "spacy",
         "nltk", "huggingface", "hugging face", "text classification", "sentiment",
         "named entity", "ner", "token"],
    ),
    # ── Computer Vision ───────────────────────────────────────────────────────
    (
        ["computer vision", "image recognition", "object detection", "opencv", "yolo"],
        ["computer vision", "opencv", "yolo", "resnet", "vgg", "image recognition",
         "object detection", "image classification", "image segmentation", "mediapipe"],
    ),
    # ── RAG / Vector Search ───────────────────────────────────────────────────
    (
        ["rag", "retrieval augmented", "vector search", "semantic search", "chromadb",
         "faiss", "embedding"],
        ["rag", "retrieval augmented", "retrieval-augmented", "vector search", "chromadb",
         "faiss", "pinecone", "weaviate", "qdrant", "semantic search", "embedding search"],
    ),
    # ── Agentic / LangChain ───────────────────────────────────────────────────
    (
        ["agent", "agentic", "langchain", "langgraph", "multi-agent", "tool use"],
        ["agent", "agentic", "langchain", "langgraph", "autogen", "crewai",
         "multi-agent", "react agent", "tool use", "function calling"],
    ),
    # ── Python ────────────────────────────────────────────────────────────────
    (
        ["python"],
        ["python", "pytorch", "tensorflow", "pandas", "numpy", "scipy",
         "fastapi", "flask", "django", ".py"],
    ),
    # ── Cloud / MLOps ─────────────────────────────────────────────────────────
    (
        ["mlops", "docker", "kubernetes", "cloud", "aws", "gcp", "azure", "deployment"],
        ["mlops", "docker", "kubernetes", "aws", "gcp", "azure", "sagemaker",
         "vertex ai", "model deployment", "model serving", "airflow", "mlflow"],
    ),
    # ── Data Science / Analytics ──────────────────────────────────────────────
    (
        ["data science", "data analysis", "data scientist", "analytics"],
        ["data science", "data analysis", "pandas", "numpy", "power bi", "tableau",
         "matplotlib", "seaborn", "sql", "statistics", "eda"],
    ),
    # ── Software Engineering fundamentals ─────────────────────────────────────
    (
        ["api", "fastapi", "rest", "backend", "software engineer"],
        ["api", "fastapi", "rest", "flask", "django", "backend", "microservice",
         "docker", "git", "software engineer"],
    ),
]


def _concept_skill_coverage(query: str, cv_text: str) -> tuple[int, int]:
    """
    Returns (hits, total_requested) where:
    - total_requested = number of _SKILL_GROUPS triggered by the query
    - hits            = how many of those requested skills appear in the CV text

    Both query and cv_text must already be lowercased.
    """
    hits = 0
    total = 0
    for triggers, evidence in _SKILL_GROUPS:
        query_mentions = any(t in query for t in triggers)
        if not query_mentions:
            continue
        total += 1
        cv_has = any(e in cv_text for e in evidence)
        if cv_has:
            hits += 1
    return hits, max(total, 1)

# ══════════════════════════════════════════════════════════════════════════════
# TEXT EXTRACTION  — native first, OCR only as fallback
# ══════════════════════════════════════════════════════════════════════════════

_tesseract_ok: bool | None = None

def _tesseract_available() -> bool:
    global _tesseract_ok
    if _tesseract_ok is not None:
        return _tesseract_ok
    import shutil
    if shutil.which("tesseract"):
        _tesseract_ok = True
        return True
    try:
        r = subprocess.run([TESSERACT_CMD, "--version"], capture_output=True, timeout=5)
        _tesseract_ok = (r.returncode == 0)
    except Exception:
        _tesseract_ok = False
    return _tesseract_ok

def _native_text(path: str) -> str:
    """Extract embedded text from a PDF using PyMuPDF. Returns '' on failure."""
    import fitz
    text = ""
    try:
        doc = fitz.open(path)
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
    except Exception as e:
        print(f"    PyMuPDF error: {e}")
    return text.strip()

def _ocr_page(img_path: str) -> str:
    out_base = img_path + "_tess"
    try:
        subprocess.run(
            [TESSERACT_CMD, img_path, out_base, "-l", "eng", "txt"],
            capture_output=True, timeout=OCR_TIMEOUT,
        )
        txt = Path(out_base + ".txt")
        if txt.exists():
            result = txt.read_text(encoding="utf-8", errors="ignore")
            txt.unlink(missing_ok=True)
            return result
    except subprocess.TimeoutExpired:
        print(f"    ⚠ Tesseract timed out ({OCR_TIMEOUT}s) — skipping page")
    except Exception as e:
        print(f"    ⚠ Tesseract error: {e}")
    return ""

def _ocr_pdf(path: str) -> str:
    if not _tesseract_available():
        return ""
    import fitz
    from PIL import Image
    text = ""
    try:
        doc = fitz.open(path)
        for page in doc:
            pix = page.get_pixmap(dpi=OCR_DPI)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                img.save(tmp.name)
                tmp_path = tmp.name
            try:
                text += _ocr_page(tmp_path) + "\n"
            finally:
                try: os.unlink(tmp_path)
                except: pass
        doc.close()
    except Exception as e:
        print(f"    OCR fallback error: {e}")
    return text.strip()

def extract_text(file_path: str) -> tuple[str, str]:
    """
    Returns (text, method) where method is 'native', 'tesseract', or 'none'.
    OCR is ONLY attempted when native extraction yields fewer than 30 characters.
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        native = _native_text(file_path)
        print(f"    native chars: {len(native)}")

        # ── KEY FIX: early return if native text is good ──────────────────────
        if len(native) >= 30:
            text, method = native, "native"
        else:
            # Scanned PDF — fall back to Tesseract
            print(f"    → OCR fallback (native too short)")
            ocr = _ocr_pdf(file_path)
            if ocr:
                text, method = ocr, "tesseract"
            elif native:
                text, method = native, "native"
            else:
                text, method = "", "none"

    elif ext in (".png", ".jpg", ".jpeg", ".webp"):
        from PIL import Image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            Image.open(file_path).save(tmp.name)
            tmp_path = tmp.name
        text = _ocr_page(tmp_path)
        try: os.unlink(tmp_path)
        except: pass
        method = "tesseract"

    else:
        return "", "none"

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text.replace("\xa0", " "))
    return text.strip(), method

# ══════════════════════════════════════════════════════════════════════════════
# CONTACT EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_contact(text: str) -> dict:
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    email_m  = re.search(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", text)
    email    = email_m.group(0).lower() if email_m else ""

    phone = ""
    m = re.search(r"\+?[\d][\d\s\-().]{8,14}[\d]", text[:600])
    if m:
        phone = m.group(0).strip()

    linkedin_m = re.search(r"linkedin\.com/in/[\w\-]+", text, re.I)
    linkedin   = linkedin_m.group(0) if linkedin_m else ""

    name = ""
    for line in lines[:10]:
        if re.search(r"[@|http|www|\d{4,}|CV|Resume|curriculum]", line, re.I):
            continue
        words = line.split()
        if 2 <= len(words) <= 5 and len(line) < 55 and line[0].isupper():
            name = line
            break

    return {"name": name, "email": email, "phone": phone, "linkedin": linkedin}

# ══════════════════════════════════════════════════════════════════════════════
# SECTION-AWARE CHUNKING
# ══════════════════════════════════════════════════════════════════════════════

SECTION_KEYWORDS = {
    "skills":     r"\b(skills?|technologies|tools|competenc|tech stack|languages|frameworks)\b",
    "experience": r"\b(experience|work history|employment|career|positions? held|professional)\b",
    "education":  r"\b(education|university|college|degree|academic|graduation|schooling|studied)\b",
    "projects":   r"\b(projects?|portfolio|github|capstone|personal work|side project)\b",
    "summary":    r"\b(summary|profile|objective|about me|overview|introduction|bio)\b",
    "contact":    r"\b(contact|email|phone|address|linkedin|reach me|get in touch)\b",
    "certif":     r"\b(certifications?|licenses?|awards?|achievements?|honors?|recognition)\b",
    "training":   r"\b(training|courses?|workshops?|bootcamp|udemy|coursera|mooc)\b",
    "languages":  r"\b(languages?|spoken|fluent|native|bilingual)\b",
    "volunteer":  r"\b(volunteer|community|non-?profit|charity|social work)\b",
}

_FALSE_POS = re.compile(
    r"\b(professional experience|work experience|education background|"
    r"technical skills|soft skills|career objective|bachelor|master|"
    r"computer science|information technology|university of|college of)\b", re.I,
)

@dataclass
class Chunk:
    section: str
    text:    str
    contact: dict = field(default_factory=dict)

def detect_section(line: str) -> str | None:
    s = line.strip()
    if not s or len(s) > 55 or s.endswith((".", ",", ";", ":")):
        return None
    if not re.search(r"[a-zA-Z]", s) or len(s.split()) > 7:
        return None
    for section, pattern in SECTION_KEYWORDS.items():
        if re.search(pattern, s, re.I):
            if _FALSE_POS.search(s):
                return None
            return section
    return None

def chunk_text(text: str, contact: dict) -> list[Chunk]:
    chunks: list[Chunk] = []
    current_section = "summary"
    buffer: list[str] = []

    for line in text.splitlines():
        hit = detect_section(line)
        if hit:
            if buffer:
                joined = "\n".join(buffer).strip()
                if joined:
                    chunks.append(Chunk(current_section, joined, contact))
            current_section = hit
            buffer = []
        else:
            buffer.append(line)

    if buffer:
        joined = "\n".join(buffer).strip()
        if joined:
            chunks.append(Chunk(current_section, joined, contact))

    # Merge tiny chunks into the previous one
    merged: list[Chunk] = []
    for c in chunks:
        real = len(re.sub(r"\s+", "", c.text))
        if real < MIN_CHUNK_CHARS and merged:
            merged[-1] = Chunk(merged[-1].section,
                               merged[-1].text + "\n" + c.text,
                               merged[-1].contact)
        elif real >= MIN_CHUNK_CHARS:
            merged.append(c)

    return merged

# ══════════════════════════════════════════════════════════════════════════════
# CHROMADB STORAGE
# ══════════════════════════════════════════════════════════════════════════════

_collection = None

def get_collection():
    global _collection
    if _collection is None:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection

def upsert_records(chunks: list[Chunk], file_name: str):
    if not chunks:
        return
    texts = [c.text for c in chunks]
    vecs  = embed_texts(texts)
    ids, docs, metas, embeddings = [], [], [], []

    for chunk, vec in zip(chunks, vecs):
        chunk_id = f"{file_name}::{chunk.section}::{uuid.uuid4().hex[:8]}"
        ids.append(chunk_id)
        docs.append(chunk.text)
        embeddings.append(vec)
        metas.append({
            "file":     file_name,
            "section":  chunk.section,
            "name":     chunk.contact.get("name",     ""),
            "email":    chunk.contact.get("email",    ""),
            "phone":    chunk.contact.get("phone",    ""),
            "linkedin": chunk.contact.get("linkedin", ""),
        })

    get_collection().upsert(
        ids=ids, embeddings=embeddings, documents=docs, metadatas=metas
    )

def count_chunks() -> int:
    return get_collection().count()

def processed_files() -> set[str]:
    """
    Primary: read from the fast skip-list file.
    Fallback: query ChromaDB (slower — used only if the file doesn't exist).
    """
    if Path(INDEXED_FILE).exists():
        return load_indexed_set()
    col = get_collection()
    if col.count() == 0:
        return set()
    result = col.get(include=["metadatas"])
    return {m["file"] for m in result["metadatas"]}

def get_all_documents() -> list[dict]:
    col = get_collection()
    if col.count() == 0:
        return []
    result = col.get(include=["documents", "metadatas"])
    return [
        {"id": did, "document": doc, "metadata": meta}
        for did, doc, meta in zip(result["ids"], result["documents"], result["metadatas"])
    ]

# ══════════════════════════════════════════════════════════════════════════════
# BM25 INDEX
# ══════════════════════════════════════════════════════════════════════════════

def rebuild_bm25():
    from rank_bm25 import BM25Okapi
    rows = get_all_documents()
    if not rows:
        return None, []
    corpus  = [r["document"].lower().split() for r in rows]
    doc_ids = [r["id"] for r in rows]
    bm25    = BM25Okapi(corpus)
    with open(BM25_FILE, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": doc_ids}, f)
    print(f"  BM25 saved — {len(doc_ids)} chunks")
    return bm25, doc_ids

def load_bm25():
    if not Path(BM25_FILE).exists():
        return None, []
    with open(BM25_FILE, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["ids"]

# ══════════════════════════════════════════════════════════════════════════════
# INTELLIGENT SEARCH  — hybrid RRF + reranker + fit percentage
# ══════════════════════════════════════════════════════════════════════════════

def _rrf(dense_ids: list, sparse_ids: list, k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for rank, doc_id in enumerate(dense_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, doc_id in enumerate(sparse_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def _extract_years(text: str) -> int:
    matches = re.findall(r"(\d+)\+?\s*year", text, re.I)
    return max((int(m) for m in matches), default=0)

def _score_candidate(candidate: dict, query: str) -> dict:
    """
    Compute a blended fit_pct from three signals:

      60% — CrossEncoder rerank_score (semantic relevance)
      30% — Concept-level skill coverage: how many AI/ML skills mentioned in
             the query are also found in the CV, using _SKILL_GROUPS for
             phrase-aware matching (avoids double-counting 'learning' in
             'machine learning deep learning').
      10% — CV section richness (breadth of CV sections covered)

    When rerank_score is absent, blend shifts to 70% skill coverage + 30%
    section richness.
    """
    query_lower = query.lower()
    all_text    = " ".join(candidate["all_chunks"]).lower()
    years       = _extract_years(all_text)

    # ── Signal 1: concept-level skill coverage ────────────────────────────────
    skill_hits, skill_total = _concept_skill_coverage(query_lower, all_text)
    skill_coverage = skill_hits / skill_total   # 0.0 – 1.0

    # keyword_hits: kept for UI display (the green chip showing "N kw hits")
    # Still uses simple word split — only used for the chip, not for fit_pct
    query_words  = [w for w in query_lower.split() if len(w) > 2]
    keyword_hits = sum(1 for w in query_words if w in all_text)

    # ── Signal 2: CV section richness (0-1) ──────────────────────────────────
    unique_sections  = set(s for s in candidate.get("sections_found", []) if s)
    section_richness = len(unique_sections) / max(len(SECTION_KEYWORDS), 1)

    sections_matched = list(dict.fromkeys(candidate.get("sections_found", [])))

    # ── Blended fit_pct ───────────────────────────────────────────────────────
    if "rerank_score" in candidate:
        rerank_pct = _sigmoid_pct(candidate["rerank_score"])
        fit_pct = round(
            0.60 * rerank_pct
            + 0.30 * skill_coverage   * 100
            + 0.10 * section_richness * 100
        )
    else:
        fit_pct = round(
            0.70 * skill_coverage   * 100
            + 0.30 * section_richness * 100
        )

    fit_pct = max(1, min(99, fit_pct))

    # ── Match quality badge ───────────────────────────────────────────────────
    if fit_pct >= 65 or skill_coverage >= 0.7:
        quality = "strong"
    elif fit_pct >= 40 or skill_coverage >= 0.4:
        quality = "good"
    else:
        quality = "partial"

    candidate.update({
        "keyword_hits":     keyword_hits,
        "years_exp":        years,
        "match_quality":    quality,
        "hit_ratio":        round(skill_coverage, 2),
        "fit_pct":          fit_pct,
        "sections_matched": sections_matched,
    })
    return candidate

def search(
    query:          str,
    top_k:          int  = 5,
    section_filter: str | None = None,
    use_reranker:   bool = True,
) -> list[dict]:
    top_k = int(top_k)   # defensive cast — callers may pass strings
    col = get_collection()
    if col.count() == 0:
        return []

    qvec = embed_query(query)

    # ── Dense retrieval (ChromaDB cosine similarity) ───────────────────────────
    n     = min(N_DENSE_FETCH, col.count())
    where = {"section": section_filter} if section_filter else None
    r     = col.query(
        query_embeddings=[qvec], n_results=n, where=where,
        include=["documents", "metadatas", "distances"],
    )
    dense_ids = r["ids"][0] if r["ids"] else []

    # ── Sparse retrieval (BM25 keyword) ───────────────────────────────────────
    bm25, bm25_ids = load_bm25()
    sparse_ids: list[str] = []
    if bm25 and bm25_ids:
        bm25_scores = bm25.get_scores(query.lower().split())
        ranked      = sorted(range(len(bm25_scores)),
                             key=lambda i: bm25_scores[i], reverse=True)
        sparse_ids  = [bm25_ids[i] for i in ranked if float(bm25_scores[i]) > 0]

    # ── RRF fusion ────────────────────────────────────────────────────────────
    fused = _rrf(dense_ids, sparse_ids)
    if not fused:
        return []

    # Fetch fused chunks
    all_ids  = [fid for fid, _ in fused]
    fetched  = col.get(ids=all_ids, include=["documents", "metadatas"])
    id_to_doc = {
        did: {"document": doc, "metadata": meta}
        for did, doc, meta in zip(
            fetched["ids"], fetched["documents"], fetched["metadatas"]
        )
    }

    # ── Deduplicate: one entry per candidate, merge all their chunks ───────────
    best: dict[str, dict] = {}
    for doc_id, rrf_score in fused:
        if doc_id not in id_to_doc:
            continue
        row   = id_to_doc[doc_id]
        meta  = row["metadata"]
        fname = meta.get("file", doc_id)

        if fname not in best:
            best[fname] = {
                "id":             doc_id,
                "text":           row["document"],
                "metadata":       meta,
                "rrf_score":      rrf_score,
                "all_chunks":     [row["document"]],
                "sections_found": [meta.get("section", "")],
            }
        else:
            best[fname]["all_chunks"].append(row["document"])
            best[fname]["sections_found"].append(meta.get("section", ""))
            best[fname]["text"] = "\n\n".join(best[fname]["all_chunks"])

    candidates = sorted(best.values(), key=lambda x: x["rrf_score"], reverse=True)

    # ── CrossEncoder reranker (singleton — loaded once) ───────────────────────
    if use_reranker and candidates:
        try:
            reranker = get_reranker()
            pairs    = [(query, c["text"][:2000]) for c in candidates]
            rscores  = reranker.predict(pairs)
            for c, s in zip(candidates, rscores):
                c["rerank_score"] = float(s)
            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        except Exception as e:
            print(f"  Reranker skipped: {e}")

    return [_score_candidate(c, query) for c in candidates[:top_k]]

# ══════════════════════════════════════════════════════════════════════════════
# INGESTION
# ══════════════════════════════════════════════════════════════════════════════

def process_file(file_path: str) -> tuple[bool, str]:
    name = Path(file_path).name
    print(f"\n  Processing: {name}")
    try:
        text, method = extract_text(file_path)
        print(f"  Extraction: {method} | {len(text)} chars")
        if not text.strip():
            return False, f"⚠ {name} — no text extracted"
        contact = extract_contact(text)
        print(f"  Contact: {contact['name']} <{contact['email']}>")
        chunks = chunk_text(text, contact)
        print(f"  Chunks: {len(chunks)}")
        if not chunks:
            return False, f"⚠ {name} — no usable chunks"
        upsert_records(chunks, name)
        mark_indexed(name)   # write to skip-list immediately
        return True, f"✅ {name} — {len(chunks)} chunks [{method}]"
    except Exception as e:
        return False, f"❌ {name} — {e}"

def process_dataset(folder_path: str, limit: int = 0, progress_cb=None):
    def log(msg):
        print(msg)
        if progress_cb:
            progress_cb(msg)

    done  = processed_files()
    files = sorted(Path(folder_path).rglob("*.pdf"))
    if limit > 0:
        files = files[:limit]

    log(f"\n  Found {len(files)} PDFs | Already indexed: {len(done)}\n")
    ok = skipped = failed = 0

    for i, fpath in enumerate(files, 1):
        if fpath.name in done:
            log(f"  [{i}/{len(files)}] SKIP: {fpath.name}")
            skipped += 1
            continue

        success, msg = process_file(str(fpath))
        log(f"  [{i}/{len(files)}] {msg}")

        if success:
            ok += 1
            # Rebuild BM25 periodically, not every single file
            if ok % BM25_REBUILD_EVERY == 0:
                log(f"  >> Rebuilding BM25 at {ok} new files…")
                rebuild_bm25()
        else:
            failed += 1

    log("\n  Final BM25 rebuild…")
    rebuild_bm25()
    log(f"\n  Done — {ok} indexed | {skipped} skipped | {failed} failed")
    log(f"  Total chunks in DB: {count_chunks()}")
    return {"ok": ok, "skipped": skipped, "failed": failed, "total": count_chunks()}

# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not _tesseract_available():
        print("⚠ Tesseract not found — scanned PDFs will be skipped.")
        print("  Install: https://github.com/UB-Mannheim/tesseract/wiki\n")

    if "--rebuild" in sys.argv:
        import shutil
        if Path(CHROMA_DIR).exists():   shutil.rmtree(CHROMA_DIR)
        if Path(BM25_FILE).exists():    Path(BM25_FILE).unlink()
        if Path(INDEXED_FILE).exists(): Path(INDEXED_FILE).unlink()
        print("Cleared existing indexes.\n")

    n = count_chunks()
    if n > 0 and "--rebuild" not in sys.argv:
        print(f"✅ {n} chunks already in ChromaDB. Run app.py for the UI.")
        print("   Use --rebuild to re-index from scratch.")
        sys.exit(0)

    if "--kaggle" in sys.argv:
        try:
            import kagglehub
            folder = kagglehub.dataset_download("hadikp/resume-data-pdf")
        except ImportError:
            print("❌ pip install kagglehub"); sys.exit(1)
    elif "--dataset" in sys.argv:
        idx    = sys.argv.index("--dataset")
        folder = sys.argv[idx + 1]
    else:
        print("Usage:")
        print("  python pipeline.py --dataset <folder>")
        print("  python pipeline.py --dataset <folder> --limit 100")
        print("  python pipeline.py --dataset <folder> --rebuild")
        print("  python pipeline.py --kaggle")
        sys.exit(1)

    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
    process_dataset(folder, limit=limit)
    print("\nDone. Run:  python app.py")