"""
DEPI GenAI CV Processor
Pipeline: Kaggle Dataset → OCR → Chunking → Embeddings → Search
"""

import re
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import sys

# ⚠️ SET THIS PATH (VERY IMPORTANT)
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ══════════════════════════════════════════════════════════════
# STEP 0 — KAGGLE DATASET
# ══════════════════════════════════════════════════════════════

def load_kaggle_dataset():
    import kagglehub
    dataset_path = kagglehub.dataset_download("hadikp/resume-data-pdf")
    print("Dataset downloaded at:", dataset_path)

    all_files = list(Path(dataset_path).rglob("*"))
    print(f"Total files found: {len(all_files)}")

    return dataset_path


# ══════════════════════════════════════════════════════════════
# STEP 1 — OCR (FIXED)
# ══════════════════════════════════════════════════════════════

def ocr_pdf(path: str) -> str:
    import fitz  # PyMuPDF
    from PIL import Image

    text = ""

    try:
        doc = fitz.open(path)

        for page in doc:
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            page_text = pytesseract.image_to_string(img, lang="eng")
            text += page_text + "\n"

    except Exception as e:
        print(f"OCR PDF failed on {path}: {e}")

    return text.strip()


def ocr_image(path: str) -> str:
    from PIL import Image
    try:
        return pytesseract.image_to_string(Image.open(path), lang="eng")
    except Exception as e:
        print(f"OCR image failed: {e}")
        return ""


def extract_text(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        text = ocr_pdf(file_path)
    elif ext in [".png", ".jpg", ".jpeg", ".webp"]:
        text = ocr_image(file_path)
    else:
        return ""

    text = text.replace("\xa0", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ══════════════════════════════════════════════════════════════
# STEP 2 — CHUNKING
# ══════════════════════════════════════════════════════════════

SECTION_KEYWORDS = {
    "skills":     r"skill|technolog|tools",
    "experience": r"experience|work|employment",
    "education":  r"education|university|college",
    "projects":   r"project|portfolio",
    "summary":    r"summary|profile|objective",
}


def detect_section(line: str):
    if len(line.strip()) > 50 or line.strip().endswith("."):
        return None

    for section, pattern in SECTION_KEYWORDS.items():
        if re.search(pattern, line, re.I):
            return section
    return None


@dataclass
class Chunk:
    section: str
    text: str


def chunk_text(text: str):
    chunks, current_section, buffer = [], "header", []

    for line in text.splitlines():
        hit = detect_section(line)

        if hit:
            if buffer:
                joined = "\n".join(buffer).strip()
                if joined:
                    chunks.append(Chunk(current_section, joined))
            current_section, buffer = hit, []
        else:
            buffer.append(line)

    if buffer:
        joined = "\n".join(buffer).strip()
        if joined:
            chunks.append(Chunk(current_section, joined))

    return chunks


# ══════════════════════════════════════════════════════════════
# STEP 3 — EMBEDDINGS
# ══════════════════════════════════════════════════════════════

_model = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print("Loading embedding model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


@dataclass
class Embedded:
    file: str
    section: str
    text: str
    embedding: list


def embed_chunks(chunks, file_name):
    model = get_model()

    texts = [c.text for c in chunks if c.text.strip()]
    if not texts:
        return []

    vectors = model.encode(texts, normalize_embeddings=True)

    return [
        Embedded(file_name, c.section, c.text, v.tolist())
        for c, v in zip(chunks, vectors)
    ]


# ══════════════════════════════════════════════════════════════
# STEP 4 — SAVE & LOAD
# ══════════════════════════════════════════════════════════════

EMBEDDINGS_FILE = "cv_embeddings.json"

def save(data):
    records = [
        {"file": d.file, "section": d.section,
         "text": d.text, "embedding": d.embedding}
        for d in data
    ]
    Path(EMBEDDINGS_FILE).write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(records)} chunks → {EMBEDDINGS_FILE}")


def load():
    records = json.loads(Path(EMBEDDINGS_FILE).read_text())
    return [Embedded(**r) for r in records]


# ══════════════════════════════════════════════════════════════
# STEP 5 — SEARCH
# ══════════════════════════════════════════════════════════════

def cosine(a, b):
    return float(np.dot(a, b))  # normalized


def search(query: str, data, top_k=5):
    model = get_model()
    qv = model.encode([query], normalize_embeddings=True)[0]

    scored = sorted(data, key=lambda x: cosine(qv, x.embedding), reverse=True)

    return [
        {"file": r.file, "section": r.section,
         "score": round(cosine(qv, r.embedding), 4),
         "text": r.text}
        for r in scored[:top_k]
    ]


def print_results(results):
    if not results:
        print("⚠️ No results found\n")
        return

    for i, r in enumerate(results, 1):
        print(f"  #{i}  {r['file']}")
        print(f"       Score   : {r['score']}")
        print(f"       Section : {r['section']}")
        print(f"       Preview : {r['text'][:120]}")
        print()


# ══════════════════════════════════════════════════════════════
# PROCESSING
# ══════════════════════════════════════════════════════════════
def process_file(file_path):
    print(f"\nProcessing: {file_path}")

    text = extract_text(file_path)

    if not text:
        print("⚠️ Empty OCR result — skipping")
        return []

    chunks = chunk_text(text)
    print(f"Chunks created: {len(chunks)}")

    return embed_chunks(chunks, Path(file_path).name)



def process_dataset(folder_path):
   
    all_data = load() if Path(EMBEDDINGS_FILE).exists() else []
    
    
    processed_files = {d['file'] for d in all_data}
    
    files = list(Path(folder_path).rglob("*.pdf"))
    print(f"\nFound {len(files)} PDF files. Already processed: {len(processed_files)}")

    for i, file in enumerate(files):
        
        if file.name in processed_files:
            continue

        try:
            data = process_file(str(file))
            all_data.extend(data)
            print(f"  [{i+1}/{len(files)}] DONE")
            
            if (i + 1) % 10 == 0:
                save(all_data)
                print(f"  >> Auto-saved at {i+1} files.")
                
        except Exception as e:
            print(f"  SKIP {file.name}: {e}")

    return all_data
"""def process_dataset(folder_path):
    all_data = []

    files = list(Path(folder_path).rglob("*.pdf"))
    print(f"\nFound {len(files)} PDF files")

    for i, file in enumerate(files):
        try:
            data = process_file(str(file))
            all_data.extend(data)
            print(f"  [{i+1}/{len(files)}] DONE")
            
            
            if (i + 1) % 10 == 0:
                save(all_data)
                print(f"  >> Auto-saved at {i+1} files.")

        except Exception as e:
            print(f"  SKIP {file.name}: {e}")

    return all_data"""


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    force = "--rebuild" in sys.argv

    if Path(EMBEDDINGS_FILE).exists() and not force:
        data = load()

        if len(data) == 0:
            print("⚠️ Empty embeddings — rebuilding...")
            Path(EMBEDDINGS_FILE).unlink()
            force = True
        else:
            print(f"Loaded {len(data)} chunks\n")

    if not Path(EMBEDDINGS_FILE).exists() or force:

        if "--kaggle" in sys.argv:
            folder = load_kaggle_dataset()
        elif "--dataset" in sys.argv:
            idx = sys.argv.index("--dataset")
            folder = sys.argv[idx + 1]
        else:
            print("Usage:")
            print("  python pipeline.py --kaggle --rebuild")
            print("  python pipeline.py --dataset <folder>")
            sys.exit(1)

        data = process_dataset(folder)
        save(data)

    # TEST SEARCH
    print("\n=== SEARCH TEST ===\n")
    print_results(search("python machine learning", data, 3))

    # INTERACTIVE
    while True:
        try:
            q = input("\nSearch > ").strip()
            if not q:
                continue
            print_results(search(q, data))
        except KeyboardInterrupt:
            print("\nDone.")
            break