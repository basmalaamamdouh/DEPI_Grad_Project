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
Resume screening automation
Recruitment candidate matching
Semantic CV search systems
HR analytics tools
