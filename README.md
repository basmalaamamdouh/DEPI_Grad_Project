# HR Assistant — AI-Powered CV Search & Recruitment System

A multi-agent recruitment assistant built on a hybrid RAG pipeline. HR teams can index CV datasets, search with natural language, get AI-reasoned candidate recommendations, and send personalized emails — all from a single web app.

---

## Table of Contents

1. [What This System Does](#1-what-this-system-does)
2. [Architecture Overview](#2-architecture-overview)
3. [Project Structure](#3-project-structure)
4. [Agent Breakdown](#4-agent-breakdown)
5. [Installation](#5-installation)
6. [Running the Pipeline (Ingestion)](#6-running-the-pipeline-ingestion)
7. [Running the Web App](#7-running-the-web-app)
8. [Pipeline Deep Dive](#8-pipeline-deep-dive)
9. [Search Quality Explained](#9-search-quality-explained)
10. [Configuration Reference](#10-configuration-reference)
11. [Team — Who Owns What](#11-team--who-owns-what)
12. [Next Steps for the Team](#12-next-steps-for-the-team)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. What This System Does

This system lets an HR team:

1. **Index** a folder of CV/resume PDFs once (runs offline, takes 30–90 min for 2,000 CVs)
2. **Search** the full dataset instantly using plain English
3. **Get AI-reasoned results** — the agent explains *why* each candidate was selected, citing specific evidence from their CV
4. **Chat conversationally** — refine searches through follow-up messages ("make it more senior", "add React too", "only Cairo-based")
5. **Email candidates** directly from the results

Example queries that work:

> *"Senior Python developer with Django and PostgreSQL, 5 years experience"*
> *"Junior data analyst proficient in Excel and Power BI, Cairo-based"*
> *"Machine learning engineer with NLP and TensorFlow background"*

Each result shows a **fit percentage** (0–100%), a **match quality badge** (Strong / Good / Partial), contact info, matched CV sections, and a text preview.

---

## 2. Architecture Overview

The system has two independent phases and one conversational agent layer on top.

```
┌──────────────────────────────────────────────────────────────────┐
│  PHASE 1 — OFFLINE INGESTION  (run once)                         │
│                                                                  │
│  PDF files → Text extraction → Chunking → Embedding             │
│               (PyMuPDF +        (section-   (BGE-small)         │
│                Tesseract OCR)    aware)                          │
│                                              ↓                  │
│                                   ┌──────────┴──────────┐       │
│                                   │  ChromaDB (vector)  │       │
│                                   │  BM25 index (.pkl)  │       │
│                                   │  indexed_files.txt  │       │
│                                   └─────────────────────┘       │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  PHASE 2 — ONLINE SEARCH  (always fast, < 1.5s)                 │
│                                                                  │
│  Query → Embed → ChromaDB dense retrieval                        │
│               → BM25 sparse retrieval                           │
│               → RRF fusion                                       │
│               → Deduplicate per candidate                        │
│               → CrossEncoder reranker                            │
│               → Fit percentage → Result cards                    │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  AGENT LAYER — on top of Phase 2                                 │
│                                                                  │
│  HR message                                                      │
│     → Groq LLaMA 3.3-70b (turn 1 — decides to search or ask)   │
│     → search_candidates tool → smart_search() → Phase 2         │
│     → Groq LLaMA 3.3-70b (turn 2 — writes summary + reasoning) │
│     → HTML reply with candidate cards                            │
│     → Updated conversation history (for follow-up turns)        │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Project Structure

```
project/
│
├── pipeline.py           # Ingestion + hybrid search core
├── query_rewriter.py     # Groq-powered query expansion (smart_search)
├── RetriveCVAgent.py     # Agent 1 — conversational search + card rendering
├── app.py                # Gradio web UI (4 tabs)
│
├── chroma_db/            # Vector database (auto-created)
├── bm25_index.pkl        # BM25 keyword index (auto-created)
├── indexed_files.txt     # Fast skip-list of processed CVs
├── uploads/              # Temp folder for individually uploaded PDFs
│
└── README.md
```

---

## 4. Agent Breakdown

### ✅ Agent 1 — Conversational Search (done: `RetriveCVAgent.py`)

Converts natural HR dialogue into structured candidate searches. Handles multi-turn conversation so HR can refine results without repeating context.

**Flow (one turn):**

```
HR message
  → Groq LLaMA 3.3-70b with search_candidates tool
  → If tool called  → execute smart_search() → render CV cards
  → If no tool call → ask one clarifying question
  → Returns (reply_html, updated_llm_history)
```

**Key behaviors:**
- Calls `search_candidates` immediately when there's enough to go on — never over-asks
- Handles follow-up refinements: "more senior", "add React", "only Cairo-based"
- Builds a rich query string: role + skills + seniority + location
- Returns an HTML summary + styled candidate cards with SVG fit gauge

**Renders each candidate with:**
- Semi-circular fit score dial (green = strong, amber = good, gray = partial)
- Name, email, phone, LinkedIn
- Years of experience + keyword hits badges
- Section chips (skills, experience, education…)
- Text preview from the most relevant CV chunks

**Used by `app.py`:**
```python
from RetriveCVAgent import run_agent_turn

reply_html, new_history = run_agent_turn(user_msg, llm_history, search_fn)
```

---

### 🔲 Agent 2 — Reasoning Agent (to be built)

Explains *why* each candidate was selected, citing specific evidence from their CV sections.

**Planned behavior:** After search results come back, this agent reads the candidate's actual CV text and the HR's query, then generates a structured justification like:

> *"Selected because her experience section explicitly mentions 3 years of FastAPI in a production environment — matching your seniority requirement — and her projects section shows two deployed ML APIs. Skills section lists PostgreSQL and Docker, which you specified. Gap: no mention of React, which you listed as preferred."*

This is explainable AI for recruitment. The reasoning is cited from the CV, not hallucinated.

---

### 🔲 Agent 3 — Email Agent (to be built)

Drafts a personalized outreach email for a selected candidate, informed by the reasoning from Agent 2.

**Planned behavior:** Reads the HR's requirements + the reasoning for why this specific candidate was chosen, then writes an email that references their actual background. Not a generic template — the email mentions specific skills or projects from their CV to make it feel personal.

---

### 🔲 Agent 4 — Candidate Gap Agent (to be built)

Analyzes a candidate's CV against a job description and produces a specific, actionable gap report.

**Planned behavior:** Compares the candidate's CV sections against the job requirements and outputs something like:

> *"Your skills section lists Python but no frameworks. Roles like this require FastAPI or Django — a small GitHub project using either would close this gap. Your experience section doesn't mention system design or architecture; consider documenting a project where you made architectural decisions. Estimated time to address: 4–6 weeks."*

Includes learning resources and a rough timeline per gap.

---

## 5. Installation

### Requirements

- Python 3.10+
- ~2 GB disk space for models (downloaded automatically on first run)
- Tesseract OCR (only needed for scanned/image-based PDFs — most CV datasets don't need it)

### Install Python dependencies

```bash
pip install gradio sentence-transformers chromadb rank-bm25 pymupdf pillow groq
```

### Set your Groq API key (free at https://console.groq.com)

```bash
# Windows
set GROQ_API_KEY=gsk_...
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  #this is the API i used 
# Mac / Linux
export GROQ_API_KEY=gsk_...
```

The agent and query rewriter both use Groq (free tier). Search and ingestion work without it.

### Install Tesseract (only for scanned PDFs)

- **Windows:** https://github.com/UB-Mannheim/tesseract/wiki
- **Linux:** `sudo apt install tesseract-ocr`
- **macOS:** `brew install tesseract`

Then update `TESSERACT_CMD` in `pipeline.py` to your install path.

---

## 6. Running the Pipeline (Ingestion)

Run once on your CV dataset. Already-indexed files are skipped automatically on every subsequent run.

```bash
# Index a full folder
python pipeline.py --dataset "D:/path/to/CVs"

# Test with a small batch first
python pipeline.py --dataset "D:/path/to/CVs" --limit 100

# Download and index the Kaggle dataset automatically
python pipeline.py --kaggle

# Wipe everything and re-index from scratch
python pipeline.py --dataset "D:/path/to/CVs" --rebuild
```

**What you'll see per file:**

```
Processing: john_smith_cv.pdf
  native chars: 3847
  Extraction: native | 3847 chars
  Contact: John Smith <john@email.com>
  Chunks: 6
✅ john_smith_cv.pdf — 6 chunks [native]
```

If `native chars` is large (> 30), OCR was skipped — this is the fast path. Small numbers mean the PDF is scanned and Tesseract will run.

**If ingestion is interrupted:** just re-run the same command. `indexed_files.txt` tracks what's done and skips it.

---

## 7. Running the Web App

```bash
python app.py
# Open: http://localhost:7860
```

**Tabs:**

| Tab | What it does |
|---|---|
| 🔍 Search Candidates | Manual search with section filter, top-k, min fit % sliders |
| 🤖 AI Assistant | Conversational chat with Agent 1 |
| 📁 Upload & Ingest | Upload individual PDFs or point to a folder |
| ✉️ Send Emails | Send interview invitations to candidates from your last search |
| ⚙️ Setup | Quick-start guide |

**Email setup (Gmail):**

```bash
set SMTP_USER=you@gmail.com
set SMTP_PASS=your_app_password   # Gmail App Password, not your login password
python app.py
```

---

## 8. Pipeline Deep Dive

### Text Extraction

Uses PyMuPDF (`fitz`) to extract embedded text first. OCR (Tesseract) only runs if native extraction yields fewer than 30 characters — meaning the PDF is genuinely scanned. This keeps ingestion fast; most CVs skip OCR entirely.

### Section-Aware Chunking

Instead of fixed-size windows, the chunker detects CV section headers (Skills, Experience, Education, Projects, etc.) using regex patterns and splits on those boundaries. Result: each chunk belongs to a named section and contains coherent content. This improves search quality significantly vs. character-based chunking.

### Hybrid Search: BM25 + ChromaDB

Two retrieval signals run in parallel on every query:

- **ChromaDB (dense):** Embeds the query with `BAAI/bge-small-en-v1.5` and finds the 500 most semantically similar chunks. Understands meaning — "senior developer" matches "experienced engineer".
- **BM25 (sparse):** Scores every chunk by keyword overlap. Excellent for rare/specific terms like "COBOL" or "Simulink" that the embedding model may underweight.

Results are merged with **Reciprocal Rank Fusion (RRF)** — a formula that combines rankings (not raw scores) from both lists.

### CrossEncoder Reranking

The top merged candidates are re-scored by `cross-encoder/ms-marco-MiniLM-L-6-v2`, a model that reads the query and candidate text *together* and produces a precise relevance score. More accurate than vector similarity alone. Runs as a singleton (loaded once, reused).

### Fit Percentage

Raw CrossEncoder scores are mapped to 0–100% using a sigmoid function. Thresholds: **Strong match ≥ 70%**, **Good match 45–69%**, **Partial < 45%**.

### Query Expansion (`query_rewriter.py`)

Before search, the HR's query is sent to Groq which expands it: abbreviations (`ML` → `machine learning`), synonyms (`React` → `React.js / ReactJS`), must-have skills, and minimum years of experience. Multiple query variants run through the full pipeline and results are merged.

### Search Performance

| Step | Time |
|---|---|
| Query embedding | ~50ms |
| ChromaDB vector search | ~100–200ms |
| BM25 scoring | ~20ms |
| RRF + dedup | ~10ms |
| CrossEncoder reranking | ~200–800ms |
| **Total** | **~0.5–1.5s** |

---

## 9. Search Quality Explained

Three signals work together:

- **Semantic similarity** — captures meaning. Finds candidates who match the *intent* of the query even with different wording.
- **Keyword matching (BM25)** — captures specificity. Surfaces candidates who use the exact rare terms in the query.
- **CrossEncoder reranking** — final arbiter. Reads query + candidate text together for the most accurate relevance score.

---

## 10. Configuration Reference

All tuneable values are at the top of `pipeline.py`:

| Variable | Default | What it does |
|---|---|---|
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model used for indexing and queries |
| `N_DENSE_FETCH` | `500` | Chunks pulled from ChromaDB before RRF (increase for better recall) |
| `BM25_REBUILD_EVERY` | `20` | Rebuild BM25 every N new files during ingestion |
| `SCORE_SCALE` | `0.35` | Sigmoid steepness — higher = scores more spread apart |
| `SCORE_SHIFT` | `2.0` | Where 50% sits on the raw score scale — higher = stricter |
| `TESSERACT_CMD` | *(your path)* | Full path to Tesseract executable |
| `OCR_DPI` | `200` | Render DPI for scanned pages (lower = faster, less accurate) |

**Common adjustments:**
- Scores clustering together → increase `SCORE_SCALE` to `0.5`
- Too many low-quality results → increase `SCORE_SHIFT` to `3.0`
- Missing relevant candidates → increase `N_DENSE_FETCH` to `1000`

---

## 11. Team — Who Owns What

| Person | Component | Status |
|---|---|---|
| Afaf | Agent 1 — Conversational Search (`RetriveCVAgent.py`) + base pipeline | ✅ Done |
| Teammate 2 | Agent 2 — Reasoning Agent | 🔲 To build |
| Teammate 3 | Agent 3 — Email Agent | 🔲 To build |
| Teammate 4 | Agent 4 — Candidate Gap Agent | 🔲 To build |

**Shared infrastructure** (everyone depends on this, don't modify without coordinating):
- `pipeline.py` — ingestion + `search()` function
- `query_rewriter.py` — `smart_search()` function
- ChromaDB schema (metadata fields: `file`, `section`, `name`, `email`, `phone`, `linkedin`)

**How new agents plug in:** Each agent imports `smart_search` from `query_rewriter.py` and calls it the same way Agent 1 does:

```python
from query_rewriter import smart_search

results, rq = smart_search(query, top_k=5, section_filter=None, use_reranker=True, min_fit_pct=25)
```

`results` is a list of dicts. Each dict contains:

```python
{
    "metadata": {"name": "...", "email": "...", "phone": "...", "linkedin": "...", "file": "..."},
    "fit_pct": 72,
    "match_quality": "strong",   # "strong" | "good" | "partial"
    "years_exp": 4,
    "keyword_hits": 7,
    "sections_found": ["skills", "experience", "projects"],
    "all_chunks": ["chunk text 1", "chunk text 2", ...],
    "text": "full merged candidate text",
}
```

---

## 12. Next Steps for the Team

### Agent 2 — Reasoning Agent
- Input: `results` list from `smart_search()` + the original HR query
- For each candidate: send `(query, candidate["text"])` to Groq with a prompt that asks for cited, evidence-based reasoning
- Output: a short structured justification per candidate, referencing specific sections
- Plug in: called after Agent 1 returns results, adds a `"reasoning"` key to each result dict

### Agent 3 — Email Agent
- Input: selected candidate dict (with reasoning from Agent 2) + HR's job requirements
- Prompt Groq to write a personalized email that references the candidate's specific background
- Output: draft email string ready for the Send Emails tab
- Should use the existing `send_email()` function in `app.py` to actually send

### Agent 4 — Candidate Gap Agent
- Input: candidate CV text (`candidate["text"]`) + job description or HR query
- Prompt Groq to compare the two and identify specific, named gaps with actionable suggestions
- Output: structured gap report (missing skills, experience gaps, recommended resources, rough timeline)
- Can be triggered from the Search tab with a "Gap Report" button per candidate card

### LangGraph Orchestration (future)
Once all four agents exist as standalone functions, wrap them in a LangGraph graph with a supervisor node for conditional routing. This is what lets you say on your CV: *"multi-agent system with graph-based orchestration"*.

---

## 13. Troubleshooting

**OCR running on every file (slow ingestion)**
Check `native chars:` in the log. If it shows `0` for normal PDFs, PyMuPDF may not be installed: `pip install pymupdf`.

**"No candidates found" on every search**
`bm25_index.pkl` may be missing. Rebuild it:
```python
from pipeline import rebuild_bm25
rebuild_bm25()
```

**ChromaDB error on startup**
Database may be corrupt from a hard crash. Run `--rebuild` to start fresh (re-ingestion required).

**Reranker not loading**
The CrossEncoder downloads from HuggingFace on first use — needs internet on the first run. If it fails, use the Reranker checkbox in the UI to disable it temporarily.

**Fit percentages all clustering near the same value**
Increase `SCORE_SCALE` in `pipeline.py` (e.g. `0.35` → `0.55`).

**Agent returns "GROQ_API_KEY not set"**
Set the environment variable before running `app.py`:
```bash
set GROQ_API_KEY=gsk_...   # Windows
export GROQ_API_KEY=gsk_... # Mac/Linux
```
