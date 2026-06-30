"""
query_rewriter.py  —  Engineer 1 task
LLM-based query intelligence layer for the HR RAG pipeline.

Drop this file next to pipeline.py and app.py.
Then follow the three integration patches at the bottom of this file.

Requires:
    pip install groq
    set GROQ_API_KEY=gsk_...        (get free key at https://console.groq.com)

Free Groq tier limits (as of 2025):
    llama-3.1-8b-instant  — 6,000 req/day, 500 req/min  ← default (fastest)
    llama-3.3-70b-versatile — 1,000 req/day, 100 req/min  ← swap for better quality
    gemma2-9b-it          — 14,400 req/day, 30 req/min
"""

import os
import json
import re
from dataclasses import dataclass, field
from typing import Optional

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

# Swap to "llama-3.3-70b-versatile" for higher quality at the cost of rate limit
REWRITE_MODEL   = "llama-3.1-8b-instant"
REWRITE_TIMEOUT = 8                     # seconds before fallback to original query
MIN_FIT_DEFAULT = 30                    # hide candidates below this % by default

# ══════════════════════════════════════════════════════════════════════════════
# REWRITTEN QUERY  (returned by rewrite_query)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RewrittenQuery:
    # The single best expanded query — fed to embed_query() and BM25
    primary: str

    # 1–2 alternative phrasings for multi-query retrieval
    alternatives: list[str] = field(default_factory=list)

    # Hard constraints extracted from the original query
    must_have_skills:   list[str] = field(default_factory=list)
    nice_to_have_skills: list[str] = field(default_factory=list)
    min_years_exp:      Optional[int]  = None
    seniority:          Optional[str]  = None   # "junior" | "mid" | "senior" | None
    location:           Optional[str]  = None

    # Whether LLM rewriting actually ran (False = fallback to original)
    rewritten: bool = False

    def all_queries(self) -> list[str]:
        """Return primary + alternatives, deduplicated."""
        seen, out = set(), []
        for q in [self.primary] + self.alternatives:
            q = q.strip()
            if q and q not in seen:
                seen.add(q)
                out.append(q)
        return out


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT  (focused and strict — forces clean JSON from Llama)
# ══════════════════════════════════════════════════════════════════════════════

_SYSTEM = """You are a query-expansion assistant for an HR CV search engine.
Given a recruiter's natural-language query, return a JSON object with these fields:

{
  "primary": "expanded, keyword-rich version of the query (1–2 sentences)",
  "alternatives": ["alternative phrasing 1", "alternative phrasing 2"],
  "must_have_skills": ["skill1", "skill2"],
  "nice_to_have_skills": ["skill3"],
  "min_years_exp": null or integer,
  "seniority": null or one of "junior" | "mid" | "senior",
  "location": null or city/country string
}

Rules:
- Expand abbreviations: ML → machine learning, NLP → natural language processing, k8s → Kubernetes
- Infer synonyms: React → React.js, ReactJS; Python dev → Python developer, Python engineer
- Extract hard constraints like "3 years", "senior", "Cairo-based" into their fields
- primary must be a full sentence, not just keywords
- Return ONLY valid JSON. No explanation, no markdown fences.
"""

# ══════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def _safe_int(val, default=None):
    """
    Convert val to int safely. Returns default if val is None, non-numeric string
    (e.g. "mid", "senior"), or anything else that can't be cleanly cast.
    """
    if val is None:
        return default
    try:
        result = int(val)
        return result if result > 0 else default
    except (ValueError, TypeError):
        return default


def rewrite_query(query: str) -> RewrittenQuery:
    """
    Call Groq to expand and structure the HR query.
    Falls back gracefully to the original query if the API call fails or key is missing.
    """
    query = query.strip()
    if not query:
        return RewrittenQuery(primary=query)

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        print("  [query_rewriter] GROQ_API_KEY not set — skipping rewrite")
        print("  Get a free key at: https://console.groq.com")
        return RewrittenQuery(primary=query)

    try:
        from groq import Groq

        client = Groq(api_key=api_key)

        completion = client.chat.completions.create(
            model=REWRITE_MODEL,
            max_tokens=400,
            timeout=REWRITE_TIMEOUT,
            temperature=0,  # deterministic — same query must always expand the same way,
                            # otherwise downstream embedding search returns different
                            # candidates/order on every identical run
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": query},
            ],
            # Force JSON output — Groq supports this natively
            response_format={"type": "json_object"},
        )

        raw = completion.choices[0].message.content.strip()

        # Safety: strip any accidental markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        data = json.loads(raw)

        return RewrittenQuery(
            primary             = data.get("primary", query),
            alternatives        = data.get("alternatives", [])[:2],
            must_have_skills    = data.get("must_have_skills", []),
            nice_to_have_skills = data.get("nice_to_have_skills", []),
            min_years_exp       = _safe_int(data.get("min_years_exp")),  # "mid"/"senior" → None
            seniority           = data.get("seniority"),
            location            = data.get("location"),
            rewritten           = True,
        )

    except Exception as e:
        print(f"  [query_rewriter] Groq rewrite failed ({e}) — using original query")
        return RewrittenQuery(primary=query)


# ══════════════════════════════════════════════════════════════════════════════
# SYNONYM MAP  —  covers common AI/ML/tech abbreviations and alternate spellings
# Add more rows here as needed; all values are checked as substrings.
# ══════════════════════════════════════════════════════════════════════════════

_SYNONYMS: dict[str, list[str]] = {
    # LLMs / GenAI
    "large language model":   ["large language model", "llm", "llms", "gpt", "chatgpt",
                                "generative ai", "genai", "gen ai", "foundation model"],
    "llm":                    ["llm", "llms", "large language model", "gpt", "generative ai"],
    "llms":                   ["llm", "llms", "large language model", "gpt", "generative ai"],
    "generative ai":          ["generative ai", "genai", "gen ai", "llm", "llms", "gpt",
                                "large language model", "foundation model"],
    "prompt engineering":     ["prompt engineering", "prompt design", "prompting"],

    # Classic ML
    "machine learning":       ["machine learning", "ml", "sklearn", "scikit-learn",
                                "supervised learning", "unsupervised learning",
                                "gradient boosting", "xgboost", "random forest"],
    "ml":                     ["ml", "machine learning", "sklearn", "scikit"],
    "deep learning":          ["deep learning", "dl", "neural network", "neural net",
                                "cnn", "rnn", "lstm", "transformer", "pytorch", "tensorflow",
                                "keras", "backpropagation"],
    "dl":                     ["dl", "deep learning", "neural network", "pytorch", "tensorflow"],
    "neural network":         ["neural network", "neural net", "deep learning", "dl",
                                "cnn", "rnn", "lstm", "transformer"],

    # Computer Vision
    "computer vision":        ["computer vision", "cv", "image recognition", "object detection",
                                "image classification", "opencv", "yolo", "resnet", "vgg",
                                "image segmentation", "mediapipe", "convolutional"],
    "cv":                     ["computer vision", "cv", "opencv", "image recognition",
                                "object detection", "yolo"],
    "object detection":       ["object detection", "yolo", "rcnn", "faster rcnn",
                                "ssd", "detection model"],
    "image recognition":      ["image recognition", "image classification", "resnet", "vgg",
                                "inception", "efficientnet", "computer vision"],

    # NLP
    "natural language processing": ["natural language processing", "nlp", "text classification",
                                     "sentiment analysis", "named entity recognition", "ner",
                                     "bert", "roberta", "transformers", "spacy", "nltk",
                                     "huggingface", "hugging face"],
    "nlp":                    ["nlp", "natural language processing", "bert", "transformers",
                                "spacy", "nltk", "text classification", "huggingface"],

    # RAG / agents
    "rag":                    ["rag", "retrieval augmented generation",
                                "retrieval-augmented generation", "vector search",
                                "chromadb", "faiss", "pinecone", "weaviate",
                                "semantic search", "embedding search"],
    "retrieval augmented generation": ["retrieval augmented generation", "rag",
                                        "retrieval-augmented generation", "vector search"],
    "agentic":                ["agentic", "agent", "agents", "multi-agent", "langgraph",
                                "langchain", "autogen", "crewai", "tool use", "react agent"],
    "langchain":              ["langchain", "lang chain", "langgraph"],

    # Frameworks / libraries
    "pytorch":                ["pytorch", "torch"],
    "tensorflow":             ["tensorflow", "tf", "keras"],
    "huggingface":            ["huggingface", "hugging face", "hf", "transformers library"],
    "opencv":                 ["opencv", "cv2", "open cv"],
    "scikit-learn":           ["scikit-learn", "sklearn", "scikit"],
    "fastapi":                ["fastapi", "fast api"],

    # Python ecosystem
    "python":                 ["python", "py", ".py", "django", "flask", "fastapi",
                                "pandas", "numpy", "scipy"],
    "pandas":                 ["pandas", "dataframe", "data wrangling"],
    "numpy":                  ["numpy", "np", "ndarray"],

    # Cloud / MLOps
    "mlops":                  ["mlops", "ml ops", "model deployment", "model serving",
                                "docker", "kubernetes", "k8s", "ci/cd", "mlflow",
                                "airflow", "model monitoring"],
    "docker":                 ["docker", "container", "containerization", "dockerfile"],
    "kubernetes":             ["kubernetes", "k8s", "kubectl", "helm"],
    "aws":                    ["aws", "amazon web services", "s3", "ec2", "sagemaker",
                                "lambda", "amazon"],
    "gcp":                    ["gcp", "google cloud", "bigquery", "vertex ai",
                                "google cloud platform"],
    "azure":                  ["azure", "microsoft azure", "azure ml"],

    # Databases / vector stores
    "sql":                    ["sql", "mysql", "postgresql", "postgres", "sqlite",
                                "database", "relational database"],
    "vector database":        ["vector database", "vector store", "chromadb", "faiss",
                                "pinecone", "weaviate", "qdrant", "milvus"],

    # Soft / role terms
    "ai engineer":            ["ai engineer", "artificial intelligence engineer",
                                "machine learning engineer", "ml engineer",
                                "deep learning engineer", "data scientist"],
    "data scientist":         ["data scientist", "data science", "ml engineer",
                                "machine learning engineer", "ai engineer"],
}


def _skill_present(skill: str, text: str) -> bool:
    """
    Return True if `skill` (or any of its synonyms) appears in `text`.
    Text must already be lowercased.
    """
    skill_lower = skill.lower().strip()

    # Direct substring match first (fast path)
    if skill_lower in text:
        return True

    # Check synonym list — both directions
    synonyms = _SYNONYMS.get(skill_lower, [])
    if any(syn in text for syn in synonyms):
        return True

    # Also check if this skill is a VALUE in the map (reverse lookup)
    for key, vals in _SYNONYMS.items():
        if skill_lower in vals:
            # Check the key itself and all other synonyms
            if key in text or any(v in text for v in vals):
                return True

    return False


# ══════════════════════════════════════════════════════════════════════════════
# SOFT FILTER  (replaced old hard filter — penalises instead of eliminates)
# ══════════════════════════════════════════════════════════════════════════════

def apply_hard_filters(
    candidates:  list[dict],
    rq:          RewrittenQuery,
    min_fit_pct: int = MIN_FIT_DEFAULT,
) -> list[dict]:
    """
    Applies skill checks with synonym expansion and soft scoring.

    Changes from the old version:
    - Skills are checked using a synonym map, so "DL" matches "deep learning",
      "LLM" matches "large language model", etc.
    - Missing must-have skills REDUCE fit_pct by a penalty instead of
      eliminating the candidate outright. Only candidates missing ALL
      must-have skills are dropped.
    - The min_fit_pct threshold is the only hard cutoff — and it is applied
      AFTER the penalty so the threshold remains meaningful.
    - Years-of-experience filter is unchanged (only fires if explicitly stated
      in query AND the candidate's CV has a parseable years field).
    """
    min_fit_pct = int(min_fit_pct)

    # Per-missing-skill penalty: spread evenly so losing half the skills
    # drops fit_pct by ~20 points. Adjust this constant to taste.
    PENALTY_PER_MISSING_SKILL = 12

    out = []
    for c in candidates:
        fit = int(c.get("fit_pct", 0))
        all_text = " ".join(c.get("all_chunks", [c.get("text", "")])).lower()

        # ── Skill check with synonym expansion ───────────────────────────────
        if rq.must_have_skills:
            found   = [s for s in rq.must_have_skills if _skill_present(s, all_text)]
            missing = [s for s in rq.must_have_skills if s not in found]

            print(
                f"  [filter] {c['metadata'].get('file', '?')[:35]:<35} "
                f"skills found={found} missing={missing}"
            )

            # Drop only if ZERO must-have skills are found at all
            if not found and rq.must_have_skills:
                print(f"  [filter] ✗ dropped — no must-have skills found")
                continue

            # Soft penalty for partial matches
            penalty = len(missing) * PENALTY_PER_MISSING_SKILL
            fit     = max(0, fit - penalty)
            c       = {**c, "fit_pct": fit}   # don't mutate original dict

        # ── Years-of-experience (unchanged — only fires if query states it) ──
        min_yrs = _safe_int(rq.min_years_exp)
        if min_yrs:
            years = _safe_int(c.get("years_exp"), default=0)
            if years and years < min_yrs:
                print(
                    f"  [filter] ✗ dropped {c['metadata'].get('file','?')} "
                    f"— exp {years}y < required {min_yrs}y"
                )
                continue

        # ── Final fit threshold (after skill penalty applied) ─────────────────
        if fit < min_fit_pct:
            print(
                f"  [filter] ✗ below threshold  {c['metadata'].get('file','?')} "
                f"fit={fit}% < {min_fit_pct}%"
            )
            continue

        out.append(c)

    print(f"  [filter] {len(out)} candidate(s) passed after soft filtering")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-QUERY SEARCH  (wraps pipeline.search)
# ══════════════════════════════════════════════════════════════════════════════

def smart_search(
    query:          str,
    top_k:          int  = 5,
    section_filter: str | None = None,
    use_reranker:   bool = True,
    min_fit_pct:    int  = MIN_FIT_DEFAULT,
) -> tuple[list[dict], RewrittenQuery]:
    """
    Full smart search pipeline:
        1. Rewrite + expand the query
        2. Run primary + alternative queries through pipeline.search()
        3. Merge results with RRF across query variants
        4. Apply hard filters
        5. Return (results, rewritten_query) so the UI can show what was searched

    Usage in app.py:
        from query_rewriter import smart_search
        results, rq = smart_search(query, top_k, section_filter, use_reranker, min_fit_pct)
    """
    from pipeline import search, embed_query
    from pipeline import _rrf, get_collection, load_bm25, get_reranker, _score_candidate

    # Cast defensively — callers may pass strings from Gradio or LangGraph state
    top_k       = int(top_k)
    min_fit_pct = int(min_fit_pct)

    # Step 1: rewrite
    rq = rewrite_query(query)
    queries = rq.all_queries()
    print(f"  [smart_search] queries: {queries}")
    if rq.must_have_skills:
        print(f"  [smart_search] must-have: {rq.must_have_skills}")
    if rq.min_years_exp:
        print(f"  [smart_search] min years: {rq.min_years_exp}")

    # Step 2: run each query variant independently
    # We call pipeline.search() with use_reranker=False here and rerank once
    # over the merged pool — reranking 3× separately wastes compute.
    all_candidate_maps: list[dict[str, dict]] = []

    for q in queries:
        partial = search(
            q,
            top_k=top_k * 4,          # fetch more before merging
            section_filter=section_filter,
            use_reranker=False,        # defer reranking to after merge
        )
        # key by filename so we can merge across query variants
        by_file = {c["metadata"].get("file", c["id"]): c for c in partial}
        all_candidate_maps.append(by_file)

    # Step 3: merge — for each candidate take the best rrf_score across variants
    merged: dict[str, dict] = {}
    for variant_map in all_candidate_maps:
        for fname, candidate in variant_map.items():
            if fname not in merged:
                merged[fname] = candidate
            else:
                # keep the richer chunk set and higher score
                existing = merged[fname]
                if candidate["rrf_score"] > existing["rrf_score"]:
                    # merge chunks from both
                    existing_chunks = set(existing["all_chunks"])
                    for chunk in candidate["all_chunks"]:
                        if chunk not in existing_chunks:
                            existing["all_chunks"].append(chunk)
                    existing["rrf_score"] = candidate["rrf_score"]
                    existing["text"] = "\n\n".join(existing["all_chunks"])

    candidates = sorted(
        merged.values(),
        key=lambda x: (x["rrf_score"], x["metadata"].get("file", "")),
        reverse=True,
    )

    # Step 4: rerank once over the merged pool (use primary query for reranking)
    if use_reranker and candidates:
        try:
            reranker = get_reranker()
            pairs = [(rq.primary, c["text"][:2000]) for c in candidates]
            rscores = reranker.predict(pairs)
            for c, s in zip(candidates, rscores):
                c["rerank_score"] = float(s)
            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        except Exception as e:
            print(f"  Reranker skipped: {e}")

    # Score all candidates
    scored = [_score_candidate(c, rq.primary) for c in candidates]

    # Step 5: apply hard filters
    filtered = apply_hard_filters(scored, rq, min_fit_pct=min_fit_pct)

    return filtered[:top_k], rq


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION PATCHES
# ══════════════════════════════════════════════════════════════════════════════
#
# ── SETUP ────────────────────────────────────────────────────────────────────
#
#   pip install groq
#
#   Windows:   set GROQ_API_KEY=gsk_...
#   Linux/Mac: export GROQ_API_KEY=gsk_...
#
#   Free key:  https://console.groq.com  (sign up, click "API Keys")
#
# ── MODEL CHOICE ─────────────────────────────────────────────────────────────
#
#   REWRITE_MODEL = "llama-3.1-8b-instant"     ← default: fastest, 6k req/day
#   REWRITE_MODEL = "llama-3.3-70b-versatile"  ← better quality, 1k req/day
#   REWRITE_MODEL = "gemma2-9b-it"             ← balanced, 14.4k req/day
#
# ── PATCH 1: app.py — update imports ─────────────────────────────────────────
#
# Replace:
#   from pipeline import (
#       process_file, process_dataset,
#       rebuild_bm25, count_chunks, processed_files, search,
#   )
#
# With:
#   from pipeline import (
#       process_file, process_dataset,
#       rebuild_bm25, count_chunks, processed_files,
#   )
#   from query_rewriter import smart_search, MIN_FIT_DEFAULT
#
# ── PATCH 2: app.py — new cb_search ─────────────────────────────────────────
#
# def cb_search(query, section, top_k, use_reranker, min_fit_pct):
#     global _last_results
#     q = query.strip()
#     if not q:
#         return "<p style='color:#94a3b8;text-align:center;padding:48px'>Enter a query above.</p>", ""
#
#     section_filter = None if section == "Any" else section
#     results, rq = smart_search(
#         q,
#         top_k=int(top_k),
#         section_filter=section_filter,
#         use_reranker=use_reranker,
#         min_fit_pct=int(min_fit_pct),
#     )
#     _last_results = results
#
#     rewrite_note = ""
#     if rq.rewritten:
#         pills = "".join(
#             f"<span style='background:#f1f5f9;color:#475569;padding:2px 8px;"
#             f"border-radius:10px;font-size:11px;margin-right:4px'>{s}</span>"
#             for s in rq.must_have_skills
#         )
#         rewrite_note = (
#             f"<div style='font-size:12px;color:#64748b;margin-bottom:12px'>"
#             f"<b>Searched:</b> {rq.primary}"
#             + (f"<br><b>Must-have skills:</b> {pills}" if pills else "")
#             + (f"<br><b>Min experience:</b> {rq.min_years_exp}y" if rq.min_years_exp else "")
#             + "</div>"
#         )
#
#     if not results:
#         status = f"<span style='color:#94a3b8'>No candidates above {int(min_fit_pct)}% fit.</span>"
#         return rewrite_note + "<p style='text-align:center;padding:32px;color:#94a3b8'>No results</p>", status
#
#     cards = "".join(_candidate_card(i + 1, r) for i, r in enumerate(results))
#     status = (
#         f"<span style='color:#16a34a'>✓ {len(results)} candidate(s) found"
#         + (" · query rewritten by Groq" if rq.rewritten else "")
#         + "</span>"
#     )
#     return rewrite_note + cards, status
#
# ── PATCH 3: app.py — add min_fit_pct slider to the Search tab ───────────────
#
#   min_fit_sl = gr.Slider(
#       label="Min fit %", minimum=0, maximum=80, value=MIN_FIT_DEFAULT, step=5, scale=1
#   )
#
#   # Update both .click() and .submit() inputs:
#   inputs=[query_box, section_dd, top_k_sl, rerank_cb, min_fit_sl],