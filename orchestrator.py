"""
orchestrator.py — LangGraph Multi-Agent Orchestrator
=====================================================
Connects all 4 HR Assistant agents in a directed graph:

  HR query
    → Supervisor (decides what to do)
    → Agent 1: RetriveCVAgent  (search + find candidates)
    → Agent 2: ReasoningAgent  (explain WHY each candidate fits)
    → Agent 3: EmailAgent      (draft personalized email)
    → Agent 4: GapAgent        (skill gap report)
    → END

Usage from app.py:
    from orchestrator import run_pipeline
    result = run_pipeline(user_query, job_description="...", mode="full")

Install:
    pip install langgraph langchain-core
"""

import os
from typing import TypedDict, Annotated, Literal
import operator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ── Import your existing agents ────────────────────────────────────────────────
from query_rewriter import smart_search
from RetriveCVAgent import run_agent_turn
from ReasoningAgent import generate_candidate_reasoning   # ← use the real Agent 2


# ══════════════════════════════════════════════════════════════════════════════
# SHARED STATE
# This dict flows through every node. Each agent reads from it and writes to it.
# ══════════════════════════════════════════════════════════════════════════════

class HRState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    user_query:        str               # What HR typed
    job_description:   str               # Full JD text (optional, for gap agent)
    top_k:             int               # How many candidates to retrieve
    min_fit_pct:       int               # Minimum fit threshold

    # ── Agent 1 output ─────────────────────────────────────────────────────
    candidates:        list[dict]        # Raw results from smart_search()
    search_html:       str               # Rendered candidate cards HTML
    llm_history:       list[dict]        # Groq conversation history

    # ── Agent 2 output ─────────────────────────────────────────────────────
    reasoning:         list[dict]        # [{candidate_file, reasoning_text}, ...]

    # ── Agent 3 output ─────────────────────────────────────────────────────
    email_drafts:      list[dict]        # [{candidate_file, email_subject, email_body}, ...]

    # ── Agent 4 output ─────────────────────────────────────────────────────
    gap_reports:       list[dict]        # [{candidate_file, gap_html}, ...]

    # ── Routing ────────────────────────────────────────────────────────────
    next_agent:        str               # Supervisor sets this to route to next node
    mode:              str               # "search" | "reason" | "email" | "full"
    errors:            Annotated[list, operator.add]   # Accumulates any errors


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1 — SUPERVISOR
# Reads the mode and decides which agents to run
# ══════════════════════════════════════════════════════════════════════════════

def supervisor_node(state: HRState) -> HRState:
    """
    Routes the pipeline based on `mode`:
      "search" → only Agent 1
      "reason" → Agent 1 + Agent 2
      "email"  → Agent 1 + Agent 2 + Agent 3
      "full"   → All 4 agents
    Always starts with Agent 1 (search).
    """
    print(f"\n[Supervisor] mode={state['mode']} | query={state['user_query'][:60]}")
    return {**state, "next_agent": "search"}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 2 — AGENT 1: Retrieval & Conversational Search
# ══════════════════════════════════════════════════════════════════════════════

def search_agent_node(state: HRState) -> HRState:
    """
    Calls smart_search() to find matching candidates.
    Also calls run_agent_turn() to get the LLM-generated HTML reply.
    """
    print(f"[Agent 1] Searching for: {state['user_query']}")
    errors = []

    # ── Defensive int casting — Gradio sliders and LangGraph state can
    #    serialize ints as strings; pipeline.py does > comparisons on these
    top_k       = int(state.get("top_k", 5))
    min_fit_pct = int(state.get("min_fit_pct", 10))

    try:
        # Run the hybrid search pipeline
        candidates, rq = smart_search(
            query       = state["user_query"],
            top_k       = top_k,
            min_fit_pct = min_fit_pct,
            use_reranker= True,
        )

        # Normalize fit_pct on every candidate to int — ChromaDB metadata
        # sometimes returns numeric fields as strings
        for c in candidates:
            c["fit_pct"]      = int(c.get("fit_pct", 0))
            c["years_exp"]    = int(c.get("years_exp", 0))
            c["keyword_hits"] = int(c.get("keyword_hits", 0))

        print(f"[Agent 1] Found {len(candidates)} candidates")

        # Run the conversational agent (LLM summary + HTML cards)
        def _search_adapter(query, top_k, min_fit_pct):
            return smart_search(query, top_k=int(top_k), min_fit_pct=int(min_fit_pct))

        search_html, llm_history = run_agent_turn(
            user_message = state["user_query"],
            llm_history  = state.get("llm_history", []),
            search_fn    = _search_adapter,
        )

    except Exception as e:
        print(f"[Agent 1] ERROR: {e}")
        errors.append(f"Agent1: {e}")
        candidates   = []
        search_html  = f"<div style='color:red'>Search failed: {e}</div>"
        llm_history  = state.get("llm_history", [])

    # Decide next node
    mode = state.get("mode", "search")
    if not candidates or mode == "search":
        next_agent = "end"
    elif mode == "reason":
        next_agent = "reason"
    elif mode == "email":
        next_agent = "reason"   # reason runs before email
    elif mode == "full":
        next_agent = "reason"
    else:
        next_agent = "end"

    return {
        **state,
        "candidates":   candidates,
        "search_html":  search_html,
        "llm_history":  llm_history,
        "next_agent":   next_agent,
        "errors":       errors,
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3 — AGENT 2: Reasoning Agent
# Explains WHY each candidate fits the query
# ══════════════════════════════════════════════════════════════════════════════

def _find_named_candidate(message: str, candidates: list) -> list:
    """
    If the message mentions a specific candidate's name, return only that candidate.
    Otherwise return all candidates (for "why did you choose them all" style questions).
    """
    msg_lower = message.lower().strip()
    for c in candidates:
        name = (c.get("metadata", {}).get("name", "") or
                c.get("metadata", {}).get("file", "")).lower()
        name_parts = [p for p in name.replace("_", " ").replace(".", " ").split() if len(p) >= 4]
        if any(part in msg_lower for part in name_parts):
            return [c]   # ← only the named candidate
    return candidates    # ← all candidates (generic "why them" question)


def reasoning_agent_node(state: HRState) -> HRState:
    """
    For each candidate in state['candidates'], calls ReasoningAgent.generate_candidate_reasoning()
    and attaches the result to the candidate dict and to the 'reasoning' list.
    """
    candidates_to_reason = state["candidates"]
    print(f"[Agent 2] Reasoning over {len(candidates_to_reason)} candidates")
    errors  = []
    results = []

    try:
        for c in candidates_to_reason:
            cv_text = " ".join(c.get("all_chunks", [c.get("text", "")]))[:3000]
            name    = c["metadata"].get("name") or c["metadata"].get("file", "Candidate")

            try:
                reasoning_text = generate_candidate_reasoning(
                    hr_query       = state["user_query"],
                    candidate_text = cv_text,
                )
            except Exception as e:
                reasoning_text = f"Reasoning unavailable: {e}"
                errors.append(f"Agent2/{name}: {e}")

            c["reasoning"] = reasoning_text
            results.append({
                "candidate_file": c["metadata"].get("file", ""),
                "candidate_name": name,
                "reasoning_text": reasoning_text,
            })
            print(f"  [Agent 2] ✓ {name}")

    except Exception as e:
        print(f"[Agent 2] ERROR: {e}")
        errors.append(f"Agent2: {e}")

    mode = state.get("mode", "reason")
    next_agent = "email" if mode in ("email", "full") else "end"

    return {
        **state,
        "reasoning":  results,
        "candidates": state["candidates"],
        "next_agent": next_agent,
        "errors":     errors,
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 4 — AGENT 3: Email Agent
# Drafts a personalized outreach email per candidate
# ══════════════════════════════════════════════════════════════════════════════

def email_agent_node(state: HRState) -> HRState:
    """
    For each candidate, uses their reasoning + CV background to draft
    a personalized interview invitation email.
    """
    print(f"[Agent 3] Drafting emails for {len(state['candidates'])} candidates")
    errors       = []
    email_drafts = []

    try:
        from groq import Groq
        client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

        EMAIL_SYSTEM = """You are a professional HR recruiter writing personalized interview invitation emails.

Rules:
- Reference 1-2 SPECIFIC skills or experiences from the candidate's background (mentioned in the reasoning)
- Keep it under 150 words
- Professional but warm tone
- End with a call to action (reply with availability)
- Use {name} as placeholder for the candidate's name
- Return ONLY the email body, no subject line"""

        for c in state["candidates"]:
            name      = c["metadata"].get("name") or c["metadata"].get("file", "Candidate")
            reasoning = c.get("reasoning", "")
            cv_snippet= " ".join(c.get("all_chunks", []))[:800]

            try:
                # Subject line
                subj_resp = client.chat.completions.create(
                    model      = "llama-3.1-8b-instant",
                    max_tokens = 20,
                    temperature= 0.3,
                    messages   = [{
                        "role": "user",
                        "content": f"Write a short professional email subject line (max 8 words) for "
                                   f"inviting this candidate to interview for: {state['user_query']}"
                    }],
                )
                subject = subj_resp.choices[0].message.content.strip().strip('"')

                # Email body
                body_resp = client.chat.completions.create(
                    model      = "llama-3.3-70b-versatile",
                    max_tokens = 250,
                    temperature= 0.4,
                    messages   = [
                        {"role": "system", "content": EMAIL_SYSTEM},
                        {"role": "user",   "content":
                            f"Role we're hiring for: {state['user_query']}\n\n"
                            f"Why this candidate was selected: {reasoning}\n\n"
                            f"CV background: {cv_snippet}"},
                    ],
                )
                body = body_resp.choices[0].message.content.strip()

            except Exception as e:
                subject = "Interview Invitation"
                body    = "Dear {name},\n\nWe were impressed by your profile and would like to invite you for an interview.\n\nPlease reply with your availability.\n\nBest regards,\nHR Team"
                errors.append(f"Agent3/{name}: {e}")

            email_drafts.append({
                "candidate_file":  c["metadata"].get("file", ""),
                "candidate_name":  name,
                "candidate_email": c["metadata"].get("email", ""),
                "email_subject":   subject,
                "email_body":      body,
            })
            print(f"  [Agent 3] ✓ {name}")

    except Exception as e:
        print(f"[Agent 3] ERROR: {e}")
        errors.append(f"Agent3: {e}")

    # Decide next node
    next_agent = "gap" if state.get("mode") == "full" else "end"

    return {
        **state,
        "email_drafts": email_drafts,
        "next_agent":   next_agent,
        "errors":       errors,
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 5 — AGENT 4: Gap Agent
# Produces skill gap report per candidate vs job description
# ══════════════════════════════════════════════════════════════════════════════

def gap_agent_node(state: HRState) -> HRState:
    """
    Compares each candidate's CV against the job description (or query)
    and produces an actionable gap report.
    Tries to import generate_gap_report from gap_agent.py first.
    Falls back to inline Groq call if not available.
    """
    print(f"[Agent 4] Gap analysis for {len(state['candidates'])} candidates")
    errors      = []
    gap_reports = []

    jd = state.get("job_description", "") or state.get("user_query", "")

    for c in state["candidates"]:
        name    = c["metadata"].get("name") or c["metadata"].get("file", "Candidate")
        cv_text = " ".join(c.get("all_chunks", [c.get("text", "")]))[:3000]

        try:
            # Try your teammate's gap_agent.py first
            try:
                from gap_agent import generate_gap_report
                gap_html = generate_gap_report(
                    candidate_text = cv_text,
                    candidate_name = name,
                    job_description= jd,
                )
            except ImportError:
                # Fallback: inline Groq call
                from groq import Groq
                client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

                GAP_SYSTEM = """You are an expert career coach analyzing a candidate's fit gap.
Given a job description and a candidate's CV, identify:
1. Missing must-have skills (cite what the JD needs vs what the CV shows)
2. Experience gaps (years, domains, seniority level)
3. Actionable recommendations (specific courses, projects, or certifications)
4. Estimated time to close each gap

Format as clean HTML with sections. Be specific — never generic advice."""

                resp = client.chat.completions.create(
                    model      = "llama-3.3-70b-versatile",
                    max_tokens = 600,
                    temperature= 0.2,
                    messages   = [
                        {"role": "system", "content": GAP_SYSTEM},
                        {"role": "user",   "content":
                            f"Job Description / Requirements:\n{jd}\n\n"
                            f"Candidate CV ({name}):\n{cv_text}"},
                    ],
                )
                gap_html = f"<div style='font-family:monospace;font-size:.85em'>{resp.choices[0].message.content}</div>"

        except Exception as e:
            gap_html = f"<div style='color:red'>Gap report failed for {name}: {e}</div>"
            errors.append(f"Agent4/{name}: {e}")

        gap_reports.append({
            "candidate_file": c["metadata"].get("file", ""),
            "candidate_name": name,
            "gap_html":       gap_html,
        })
        print(f"  [Agent 4] ✓ {name}")

    return {
        **state,
        "gap_reports": gap_reports,
        "next_agent":  "end",
        "errors":      errors,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ROUTING FUNCTION
# LangGraph calls this after each node to decide the next node
# ══════════════════════════════════════════════════════════════════════════════

def route(state: HRState) -> Literal["search", "reason", "email", "gap", "__end__"]:
    next_node = state.get("next_agent", "end")
    if next_node == "end":
        return END
    return next_node


# ══════════════════════════════════════════════════════════════════════════════
# BUILD THE GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def build_graph():
    graph = StateGraph(HRState)

    # Register nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("search",     search_agent_node)
    graph.add_node("reason",     reasoning_agent_node)
    graph.add_node("email",      email_agent_node)
    graph.add_node("gap",        gap_agent_node)

    # Entry point
    graph.set_entry_point("supervisor")

    # Supervisor always goes to search
    graph.add_edge("supervisor", "search")

    # After each agent, use the routing function to decide next
    graph.add_conditional_edges("search", route)
    graph.add_conditional_edges("reason", route)
    graph.add_conditional_edges("email",  route)
    graph.add_conditional_edges("gap",    route)

    # Compile with in-memory checkpointing (enables multi-turn memory)
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# Singleton — build once, reuse
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API — call this from app.py
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    user_query:      str,
    job_description: str  = "",
    mode:            str  = "search",   # "search" | "reason" | "email" | "full"
    top_k:           int  = 5,
    min_fit_pct:     int  = 10,
    thread_id:       str  = "default",  # use per-user thread_id for multi-turn memory
    llm_history:     list = None,
) -> dict:
    """
    Run the multi-agent pipeline and return all outputs.

    Parameters
    ----------
    user_query      : HR's natural language query
    job_description : Full JD text (used by gap agent — optional)
    mode            : Which agents to run:
                      "search" → Agent 1 only
                      "reason" → Agents 1 + 2
                      "email"  → Agents 1 + 2 + 3
                      "full"   → All 4 agents
    top_k           : Number of candidates to retrieve
    min_fit_pct     : Minimum fit percentage filter
    thread_id       : Conversation thread ID (for multi-turn memory via checkpointer)
    llm_history     : Existing Groq message history (for follow-up turns)

    Returns
    -------
    dict with keys:
        candidates    : list of candidate dicts (with reasoning attached if mode != "search")
        search_html   : HTML string of candidate cards
        reasoning     : list of {candidate_name, reasoning_text}
        email_drafts  : list of {candidate_name, email_subject, email_body, candidate_email}
        gap_reports   : list of {candidate_name, gap_html}
        llm_history   : updated conversation history
        errors        : list of any errors encountered
    """
    initial_state: HRState = {
        "user_query":      user_query,
        "job_description": job_description,
        "top_k":           int(top_k),
        "min_fit_pct":     int(min_fit_pct),
        "mode":            mode,
        "llm_history":     llm_history or [],
        # outputs start empty
        "candidates":      [],
        "search_html":     "",
        "reasoning":       [],
        "email_drafts":    [],
        "gap_reports":     [],
        "next_agent":      "",
        "errors":          [],
    }

    config = {"configurable": {"thread_id": thread_id}}

    final_state = get_graph().invoke(initial_state, config=config)

    return {
        "candidates":   final_state.get("candidates",    []),
        "search_html":  final_state.get("search_html",   ""),
        "reasoning":    final_state.get("reasoning",     []),
        "email_drafts": final_state.get("email_drafts",  []),
        "gap_reports":  final_state.get("gap_reports",   []),
        "llm_history":  final_state.get("llm_history",   []),
        "errors":       final_state.get("errors",        []),
    }


# ══════════════════════════════════════════════════════════════════════════════
# INTENT DETECTION
# Decides whether a follow-up message is a "why / reasoning" request
# or a new search query — so app.py doesn't have to guess.
# ══════════════════════════════════════════════════════════════════════════════

_REASONING_TRIGGERS = [
    # "why" family
    "why", "explain", "reason", "justify", "how did you", "what made",
    "tell me more about", "elaborate", "basis", "criteria", "rationale",
    "why them", "why these", "why did you choose", "why chosen",
    # candidate-specific info questions
    "what is the skill", "what are the skill", "what is her skill", "what is his skill",
    "what are her skill", "what are his skill", "skills of", "experience of",
    "background of", "tell me about", "more about", "show me", "describe",
    "what does she", "what does he", "what can she", "what can he",
    "qualifications", "profile of", "summary of",
    # Arabic
    "ليه", "ليه اخترت", "علي اساس ايه", "وضح", "اشرح",
    "مهاراتها", "مهاراته", "خبرتها", "خبرته", "عنها", "عنه", "اعرف عن",
]

# Questions that compare/filter within the already-retrieved results
# rather than triggering a new search
_COMPARE_TRIGGERS = [
    "who is better", "who's better", "who is best", "who's best",
    "compare", "which one", "which candidate", "between them",
    "who has", "who have", "who knows", "who worked", "who built",
    "who used", "who did", "who can",
    "rank them", "rank these", "order them",
    "strongest", "weakest", "most experienced", "least experienced",
    "most relevant", "best fit", "worst fit",
    "among them", "among these", "from these", "from the results",
    "out of these", "out of them", "of the candidates",
    # Arabic
    "مين أحسن", "مين أفضل", "مين عنده", "مين بنى", "مين شتغل",
    "قارن", "ايه الفرق", "مين منهم",
]

def _is_reasoning_request(message: str, last_candidates: list) -> bool:
    """
    Return True if the message is a follow-up about already-retrieved candidates
    rather than a brand-new search query.
    Signals:
    1. Contains a reasoning/explanation trigger word.
    2. Mentions a retrieved candidate's name.
    (Compare intent is handled separately by _is_compare_request.)
    """
    if not last_candidates:
        return False

    msg_lower = message.lower().strip()

    if any(trigger in msg_lower for trigger in _REASONING_TRIGGERS):
        return True

    # Name match — at least 4-char name parts to avoid short false positives
    for c in last_candidates:
        name = (c.get("metadata", {}).get("name", "") or
                c.get("metadata", {}).get("file", "")).lower()
        name_parts = [p for p in name.replace("_", " ").replace(".", " ").split() if len(p) >= 4]
        if any(part in msg_lower for part in name_parts):
            return True

    return False


def _is_compare_request(message: str, last_candidates: list) -> bool:
    """
    Return True if the user is asking to compare, rank, or filter
    among the already-retrieved candidates.
    Examples: "who is better in GenAI", "who has built RAG systems",
              "which one knows FastAPI", "rank them by seniority"
    """
    if not last_candidates:
        return False
    msg_lower = message.lower().strip()
    return any(trigger in msg_lower for trigger in _COMPARE_TRIGGERS)


def _compare_candidates(user_message: str, candidates: list) -> str:
    """
    Ask Groq to answer a comparison/filter question using the candidate
    data already in memory — no new search needed.
    Returns an HTML string.
    """
    from groq import Groq
    import os

    client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

    # Build a compact text summary of all candidates for Groq to reason over
    summaries = []
    for i, c in enumerate(candidates, 1):
        name    = c.get("metadata", {}).get("name") or c.get("metadata", {}).get("file", f"Candidate {i}")
        cv_text = " ".join(c.get("all_chunks", [c.get("text", "")]))[:1500]
        summaries.append(f"--- Candidate {i}: {name} ---\n{cv_text}")

    candidates_block = "\n\n".join(summaries)

    COMPARE_SYSTEM = """You are an expert HR analyst. You have a set of candidates already retrieved from a CV database.
Answer the HR's question by analyzing ONLY the candidate information provided — do not suggest new searches.

Rules:
- Reference candidates by name
- Cite specific evidence from their CV text
- If comparing, be direct about who ranks higher and why
- If filtering (e.g. "who has built RAG"), list only candidates with that evidence, or clearly state if none have it
- Keep the answer concise and structured
- Never hallucinate skills not present in the CV text provided"""

    try:
        resp = client.chat.completions.create(
            model       = "llama-3.3-70b-versatile",
            temperature = 0.1,
            max_tokens  = 600,
            messages    = [
                {"role": "system", "content": COMPARE_SYSTEM},
                {"role": "user",   "content":
                    f"HR Question: {user_message}\n\n"
                    f"Retrieved Candidates:\n{candidates_block}"},
            ],
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        answer = f"Comparison failed: {e}"

    # Render as a clean HTML block
    formatted = answer.replace("\n\n", "</p><p style='margin:8px 0'>") \
                      .replace("\n•", "<br>•") \
                      .replace("\n-", "<br>-") \
                      .replace("\n**", "<br><strong>").replace("**", "</strong>")
    return (
        "<div style='font-family:Inter,system-ui,sans-serif;padding:4px 0'>"
        "<div style='font-weight:600;color:#111827;margin-bottom:14px;font-size:.95em'>"
        "Comparing retrieved candidates:</div>"
        "<div style='padding:16px 18px;background:#F9FAFB;border:1px solid #E5E7EB;"
        "border-radius:8px;border-left:3px solid #059669;"
        "font-size:.87em;color:#374151;line-height:1.7'>"
        f"<p style='margin:0'>{formatted}</p>"
        "</div></div>"
    )


# ══════════════════════════════════════════════════════════════════════════════
# CONVERSATIONAL TURN — the function app.py should call
#
# Replaces the direct call to run_agent_turn() in app.py.
# Handles two cases:
#   1. New search query  → mode="search", returns search HTML
#   2. "Why" follow-up   → mode="reason", runs Agent 2 over stored candidates
#                          and returns reasoning HTML injected into the reply
# ══════════════════════════════════════════════════════════════════════════════

def run_conversational_turn(
    user_message:  str,
    llm_history:   list,
    last_candidates: list = None,
    thread_id:     str   = "default",
    top_k:         int   = 5,
    min_fit_pct:   int   = 10,
) -> tuple[str, list, list]:
    # Cast to int defensively — Gradio slider values can arrive as strings
    top_k       = int(top_k)
    min_fit_pct = int(min_fit_pct)

    # ── Case 1: Follow-up reasoning about specific candidate(s) ─────────────
    if _is_reasoning_request(user_message, last_candidates or []):

        targets = _find_named_candidate(user_message, last_candidates or [])
        print(f"[Orchestrator] Reasoning request → targeting {len(targets)} candidate(s)")

        dummy_state: HRState = {
            "user_query":      user_message,
            "job_description": "",
            "top_k":           top_k,
            "min_fit_pct":     min_fit_pct,
            "mode":            "reason",
            "candidates":      targets,
            "search_html":     "",
            "llm_history":     llm_history,
            "reasoning":       [],
            "email_drafts":    [],
            "gap_reports":     [],
            "next_agent":      "",
            "errors":          [],
        }

        result_state = reasoning_agent_node(dummy_state)
        reasoning    = result_state.get("reasoning", [])
        reply_html   = _render_reasoning_html(reasoning)

        updated_history = llm_history + [
            {"role": "user",      "content": user_message},
            {"role": "assistant", "content": "[Reasoning response provided]"},
        ]
        return reply_html, updated_history, last_candidates

    # ── Case 2: Compare / filter within already-retrieved results ────────────
    # e.g. "who is better in GenAI", "who has built RAG systems", "rank them"
    if _is_compare_request(user_message, last_candidates or []):
        print(f"[Orchestrator] Compare request → answering from {len(last_candidates)} cached candidates")

        reply_html = _compare_candidates(user_message, last_candidates or [])

        updated_history = llm_history + [
            {"role": "user",      "content": user_message},
            {"role": "assistant", "content": "[Comparison response provided]"},
        ]
        return reply_html, updated_history, last_candidates  # candidates unchanged

    # ── Case 3: Brand-new search query ───────────────────────────────────────
    print(f"[Orchestrator] New search query: {user_message[:60]}")
    result = run_pipeline(
        user_query   = user_message,
        mode         = "search",
        top_k        = top_k,
        min_fit_pct  = min_fit_pct,
        thread_id    = thread_id,
        llm_history  = llm_history,
    )
    return result["search_html"], result["llm_history"], result["candidates"]


def _last_query_from_history(llm_history: list) -> str:
    """Extract the last user message from LLM history (used as context for reasoning)."""
    for msg in reversed(llm_history):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def _render_reasoning_html(reasoning: list[dict]) -> str:
    """Render the reasoning list from Agent 2 (ReasoningAgent.py) into clean HTML.
    ReasoningAgent returns structured text with sections like 'Why Selected:', 'Evidence Found:', etc.
    We preserve the structure and render it cleanly.
    """
    if not reasoning:
        return "<div style='color:#6B7280;font-size:.9em'>No reasoning available — try searching first.</div>"

    parts = [
        "<div style='font-family:Inter,system-ui,sans-serif;padding:4px 0'>"
        "<div style='font-weight:600;color:#111827;margin-bottom:14px;font-size:.95em'>"
        "Here's the reasoning for each candidate:</div>"
    ]
    for item in reasoning:
        name = item.get("candidate_name", "Candidate")
        text = item.get("reasoning_text", "No reasoning available.")

        # Convert the plain-text structured output from ReasoningAgent into HTML
        # It uses sections like "Why Selected:\n...\n\nEvidence Found:\n..."
        formatted = text.replace("\n\n", "</p><p style='margin:8px 0'>") \
                        .replace("\n•", "<br>•") \
                        .replace("\n-", "<br>-")

        parts.append(
            f"<div style='margin-bottom:16px;padding:16px 18px;"
            f"background:#F9FAFB;border:1px solid #E5E7EB;border-radius:8px;"
            f"border-left:3px solid #4F46E5'>"
            f"<div style='font-weight:600;color:#4F46E5;font-size:.88em;margin-bottom:10px'>"
            f"📋 {name}</div>"
            f"<div style='font-size:.86em;color:#374151;line-height:1.7'>"
            f"<p style='margin:0'>{formatted}</p>"
            f"</div>"
            f"</div>"
        )
    parts.append("</div>")
    return "".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# CLI TEST — python orchestrator.py
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing orchestrator in 'search' mode...\n")

    result = run_pipeline(
        user_query  = "machine learning engineer with Python and TensorFlow",
        mode        = "search",
        top_k       = 3,
        min_fit_pct = 5,
    )

    print(f"\n{'='*50}")
    print(f"Candidates found : {len(result['candidates'])}")
    print(f"Errors           : {result['errors']}")
    for c in result["candidates"]:
        print(f"  - {c['metadata'].get('file')} | fit={c.get('fit_pct')}%")

    print("\nTest 'full' mode with a job description...\n")
    result2 = run_pipeline(
        user_query      = "senior Python developer Django REST",
        job_description = "We need a senior Python developer with 5+ years, Django, REST APIs, PostgreSQL.",
        mode            = "full",
        top_k           = 2,
        min_fit_pct     = 5,
    )
    print(f"Emails drafted   : {len(result2['email_drafts'])}")
    print(f"Gap reports      : {len(result2['gap_reports'])}")
    if result2["email_drafts"]:
        d = result2["email_drafts"][0]
        print(f"\nSample email for {d['candidate_name']}:")
        print(f"Subject: {d['email_subject']}")
        print(f"Body:\n{d['email_body'][:300]}")