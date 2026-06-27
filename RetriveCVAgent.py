<<<<<<< HEAD
"""
Agent 1 — Job Description Analyzer & Conversational Search
===========================================================
Converts natural HR dialogue into structured candidate searches.

Flow (one turn):
  HR message
    → Groq LLaMA 3.3 (with search_candidates tool)
    → If tool called  → execute smart_search → render CV cards
    → If no tool call → ask one clarifying question
    → Returns (reply_html, updated_llm_history)

Usage from app.py:
    from agent1 import run_agent_turn
    reply_html, new_history = run_agent_turn(user_msg, llm_history, search_fn)
"""

import os, json, math
from pathlib import Path
from groq import Groq

# ─── Config ──────────────────────────────────────────────────────────────────
MODEL = "llama-3.3-70b-versatile"

# ─── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an AI recruitment assistant helping HR professionals search a CV database.

YOUR BEHAVIOR:
- When the HR describes a role or candidate, extract the requirements and call search_candidates IMMEDIATELY.
- Only ask ONE clarifying question if the request is genuinely too vague to search (e.g. just "hi" or "I need help").
- After results come back, write a SHORT natural summary (2-3 sentences max). No bullet lists.
- Handle follow-up refinements naturally: "more senior", "add React too", "only Cairo-based" — refine the previous search.
- Remember context across the conversation: if HR said "Python dev" earlier, follow-ups build on that.
- Be professional but conversational. Speak as if you're a smart colleague, not a chatbot.

SEARCH QUERY TIPS:
- Build a rich query string: role + skills + seniority + context.
- For Arabic/MENA context: include location keywords if mentioned (Cairo, Egypt, Alexandria, etc.).
- If the HR mentions years of experience explicitly, include it in the query.

EXAMPLES:
HR: "I'm looking for a senior Python dev with FastAPI, 3+ years, Cairo"
→ Call search_candidates with query="senior Python developer FastAPI Cairo 3 years experience"

HR: "make it more junior and add Django"
→ Call search_candidates with query="junior Python developer FastAPI Django Cairo"

HR: "what about ML engineers?"
→ Call search_candidates with query="machine learning engineer Python TensorFlow Cairo"
"""

# ─── Tool definition (what Groq can call) ─────────────────────────────────────
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_candidates",
        "description": (
            "Search the CV database for candidates matching the HR's requirements. "
            "Call this whenever you have enough information to run a meaningful search. "
            "Prefer calling it with the information you have rather than asking for more."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Rich natural language query combining role, skills, seniority, location, "
                        "and any other relevant context. "
                        "Example: 'senior Python developer FastAPI PostgreSQL Cairo 3 years experience'"
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top candidates to retrieve. Default 5, max 10.",
                    "default": 5,
                },
                "min_fit_pct": {
                    "type": "integer",
                    "description": "Minimum fit percentage threshold (0-80). Default 25.",
                    "default": 25,
                },
            },
            "required": ["query"],
        },
    },
}

# ─── Card rendering ────────────────────────────────────────────────────────────
# Self-contained inline-style rendering so agent1.py has no dependency on app.py

_COLORS = {
    "strong":  "#2F6F4E",
    "good":    "#9A6B12",
    "partial": "#6B7177",
}
_LABELS = {
    "strong":  "STRONG MATCH",
    "good":    "GOOD MATCH",
    "partial": "PARTIAL",
}


def _gauge(pct: int, color: str) -> str:
    """Analog semi-circle dial SVG."""
    cx, cy, r = 60, 56, 46
    angle     = math.radians(180 - 1.8 * pct)
    ex, ey    = cx + r * math.cos(angle), cy - r * math.sin(angle)
    nx, ny    = cx + 37 * math.cos(angle), cy - 37 * math.sin(angle)
    ticks     = "".join(
        (lambda a: f'<line x1="{cx+46*math.cos(a):.1f}" y1="{cy-46*math.sin(a):.1f}"'
                   f' x2="{cx+38*math.cos(a):.1f}" y2="{cy-38*math.sin(a):.1f}"'
                   f' stroke="#B8BBB3" stroke-width="2"/>')
        (math.radians(180 - 1.8 * t))
        for t in (0, 25, 50, 75, 100)
    )
    return (
        f'<svg width="108" height="58" viewBox="0 0 120 62" style="flex-shrink:0">'
        f'<path d="M{cx-r},{cy} A{r},{r} 0 0,1 {cx+r},{cy}"'
        f' fill="none" stroke="#DCDDD5" stroke-width="8" stroke-linecap="round"/>'
        f'<path d="M{cx-r},{cy} A{r},{r} 0 0,1 {ex:.1f},{ey:.1f}"'
        f' fill="none" stroke="{color}" stroke-width="8" stroke-linecap="round"/>'
        f'{ticks}'
        f'<line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}"'
        f' stroke="#1C1F1B" stroke-width="2.4" stroke-linecap="round"/>'
        f'<circle cx="{cx}" cy="{cy}" r="4.5" fill="#1C1F1B"/>'
        f'</svg>'
    )


def _chip(text: str, filled: bool = False) -> str:
    if filled:
        style = ("background:#E3EFE7;color:#2F6F4E;padding:2px 8px;border-radius:3px;"
                 "font-size:.68em;font-family:monospace;font-weight:600;"
                 "margin-right:4px;margin-top:5px;display:inline-block")
    else:
        style = ("border:1px solid #D9DAD3;color:#5C6259;padding:2px 8px;border-radius:3px;"
                 "font-size:.68em;font-family:monospace;"
                 "margin-right:4px;margin-top:5px;display:inline-block")
    return f'<span style="{style}">{text}</span>'


def render_card(rank: int, result: dict) -> str:
    """Render one candidate result as an HTML card string."""
    m       = result["metadata"]
    name    = m.get("name") or Path(m.get("file", "—")).stem
    email   = m.get("email", "")
    phone   = m.get("phone", "")
    linkedin= m.get("linkedin", "")
    file_nm = m.get("file", "—")
    quality = result.get("match_quality", "partial")
    pct     = result.get("fit_pct", 0)
    years   = result.get("years_exp", 0)
    hits    = result.get("keyword_hits", 0)
    secs    = list(dict.fromkeys(result.get("sections_found", [])))[:5]

    color  = _COLORS.get(quality, "#6B7177")
    label  = _LABELS.get(quality, "PARTIAL")
    gauge  = _gauge(pct, color)

    email_html = (f'<a href="mailto:{email}" style="color:#2F6F4E;text-decoration:none">'
                  f'{email}</a>') if email else "—"
    li_html    = (f'<a href="https://{linkedin}" target="_blank"'
                  f' style="color:#2F6F4E;text-decoration:none">LinkedIn ↗</a>') if linkedin else "—"

    badges = (_chip(f"{years}y exp", filled=True) if years else "") + \
             (_chip(f"{hits} kw hits", filled=True) if hits else "")
    chips  = "".join(_chip(s) for s in secs if s)

    preview = (
        " · ".join(result.get("all_chunks", [result.get("text", "")]))[:420]
        .replace("\n", " ")
        .encode("ascii", errors="ignore").decode("ascii")
        .strip()
    )

    # Corner-mark spans (pseudo-elements not available inline so we use <span>)
    cm = "position:absolute;width:11px;height:11px;border:2px solid #1C1F1B"
    return f"""
<div style="position:relative;background:#FFFFFF;border:1px solid #D9DAD3;border-radius:6px;
            padding:16px 18px 14px;margin:8px 0 16px;font-family:system-ui,sans-serif">
  <span style="{cm};top:-5px;left:-5px;border-right:none;border-bottom:none"></span>
  <span style="{cm};top:-5px;right:-5px;border-left:none;border-bottom:none"></span>
  <span style="{cm};bottom:-5px;left:-5px;border-right:none;border-top:none"></span>
  <span style="{cm};bottom:-5px;right:-5px;border-left:none;border-top:none"></span>

  <div style="display:flex;align-items:flex-start;gap:12px">
    <div style="flex:1;min-width:0">
      <div style="font-family:monospace;font-size:.7em;color:#6B7177;letter-spacing:.06em">FR.{rank:02d}</div>
      <div style="font-family:'Space Grotesk',sans-serif;font-weight:600;font-size:1.06em;
                  color:#1C1F1B;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{name}</div>
      <div style="font-family:monospace;font-size:.7em;color:#5C6259;margin-top:2px">{file_nm}</div>
    </div>
    <div style="display:flex;flex-direction:column;align-items:center;flex-shrink:0">
      {gauge}
      <div style="font-family:monospace;font-weight:700;font-size:1.25em;color:{color};
                  margin-top:-12px">{pct}%</div>
      <div style="font-family:monospace;font-size:.6em;letter-spacing:.12em;
                  color:#6B7177;text-transform:uppercase">fit score</div>
      <div style="font-family:monospace;font-size:.68em;letter-spacing:.08em;
                  text-transform:uppercase;font-weight:700;color:{color};
                  display:flex;align-items:center;gap:5px;margin-top:5px">
        <span style="width:7px;height:7px;border-radius:50%;
                     background:{color};display:inline-block"></span>{label}
      </div>
    </div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;
              margin-top:12px;padding-top:10px;border-top:1px solid #D9DAD3">
    <div>
      <div style="font-family:monospace;font-size:.63em;letter-spacing:.08em;
                  text-transform:uppercase;color:#6B7177;margin-bottom:2px">Email</div>
      <div style="font-size:.84em;overflow:hidden;text-overflow:ellipsis">{email_html}</div>
    </div>
    <div>
      <div style="font-family:monospace;font-size:.63em;letter-spacing:.08em;
                  text-transform:uppercase;color:#6B7177;margin-bottom:2px">Phone</div>
      <div style="font-size:.84em;color:#1C1F1B">{phone or "—"}</div>
    </div>
    <div>
      <div style="font-family:monospace;font-size:.63em;letter-spacing:.08em;
                  text-transform:uppercase;color:#6B7177;margin-bottom:2px">LinkedIn</div>
      <div style="font-size:.84em">{li_html}</div>
    </div>
  </div>

  <div style="margin-top:4px">{badges}{chips}</div>

  <div style="margin-top:10px;padding:9px 12px;background:#EDEEEA;
              border-left:3px solid #1C1F1B;font-family:monospace;
              font-size:.72em;color:#5C6259;line-height:1.6;border-radius:0 5px 5px 0">
    {preview}
  </div>
</div>
"""


# ─── Agent core ───────────────────────────────────────────────────────────────

def run_agent_turn(
    user_message: str,
    llm_history: list,
    search_fn,
) -> tuple[str, list]:
    """
    Process one HR message.

    Parameters
    ----------
    user_message : str
        What the HR typed.
    llm_history : list[dict]
        Groq-format message history (text only, no HTML).
        Pass [] on the first turn; store and pass back on every subsequent turn.
    search_fn : callable
        Function matching signature:
            (query: str, top_k: int, min_fit_pct: int) -> (results: list, rq: any)
        Typically a thin wrapper around smart_search().

    Returns
    -------
    reply_html : str
        Full HTML string to display in gr.Chatbot.
    updated_llm_history : list[dict]
        Updated Groq-format history for the next turn.
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return (
            "<div style='color:#9A1F1F;font-family:monospace;font-size:.85em;padding:10px;"
            "border-left:3px solid #9A1F1F;background:#FBE6DB'>"
            "⚠ GROQ_API_KEY not set. Add it to your environment variables and restart.</div>",
            llm_history,
        )

    client = Groq(api_key=api_key)

    # Build Groq messages: system + rolling history + new user turn
    groq_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    groq_messages.extend(llm_history)
    groq_messages.append({"role": "user", "content": user_message})

    # ── Turn 1: let agent decide whether to search or ask ──────────────────
    resp1 = client.chat.completions.create(
        model=MODEL,
        messages=groq_messages,
        tools=[SEARCH_TOOL],
        tool_choice="auto",
        temperature=0.2,
        max_tokens=512,
    )
    msg1 = resp1.choices[0].message

    # ── No tool call → agent wants to ask a clarifying question ────────────
    if not msg1.tool_calls:
        text = msg1.content or ""
        updated = llm_history + [
            {"role": "user",      "content": user_message},
            {"role": "assistant", "content": text},
        ]
        reply_html = (
            f'<div style="line-height:1.75;font-family:system-ui,sans-serif;'
            f'color:#1C1F1B">{text}</div>'
        )
        return reply_html, updated

    # ── Tool call → execute search ──────────────────────────────────────────
    tool_call = msg1.tool_calls[0]
    args      = json.loads(tool_call.function.arguments)
    query     = args.get("query", user_message)
    top_k     = min(int(args.get("top_k", 5)), 10)
    min_fit   = int(args.get("min_fit_pct", 25))

    results, _rq = search_fn(query=query, top_k=top_k, min_fit_pct=min_fit)

    # Compact text summary for Groq turn 2 (not HTML)
    if results:
        names_scores = ", ".join(
            f"{r['metadata'].get('name', Path(r['metadata'].get('file','?')).stem)}"
            f" ({r.get('fit_pct', 0)}%)"
            for r in results
        )
        tool_result = f"Found {len(results)} candidates: {names_scores}."
    else:
        tool_result = "No candidates found matching those criteria."

    # ── Turn 2: agent summarises results in natural language ────────────────
    groq_messages.append({
        "role": "assistant",
        "content": msg1.content,   # may be None — Groq accepts that
        "tool_calls": [{
            "id":   tool_call.id,
            "type": "function",
            "function": {
                "name":      tool_call.function.name,
                "arguments": tool_call.function.arguments,
            },
        }],
    })
    groq_messages.append({
        "role":         "tool",
        "tool_call_id": tool_call.id,
        "content":      tool_result,
    })

    resp2   = client.chat.completions.create(
        model=MODEL,
        messages=groq_messages,
        temperature=0.3,
        max_tokens=256,
    )
    summary = resp2.choices[0].message.content or tool_result

    # ── Build reply HTML ────────────────────────────────────────────────────
    summary_div = (
        f'<div style="line-height:1.75;font-family:system-ui,sans-serif;'
        f'color:#1C1F1B;margin-bottom:10px">{summary}</div>'
    )

    if results:
        cards_html = "".join(render_card(i + 1, r) for i, r in enumerate(results))
        # Show the query that was actually used
        query_tag = (
            f'<div style="font-family:monospace;font-size:.72em;color:#5C6259;'
            f'margin-bottom:8px">Searched: <em>{query}</em></div>'
        )
        reply_html = summary_div + query_tag + cards_html
    else:
        no_results = (
            '<div style="text-align:center;padding:20px;color:#6B7177;'
            'font-family:monospace;font-size:.85em;border:1px dashed #D9DAD3;'
            'border-radius:6px;margin-top:8px">'
            'No candidates matched. Try broader terms or lower the fit threshold.</div>'
        )
        reply_html = summary_div + no_results

    # Update history with TEXT-ONLY versions (no HTML — Groq doesn't need it)
    updated = llm_history + [
        {"role": "user",      "content": user_message},
        {"role": "assistant", "content": summary},
    ]
    return reply_html, updated
=======
"""
Agent 1 — Job Description Analyzer & Conversational Search
===========================================================
Converts natural HR dialogue into structured candidate searches.

Flow (one turn):
  HR message
    → Groq LLaMA 3.3 (with search_candidates tool)
    → If tool called  → execute smart_search → render CV cards
    → If no tool call → ask one clarifying question
    → Returns (reply_html, updated_llm_history)

Usage from app.py:
    from agent1 import run_agent_turn
    reply_html, new_history = run_agent_turn(user_msg, llm_history, search_fn)
"""

import os, json, math
from pathlib import Path
from groq import Groq

# ─── Config ──────────────────────────────────────────────────────────────────
MODEL = "llama-3.3-70b-versatile"

# ─── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an AI recruitment assistant helping HR professionals search a CV database.

YOUR BEHAVIOR:
- When the HR describes a role or candidate, extract the requirements and call search_candidates IMMEDIATELY.
- Only ask ONE clarifying question if the request is genuinely too vague to search (e.g. just "hi" or "I need help").
- After results come back, write a SHORT natural summary (2-3 sentences max). No bullet lists.
- Handle follow-up refinements naturally: "more senior", "add React too", "only Cairo-based" — refine the previous search.
- Remember context across the conversation: if HR said "Python dev" earlier, follow-ups build on that.
- Be professional but conversational. Speak as if you're a smart colleague, not a chatbot.

SEARCH QUERY TIPS:
- Build a rich query string: role + skills + seniority + context.
- For Arabic/MENA context: include location keywords if mentioned (Cairo, Egypt, Alexandria, etc.).
- If the HR mentions years of experience explicitly, include it in the query.

EXAMPLES:
HR: "I'm looking for a senior Python dev with FastAPI, 3+ years, Cairo"
→ Call search_candidates with query="senior Python developer FastAPI Cairo 3 years experience"

HR: "make it more junior and add Django"
→ Call search_candidates with query="junior Python developer FastAPI Django Cairo"

HR: "what about ML engineers?"
→ Call search_candidates with query="machine learning engineer Python TensorFlow Cairo"
"""

# ─── Tool definition (what Groq can call) ─────────────────────────────────────
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_candidates",
        "description": (
            "Search the CV database for candidates matching the HR's requirements. "
            "Call this whenever you have enough information to run a meaningful search. "
            "Prefer calling it with the information you have rather than asking for more."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Rich natural language query combining role, skills, seniority, location, "
                        "and any other relevant context. "
                        "Example: 'senior Python developer FastAPI PostgreSQL Cairo 3 years experience'"
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top candidates to retrieve. Default 5, max 10.",
                    "default": 5,
                },
                "min_fit_pct": {
                    "type": "integer",
                    "description": "Minimum fit percentage threshold (0-80). Default 25.",
                    "default": 25,
                },
            },
            "required": ["query"],
        },
    },
}

# ─── Card rendering ────────────────────────────────────────────────────────────
# Self-contained inline-style rendering so agent1.py has no dependency on app.py

_COLORS = {
    "strong":  "#2F6F4E",
    "good":    "#9A6B12",
    "partial": "#6B7177",
}
_LABELS = {
    "strong":  "STRONG MATCH",
    "good":    "GOOD MATCH",
    "partial": "PARTIAL",
}


def _gauge(pct: int, color: str) -> str:
    """Analog semi-circle dial SVG."""
    cx, cy, r = 60, 56, 46
    angle     = math.radians(180 - 1.8 * pct)
    ex, ey    = cx + r * math.cos(angle), cy - r * math.sin(angle)
    nx, ny    = cx + 37 * math.cos(angle), cy - 37 * math.sin(angle)
    ticks     = "".join(
        (lambda a: f'<line x1="{cx+46*math.cos(a):.1f}" y1="{cy-46*math.sin(a):.1f}"'
                   f' x2="{cx+38*math.cos(a):.1f}" y2="{cy-38*math.sin(a):.1f}"'
                   f' stroke="#B8BBB3" stroke-width="2"/>')
        (math.radians(180 - 1.8 * t))
        for t in (0, 25, 50, 75, 100)
    )
    return (
        f'<svg width="108" height="58" viewBox="0 0 120 62" style="flex-shrink:0">'
        f'<path d="M{cx-r},{cy} A{r},{r} 0 0,1 {cx+r},{cy}"'
        f' fill="none" stroke="#DCDDD5" stroke-width="8" stroke-linecap="round"/>'
        f'<path d="M{cx-r},{cy} A{r},{r} 0 0,1 {ex:.1f},{ey:.1f}"'
        f' fill="none" stroke="{color}" stroke-width="8" stroke-linecap="round"/>'
        f'{ticks}'
        f'<line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}"'
        f' stroke="#1C1F1B" stroke-width="2.4" stroke-linecap="round"/>'
        f'<circle cx="{cx}" cy="{cy}" r="4.5" fill="#1C1F1B"/>'
        f'</svg>'
    )


def _chip(text: str, filled: bool = False) -> str:
    if filled:
        style = ("background:#E3EFE7;color:#2F6F4E;padding:2px 8px;border-radius:3px;"
                 "font-size:.68em;font-family:monospace;font-weight:600;"
                 "margin-right:4px;margin-top:5px;display:inline-block")
    else:
        style = ("border:1px solid #D9DAD3;color:#5C6259;padding:2px 8px;border-radius:3px;"
                 "font-size:.68em;font-family:monospace;"
                 "margin-right:4px;margin-top:5px;display:inline-block")
    return f'<span style="{style}">{text}</span>'


def render_card(rank: int, result: dict) -> str:
    """Render one candidate result as an HTML card string."""
    m       = result["metadata"]
    name    = m.get("name") or Path(m.get("file", "—")).stem
    email   = m.get("email", "")
    phone   = m.get("phone", "")
    linkedin= m.get("linkedin", "")
    file_nm = m.get("file", "—")
    quality = result.get("match_quality", "partial")
    pct     = result.get("fit_pct", 0)
    years   = result.get("years_exp", 0)
    hits    = result.get("keyword_hits", 0)
    secs    = list(dict.fromkeys(result.get("sections_found", [])))[:5]

    color  = _COLORS.get(quality, "#6B7177")
    label  = _LABELS.get(quality, "PARTIAL")
    gauge  = _gauge(pct, color)

    email_html = (f'<a href="mailto:{email}" style="color:#2F6F4E;text-decoration:none">'
                  f'{email}</a>') if email else "—"
    li_html    = (f'<a href="https://{linkedin}" target="_blank"'
                  f' style="color:#2F6F4E;text-decoration:none">LinkedIn ↗</a>') if linkedin else "—"

    badges = (_chip(f"{years}y exp", filled=True) if years else "") + \
             (_chip(f"{hits} kw hits", filled=True) if hits else "")
    chips  = "".join(_chip(s) for s in secs if s)

    preview = (
        " · ".join(result.get("all_chunks", [result.get("text", "")]))[:420]
        .replace("\n", " ")
        .encode("ascii", errors="ignore").decode("ascii")
        .strip()
    )

    # Corner-mark spans (pseudo-elements not available inline so we use <span>)
    cm = "position:absolute;width:11px;height:11px;border:2px solid #1C1F1B"
    return f"""
<div style="position:relative;background:#FFFFFF;border:1px solid #D9DAD3;border-radius:6px;
            padding:16px 18px 14px;margin:8px 0 16px;font-family:system-ui,sans-serif">
  <span style="{cm};top:-5px;left:-5px;border-right:none;border-bottom:none"></span>
  <span style="{cm};top:-5px;right:-5px;border-left:none;border-bottom:none"></span>
  <span style="{cm};bottom:-5px;left:-5px;border-right:none;border-top:none"></span>
  <span style="{cm};bottom:-5px;right:-5px;border-left:none;border-top:none"></span>

  <div style="display:flex;align-items:flex-start;gap:12px">
    <div style="flex:1;min-width:0">
      <div style="font-family:monospace;font-size:.7em;color:#6B7177;letter-spacing:.06em">FR.{rank:02d}</div>
      <div style="font-family:'Space Grotesk',sans-serif;font-weight:600;font-size:1.06em;
                  color:#1C1F1B;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{name}</div>
      <div style="font-family:monospace;font-size:.7em;color:#5C6259;margin-top:2px">{file_nm}</div>
    </div>
    <div style="display:flex;flex-direction:column;align-items:center;flex-shrink:0">
      {gauge}
      <div style="font-family:monospace;font-weight:700;font-size:1.25em;color:{color};
                  margin-top:-12px">{pct}%</div>
      <div style="font-family:monospace;font-size:.6em;letter-spacing:.12em;
                  color:#6B7177;text-transform:uppercase">fit score</div>
      <div style="font-family:monospace;font-size:.68em;letter-spacing:.08em;
                  text-transform:uppercase;font-weight:700;color:{color};
                  display:flex;align-items:center;gap:5px;margin-top:5px">
        <span style="width:7px;height:7px;border-radius:50%;
                     background:{color};display:inline-block"></span>{label}
      </div>
    </div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;
              margin-top:12px;padding-top:10px;border-top:1px solid #D9DAD3">
    <div>
      <div style="font-family:monospace;font-size:.63em;letter-spacing:.08em;
                  text-transform:uppercase;color:#6B7177;margin-bottom:2px">Email</div>
      <div style="font-size:.84em;overflow:hidden;text-overflow:ellipsis">{email_html}</div>
    </div>
    <div>
      <div style="font-family:monospace;font-size:.63em;letter-spacing:.08em;
                  text-transform:uppercase;color:#6B7177;margin-bottom:2px">Phone</div>
      <div style="font-size:.84em;color:#1C1F1B">{phone or "—"}</div>
    </div>
    <div>
      <div style="font-family:monospace;font-size:.63em;letter-spacing:.08em;
                  text-transform:uppercase;color:#6B7177;margin-bottom:2px">LinkedIn</div>
      <div style="font-size:.84em">{li_html}</div>
    </div>
  </div>

  <div style="margin-top:4px">{badges}{chips}</div>

  <div style="margin-top:10px;padding:9px 12px;background:#EDEEEA;
              border-left:3px solid #1C1F1B;font-family:monospace;
              font-size:.72em;color:#5C6259;line-height:1.6;border-radius:0 5px 5px 0">
    {preview}
  </div>
</div>
"""


# ─── Agent core ───────────────────────────────────────────────────────────────

def run_agent_turn(
    user_message: str,
    llm_history: list,
    search_fn,
) -> tuple[str, list]:
    """
    Process one HR message.

    Parameters
    ----------
    user_message : str
        What the HR typed.
    llm_history : list[dict]
        Groq-format message history (text only, no HTML).
        Pass [] on the first turn; store and pass back on every subsequent turn.
    search_fn : callable
        Function matching signature:
            (query: str, top_k: int, min_fit_pct: int) -> (results: list, rq: any)
        Typically a thin wrapper around smart_search().

    Returns
    -------
    reply_html : str
        Full HTML string to display in gr.Chatbot.
    updated_llm_history : list[dict]
        Updated Groq-format history for the next turn.
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return (
            "<div style='color:#9A1F1F;font-family:monospace;font-size:.85em;padding:10px;"
            "border-left:3px solid #9A1F1F;background:#FBE6DB'>"
            "⚠ GROQ_API_KEY not set. Add it to your environment variables and restart.</div>",
            llm_history,
        )

    client = Groq(api_key=api_key)

    # Build Groq messages: system + rolling history + new user turn
    groq_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    groq_messages.extend(llm_history)
    groq_messages.append({"role": "user", "content": user_message})

    # ── Turn 1: let agent decide whether to search or ask ──────────────────
    resp1 = client.chat.completions.create(
        model=MODEL,
        messages=groq_messages,
        tools=[SEARCH_TOOL],
        tool_choice="auto",
        temperature=0.2,
        max_tokens=512,
    )
    msg1 = resp1.choices[0].message

    # ── No tool call → agent wants to ask a clarifying question ────────────
    if not msg1.tool_calls:
        text = msg1.content or ""
        updated = llm_history + [
            {"role": "user",      "content": user_message},
            {"role": "assistant", "content": text},
        ]
        reply_html = (
            f'<div style="line-height:1.75;font-family:system-ui,sans-serif;'
            f'color:#1C1F1B">{text}</div>'
        )
        return reply_html, updated

    # ── Tool call → execute search ──────────────────────────────────────────
    tool_call = msg1.tool_calls[0]
    args      = json.loads(tool_call.function.arguments)
    query     = args.get("query", user_message)
    top_k     = min(int(args.get("top_k", 5)), 10)
    min_fit   = int(args.get("min_fit_pct", 25))

    results, _rq = search_fn(query=query, top_k=top_k, min_fit_pct=min_fit)

    # Compact text summary for Groq turn 2 (not HTML)
    if results:
        names_scores = ", ".join(
            f"{r['metadata'].get('name', Path(r['metadata'].get('file','?')).stem)}"
            f" ({r.get('fit_pct', 0)}%)"
            for r in results
        )
        tool_result = f"Found {len(results)} candidates: {names_scores}."
    else:
        tool_result = "No candidates found matching those criteria."

    # ── Turn 2: agent summarises results in natural language ────────────────
    groq_messages.append({
        "role": "assistant",
        "content": msg1.content,   # may be None — Groq accepts that
        "tool_calls": [{
            "id":   tool_call.id,
            "type": "function",
            "function": {
                "name":      tool_call.function.name,
                "arguments": tool_call.function.arguments,
            },
        }],
    })
    groq_messages.append({
        "role":         "tool",
        "tool_call_id": tool_call.id,
        "content":      tool_result,
    })

    resp2   = client.chat.completions.create(
        model=MODEL,
        messages=groq_messages,
        temperature=0.3,
        max_tokens=256,
    )
    summary = resp2.choices[0].message.content or tool_result

    # ── Build reply HTML ────────────────────────────────────────────────────
    summary_div = (
        f'<div style="line-height:1.75;font-family:system-ui,sans-serif;'
        f'color:#1C1F1B;margin-bottom:10px">{summary}</div>'
    )

    if results:
        cards_html = "".join(render_card(i + 1, r) for i, r in enumerate(results))
        # Show the query that was actually used
        query_tag = (
            f'<div style="font-family:monospace;font-size:.72em;color:#5C6259;'
            f'margin-bottom:8px">Searched: <em>{query}</em></div>'
        )
        reply_html = summary_div + query_tag + cards_html
    else:
        no_results = (
            '<div style="text-align:center;padding:20px;color:#6B7177;'
            'font-family:monospace;font-size:.85em;border:1px dashed #D9DAD3;'
            'border-radius:6px;margin-top:8px">'
            'No candidates matched. Try broader terms or lower the fit threshold.</div>'
        )
        reply_html = summary_div + no_results

    # Update history with TEXT-ONLY versions (no HTML — Groq doesn't need it)
    updated = llm_history + [
        {"role": "user",      "content": user_message},
        {"role": "assistant", "content": summary},
    ]
    return reply_html, updated
>>>>>>> ccd2dadd5377540db68e843a6ff95315064a1051
