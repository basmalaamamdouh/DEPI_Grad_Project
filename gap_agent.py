"""
Agent 4 — Candidate Gap Agent
==============================
Compares a candidate's CV text against a job description (or HR query) and
produces a structured gap report:
  - Missing / weak skills
  - Experience gaps
  - Recommended learning resources
  - Rough timeline to bridge the gap

Public API
----------
    from gap_agent import generate_gap_report

    html = generate_gap_report(
        candidate_text="...",   # r["text"] or " ".join(r["all_chunks"])
        candidate_name="Jane",  # for display
        job_description="...",  # free-form HR query or full JD
    )
    # Returns an HTML string ready to inject into Gradio gr.HTML
"""

import os, json
from groq import Groq

MODEL  = "llama-3.3-70b-versatile"
_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
    return _client


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM = """You are an expert HR analyst. Given a candidate's CV text and a job
description, you identify concrete gaps and produce a structured JSON report.

You MUST respond with ONLY valid JSON — no markdown fences, no extra text.

Schema:
{
  "overall_readiness": "strong" | "partial" | "significant_gap",
  "readiness_pct": <integer 0-100>,
  "summary": "<2-3 sentence plain-English summary>",
  "missing_skills": [
    {"skill": "...", "importance": "critical"|"important"|"nice_to_have",
     "candidate_level": "none"|"beginner"|"some", "notes": "..."}
  ],
  "experience_gaps": [
    {"area": "...", "required": "...", "candidate_has": "...", "gap": "..."}
  ],
  "strengths": ["...", "..."],
  "recommended_resources": [
    {"title": "...", "type": "course"|"book"|"project"|"certification",
     "url_hint": "...", "covers": "..."}
  ],
  "timeline_weeks": <integer>,
  "timeline_breakdown": [
    {"phase": "...", "weeks": <int>, "focus": "..."}
  ]
}
"""

_USER_TEMPLATE = """JOB DESCRIPTION / HR QUERY:
{jd}

---

CANDIDATE CV TEXT:
{cv}
"""


# ── Core call ─────────────────────────────────────────────────────────────────

def _call_groq(cv_text: str, jd_text: str) -> dict:
    client = _get_client()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system",  "content": _SYSTEM},
            {"role": "user",    "content": _USER_TEMPLATE.format(
                jd=jd_text[:3000], cv=cv_text[:4000])},
        ],
        temperature=0.3,
        max_tokens=1800,
    )
    raw = resp.choices[0].message.content.strip()
    # Strip accidental markdown fences if model adds them
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


# ── HTML renderer ─────────────────────────────────────────────────────────────

_IMPORTANCE_COLOR = {
    "critical":      "#C0392B",
    "important":     "#9A6B12",
    "nice_to_have":  "#5C6259",
}
_READINESS_COLOR = {
    "strong":          "#2F6F4E",
    "partial":         "#9A6B12",
    "significant_gap": "#C0392B",
}


def _pill(text: str, color: str = "#5C6259", bg: str = "#F0F0EC") -> str:
    return (
        f'<span style="display:inline-block;padding:2px 10px;border-radius:12px;'
        f'background:{bg};color:{color};font-size:.78em;font-weight:600;'
        f'margin:2px 3px 2px 0;border:1px solid {color}30">{text}</span>'
    )


def _section_header(title: str) -> str:
    return (
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.72em;'
        f'color:#5C6259;letter-spacing:.08em;text-transform:uppercase;'
        f'margin:18px 0 8px;padding-bottom:4px;border-bottom:1px solid #D9DAD3">'
        f'{title}</div>'
    )


def _render_report(name: str, data: dict) -> str:
    rc      = data.get("readiness_pct", 0)
    rstatus = data.get("overall_readiness", "partial")
    rcolor  = _READINESS_COLOR.get(rstatus, "#9A6B12")

    # ── Header ────────────────────────────────────────────────────────────────
    html = (
        f'<div style="border:1px solid #D9DAD3;border-radius:10px;padding:20px 24px;'
        f'background:#F7F7F4;margin-bottom:16px;font-family:Inter,sans-serif">'

        # Title row
        f'<div style="display:flex;align-items:center;gap:14px;margin-bottom:12px">'
        f'<div style="flex:1">'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.72em;'
        f'color:#5C6259;letter-spacing:.06em">GAP REPORT — AGENT 4</div>'
        f'<div style="font-size:1.15em;font-weight:700;color:#1C1F1B;margin-top:2px">'
        f'{name}</div>'
        f'</div>'
        # Readiness badge
        f'<div style="text-align:center">'
        f'<div style="font-size:2em;font-weight:800;color:{rcolor};line-height:1">{rc}%</div>'
        f'<div style="font-size:.72em;color:{rcolor};font-weight:600;'
        f'letter-spacing:.06em;text-transform:uppercase">{rstatus.replace("_"," ")}</div>'
        f'</div>'
        f'</div>'

        # Progress bar
        f'<div style="background:#E4E5E0;border-radius:4px;height:6px;margin-bottom:12px">'
        f'<div style="width:{rc}%;background:{rcolor};height:6px;border-radius:4px;'
        f'transition:width .4s"></div></div>'

        # Summary
        f'<div style="font-size:.9em;color:#3A3D38;line-height:1.6">'
        f'{data.get("summary","")}</div>'
    )

    # ── Strengths ─────────────────────────────────────────────────────────────
    strengths = data.get("strengths", [])
    if strengths:
        html += _section_header("✅ Strengths")
        html += '<div style="display:flex;flex-wrap:wrap">'
        for s in strengths:
            html += _pill(s, "#2F6F4E", "#EAF3EE")
        html += '</div>'

    # ── Missing skills ────────────────────────────────────────────────────────
    missing = data.get("missing_skills", [])
    if missing:
        html += _section_header("⚠️ Missing / Weak Skills")
        html += (
            '<table style="width:100%;border-collapse:collapse;font-size:.85em">'
            '<tr style="background:#EDEEEA">'
            '<th style="padding:6px 10px;text-align:left;font-weight:600">Skill</th>'
            '<th style="padding:6px 10px;text-align:left;font-weight:600">Importance</th>'
            '<th style="padding:6px 10px;text-align:left;font-weight:600">Candidate level</th>'
            '<th style="padding:6px 10px;text-align:left;font-weight:600">Notes</th>'
            '</tr>'
        )
        for i, sk in enumerate(missing):
            imp   = sk.get("importance", "important")
            color = _IMPORTANCE_COLOR.get(imp, "#5C6259")
            bg    = "#FFF9F9" if imp == "critical" else "#FFFDF5" if imp == "important" else "#F7F7F4"
            html += (
                f'<tr style="background:{bg};border-top:1px solid #E4E5E0">'
                f'<td style="padding:7px 10px;font-weight:600;color:#1C1F1B">{sk.get("skill","")}</td>'
                f'<td style="padding:7px 10px">'
                f'<span style="color:{color};font-weight:700;font-size:.8em;'
                f'text-transform:uppercase">{imp.replace("_"," ")}</span></td>'
                f'<td style="padding:7px 10px;color:#5C6259">{sk.get("candidate_level","—")}</td>'
                f'<td style="padding:7px 10px;color:#3A3D38">{sk.get("notes","")}</td>'
                f'</tr>'
            )
        html += '</table>'

    # ── Experience gaps ───────────────────────────────────────────────────────
    exp_gaps = data.get("experience_gaps", [])
    if exp_gaps:
        html += _section_header("📋 Experience Gaps")
        for g in exp_gaps:
            html += (
                f'<div style="border-left:3px solid #9A6B12;padding:8px 14px;'
                f'margin-bottom:8px;background:#FFFCF5;border-radius:0 6px 6px 0">'
                f'<div style="font-weight:700;color:#1C1F1B">{g.get("area","")}</div>'
                f'<div style="font-size:.82em;color:#5C6259;margin-top:3px">'
                f'<b>Required:</b> {g.get("required","")} &nbsp;|&nbsp; '
                f'<b>Has:</b> {g.get("candidate_has","")} &nbsp;|&nbsp; '
                f'<b>Gap:</b> {g.get("gap","")}</div>'
                f'</div>'
            )

    # ── Resources ─────────────────────────────────────────────────────────────
    resources = data.get("recommended_resources", [])
    if resources:
        html += _section_header("📚 Recommended Resources")
        TYPE_ICON = {"course": "🎓", "book": "📖", "project": "🛠️", "certification": "🏅"}
        html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">'
        for res in resources:
            icon = TYPE_ICON.get(res.get("type",""), "🔗")
            html += (
                f'<div style="border:1px solid #E4E5E0;border-radius:7px;padding:10px 12px;'
                f'background:#FAFAF7">'
                f'<div style="font-weight:600;color:#1C1F1B;font-size:.88em">'
                f'{icon} {res.get("title","")}</div>'
                f'<div style="font-size:.78em;color:#6B7177;margin-top:3px">'
                f'{res.get("url_hint","")}</div>'
                f'<div style="font-size:.8em;color:#3A3D38;margin-top:4px">'
                f'{res.get("covers","")}</div>'
                f'</div>'
            )
        html += '</div>'

    # ── Timeline ──────────────────────────────────────────────────────────────
    timeline_weeks = data.get("timeline_weeks")
    breakdown      = data.get("timeline_breakdown", [])
    if timeline_weeks:
        html += _section_header(f"🗓️ Estimated Timeline — {timeline_weeks} weeks")
        if breakdown:
            total = sum(p.get("weeks", 0) for p in breakdown) or 1
            for phase in breakdown:
                pw  = phase.get("weeks", 0)
                pct = int(pw / total * 100)
                html += (
                    f'<div style="margin-bottom:7px">'
                    f'<div style="display:flex;justify-content:space-between;'
                    f'font-size:.83em;margin-bottom:3px">'
                    f'<span style="font-weight:600;color:#1C1F1B">{phase.get("phase","")}</span>'
                    f'<span style="color:#5C6259">{pw}w — {phase.get("focus","")}</span>'
                    f'</div>'
                    f'<div style="background:#E4E5E0;border-radius:3px;height:5px">'
                    f'<div style="width:{pct}%;background:#1C1F1B;height:5px;border-radius:3px"></div>'
                    f'</div></div>'
                )

    html += '</div>'  # close main card
    return html


# ── Public function ───────────────────────────────────────────────────────────

def generate_gap_report(
    candidate_text: str,
    candidate_name: str,
    job_description: str,
) -> str:
    """
    Returns an HTML string (gap report card) ready for gr.HTML.
    Raises on Groq or JSON errors — callers should wrap in try/except.
    """
    data = _call_groq(candidate_text, job_description)
    return _render_report(candidate_name, data)