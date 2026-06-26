"""
HR Assistant — CV RAG System
Run:  python app.py
Open: http://localhost:7860
"""

import os, math, smtplib
from pathlib import Path
from email.mime.text      import MIMEText
from email.mime.multipart import MIMEMultipart
from EmailAgent import run_email_agent_turn, send_personalized_email

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import gradio as gr
from pipeline import (
    process_file, process_dataset,
    rebuild_bm25, count_chunks, processed_files,
)
from query_rewriter import smart_search, MIN_FIT_DEFAULT
from RetriveCVAgent import run_agent_turn

# ══════════════════════════════════════════════════════════════════════════════
# EMAIL CONFIG
# ══════════════════════════════════════════════════════════════════════════════

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "email")
SMTP_PASS = os.getenv("SMTP_PASS", "pass")
HR_NAME   = os.getenv("HR_NAME",   "HR Team")

SECTIONS = ["Any", "skills", "experience", "education", "projects",
            "summary", "certif", "training", "languages", "volunteer"]

_last_results: list[dict] = []

# ══════════════════════════════════════════════════════════════════════════════
# EMAIL
# ══════════════════════════════════════════════════════════════════════════════

def send_email(to_addr: str, name: str, subject: str, body: str) -> str:
    if not SMTP_USER or not SMTP_PASS:
        return "❌ Email not configured — set SMTP_USER and SMTP_PASS env vars."
    if not to_addr:
        return "❌ No email address on file for this candidate."
    try:
        msg            = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = f"{HR_NAME} <{SMTP_USER}>"
        msg["To"]      = to_addr
        msg.attach(MIMEText(body.replace("{name}", name or "Candidate"), "plain"))
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
            s.ehlo(); s.starttls(); s.login(SMTP_USER, SMTP_PASS)
            s.sendmail(SMTP_USER, to_addr, msg.as_string())
        return f"✅ Email sent to {to_addr}"
    except Exception as e:
        return f"❌ Failed: {e}"

# ══════════════════════════════════════════════════════════════════════════════
# UPLOAD CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

def cb_upload(files):
    if not files:
        return "No files selected.", db_stats()
    msgs = []
    for f in files:
        ok, msg = process_file(f.name)
        msgs.append(msg)
    rebuild_bm25()
    return "\n".join(msgs), db_stats()

def cb_ingest_folder(folder: str, limit: int):
    folder = folder.strip()
    if not folder:
        return "Please enter a folder path.", db_stats()
    if not Path(folder).is_dir():
        return f"❌ Folder not found:\n{folder}", db_stats()
    log_lines: list[str] = []
    process_dataset(folder, limit=int(limit),
                    progress_cb=lambda m: log_lines.append(m))
    return "\n".join(log_lines), db_stats()

def db_stats() -> str:
    n = count_chunks()
    d = len(processed_files())
    return f"📦 {n:,} chunks  |  {d:,} CVs indexed"

# ══════════════════════════════════════════════════════════════════════════════
# FIT SCORE GAUGE  (analog light-meter style SVG dial)
# ══════════════════════════════════════════════════════════════════════════════

QUALITY_STYLE = {
    "strong":  ("#2F6F4E", "STRONG MATCH"),
    "good":    ("#9A6B12", "GOOD MATCH"),
    "partial": ("#6B7177", "PARTIAL"),
}

def _gauge_svg(pct: int, color: str) -> str:
    """Semi-circular instrument dial (0-100 sweeping left-to-right) for the fit score."""
    cx, cy, r = 60, 56, 46
    theta = math.radians(180 - 1.8 * pct)
    end_x, end_y = cx + r * math.cos(theta), cy - r * math.sin(theta)
    nx, ny = cx + 37 * math.cos(theta), cy - 37 * math.sin(theta)

    ticks = ""
    for t in (0, 25, 50, 75, 100):
        a = math.radians(180 - 1.8 * t)
        x1, y1 = cx + 46 * math.cos(a), cy - 46 * math.sin(a)
        x2, y2 = cx + 38 * math.cos(a), cy - 38 * math.sin(a)
        ticks += f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#B8BBB3" stroke-width="2"/>'

    return (
        f'<svg width="108" height="58" viewBox="0 0 120 62" style="flex-shrink:0">'
        f'<path d="M {cx-r},{cy} A {r},{r} 0 0,1 {cx+r},{cy}" fill="none" stroke="#DCDDD5" stroke-width="8" stroke-linecap="round"/>'
        f'<path d="M {cx-r},{cy} A {r},{r} 0 0,1 {end_x:.1f},{end_y:.1f}" fill="none" stroke="{color}" stroke-width="8" stroke-linecap="round"/>'
        f'{ticks}'
        f'<line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}" stroke="#1C1F1B" stroke-width="2.4" stroke-linecap="round"/>'
        f'<circle cx="{cx}" cy="{cy}" r="4.5" fill="#1C1F1B"/>'
        f'</svg>'
    )

# ══════════════════════════════════════════════════════════════════════════════
# CANDIDATE CARD  (rich HTML)
# ══════════════════════════════════════════════════════════════════════════════

def _candidate_card(rank: int, r: dict) -> str:
    m        = r["metadata"]
    name     = m.get("name") or Path(m.get("file", "—")).stem
    email    = m.get("email", "")
    phone    = m.get("phone", "")
    linkedin = m.get("linkedin", "")
    file_nm  = m.get("file", "—")
    quality  = r.get("match_quality", "partial")
    fit_pct  = r.get("fit_pct", 0)
    years    = r.get("years_exp", 0)
    hits     = r.get("keyword_hits", 0)
    sections = list(dict.fromkeys(r.get("sections_found", [])))[:5]

    color, q_label = QUALITY_STYLE.get(quality, QUALITY_STYLE["partial"])
    gauge = _gauge_svg(fit_pct, color)

    email_link = (f'<a href="mailto:{email}">{email}</a>') if email else "—"
    li_link    = (f'<a href="https://{linkedin}" target="_blank">LinkedIn ↗</a>') if linkedin else "—"

    section_pills = "".join(
        f'<span class="chip chip-outline">{s}</span>' for s in sections if s
    )
    badges = ""
    if years:
        badges += f'<span class="chip chip-filled">{years}y exp</span>'
    if hits:
        badges += f'<span class="chip chip-filled">{hits} keyword hits</span>'

    preview = (
        " · ".join(r.get("all_chunks", [r["text"]]))[:480]
        .replace("\n", " ")
        .encode("ascii", errors="ignore").decode("ascii")
        .strip()
    )

    return f"""
<div class="frame-card">
  <div class="corner-bl"></div><div class="corner-br"></div>

  <!-- Header row -->
  <div class="frame-head">
    <div style="flex:1;min-width:0">
      <span class="frame-index">FR.{rank:02d}</span>
      <div class="frame-name" style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{name}</div>
      <div class="frame-file">{file_nm}</div>
    </div>

    <!-- Fit gauge -->
    <div class="gauge-col">
      {gauge}
      <div class="gauge-pct" style="color:{color}">{fit_pct}%</div>
      <div class="gauge-label">fit score</div>
      <div class="quality-tag" style="color:{color}">
        <span class="quality-dot" style="background:{color}"></span>{q_label}
      </div>
    </div>
  </div>

  <!-- Contact row -->
  <div class="contact-grid">
    <div>
      <div class="contact-label">Email</div>
      <div class="contact-value" style="overflow:hidden;text-overflow:ellipsis">{email_link}</div>
    </div>
    <div>
      <div class="contact-label">Phone</div>
      <div class="contact-value">{phone or "—"}</div>
    </div>
    <div>
      <div class="contact-label">LinkedIn</div>
      <div class="contact-value">{li_link}</div>
    </div>
  </div>

  <!-- Badges + sections -->
  <div style="margin-top:4px">
    {badges}{section_pills}
  </div>

  <!-- Preview -->
  <div class="scan-log">
    {preview}
  </div>
</div>
"""

# ══════════════════════════════════════════════════════════════════════════════
# SEARCH CALLBACK  (now uses smart_search with Groq query rewriting)
# ══════════════════════════════════════════════════════════════════════════════

def cb_search(query: str, section: str, top_k: int, use_reranker: bool, min_fit_pct: int):
    global _last_results
    query = query.strip()
    if not query:
        return (
            "<div class='empty-state'>Enter a search query above to find candidates.</div>",
            "",
        )

    sec = section if section != "Any" else None
    results, rq = smart_search(
        query,
        top_k=int(top_k),
        section_filter=sec,
        use_reranker=use_reranker,
        min_fit_pct=int(min_fit_pct),
    )
    _last_results = results

    # Show what Groq actually searched (only visible when rewrite ran)
    rewrite_note = ""
    if rq.rewritten:
        pills = "".join(
            f"<span class='chip chip-outline'>{s}</span>"
            for s in rq.must_have_skills
        )
        rewrite_note = (
            f"<div class='rewrite-note'>"
            f"<b>Searched:</b> {rq.primary}"
            + (f"<br><b>Must-have skills:</b> {pills}" if pills else "")
            + (f"<br><b>Min experience:</b> {rq.min_years_exp} years" if rq.min_years_exp else "")
            + "</div>"
        )

    if not results:
        return (
            rewrite_note +
            f"<div class='empty-state'>No candidates found above {int(min_fit_pct)}% fit.</div>",
            "",
        )

    cards   = "".join(_candidate_card(i + 1, r) for i, r in enumerate(results))
    top_fit = results[0].get("fit_pct", 0)
    status  = (
        f"<span class='status-line'>"
        f"Found <strong>{len(results)}</strong> candidate(s) for "
        f"<em>\"{query}\"</em> — top fit: <strong>{top_fit}%</strong>"
        + (" · <em>query expanded by Groq</em>" if rq.rewritten else "")
        + "</span>"
    )
    return rewrite_note + cards, status

# ══════════════════════════════════════════════════════════════════════════════
# EMAIL CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

def cb_load_candidates():
    if not _last_results:
        return gr.update(choices=[], value=None)
    choices = []
    for r in _last_results:
        m     = r["metadata"]
        name  = m.get("name") or m.get("file", "—")
        email = m.get("email", "")
        pct   = r.get("fit_pct", 0)
        label = f"[{pct}%] {name}  <{email}>" if email else f"[{pct}%] {name}"
        choices.append((label, email))
    return gr.update(choices=choices, value=choices[0][1] if choices else None)

def cb_send_one(email_addr: str, subject: str, body: str) -> str:
    if not email_addr:
        return "❌ No candidate selected."
    name = next(
        (r["metadata"].get("name", "") for r in _last_results
         if r["metadata"].get("email") == email_addr),
        "",
    )
    return send_email(email_addr, name, subject, body)

def cb_send_all(subject: str, body: str) -> str:
    if not _last_results:
        return "❌ Run a search first."
    return "\n".join(
        send_email(
            r["metadata"].get("email", ""),
            r["metadata"].get("name", "Candidate"),
            subject, body,
        )
        for r in _last_results
    )

# ══════════════════════════════════════════════════════════════════════════════
# DEFAULT EMAIL TEMPLATE
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_SUBJECT = "Interview Invitation — We'd Love to Connect"
DEFAULT_BODY = """\
Dear {name},

We have reviewed your CV and are genuinely impressed by your background and experience.

We would like to invite you for an interview for a role that closely matches your profile.

Please reply with your availability, or book a time directly at:
[CALENDAR LINK]

We look forward to speaking with you.

Best regards,
The HR Team"""

# ══════════════════════════════════════════════════════════════════════════════
# THEME  — "light table" identity: warm instrument-panel neutrals + signal green
# ══════════════════════════════════════════════════════════════════════════════

THEME = gr.themes.Default(
    font=[gr.themes.GoogleFont("Space Grotesk"), "ui-sans-serif", "system-ui"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace"],
).set(
    body_background_fill="#EDEEEA",
    body_background_fill_dark="#EDEEEA",
    body_text_color="#1C1F1B",
    body_text_color_dark="#1C1F1B",
    block_background_fill="#F7F7F4",
    block_background_fill_dark="#F7F7F4",
    block_border_color="#1C1F1B",
    block_border_color_dark="#1C1F1B",
    block_label_text_color="#5C6259",
    block_label_text_color_dark="#5C6259",
    input_background_fill="#FFFFFF",
    input_background_fill_dark="#FFFFFF",
    input_border_color="#D9DAD3",
    input_border_color_dark="#D9DAD3",
    border_color_primary="#1C1F1B",
    border_color_primary_dark="#1C1F1B",
    color_accent="#2F6F4E",
    color_accent_soft="#E3EFE7",
    color_accent_soft_dark="#E3EFE7",
    slider_color="#2F6F4E",
    slider_color_dark="#2F6F4E",
    checkbox_background_color_selected="#2F6F4E",
    checkbox_background_color_selected_dark="#2F6F4E",
    checkbox_border_color_selected="#2F6F4E",
    checkbox_border_color_selected_dark="#2F6F4E",
    button_primary_background_fill="#C2410C",
    button_primary_background_fill_hover="#9A330A",
    button_primary_background_fill_dark="#C2410C",
    button_primary_text_color="#FFFFFF",
    button_primary_text_color_dark="#FFFFFF",
    button_secondary_background_fill="#1C1F1B",
    button_secondary_background_fill_dark="#1C1F1B",
    button_secondary_text_color="#FFFFFF",
    button_secondary_text_color_dark="#FFFFFF",
    button_cancel_background_fill="#9A1F1F",
    button_cancel_background_fill_dark="#9A1F1F",
    button_cancel_text_color="#FFFFFF",
    button_cancel_text_color_dark="#FFFFFF",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');


.gradio-container, .gradio-container * {
  --canvas:#EDEEEA; --panel-2:#FFFFFF;
  --ink:#1C1F1B; --ink-soft:#5C6259; --line:#D9DAD3;
  --signal:#2F6F4E; --signal-soft:#E3EFE7;
  --mute:#6B7177; --mute-soft:#E7E8E5;
  --spotlight:#C2410C;
}

/* ── Header strip ───────────────────────────────────────────────────────── */
#header-strip {
  background:var(--ink); color:#F4F4F0; border-radius:14px;
  padding:22px 28px 20px; margin-bottom:4px; position:relative; overflow:hidden;
}
#header-strip::before, #header-strip::after {
  content:""; position:absolute; left:0; right:0; height:7px;
  background-image:repeating-linear-gradient(90deg, rgba(244,244,240,.35) 0 5px, transparent 5px 13px);
}
#header-strip::before { top:0; }
#header-strip::after  { bottom:0; }
#eyebrow     { font-family:'JetBrains Mono',monospace; font-size:.72em; letter-spacing:.14em;
               text-transform:uppercase; color:#A9AFA4; margin:0 0 6px; }
#title-text  { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.7em; margin:0; }
#subtitle-text { font-family:'Inter',sans-serif; font-size:.93em; color:#C7CBC2; margin:8px 0 0; max-width:680px; }

/* ── Tabs styled as an instrument panel ─────────────────────────────────── */
.tabs > .tab-nav button, button[role="tab"] {
  font-family:'JetBrains Mono',monospace !important; font-size:.78em !important;
  letter-spacing:.05em; text-transform:uppercase;
}
.tabs > .tab-nav button.selected, button[role="tab"][aria-selected="true"] {
  color:var(--signal) !important; border-color:var(--signal) !important;
}

.log-box textarea { font-family:'JetBrains Mono',monospace !important; font-size:.82em; }
#results-html { max-height:760px; overflow-y:auto; padding-right:6px; }

/* ── Status line / rewrite note / empty states ──────────────────────────── */
.status-line  { font-family:'JetBrains Mono',monospace; font-size:.85em; color:var(--signal); font-weight:600; }
.rewrite-note { font-size:12px; color:var(--ink-soft); margin-bottom:12px; padding:8px 12px;
                background:var(--mute-soft); border-radius:8px; }
.empty-state  { text-align:center; padding:40px; color:var(--ink-soft); font-family:'Inter',sans-serif; }

/* ── Candidate frame card ───────────────────────────────────────────────── */
.frame-card {
  position:relative; background:var(--panel-2); border:1px solid var(--line);
  border-radius:6px; padding:18px 20px 16px; margin:0 8px 20px;
}
.frame-card::before, .frame-card::after, .frame-card .corner-bl, .frame-card .corner-br {
  content:""; position:absolute; width:13px; height:13px; border:2px solid var(--ink); opacity:.85;
}
.frame-card::before    { top:-6px; left:-6px;  border-right:none; border-bottom:none; }
.frame-card::after     { top:-6px; right:-6px; border-left:none;  border-bottom:none; }
.frame-card .corner-bl { bottom:-6px; left:-6px;  border-right:none; border-top:none; }
.frame-card .corner-br { bottom:-6px; right:-6px; border-left:none;  border-top:none; }

.frame-head  { display:flex; align-items:flex-start; gap:14px; }
.frame-index { font-family:'JetBrains Mono',monospace; font-size:.72em; color:var(--mute);
               letter-spacing:.06em; display:block; margin-bottom:2px; }
.frame-name  { font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:1.08em; color:var(--ink); }
.frame-file  { font-family:'JetBrains Mono',monospace; font-size:.72em; color:var(--ink-soft); margin-top:2px; }

.gauge-col   { display:flex; flex-direction:column; align-items:center; margin-left:auto; flex-shrink:0; }
.gauge-pct   { font-family:'JetBrains Mono',monospace; font-weight:700; font-size:1.3em; margin-top:-14px; }
.gauge-label { font-family:'JetBrains Mono',monospace; font-size:.62em; letter-spacing:.12em;
               color:var(--mute); text-transform:uppercase; margin-top:1px; }

.quality-tag { font-family:'JetBrains Mono',monospace; font-size:.7em; letter-spacing:.1em;
               text-transform:uppercase; font-weight:700; display:flex; align-items:center; gap:6px; margin-top:6px; }
.quality-dot { width:8px; height:8px; border-radius:50%; display:inline-block; }

.contact-grid  { display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; margin-top:14px;
                  padding-top:12px; border-top:1px solid var(--line); }
.contact-label { font-family:'JetBrains Mono',monospace; font-size:.65em; letter-spacing:.08em;
                  text-transform:uppercase; color:var(--mute); margin-bottom:3px; }
.contact-value { font-size:.86em; color:var(--ink); }
.contact-value a { color:var(--signal); text-decoration:none; }

.chip { font-family:'JetBrains Mono',monospace; font-size:.68em; letter-spacing:.02em; padding:2px 8px;
        border-radius:3px; margin-right:5px; display:inline-block; margin-top:6px; }
.chip-outline { border:1px solid var(--line); color:var(--ink-soft); background:transparent; }
.chip-filled  { background:var(--signal-soft); color:var(--signal); font-weight:600; }

.scan-log {
  margin-top:12px; padding:10px 14px; background:var(--canvas); border-left:3px solid var(--ink);
  font-family:'JetBrains Mono',monospace; font-size:.74em; color:var(--ink-soft); line-height:1.6;
  border-radius:0 6px 6px 0;
}

/* ── Agent chat tab ─────────────────────────────────────────────────────── */
#agent-chatbot { border:1px solid var(--line) !important; border-radius:8px !important; }
.chat-input-row textarea { font-family:'Inter',sans-serif !important; }
.agent-thinking { font-family:'JetBrains Mono',monospace; font-size:.82em; color:var(--mute);
                  padding:10px 0; animation: blink 1s step-end infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }
.chat-tip { font-family:'JetBrains Mono',monospace; font-size:.75em; color:var(--mute);
            padding:6px 12px; border:1px solid var(--line); border-radius:4px;
            cursor:pointer; display:inline-block; margin:4px 4px 0 0; }
.chat-tip:hover { border-color:var(--signal); color:var(--signal); }
"""

# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

with gr.Blocks(title="HR Assistant — CV RAG") as demo:

    gr.HTML(
        "<div id='header-strip'>"
        "<div id='eyebrow'>AI-powered candidate retrieval</div>"
        "<div id='title-text'>HR Assistant — CV Search</div>"
        "<div id='subtitle-text'>Index your CV dataset once. Search with natural language. "
        "Each result shows a fit score, match quality, years of experience, and contact info.</div>"
        "</div>"
    )

    with gr.Tabs():

        # ── Tab 1: Search ─────────────────────────────────────────────────────
        with gr.Tab("🔍 Search Candidates"):
            gr.Markdown("### Search by skills, role, experience, or any requirement")
            gr.Markdown(
                "Natural language works best — e.g. *'Senior Python developer Django PostgreSQL'* "
                "or *'junior data analyst Excel Cairo'* or *'machine learning NLP 3 years'*"
            )
            with gr.Row():
                query_box  = gr.Textbox(
                    label="Search query", scale=5,
                    placeholder="e.g.  Python machine learning engineer with TensorFlow",
                )
                section_dd = gr.Dropdown(
                    label="Section filter", choices=SECTIONS, value="Any", scale=1,
                )
                top_k_sl   = gr.Slider(
                    label="Top K results", minimum=1, maximum=20, value=5, step=1, scale=1,
                )
                rerank_cb  = gr.Checkbox(label="Reranker", value=True, scale=1)
                min_fit_sl = gr.Slider(
                    label="Min fit %", minimum=0, maximum=80,
                    value=MIN_FIT_DEFAULT, step=5, scale=1,
                )

            search_btn    = gr.Button("Search", variant="primary", size="lg")
            search_status = gr.HTML("")
            results_html  = gr.HTML(
                "<div class='empty-state'>"
                "Your search results will appear here.</div>",
                elem_id="results-html",
            )

            search_btn.click(
                cb_search,
                inputs=[query_box, section_dd, top_k_sl, rerank_cb, min_fit_sl],
                outputs=[results_html, search_status],
            )
            query_box.submit(
                cb_search,
                inputs=[query_box, section_dd, top_k_sl, rerank_cb, min_fit_sl],
                outputs=[results_html, search_status],
            )

        # ── Tab 2: AI Assistant (Agent 1) ─────────────────────────────────────
        with gr.Tab("🤖 AI Assistant"):

            gr.HTML(
                "<div style='font-family:JetBrains Mono,monospace;font-size:.75em;"
                "color:#5C6259;padding:8px 0 4px;letter-spacing:.04em'>"
                "AGENT 1 — JOB DESCRIPTION ANALYZER &amp; SEARCH</div>"
                "<p style='font-size:.92em;color:#1C1F1B;margin:0 0 14px'>"
                "Chat naturally. Describe the role you're hiring for and the agent "
                "will extract requirements, search the CV database, and return "
                "matching candidates directly in this window."
                "</p>"
            )

            # Quick-start example prompts
            gr.HTML(
                "<div style='margin-bottom:14px'>"
                "<span style='font-family:monospace;font-size:.72em;color:#6B7177;"
                "letter-spacing:.06em;text-transform:uppercase'>Try asking:</span><br>"
                "<span class='chat-tip' onclick=\"document.querySelector('#chat-input textarea').value="
                "'I need a senior Python developer with FastAPI and PostgreSQL, 3+ years experience';"
                "document.querySelector('#chat-input textarea').dispatchEvent(new Event('input'))\">"
                "Senior Python dev, FastAPI, 3+ years</span>"
                "<span class='chat-tip' onclick=\"document.querySelector('#chat-input textarea').value="
                "'Find me AI/ML engineers with NLP or LLM experience';"
                "document.querySelector('#chat-input textarea').dispatchEvent(new Event('input'))\">"
                "AI/ML engineer with LLM experience</span>"
                "<span class='chat-tip' onclick=\"document.querySelector('#chat-input textarea').value="
                "'Junior data analyst with Excel and Power BI, fresh graduate is fine';"
                "document.querySelector('#chat-input textarea').dispatchEvent(new Event('input'))\">"
                "Junior data analyst, fresh grad OK</span>"
                "</div>"
            )

            chatbot = gr.Chatbot(
                value=[],
                render_markdown=True,
                sanitize_html=False,
                height=620,
                elem_id="agent-chatbot",
                label="",
                show_label=False,
            )

            with gr.Row(elem_classes=["chat-input-row"]):
                chat_input = gr.Textbox(
                    placeholder=(
                        "E.g. 'I need a senior Python dev with FastAPI, 3+ years, based in Cairo'"
                    ),
                    show_label=False,
                    scale=8,
                    elem_id="chat-input",
                    lines=1,
                    max_lines=4,
                )
                send_btn = gr.Button("Send ↵", variant="primary", scale=1, min_width=80)

            with gr.Row():
                clear_btn = gr.Button(
                    "Clear conversation", variant="secondary", size="sm", scale=1
                )
                gr.HTML(
                    "<div style='font-family:monospace;font-size:.72em;color:#6B7177;"
                    "padding:6px 0;flex:4'>Powered by Groq LLaMA 3.3-70b · "
                    "Results pulled from your indexed CV database</div>"
                )

            # State: Groq-format history (plain text, no HTML)
            llm_hist = gr.State([])

            def _search_adapter(query: str, top_k: int, min_fit_pct: int):
                """Thin wrapper so agent1 doesn't need to know about section_filter / reranker."""
                return smart_search(
                    query,
                    top_k=top_k,
                    section_filter=None,
                    use_reranker=True,
                    min_fit_pct=min_fit_pct,
                )

            def on_send(user_msg: str, history: list, llm_history: list):
                if not user_msg.strip():
                    yield history, llm_history, ""
                    return

                # Gradio 6.x Chatbot uses messages dict format:
                # [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
                history = history + [
                    {"role": "user",      "content": user_msg},
                    {"role": "assistant", "content":
                        "<span style='font-family:monospace;font-size:.82em;color:#6B7177'>"
                        "⏳ Analyzing requirements and searching candidates…</span>"},
                ]
                yield history, llm_history, ""

                # Run the agent (blocking LLM + search)
                reply_html, updated_llm_history = run_agent_turn(
                    user_message=user_msg,
                    llm_history=llm_history,
                    search_fn=_search_adapter,
                )

                # Dicts are mutable — update in place
                history[-1]["content"] = reply_html
                yield history, updated_llm_history, ""

            send_btn.click(
                on_send,
                inputs=[chat_input, chatbot, llm_hist],
                outputs=[chatbot, llm_hist, chat_input],
            )
            chat_input.submit(
                on_send,
                inputs=[chat_input, chatbot, llm_hist],
                outputs=[chatbot, llm_hist, chat_input],
            )
            clear_btn.click(
                lambda: ([], [], ""),
                outputs=[chatbot, llm_hist, chat_input],
            )

        # ── Tab 3: Upload & Ingest ────────────────────────────────────────────
        with gr.Tab("📁 Upload & Ingest"):
            gr.Markdown(
                "**Run the dataset once** — already-indexed files are automatically skipped "
                "on every subsequent run. Use the CLI for large datasets."
            )
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("### Ingest a full folder of CVs")
                    folder_box = gr.Textbox(
                        label="Dataset folder path",
                        placeholder=r"D:\Data\Resumes",
                    )
                    limit_sl   = gr.Slider(
                        label="Limit (0 = all files)", minimum=0, maximum=2000,
                        value=0, step=10,
                    )
                    ingest_btn = gr.Button("Start ingestion", variant="primary")
                    ingest_log = gr.Textbox(
                        label="Ingestion log", lines=22, interactive=False,
                        elem_classes=["log-box"],
                    )

                with gr.Column(scale=2):
                    gr.Markdown("### Upload individual CVs")
                    upload_box = gr.File(
                        label="Drop PDF files here", file_types=[".pdf"],
                        file_count="multiple",
                    )
                    upload_btn = gr.Button("Index selected files", variant="secondary")
                    upload_log = gr.Textbox(label="Result", lines=6, interactive=False)
                    gr.Markdown("---")
                    stats_out  = gr.Textbox(label="Database status", interactive=False)
                    refresh_btn = gr.Button("Refresh stats")

            ingest_btn.click(
                cb_ingest_folder,
                inputs=[folder_box, limit_sl],
                outputs=[ingest_log, stats_out],
            )
            upload_btn.click(cb_upload, inputs=upload_box, outputs=[upload_log, stats_out])
            refresh_btn.click(db_stats, outputs=stats_out)
            demo.load(db_stats, outputs=stats_out)

        # ── Tab 3: Email ──────────────────────────────────────────────────────
        with gr.Tab("✉️ Send Emails"):
            gr.HTML("<h3 style='color:#1C1F1B;font-family:Space Grotesk,sans-serif;margin:0 0 6px'>Send interview invitations from your last search</h3>")
            gr.HTML("<p style='color:#5C6259;font-family:monospace;font-size:.85em;margin:0 0 14px'>Use <code style='background:#E7E8E5;padding:2px 6px;border-radius:3px;color:#1C1F1B'>{name}</code> in the body for personalisation.</p>") 
            load_btn     = gr.Button("Load candidates from last search")
            candidate_dd = gr.Dropdown(label="Select candidate", choices=[], interactive=True)
            subject_box  = gr.Textbox(label="Subject", value=DEFAULT_SUBJECT)
            body_box     = gr.Textbox(label="Email body", value=DEFAULT_BODY, lines=14)
            with gr.Row():
                send_one_btn = gr.Button("Send to selected",    variant="primary")
                send_all_btn = gr.Button("Send to ALL results", variant="stop")
            email_out = gr.Textbox(label="Status", lines=8, interactive=False)

            load_btn.click(cb_load_candidates, outputs=candidate_dd)
            send_one_btn.click(
                cb_send_one,
                inputs=[candidate_dd, subject_box, body_box],
                outputs=email_out,
            )
            send_all_btn.click(
                cb_send_all,
                inputs=[subject_box, body_box],
                outputs=email_out,
            )

        # ── Tab 4: Setup guide ────────────────────────────────────────────────
        with gr.Tab("⚙️ Setup"):
            gr.Markdown("""
## Quick-start guide

### 1 — Install dependencies
```bash
pip install gradio sentence-transformers chromadb rank-bm25 pymupdf pillow groq
```

### 2 — Set your Groq API key (free at https://console.groq.com)
```bash
set GROQ_API_KEY=gsk_...        # Windows
export GROQ_API_KEY=gsk_...     # Mac / Linux
```

### 3 — Configure Tesseract (only needed for scanned PDFs)
Download from: https://github.com/UB-Mannheim/tesseract/wiki  
Then set `TESSERACT_CMD` in `pipeline.py` to your install path.

### 4 — Index your dataset (run once, resumes on interrupt)
```bash
# From a local folder
python pipeline.py --dataset "D:/path/to/CVs"

# With a file limit (great for testing)
python pipeline.py --dataset "D:/path/to/CVs" --limit 100

# From Kaggle directly
python pipeline.py --kaggle

# Start completely fresh
python pipeline.py --dataset "D:/path/to/CVs" --rebuild
```
Already-indexed files are tracked in `indexed_files.txt` and skipped automatically.

### 5 — Start the web app
```bash
python app.py
# Open: http://localhost:7860
```

### 6 — Email setup (Gmail)
1. Enable 2-step verification on your Google account  
2. Go to Google Account → Security → App Passwords → create one  
3. Set before running:
```bash
set SMTP_USER=you@gmail.com
set SMTP_PASS=your_app_password
python app.py
```

---

### How smart search works
Every query is first sent to **Groq (free)** which expands it:
- Abbreviations expanded: ML → machine learning, k8s → Kubernetes
- Synonyms added: React → React.js / ReactJS
- Must-have skills extracted and used as hard filters
- Min years of experience parsed and enforced

Then 2–3 query variants run through the full hybrid pipeline (BM25 + ChromaDB → RRF → reranker).

### How fit percentage works
Each candidate gets a **0–100% fit score** calculated from:
- **Semantic similarity** — how closely the CV meaning matches your query (via the BGE embedding model)
- **Keyword overlap** — how many of your query words appear in the CV text
- **CrossEncoder reranking** — a second model that reads the query and CV together for a more precise score

The final percentage is mapped through a sigmoid function so scores spread naturally across the range.

| Badge | Fit % | Meaning |
|---|---|---|
| Strong match | ≥ 70% | High semantic + keyword match |
| Good match   | 45–69% | Moderate relevance |
| Partial      | < 45%  | Low match, shown for completeness |
""")

if __name__ == "__main__":
    demo.launch(server_name="localhost", server_port=7860, share=False,
                theme=THEME, css=CSS)
