"""
HR Assistant — CV RAG System
Run:  python app.py
Open: http://localhost:7860
"""

import os, smtplib
from pathlib import Path
from email.mime.text      import MIMEText
from email.mime.multipart import MIMEMultipart

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import gradio as gr
from pipeline import (
    process_file, process_dataset,
    rebuild_bm25, count_chunks, processed_files, search,
)

# ══════════════════════════════════════════════════════════════════════════════
# EMAIL CONFIG
# ══════════════════════════════════════════════════════════════════════════════

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
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
# FIT PERCENTAGE RING  (SVG donut)
# ══════════════════════════════════════════════════════════════════════════════

def _fit_ring(pct: int, quality: str) -> str:
    """Return a small inline SVG donut showing the fit percentage."""
    color = {"strong": "#16a34a", "good": "#d97706", "partial": "#3b82f6"}.get(quality, "#64748b")
    r     = 18
    circ  = 2 * 3.14159 * r
    dash  = circ * pct / 100
    return (
        f'<svg width="52" height="52" viewBox="0 0 52 52" style="flex-shrink:0">'
        f'<circle cx="26" cy="26" r="{r}" fill="none" stroke="#e2e8f0" stroke-width="5"/>'
        f'<circle cx="26" cy="26" r="{r}" fill="none" stroke="{color}" stroke-width="5"'
        f' stroke-dasharray="{dash:.1f} {circ:.1f}"'
        f' stroke-linecap="round" transform="rotate(-90 26 26)"/>'
        f'<text x="26" y="30" text-anchor="middle" font-size="11" font-weight="600"'
        f' font-family="sans-serif" fill="{color}">{pct}%</text>'
        f'</svg>'
    )

# ══════════════════════════════════════════════════════════════════════════════
# CANDIDATE CARD  (rich HTML)
# ══════════════════════════════════════════════════════════════════════════════

QUALITY_STYLE = {
    "strong":  ("background:#dcfce7;color:#166534", "Strong match"),
    "good":    ("background:#fef9c3;color:#92400e", "Good match"),
    "partial": ("background:#dbeafe;color:#1e40af", "Partial match"),
}

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

    q_style, q_label = QUALITY_STYLE.get(quality, QUALITY_STYLE["partial"])

    ring       = _fit_ring(fit_pct, quality)
    email_link = (f'<a href="mailto:{email}" style="color:#3b82f6;text-decoration:none">'
                  f'{email}</a>') if email else "—"
    li_link    = (f'<a href="https://{linkedin}" target="_blank" '
                  f'style="color:#0077b5;text-decoration:none">LinkedIn ↗</a>') if linkedin else "—"

    section_pills = "".join(
        f'<span style="background:#f1f5f9;color:#475569;padding:2px 9px;'
        f'border-radius:12px;font-size:11px;margin-right:4px;white-space:nowrap">{s}</span>'
        for s in sections if s
    )
    badges = ""
    if years:
        badges += (f'<span style="background:#f0fdf4;color:#166534;padding:2px 9px;'
                   f'border-radius:12px;font-size:11px;margin-right:4px">{years}y exp</span>')
    if hits:
        badges += (f'<span style="background:#eff6ff;color:#1d4ed8;padding:2px 9px;'
                   f'border-radius:12px;font-size:11px">{hits} keyword hits</span>')

    preview = (
        " · ".join(r.get("all_chunks", [r["text"]]))[:480]
        .replace("\n", " ")
        .encode("ascii", errors="ignore").decode("ascii")
        .strip()
    )

    return f"""
<div style="border:1px solid #e2e8f0;border-radius:14px;padding:18px 20px;
            margin-bottom:14px;background:#ffffff;font-family:system-ui,sans-serif;
            box-shadow:0 1px 4px rgba(0,0,0,.05)">

  <!-- Header row -->
  <div style="display:flex;align-items:center;gap:14px">
    <!-- Rank badge -->
    <div style="width:38px;height:38px;border-radius:50%;background:#6366f1;
                display:flex;align-items:center;justify-content:center;
                color:#fff;font-weight:700;font-size:15px;flex-shrink:0">
      {rank}
    </div>

    <!-- Name + file -->
    <div style="flex:1;min-width:0">
      <div style="font-size:16px;font-weight:600;color:#0f172a;
                  white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{name}</div>
      <div style="font-size:12px;color:#94a3b8;margin-top:2px">{file_nm}</div>
    </div>

    <!-- Fit ring -->
    <div style="display:flex;flex-direction:column;align-items:center;gap:2px">
      {ring}
      <span style="font-size:10px;color:#94a3b8">fit score</span>
    </div>

    <!-- Quality badge -->
    <span style="{q_style};padding:4px 12px;border-radius:20px;
                 font-size:12px;font-weight:600;white-space:nowrap;flex-shrink:0">
      {q_label}
    </span>
  </div>

  <!-- Contact row -->
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;
              margin-top:14px;padding-top:12px;border-top:1px solid #f1f5f9">
    <div>
      <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;
                  letter-spacing:.4px;margin-bottom:3px">Email</div>
      <div style="font-size:13px;overflow:hidden;text-overflow:ellipsis">{email_link}</div>
    </div>
    <div>
      <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;
                  letter-spacing:.4px;margin-bottom:3px">Phone</div>
      <div style="font-size:13px;color:#0f172a">{phone or "—"}</div>
    </div>
    <div>
      <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;
                  letter-spacing:.4px;margin-bottom:3px">LinkedIn</div>
      <div style="font-size:13px">{li_link}</div>
    </div>
  </div>

  <!-- Badges + sections -->
  <div style="margin-top:10px;display:flex;flex-wrap:wrap;gap:4px">
    {badges}{section_pills}
  </div>

  <!-- Preview -->
  <div style="margin-top:12px;padding:10px 14px;background:#f8fafc;border-radius:8px;
              font-size:12px;color:#64748b;line-height:1.65">
    {preview}
  </div>
</div>
"""

# ══════════════════════════════════════════════════════════════════════════════
# SEARCH CALLBACK
# ══════════════════════════════════════════════════════════════════════════════

def cb_search(query: str, section: str, top_k: int, use_reranker: bool):
    global _last_results
    query = query.strip()
    if not query:
        return (
            "<div style='text-align:center;padding:40px;color:#94a3b8;font-family:sans-serif'>"
            "Enter a search query above to find candidates.</div>",
            "",
        )

    sec     = section if section != "Any" else None
    results = search(query, top_k=int(top_k), section_filter=sec,
                     use_reranker=use_reranker)
    _last_results = results

    if not results:
        return (
            "<div style='text-align:center;padding:40px;color:#94a3b8;font-family:sans-serif'>"
            "No candidates found for this query.</div>",
            "",
        )

    cards  = "".join(_candidate_card(i + 1, r) for i, r in enumerate(results))
    top_fit = results[0].get("fit_pct", 0)
    status = (
        f"<span style='color:#166534;font-weight:500'>"
        f"Found <strong>{len(results)}</strong> candidate(s) for "
        f"<em>\"{query}\"</em> — top fit: <strong>{top_fit}%</strong></span>"
    )
    return cards, status

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
# CSS
# ══════════════════════════════════════════════════════════════════════════════

CSS = """
#title    { text-align:center; margin-bottom:.2em; }
#subtitle { text-align:center; color:#64748b; font-size:.93em; margin-bottom:1.4em; }
.log-box  textarea { font-family:monospace; font-size:.82em; }
#results-html { max-height:720px; overflow-y:auto; padding-right:4px; }
"""

# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

with gr.Blocks(title="HR Assistant — CV RAG", css=CSS) as demo:

    gr.Markdown("# HR Assistant — CV Search", elem_id="title")
    gr.Markdown(
        "Index your CV dataset once. Search with natural language. "
        "Each result shows a **fit percentage**, match quality, years of experience, and contact info.",
        elem_id="subtitle",
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

            search_btn    = gr.Button("Search", variant="primary", size="lg")
            search_status = gr.HTML("")
            results_html  = gr.HTML(
                "<div style='text-align:center;padding:48px;color:#94a3b8;font-family:sans-serif'>"
                "Your search results will appear here.</div>",
                elem_id="results-html",
            )

            search_btn.click(
                cb_search,
                inputs=[query_box, section_dd, top_k_sl, rerank_cb],
                outputs=[results_html, search_status],
            )
            query_box.submit(
                cb_search,
                inputs=[query_box, section_dd, top_k_sl, rerank_cb],
                outputs=[results_html, search_status],
            )

        # ── Tab 2: Upload & Ingest ────────────────────────────────────────────
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
            gr.Markdown("### Send interview invitations from your last search")
            gr.Markdown("Use `{name}` in the body for personalisation.")
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
pip install gradio sentence-transformers chromadb rank-bm25 pymupdf pillow
```

### 2 — Configure Tesseract (only needed for scanned PDFs)
Download from: https://github.com/UB-Mannheim/tesseract/wiki  
Then set `TESSERACT_CMD` in `pipeline.py` to your install path.

### 3 — Index your dataset (run once, resumes on interrupt)
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

### 4 — Start the web app
```bash
python app.py
# Open: http://localhost:7860
```

### 5 — Email setup (Gmail)
1. Enable 2-step verification on your Google account  
2. Go to Google Account → Security → App Passwords → create one  
3. Set before running:
```bash
set SMTP_USER=you@gmail.com
set SMTP_PASS=your_app_password
python app.py
```

---

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
    demo.launch(server_name="localhost", server_port=7860, share=False)