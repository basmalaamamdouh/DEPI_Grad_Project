"""
HR Assistant — CV RAG System
Run:  python app.py
Open: http://localhost:7860
"""
from dotenv import load_dotenv
load_dotenv()
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
# EMAIL CONFIG & TEMPLATES
# ══════════════════════════════════════════════════════════════════════════════

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
HR_NAME   = os.getenv("HR_NAME",   "HR Team")

EMAIL_TEMPLATES = {
    "Manual / Custom": "",
    "Schedule Interview": (
        "Dear {name},\n\n"
        "We were impressed with your profile and would like to schedule a brief interview "
        "to discuss your experience further.\n\n"
        "Please let us know your availability for next week.\n\n"
        "Best regards,\n{hr_name}"
    ),
    "Ask Questions": (
        "Dear {name},\n\n"
        "Thank you for your application. We are currently reviewing your profile and "
        "have a few follow-up questions regarding your recent projects.\n\n"
        "Could you please provide more details on your role in these tasks?\n\n"
        "Best regards,\n{hr_name}"
    )
}

SECTIONS = ["Any", "skills", "experience", "education", "projects",
            "summary", "certif", "training", "languages", "volunteer"]

_last_results: list[dict] = []

# ══════════════════════════════════════════════════════════════════════════════
# EMAIL LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def update_template(template_name, recipient_name):
    """Updates the message body when a template is selected."""
    text = EMAIL_TEMPLATES.get(template_name, "")
    if template_name != "Manual / Custom":
        return text.format(name=recipient_name or "Candidate", hr_name=HR_NAME)
    return text

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
        # Replace {name} placeholder if user didn't use the template system
        final_body = body.replace("{name}", name or "Candidate")
        msg.attach(MIMEText(final_body, "plain"))
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
# FIT PERCENTAGE RING (SVG) & CANDIDATE CARD
# ══════════════════════════════════════════════════════════════════════════════

def _fit_ring(pct: int, quality: str) -> str:
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
    section_pills = "".join(f'<span style="background:#f1f5f9;color:#475569;padding:2px 9px;border-radius:12px;font-size:11px;margin-right:4px;white-space:nowrap">{s}</span>' for s in sections if s)
    badges = ""
    if years: badges += (f'<span style="background:#f0fdf4;color:#166534;padding:2px 9px;border-radius:12px;font-size:11px;margin-right:4px">{years}y exp</span>')
    if hits: badges += (f'<span style="background:#eff6ff;color:#1d4ed8;padding:2px 9px;border-radius:12px;font-size:11px">{hits} keyword hits</span>')
    preview = " · ".join(r.get("all_chunks", [r["text"]]))[:480].replace("\n", " ").encode("ascii", errors="ignore").decode("ascii").strip()

    return f"""
<div style="border:1px solid #e2e8f0;border-radius:14px;padding:18px 20px;margin-bottom:14px;background:#ffffff;font-family:system-ui,sans-serif;box-shadow:0 1px 4px rgba(0,0,0,.05)">
  <div style="display:flex;align-items:center;gap:14px">
    <div style="width:38px;height:38px;border-radius:50%;background:#6366f1;display:flex;align-items:center;justify-content:center;color:#fff;font-weight:700;font-size:15px;flex-shrink:0">{rank}</div>
    <div style="flex:1;min-width:0">
      <div style="font-size:16px;font-weight:600;color:#0f172a;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{name}</div>
      <div style="font-size:12px;color:#94a3b8;margin-top:2px">{file_nm}</div>
    </div>
    <div style="display:flex;flex-direction:column;align-items:center;gap:2px">{ring}<span style="font-size:10px;color:#94a3b8">fit score</span></div>
    <span style="{q_style};padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600;white-space:nowrap;flex-shrink:0">{q_label}</span>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-top:14px;padding-top:12px;border-top:1px solid #f1f5f9">
    <div><div style="font-size:10px;color:#94a3b8;text-transform:uppercase;margin-bottom:3px">Email</div><div style="font-size:13px">{email_link}</div></div>
    <div><div style="font-size:10px;color:#94a3b8;text-transform:uppercase;margin-bottom:3px">Phone</div><div style="font-size:13px;color:#0f172a">{phone or "—"}</div></div>
    <div><div style="font-size:10px;color:#94a3b8;text-transform:uppercase;margin-bottom:3px">LinkedIn</div><div style="font-size:13px">{li_link}</div></div>
  </div>
  <div style="margin-top:10px;display:flex;flex-wrap:wrap;gap:4px">{badges}{section_pills}</div>
  <div style="margin-top:12px;padding:10px 14px;background:#f8fafc;border-radius:8px;font-size:12px;color:#64748b;line-height:1.65">{preview}</div>
</div>"""

# ══════════════════════════════════════════════════════════════════════════════
# SEARCH & EMAIL CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

def cb_search(query: str, section: str, top_k: int, use_reranker: bool):
    global _last_results
    query = query.strip()
    if not query: return "<div style='text-align:center;padding:40px;color:#94a3b8'>Enter query...</div>", ""
    sec = section if section != "Any" else None
    results = search(query, top_k=int(top_k), section_filter=sec, use_reranker=use_reranker)
    _last_results = results
    if not results: return "<div style='text-align:center;padding:40px;color:#94a3b8'>No candidates found.</div>", ""
    cards = "".join(_candidate_card(i + 1, r) for i, r in enumerate(results))
    status = f"<span style='color:#166534;font-weight:500'>Found <strong>{len(results)}</strong> candidate(s) for \"{query}\"</span>"
    return cards, status

def cb_load_candidates():
    if not _last_results: return gr.update(choices=[], value=None)
    choices = []
    for r in _last_results:
        m = r["metadata"]
        name = m.get("name") or m.get("file", "—")
        email = m.get("email", "")
        pct = r.get("fit_pct", 0)
        label = f"[{pct}%] {name} <{email}>" if email else f"[{pct}%] {name}"
        choices.append((label, email))
    return gr.update(choices=choices, value=choices[0][1] if choices else None)

def cb_send_manual(to_addr: str, name: str, template: str, subject: str, body: str) -> str:
    if not to_addr or "@" not in to_addr: return "❌ Invalid email."
    return send_email(to_addr, name, subject, body)

# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

CSS = """
#title { text-align:center; margin-bottom:.2em; }
#results-html { max-height:720px; overflow-y:auto; padding-right:4px; }
"""

with gr.Blocks(title="HR Assistant — CV RAG", css=CSS) as demo:
    gr.Markdown("# HR Assistant — CV Search", elem_id="title")

    with gr.Tabs():
        # SEARCH TAB
        with gr.Tab("🔍 Search Candidates"):
            with gr.Row():
                query_box = gr.Textbox(label="Search query", scale=5)
                section_dd = gr.Dropdown(label="Section filter", choices=SECTIONS, value="Any", scale=1)
                top_k_sl = gr.Slider(label="Top K", minimum=1, maximum=20, value=5, step=1, scale=1)
                rerank_cb = gr.Checkbox(label="Reranker", value=True, scale=1)
            search_btn = gr.Button("Search", variant="primary")
            search_status = gr.HTML("")
            results_html = gr.HTML("<div style='text-align:center;padding:48px;color:#94a3b8'>Results...</div>", elem_id="results-html")
            search_btn.click(cb_search, inputs=[query_box, section_dd, top_k_sl, rerank_cb], outputs=[results_html, search_status])

        # UPLOAD TAB
        with gr.Tab("📁 Upload & Ingest"):
            with gr.Row():
                with gr.Column(scale=3):
                    folder_box = gr.Textbox(label="Dataset folder path")
                    limit_sl = gr.Slider(label="Limit", minimum=0, maximum=2000, value=0, step=10)
                    ingest_btn = gr.Button("Start ingestion", variant="primary")
                    ingest_log = gr.Textbox(label="Log", lines=15)
                with gr.Column(scale=2):
                    upload_box = gr.File(label="Drop PDF files", file_count="multiple")
                    upload_btn = gr.Button("Index selected files")
                    stats_out = gr.Textbox(label="Database status")
            ingest_btn.click(cb_ingest_folder, inputs=[folder_box, limit_sl], outputs=[ingest_log, stats_out])
            upload_btn.click(cb_upload, inputs=upload_box, outputs=[stats_out, stats_out])

        # EMAIL TAB (UPGRADED)
        with gr.Tab("✉️ Send Emails"):
            gr.Markdown("### Email Configuration")
            with gr.Row():
                manual_to = gr.Textbox(label="Recipient email", placeholder="candidate@example.com", scale=2)
                manual_name = gr.Textbox(label="Recipient name (optional)", scale=1)
            
            # Template Dropdown
            template_dd = gr.Dropdown(
                label="Choose a template", 
                choices=list(EMAIL_TEMPLATES.keys()), 
                value="Manual / Custom"
            )
            
            manual_subject = gr.Textbox(label="Subject", value="Regarding your application")
            manual_body = gr.Textbox(label="Email body", lines=10, placeholder="Choose a template or write here...")
            
            # Logic: When template or name changes, update body
            template_dd.change(fn=update_template, inputs=[template_dd, manual_name], outputs=manual_body)
            manual_name.change(fn=update_template, inputs=[template_dd, manual_name], outputs=manual_body)

            manual_send_btn = gr.Button("Send email", variant="primary")
            manual_out = gr.Textbox(label="Status", interactive=False)
            
            manual_send_btn.click(
                cb_send_manual,
                inputs=[manual_to, manual_name, template_dd, manual_subject, manual_body],
                outputs=manual_out
            )

if __name__ == "__main__":
    demo.launch(server_name="localhost", server_port=7860, share=False)