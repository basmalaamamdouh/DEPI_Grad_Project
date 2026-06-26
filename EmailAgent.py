"""
Agent 3 — Personalized Email Agent
====================================
Drafts and sends a fully personalized interview invitation email
for a specific candidate, informed by their actual CV content.

Flow (one turn):
  HR message  (e.g. "send an email to sara@example.com")
    → Extract target email from HR message
    → Match candidate in _last_results (or do a fresh CV lookup)
    → Groq LLaMA 3.3 reads (CV text + HR requirements) → drafts email
    → HR can approve / edit → send via SMTP
    → Returns (reply_html, updated_llm_history, draft_subject, draft_body)

Usage from app.py:
    from EmailAgent import run_email_agent_turn, send_personalized_email

Key design decisions:
- The email is generated from the candidate's ACTUAL CV text, not a template.
- HR can review/edit the draft before sending — no surprise sends.
- The agent remembers the conversation so HR can say
  "make it more casual" or "add our calendar link" in follow-ups.
"""

import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from groq import Groq

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL = "llama-3.3-70b-versatile"

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
HR_NAME   = os.getenv("HR_NAME", "HR Team")

# ─── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an AI recruitment assistant helping HR professionals write personalized outreach emails to candidates.

YOUR BEHAVIOR:
- You are given the candidate's CV text and the HR's job requirements.
- Write a genuinely personalized email that references specific details from the candidate's background.
- Do NOT write generic templates. Mention actual skills, projects, or experience from their CV.
- Keep the tone professional but warm — like a real person reaching out, not a robot.
- The email should feel like it was written specifically for THIS candidate.
- Use {name} as a placeholder for the candidate's name (it will be filled automatically).
- Keep the email concise: a short intro, 2-3 sentences about why they specifically stand out, and a clear call to action.
- End with a professional sign-off.

OUTPUT FORMAT:
Return ONLY a JSON object with exactly two keys:
{
  "subject": "the email subject line",
  "body": "the full email body (plain text, use \\n for line breaks)"
}

Do not include any explanation, preamble, or markdown — just the raw JSON object.
"""

# ─── Tool definition ──────────────────────────────────────────────────────────
DRAFT_TOOL = {
    "type": "function",
    "function": {
        "name": "draft_email",
        "description": (
            "Draft a personalized email for a specific candidate based on their CV and the HR's requirements. "
            "Call this when the HR asks to send or draft an email to a candidate."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target_email": {
                    "type": "string",
                    "description": "The candidate's email address to send to.",
                },
                "job_context": {
                    "type": "string",
                    "description": (
                        "The role/requirements the HR is hiring for. "
                        "Extract from the current conversation context."
                    ),
                },
            },
            "required": ["target_email"],
        },
    },
}

SEND_TOOL = {
    "type": "function",
    "function": {
        "name": "confirm_send",
        "description": (
            "Actually send the email after HR approves the draft. "
            "Only call this if the HR explicitly says 'send it', 'looks good', 'yes', or approves."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "confirmed": {
                    "type": "boolean",
                    "description": "Whether the HR has confirmed they want to send.",
                }
            },
            "required": ["confirmed"],
        },
    },
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _find_candidate(email_addr: str, last_results: list[dict]) -> dict | None:
    """Find a candidate in last_results by email address (case-insensitive)."""
    email_addr = email_addr.strip().lower()
    for r in last_results:
        candidate_email = r["metadata"].get("email", "").strip().lower()
        if candidate_email == email_addr:
            return r
    return None


def _extract_email_from_text(text: str) -> str | None:
    """Simple regex-free email extraction — finds first token with @ and a dot after."""
    for token in text.split():
        token = token.strip(".,;:!?\"'()")
        if "@" in token and "." in token.split("@")[-1]:
            return token
    return None


def _candidate_cv_summary(candidate: dict) -> str:
    """Build a concise CV summary to feed into the email drafting prompt."""
    m = candidate["metadata"]
    name     = m.get("name", "the candidate")
    email    = m.get("email", "")
    phone    = m.get("phone", "")
    linkedin = m.get("linkedin", "")
    fit_pct  = candidate.get("fit_pct", 0)
    quality  = candidate.get("match_quality", "partial")
    years    = candidate.get("years_exp", 0)
    sections = candidate.get("sections_found", [])

    # Full CV text is most valuable
    cv_text = candidate.get("text", "")
    # If there are separate chunks, join them (richer context)
    chunks = candidate.get("all_chunks", [])
    if chunks:
        cv_text = "\n\n".join(chunks)

    summary = (
        f"CANDIDATE PROFILE\n"
        f"Name: {name}\n"
        f"Email: {email}\n"
        f"Phone: {phone or 'not provided'}\n"
        f"LinkedIn: {linkedin or 'not provided'}\n"
        f"Match quality: {quality} ({fit_pct}% fit)\n"
        f"Years of experience detected: {years}\n"
        f"CV sections found: {', '.join(sections) if sections else 'general'}\n\n"
        f"CV TEXT:\n{cv_text[:3000]}"  # Cap at 3000 chars to stay within token budget
    )
    return summary


def _draft_email_with_groq(
    candidate: dict,
    job_context: str,
    extra_instructions: str = "",
    conversation_history: list | None = None,
) -> tuple[str, str]:
    """
    Use Groq to draft a personalized email.
    Returns (subject, body).
    """
    api_key = os.environ.get("GROQ_API_KEY", "")

    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    client = Groq(api_key=api_key)

    cv_summary = _candidate_cv_summary(candidate)
    name = candidate["metadata"].get("name", "the candidate")

    user_content = (
        f"Write a personalized interview invitation email for this candidate.\n\n"
        f"JOB / ROLE CONTEXT:\n{job_context or 'A relevant role at our company.'}\n\n"
        f"{cv_summary}\n\n"
        f"Remember:\n"
        f"- Use {{name}} as the placeholder for the candidate's name (it will be auto-filled).\n"
        f"- Reference specific things from their CV — skills, experience, projects.\n"
        f"- Do NOT just say 'we reviewed your CV and are impressed' — be specific.\n"
        f"- Sign off as: {HR_NAME}\n"
        + (f"\nAdditional instructions from HR: {extra_instructions}" if extra_instructions else "")
    )

    # Build messages — include conversation history for follow-up edits
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_content})

    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.4,
        max_tokens=800,
    )
    raw = resp.choices[0].message.content or "{}"

    # Strip markdown fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        parsed = json.loads(raw)
        subject = parsed.get("subject", f"Interview Invitation — {name}")
        body    = parsed.get("body", raw)
    except json.JSONDecodeError:
        # Fallback: use raw text as body
        subject = f"Interview Invitation — {name}"
        body    = raw

    return subject, body


def _do_send(to_addr: str, name: str, subject: str, body: str) -> str:
    """Send the email via SMTP. Returns status message."""
    if not SMTP_USER or not SMTP_PASS:
        return "❌ Email not configured — set SMTP_USER and SMTP_PASS environment variables."
    if not to_addr:
        return "❌ No email address provided."
    try:
        filled_body = body.replace("{name}", name or "Candidate")
        msg            = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = f"{HR_NAME} <{SMTP_USER}>"
        msg["To"]      = to_addr
        msg.attach(MIMEText(filled_body, "plain"))
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
            s.ehlo()
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.sendmail(SMTP_USER, to_addr, msg.as_string())
        return f"✅ Email sent to {to_addr}"
    except Exception as e:
        return f"❌ Send failed: {e}"


# ─── Card-style HTML builders ─────────────────────────────────────────────────

def _draft_preview_html(subject: str, body: str, to_addr: str, name: str) -> str:
    """Render the email draft as a styled preview card."""
    filled_body = body.replace("{name}", name or "Candidate")
    # Escape HTML special chars and convert newlines
    safe_body = (
        filled_body
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )
    return f"""
<div style="background:#FFFFFF;border:1px solid #D9DAD3;border-radius:6px;
            padding:20px 22px;margin:10px 0;font-family:system-ui,sans-serif;
            position:relative">

  <!-- Corner marks -->
  <span style="position:absolute;width:11px;height:11px;border:2px solid #1C1F1B;
               top:-5px;left:-5px;border-right:none;border-bottom:none"></span>
  <span style="position:absolute;width:11px;height:11px;border:2px solid #1C1F1B;
               top:-5px;right:-5px;border-left:none;border-bottom:none"></span>
  <span style="position:absolute;width:11px;height:11px;border:2px solid #1C1F1B;
               bottom:-5px;left:-5px;border-right:none;border-top:none"></span>
  <span style="position:absolute;width:11px;height:11px;border:2px solid #1C1F1B;
               bottom:-5px;right:-5px;border-left:none;border-top:none"></span>

  <!-- Header -->
  <div style="font-family:monospace;font-size:.7em;letter-spacing:.1em;
              text-transform:uppercase;color:#6B7177;margin-bottom:12px">
    EMAIL DRAFT — AGENT 3
  </div>

  <!-- Email metadata -->
  <div style="display:grid;grid-template-columns:auto 1fr;gap:4px 12px;
              margin-bottom:14px;padding-bottom:12px;border-bottom:1px solid #D9DAD3">
    <div style="font-family:monospace;font-size:.72em;color:#6B7177;text-transform:uppercase;
                letter-spacing:.06em;padding-top:2px">To</div>
    <div style="font-size:.9em;color:#1C1F1B;font-weight:500">{to_addr}</div>

    <div style="font-family:monospace;font-size:.72em;color:#6B7177;text-transform:uppercase;
                letter-spacing:.06em;padding-top:2px">Subject</div>
    <div style="font-size:.9em;color:#1C1F1B;font-weight:600">{subject}</div>
  </div>

  <!-- Email body -->
  <div style="font-size:.88em;color:#1C1F1B;line-height:1.75;
              padding:12px 14px;background:#F7F7F4;border-radius:4px;
              border-left:3px solid #2F6F4E">
    {safe_body}
  </div>

  <!-- Action hint -->
  <div style="margin-top:12px;font-family:monospace;font-size:.7em;color:#6B7177;
              letter-spacing:.04em">
    💬 Reply <strong>"send it"</strong> to send · or tell me what to change
    (e.g. "make it more casual", "add our Calendly link")
  </div>
</div>
"""


def _sent_confirmation_html(to_addr: str, name: str) -> str:
    return f"""
<div style="background:#E3EFE7;border:1px solid #2F6F4E;border-radius:6px;
            padding:14px 18px;margin:10px 0;font-family:system-ui,sans-serif">
  <div style="font-family:monospace;font-size:.72em;letter-spacing:.1em;
              text-transform:uppercase;color:#2F6F4E;margin-bottom:4px">EMAIL SENT ✓</div>
  <div style="font-size:.92em;color:#1C1F1B">
    Your personalized email to <strong>{name}</strong> (<code>{to_addr}</code>) has been sent.
  </div>
</div>
"""


def _error_html(message: str) -> str:
    return f"""
<div style="background:#FBE6DB;border:1px solid #C2410C;border-radius:6px;
            padding:12px 16px;margin:10px 0;font-family:monospace;font-size:.82em;color:#9A1F1F">
  ⚠ {message}
</div>
"""


# ─── Agent state (per-conversation) ───────────────────────────────────────────
# These are passed in from app.py as part of gr.State — see usage below.

class EmailAgentState:
    """Holds the in-progress draft for a single email conversation."""
    def __init__(self):
        self.target_email: str        = ""
        self.candidate: dict | None   = None
        self.draft_subject: str       = ""
        self.draft_body: str          = ""
        self.job_context: str         = ""
        self.groq_history: list[dict] = []  # drafting sub-conversation

    def has_draft(self) -> bool:
        return bool(self.draft_subject and self.draft_body)

    def to_dict(self) -> dict:
        return {
            "target_email":  self.target_email,
            "draft_subject": self.draft_subject,
            "draft_body":    self.draft_body,
            "job_context":   self.job_context,
            "groq_history":  self.groq_history,
            # candidate is heavy — store only metadata + text
            "candidate_meta": self.candidate["metadata"] if self.candidate else None,
            "candidate_text": self.candidate.get("text", "") if self.candidate else "",
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EmailAgentState":
        obj = cls()
        obj.target_email  = d.get("target_email", "")
        obj.draft_subject = d.get("draft_subject", "")
        obj.draft_body    = d.get("draft_body", "")
        obj.job_context   = d.get("job_context", "")
        obj.groq_history  = d.get("groq_history", [])
        meta = d.get("candidate_meta")
        text = d.get("candidate_text", "")
        if meta:
            obj.candidate = {"metadata": meta, "text": text}
        return obj


# ─── Main agent entry point ────────────────────────────────────────────────────

def run_email_agent_turn(
    user_message: str,
    llm_history: list,        # main chat history (from app.py Agent3 tab)
    last_results: list[dict], # _last_results from app.py
    agent_state: dict,        # EmailAgentState serialized as dict (gr.State)
    job_context: str = "",    # optional: what role is being hired for
) -> tuple[str, list, dict]:
    """
    Process one HR message in the email agent tab.

    Parameters
    ----------
    user_message : str
    llm_history  : list[dict]   Main chat history for the email agent tab.
    last_results : list[dict]   Results from the last search (from _last_results).
    agent_state  : dict         Serialized EmailAgentState (gr.State).
    job_context  : str          Optional role description from Agent 1 context.

    Returns
    -------
    reply_html   : str          HTML to display in the chatbot.
    updated_history : list      Updated main chat history.
    updated_state   : dict      Updated serialized EmailAgentState.
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        err = _error_html("GROQ_API_KEY not set. Add it to your environment and restart.")
        updated = llm_history + [
            {"role": "user",      "content": user_message},
            {"role": "assistant", "content": "GROQ_API_KEY not set."},
        ]
        return err, updated, agent_state

    state = EmailAgentState.from_dict(agent_state) if agent_state else EmailAgentState()
    client = Groq(api_key=api_key)

    msg_lower = user_message.lower().strip()

    # ── 1. HR is approving / confirming to send ────────────────────────────
    SEND_TRIGGERS = {"send it", "yes send", "go ahead", "confirm", "send", "looks good", "approved", "yes"}
    is_approval = any(t in msg_lower for t in SEND_TRIGGERS)

    if is_approval and state.has_draft() and state.candidate:
        name       = state.candidate["metadata"].get("name", "Candidate")
        to_addr    = state.target_email
        send_status = _do_send(to_addr, name, state.draft_subject, state.draft_body)

        if send_status.startswith("✅"):
            reply_html = _sent_confirmation_html(to_addr, name)
            summary    = f"Email sent to {name} ({to_addr})."
        else:
            reply_html = _error_html(send_status)
            summary    = f"Failed to send email: {send_status}"

        updated = llm_history + [
            {"role": "user",      "content": user_message},
            {"role": "assistant", "content": summary},
        ]
        # Reset draft state after send
        new_state = EmailAgentState()
        return reply_html, updated, new_state.to_dict()

    # ── 2. HR wants to revise the existing draft ───────────────────────────
    REVISE_TRIGGERS = {"change", "make it", "rewrite", "revise", "edit", "adjust",
                       "more casual", "more formal", "add", "remove", "shorter", "longer"}
    is_revision = state.has_draft() and any(t in msg_lower for t in REVISE_TRIGGERS)

    if is_revision and state.candidate:
        try:
            # Add the revision instruction to the sub-conversation
            state.groq_history.append({"role": "user", "content": user_message})
            subj, body = _draft_email_with_groq(
                candidate=state.candidate,
                job_context=state.job_context or job_context,
                extra_instructions=user_message,
                conversation_history=state.groq_history,
            )
            state.draft_subject = subj
            state.draft_body    = body
            state.groq_history.append({
                "role": "assistant",
                "content": json.dumps({"subject": subj, "body": body}),
            })

            name      = state.candidate["metadata"].get("name", "Candidate")
            to_addr   = state.target_email
            preview   = _draft_preview_html(subj, body, to_addr, name)
            intro     = (
                f'<div style="font-family:system-ui,sans-serif;color:#1C1F1B;'
                f'margin-bottom:8px;line-height:1.6">'
                f"I've revised the email. Here's the updated draft:"
                f"</div>"
            )
            reply_html = intro + preview
            summary    = f"Draft revised for {name} ({to_addr})."
        except Exception as e:
            reply_html = _error_html(f"Could not revise draft: {e}")
            summary    = f"Error revising draft: {e}"

        updated = llm_history + [
            {"role": "user",      "content": user_message},
            {"role": "assistant", "content": summary},
        ]
        return reply_html, updated, state.to_dict()

    # ── 3. HR is asking to send/draft an email — extract target ───────────
    # Try to extract email address from the message
    target_email = _extract_email_from_text(user_message)

    # If no email in this message, check if we already have a target in state
    if not target_email and state.target_email:
        target_email = state.target_email

    # If still no email, use Groq to understand the intent
    if not target_email:
        # Use Groq to decide what to ask
        groq_msgs = [
            {"role": "system", "content": (
                "You are an HR assistant helping draft recruitment emails. "
                "If the HR wants to send an email to a candidate but hasn't specified the email address, "
                "ask for it concisely. If the request is unclear, ask one focused clarifying question."
            )},
            *llm_history,
            {"role": "user", "content": user_message},
        ]
        resp = client.chat.completions.create(
            model=MODEL, messages=groq_msgs, temperature=0.3, max_tokens=200
        )
        text = resp.choices[0].message.content or "Which candidate's email address should I send to?"
        updated = llm_history + [
            {"role": "user",      "content": user_message},
            {"role": "assistant", "content": text},
        ]
        reply_html = (
            f'<div style="font-family:system-ui,sans-serif;color:#1C1F1B;line-height:1.75">'
            f"{text}</div>"
        )
        return reply_html, updated, state.to_dict()

    # ── 4. We have a target email — find the candidate ─────────────────────
    candidate = _find_candidate(target_email, last_results)

    if not candidate:
        # No match in search results
        not_found_html = (
            f'<div style="font-family:system-ui,sans-serif;color:#1C1F1B;line-height:1.75">'
            f"I couldn't find <strong>{target_email}</strong> in your last search results.<br><br>"
            f"Please run a search first (in the 🔍 Search or 🤖 AI Assistant tab) "
            f"so I have the candidate's CV details to personalize the email."
            f"</div>"
        )
        updated = llm_history + [
            {"role": "user",      "content": user_message},
            {"role": "assistant", "content": f"Candidate {target_email} not found in last results."},
        ]
        return not_found_html, updated, state.to_dict()

    # ── 5. Draft the personalized email ───────────────────────────────────
    state.target_email = target_email
    state.candidate    = candidate
    state.job_context  = job_context

    try:
        subj, body = _draft_email_with_groq(
            candidate=candidate,
            job_context=job_context,
            extra_instructions="",
        )
        state.draft_subject = subj
        state.draft_body    = body
        state.groq_history  = []  # fresh sub-conversation for this draft

        name    = candidate["metadata"].get("name", "Candidate")
        preview = _draft_preview_html(subj, body, target_email, name)
        intro   = (
            f'<div style="font-family:system-ui,sans-serif;color:#1C1F1B;'
            f'margin-bottom:8px;line-height:1.6">'
            f"I drafted a personalized email for <strong>{name}</strong> "
            f"based on their CV. Review it below and say <strong>\"send it\"</strong> "
            f"when you're happy, or tell me what to change."
            f"</div>"
        )
        reply_html = intro + preview
        summary    = f"Draft prepared for {name} ({target_email})."

    except Exception as e:
        reply_html = _error_html(f"Could not draft email: {e}")
        summary    = f"Error drafting email: {e}"

    updated = llm_history + [
        {"role": "user",      "content": user_message},
        {"role": "assistant", "content": summary},
    ]
    return reply_html, updated, state.to_dict()


# ─── Convenience: send without agent (called from existing Send Emails tab) ───

def send_personalized_email(
    candidate: dict,
    job_context: str = "",
    calendar_link: str = "",
) -> tuple[str, str, str]:
    """
    Draft + return a personalized email for a candidate.
    Does NOT send — caller decides whether to send.

    Returns (subject, body, status_message)
    """
    try:
        extra = f"Include this calendar link for scheduling: {calendar_link}" if calendar_link else ""
        subject, body = _draft_email_with_groq(candidate, job_context, extra)
        name = candidate["metadata"].get("name", "Candidate")
        return subject, body, f"✅ Draft ready for {name}"
    except Exception as e:
        return "", "", f"❌ Draft failed: {e}"
