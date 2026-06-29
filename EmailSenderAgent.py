"""
Email Sender Agent - Handles email generation and sending
===========================================================
This agent generates personalized emails for candidates and sends them.
"""

import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ─── Config ──────────────────────────────────────────────────────────────────
MODEL = "llama-3.3-70b-versatile"

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
HR_NAME   = os.getenv("HR_NAME", "HR Team")

# ─── System prompt for email generation ──────────────────────────────────────
EMAIL_SYSTEM_PROMPT = """You are an AI recruitment assistant that generates professional interview invitation emails.

Given a candidate's information and the HR's request, generate a formal, personalized email.

Rules:
- Use a professional and warm tone
- Reference specific details from the candidate's background if available
- Keep the email concise (150-200 words)
- Include a clear call to action
- Use {name} as a placeholder for the candidate's name
- The email should feel personal, not generic

Return ONLY a JSON object with:
{
    "subject": "email subject line",
    "body": "full email body with {name} placeholder"
}
"""


# ─── Email sending function ────────────────────────────────────────────────────

def send_email(to_addr: str, name: str, subject: str, body: str) -> str:
    """
    Actually send the email via SMTP.
    Returns status message.
    """
    if not SMTP_USER or not SMTP_PASS:
        return "❌ Email not configured. Set SMTP_USER and SMTP_PASS in .env file."
    if not to_addr:
        return "❌ No email address provided."

    try:
        # Fill the {name} placeholder
        filled_body = body.replace("{name}", name or "Candidate")

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"{HR_NAME} <{SMTP_USER}>"
        msg["To"] = to_addr
        msg.attach(MIMEText(filled_body, "plain"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
            s.ehlo()
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.sendmail(SMTP_USER, to_addr, msg.as_string())

        return f"✅ Email sent successfully to {to_addr}"

    except Exception as e:
        return f"❌ Failed to send email: {str(e)}"


# ─── Email generation function ─────────────────────────────────────────────────

def generate_and_send_email(
    user_message: str,
    candidate_info: dict = None,
) -> tuple[str, str, str]:
    """
    Generate a personalized email and send it.

    Parameters
    ----------
    user_message : str
        The HR's request (e.g., "send an email to john@email.com")
    candidate_info : dict
        Optional candidate information dict with name, email, etc.

    Returns
    -------
    subject : str
        The email subject line
    body : str
        The generated email body
    status : str
        Status message (success or error)
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "", "", "❌ GROQ_API_KEY not set in .env file."

    client = Groq(api_key=api_key)

    # Extract email address from user message if not provided
    to_email = None
    name = "Candidate"

    if candidate_info:
        to_email = candidate_info.get("email")
        name = candidate_info.get("name", "Candidate")

    # If no candidate info, try to extract email from the message
    if not to_email:
        import re
        email_match = re.search(r'[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}', user_message)
        if email_match:
            to_email = email_match.group(0)

    if not to_email:
        return "", "", "❌ No email address found. Please specify the candidate's email."

    try:
        # Generate email using Groq
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": EMAIL_SYSTEM_PROMPT},
                {"role": "user", "content": f"""
HR Request: {user_message}

Candidate Name: {name}
Candidate Email: {to_email}

Generate a professional interview invitation email for this candidate.
Use {name} as the placeholder for their name.
                """}
            ],
            temperature=0.3,
            max_tokens=500,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        data = json.loads(raw)
        subject = data.get("subject", "Interview Invitation")
        body = data.get("body", "Dear {name},\n\nWe would like to invite you for an interview.")

        # Send the email
        status = send_email(to_email, name, subject, body)

        return subject, body, status

    except Exception as e:
        return "", "", f"❌ Error generating/sending email: {str(e)}"


# ─── Format email preview for display ─────────────────────────────────────────

def format_email_preview(subject: str, body: str, to_email: str, name: str) -> str:
    """Format the email as a preview HTML card."""
    if not body:
        return "<div style='color:red'>No email content to preview.</div>"

    filled_body = body.replace("{name}", name or "Candidate")
    safe_body = (filled_body
                 .replace("&", "&amp;")
                 .replace("<", "&lt;")
                 .replace(">", "&gt;")
                 .replace("\n", "<br>"))

    return f"""
<div style="background:#FFFFFF;border:1px solid #D9DAD3;border-radius:6px;
            padding:16px 20px;margin:10px 0;font-family:system-ui,sans-serif">
    <div style="font-family:monospace;font-size:.7em;letter-spacing:.1em;
                text-transform:uppercase;color:#6B7177;margin-bottom:10px">
        📧 EMAIL PREVIEW
    </div>
    <div style="display:grid;grid-template-columns:auto 1fr;gap:4px 12px;
                margin-bottom:12px;padding-bottom:10px;border-bottom:1px solid #D9DAD3">
        <div style="font-family:monospace;font-size:.72em;color:#6B7177;">To</div>
        <div style="font-size:.9em;color:#1C1F1B;font-weight:500">{to_email}</div>
        <div style="font-family:monospace;font-size:.72em;color:#6B7177;">Subject</div>
        <div style="font-size:.9em;color:#1C1F1B;font-weight:600">{subject}</div>
    </div>
    <div style="font-size:.88em;color:#1C1F1B;line-height:1.8;
                padding:12px 16px;background:#F7F7F4;border-radius:4px;
                border-left:3px solid #2F6F4E">
        {safe_body}
    </div>
</div>
"""