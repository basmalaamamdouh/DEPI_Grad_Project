# Add this near the top of orchestrator.py after the imports
from EmailSenderAgent import generate_and_send_email, format_email_preview

# Add these trigger words
_EMAIL_TRIGGERS = [
    "send email", "send an email", "email to", "mail to",
    "invite", "interview invitation", "contact", "reach out",
    "send to", "email them", "email the candidate",
]

def _is_email_request(message: str) -> bool:
    """Return True if the message is asking to send an email."""
    msg_lower = message.lower().strip()
    return any(trigger in msg_lower for trigger in _EMAIL_TRIGGERS)

def _extract_email_from_message(message: str) -> str | None:
    """Extract email address from message."""
    import re
    email_match = re.search(r'[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}', message)
    if email_match:
        return email_match.group(0)
    return None

# Update run_conversational_turn function in orchestrator.py
# Add this as the first check after the reasoning/compare checks:

def run_conversational_turn(
    user_message: str,
    llm_history: list,
    last_candidates: list = None,
    thread_id: str = "default",
    top_k: int = 5,
    min_fit_pct: int = 10,
) -> tuple[str, list, list]:
    # Cast to int defensively
    top_k = int(top_k)
    min_fit_pct = int(min_fit_pct)

    # ── NEW: Check for email request ───────────────────────────────────────────
    if _is_email_request(user_message):
        print(f"[Orchestrator] Email request detected")

        # Try to find the candidate in last_results by email
        to_email = _extract_email_from_message(user_message)
        candidate = None

        if to_email and last_candidates:
            for c in last_candidates:
                if c["metadata"].get("email", "").lower() == to_email.lower():
                    candidate = c
                    break

        # If we found the candidate, use their info
        if candidate:
            subject, body, status = generate_and_send_email(
                user_message,
                candidate_info={
                    "name": candidate["metadata"].get("name", "Candidate"),
                    "email": candidate["metadata"].get("email", ""),
                }
            )
        else:
            # Try to send without candidate info (using just email from message)
            subject, body, status = generate_and_send_email(user_message)

        # Build the reply
        if status.startswith("✅"):
            # Success - show preview and confirmation
            preview = format_email_preview(subject, body, to_email or "Candidate", "Candidate")
            reply_html = f"""
<div style='font-family:system-ui,sans-serif;color:#1C1F1B;line-height:1.75'>
    <div style='padding:12px 16px;background:#E3EFE7;border:1px solid #2F6F4E;
                border-radius:6px;margin-bottom:12px'>
        ✅ Email sent successfully!
    </div>
    {preview}
</div>
"""
        else:
            reply_html = f"""
<div style='font-family:system-ui,sans-serif;color:#1C1F1B;line-height:1.75'>
    <div style='padding:12px 16px;background:#FBE6DB;border:1px solid #C2410C;
                border-radius:6px;'>
        ❌ {status}
    </div>
</div>
"""

        updated_history = llm_history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": status},
        ]
        return reply_html, updated_history, last_candidates

    # ── Case 1: Follow-up reasoning about specific candidate(s) ─────────────
    if _is_reasoning_request(user_message, last_candidates or []):
        # ... existing reasoning code ...
        pass

    # ── Case 2: Compare / filter within already-retrieved results ────────────
    if _is_compare_request(user_message, last_candidates or []):
        # ... existing compare code ...
        pass

    # ── Case 3: "more results" / pagination request ───────────────────────────
    # ... existing more results code ...
    pass

    # ── Case 4: Brand-new search query ───────────────────────────────────────
    # ... existing search code ...
    pass