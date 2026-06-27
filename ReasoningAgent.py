import os
from groq import Groq


def generate_candidate_reasoning(hr_query, candidate_text):
    """
    Agent 2 - Candidate Reasoning Agent
    """

    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        return "Groq API Key not found."

    client = Groq(api_key=api_key)

    system_prompt = """
You are Agent 2 in an AI Recruitment System.

Your job is to explain why this candidate matches the HR request.

Rules:
- Use ONLY information explicitly written in the CV.
- Never invent skills or experience.
- Never guess.
- If something is missing, write "Not mentioned in the CV."

Return exactly:

Why Selected:
...

Evidence Found:
• Experience:
...
• Skills:
...
• Projects:
...
• Education:
...

Key Strengths:
• ...
• ...

Gap(s):
...
"""

    try:

        response = client.chat.completions.create(

            model="llama-3.3-70b-versatile",

            temperature=0,

            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": f"""
HR Query:
{hr_query}

Candidate CV:

{candidate_text}
""",
                },
            ],
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Reasoning Agent Error: {str(e)}"
