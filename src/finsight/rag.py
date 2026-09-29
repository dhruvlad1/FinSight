import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


SYSTEM_PROMPT = """You are FinSight, a financial research assistant.

Answer questions using only the provided earnings-call transcript context.
Do not invent facts or use information outside the supplied context.

If the context does not contain enough evidence to answer the question,
say that the available transcript evidence is insufficient.

Cite the relevant source chunks using their chunk identifiers.
"""


def create_groq_client() -> Groq:
    """Create a Groq client using the environment API key."""
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is not set.")

    return Groq(api_key=api_key)


def generate_answer(
    question: str,
    context: str,
    model: str = "openai/gpt-oss-120b",
) -> str:
    """Generate a grounded answer from retrieved transcript context."""
    client = create_groq_client()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Transcript context:\n\n{context}\n\n"
                    f"Question:\n{question}"
                ),
            },
        ],
        temperature=0,
    )

    return response.choices[0].message.content or ""
