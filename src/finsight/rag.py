import os

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from .embeddings import generate_embeddings, load_embedding_model
from .retriever import FaissRetriever

load_dotenv()

SYSTEM_PROMPT = """You are a financial research assistant.

Answer ONLY using facts explicitly stated in the provided transcript context.

Strict grounding rules:
1. Do not use outside knowledge or information from your training data.
2. Do not infer a number that is not explicitly present in the context.
3. Do not invent citations, speakers, quarters, or source references.
4. If the context does not contain enough evidence to answer the question, say:
   "The retrieved transcript context does not contain enough evidence to answer this question."
5. When answering with a specific number, make sure that exact number appears in the provided context.
6. If the question asks about a specific quarter, do not substitute annual or another-quarter information.

Keep the answer concise and factual.
"""

EMBEDDING_MODEL = load_embedding_model()


def retrieve_context(
    question: str,
    chunks: pd.DataFrame,
    retriever: FaissRetriever,
    top_k: int = 5,
    ticker: str | None = None,
    quarter: int | None = None,
) -> tuple[str, pd.DataFrame]:
    """Retrieve transcript chunks and format them as LLM context."""
    query_embedding = generate_embeddings(
        [question],
        EMBEDDING_MODEL,
    )

    scores, indexes = retriever.search_filtered(
        query_embedding,
        chunks,
        top_k=top_k,
        ticker=ticker,
        quarter=quarter,
    )

    if len(indexes[0]) == 0:
        return "", chunks.iloc[0:0].copy()

    retrieved = chunks.iloc[indexes[0]].copy()
    retrieved["score"] = scores[0]

    context_parts = []

    for _, row in retrieved.iterrows():
        context_parts.append(
            f"[{row['ticker']} {row['quarter']} {row['year']} | "
            f"{row['speaker']}]\n{row['text']}"
        )

    return "\n\n".join(context_parts), retrieved


def generate_answer(
    question: str,
    context: str,
    sources: pd.DataFrame | None = None,
    model: str = "openai/gpt-oss-120b",
) -> dict:
    """Generate a grounded answer with source evidence."""
    if not context.strip():
        return {
            "answer": "I could not find sufficient transcript evidence to answer this question.",
            "sources": [],
        }

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is not configured.")

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question: {question}\n\nContext:\n{context}",
            },
        ],
        temperature=0,
    )

    source_list = []

    if sources is not None:
        for _, row in sources.iterrows():
            source_list.append(
                {
                    "ticker": row["ticker"],
                    "company": row["company"],
                    "quarter": row["quarter"],
                    "year": row["year"],
                    "speaker": row["speaker"],
                    "chunk_id": row["chunk_id"],
                    "score": float(row["score"]),
                    "text": row["text"],
                }
            )

    return {
        "answer": response.choices[0].message.content,
        "sources": source_list,
    }

def format_sources(sources: list[dict]) -> list[str]:
    """Format retrieved chunks for display in the UI."""
    formatted = []

    for source in sources:
        formatted.append(
            f"{source['ticker']} — Q{source['quarter']} {source['year']} "
            f"— {source['speaker']} "
            f"(score: {source['score']:.3f})"
        )

    return formatted