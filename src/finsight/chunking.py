import pandas as pd


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 120) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()

    if not words:
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))

        if end == len(words):
            break

        start = end - overlap

    return chunks


def create_chunks(df: pd.DataFrame) -> pd.DataFrame:
    """Create retrieval chunks while preserving transcript metadata."""
    records = []

    for _, row in df.iterrows():
        chunks = chunk_text(row["text"])

        for chunk_id, chunk in enumerate(chunks):
            record = row.to_dict()
            record["text"] = chunk
            record["chunk_id"] = chunk_id
            records.append(record)

    return pd.DataFrame(records)
