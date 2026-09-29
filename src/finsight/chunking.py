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
    """Create globally unique chunks while preserving transcript metadata."""
    chunks = []
    next_chunk_id = 0

    for _, row in df.iterrows():
        text_chunks = chunk_text(row["text"])

        for chunk in text_chunks:
            item = row.to_dict()
            item["text"] = chunk
            item["chunk_id"] = next_chunk_id
            chunks.append(item)
            next_chunk_id += 1

    return pd.DataFrame(chunks)
