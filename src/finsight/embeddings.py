from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model() -> SentenceTransformer:
    """Load the local sentence-transformer embedding model."""
    return SentenceTransformer(MODEL_NAME)


def generate_embeddings(
    texts: list[str],
    model: SentenceTransformer,
) -> np.ndarray:
    """Generate normalized embeddings for transcript chunks."""
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings.astype("float32")


def save_embeddings(embeddings: np.ndarray, path: str | Path) -> None:
    """Save embeddings locally as a NumPy array."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)
