from pathlib import Path

import faiss
import numpy as np


class FaissRetriever:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatIP(dimension)

    def add_embeddings(self, embeddings: np.ndarray) -> None:
        """Add normalized embeddings to the FAISS index."""
        self.index.add(embeddings.astype("float32"))

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return similarity scores and indexes for the top matches."""
        scores, indexes = self.index.search(
            query_embedding.astype("float32"),
            top_k,
        )
        return scores, indexes

    def save(self, path: str | Path) -> None:
        """Save the FAISS index locally."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "FaissRetriever":
        """Load a previously saved FAISS index."""
        index = faiss.read_index(str(path))
        retriever = cls(index.d)
        retriever.index = index
        return retriever
