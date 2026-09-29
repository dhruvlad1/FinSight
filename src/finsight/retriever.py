from pathlib import Path

import faiss
import numpy as np
import pandas as pd


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

    def search_filtered(
        self,
        query_embedding: np.ndarray,
        metadata: pd.DataFrame,
        top_k: int = 5,
        ticker: str | None = None,
        quarter: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search only chunks matching the requested metadata filters."""
        mask = np.ones(len(metadata), dtype=bool)

        if ticker is not None:
            mask &= metadata["ticker"].eq(ticker).to_numpy()

        if quarter is not None:
            mask &= metadata["quarter"].eq(quarter).to_numpy()

        candidate_indexes = np.flatnonzero(mask)

        if len(candidate_indexes) == 0:
            return np.array([[]], dtype="float32"), np.array([[]], dtype="int64")

        candidate_embeddings = np.zeros(
            (len(candidate_indexes), self.index.d),
            dtype="float32",
        )

        self.index.reconstruct_batch(
            candidate_indexes.astype("int64"),
            candidate_embeddings,
        )

        scores = query_embedding.astype("float32") @ candidate_embeddings.T

        order = np.argsort(scores[0])[::-1][:top_k]

        return (
            scores[:, order],
            candidate_indexes[order].reshape(1, -1),
        )

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