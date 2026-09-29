def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Calculate Recall@K for a single query."""
    if not relevant:
        return 0.0

    retrieved_at_k = set(retrieved[:k])
    return len(retrieved_at_k & relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """Calculate reciprocal rank for a single query."""
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(results: list[tuple[list[str], set[str]]]) -> float:
    """Calculate MRR across multiple queries."""
    if not results:
        return 0.0

    scores = [
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in results
    ]

    return sum(scores) / len(scores)
