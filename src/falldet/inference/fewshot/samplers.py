from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils.data import Dataset


def retrieve_topk(
    query_embeddings: torch.Tensor, corpus_embeddings: torch.Tensor, k=10
) -> list[dict]:
    """Retrieve top-k results based on cosine similarity.

    Args:
        query_embeddings: Tensor of shape (num_queries, embedding_dim)
        corpus_embeddings: Tensor of shape (num_corpus, embedding_dim)
        k: Number of top results to retrieve per query
    Returns:
        List of dicts with 'ranked_indices' and 'ranked_scores' for each query."""
    similarity_scores = query_embeddings @ corpus_embeddings.T
    results = []
    for i in range(len(query_embeddings)):
        scores = similarity_scores[i].cpu().float().numpy()
        ranked_indices = np.argsort(scores)[::-1][:k]
        ranked_scores = scores[ranked_indices]
        results.append(
            {"ranked_indices": ranked_indices.tolist(), "ranked_scores": ranked_scores.tolist()}
        )
    return results


class ExemplarSampler(ABC):
    def __init__(self, corpus: Dataset, num_shots: int = 5):
        self.corpus = corpus

        if num_shots <= 0:
            raise ValueError("num_shots must be positive.")
        self.num_shots = num_shots

    @abstractmethod
    def sample(self) -> list[int]:
        """Get exemplar indices based on the given video if available."""


class RandomSampler(ExemplarSampler):
    def __init__(self, corpus: Dataset, num_shots: int = 5, seed: int = 0):
        super().__init__(corpus, num_shots)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def sample(self) -> list[int]:
        """Randomly sample exemplar indices from train set."""
        total_indices = list(range(len(self.corpus)))  # ty:ignore[invalid-argument-type]
        sampled_indices = self.rng.choice(total_indices, size=self.num_shots, replace=False)
        return sampled_indices.tolist()


class BalancedRandomSampler(RandomSampler):
    def __init__(self, corpus, num_shots: int = 5, seed: int = 0):
        super().__init__(corpus, num_shots, seed)

    def sample(self):
        raise NotImplementedError(
            "BalancedRandomSampler is not implemented yet. It requires class labels to sample balanced exemplars across classes."
        )


class SimilaritySampler(ExemplarSampler):
    def __init__(self, corpus, num_shots: int = 5, query_embeddings=None, corpus_embeddings=None):
        super().__init__(corpus, num_shots)
        self.query_embeddings = query_embeddings
        self.corpus_embeddings = corpus_embeddings

    def sample(self, video: torch.Tensor | None = None) -> list[int]:
        raise NotImplementedError(
            "SimilaritySampler is not implemented yet. It requires query and corpus embeddings to retrieve similar exemplars."
        )
