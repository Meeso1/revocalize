from typing import Optional
import numpy as np
import faiss  # type: ignore


class IndexCreator:
    def __init__(self, dimension: int):
        """
        Creates a FAISS index for content vectors of given dimension.
        """
        self.dim = dimension
        self.index = faiss.IndexFlatL2(dimension)

    def add(self, vectors: np.ndarray):
        assert vectors.ndim == 2 and vectors.shape[1] == self.dim, "Vectors shape mismatch."
        if vectors.dtype != np.float32:
            vectors = vectors.astype("float32")
        self.index.add(vectors)

    def save(self, filepath: str):
        faiss.write_index(self.index, filepath)

    def load(self, filepath: str):
        self.index = faiss.read_index(filepath)
        self.dim = self.index.d

    def search(self, query_vectors: np.ndarray, k: int = 1) -> np.ndarray:
        assert query_vectors.shape[1] == self.dim, "Query vector dimension mismatch."
        if query_vectors.dtype != np.float32:
            query_vectors = query_vectors.astype("float32")
        distances, indices = self.index.search(query_vectors, k)
        return indices
