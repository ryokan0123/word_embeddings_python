from typing import List
from pathlib import Path
from .functions import softmax
import numpy as np

from .data_processing.vocabulary import Vocabulary


class SkipGram:

    def __init__(self,
                 vocab: Vocabulary,
                 vec_size: int = 300):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.vec_size = vec_size

        self.word_vectors = (np.random.randn(self.vocab_size, self.vec_size) - 0.5) / self.vec_size
        self.context_vectors = (np.random.randn(self.vocab_size, self.vec_size) - 0.5) / self.vec_size

        self._normalized_word_vectors_cache = None

    def fit(self, w: int, c: int, alpha: float):
        """
        Perform fitting for a given word-context pair
        """

        w_vec = self.word_vectors[w]
        c_vec = self.context_vectors[c]
        dot_scores = self.context_vectors.dot(w_vec)
        softmax_scores = softmax(dot_scores)

        # compute derivative and perform gradient ascent
        self.word_vectors[w] += alpha * (1 - softmax_scores[c]) * c_vec

        # the true derivative for self.context_vectors[c] should be `(1 - softmax_scores[c]) * w_vec`,
        # but we merge the term `- softmax_scores[c]) * w_vec` in the following computation.
        self.context_vectors[c] += alpha * w_vec
        self.context_vectors -= alpha * np.outer(softmax_scores, w_vec)

    def get_cosine_neighbours(self, word: str, top_k: int = 10) -> List[str]:
        if self._normalized_word_vectors_cache is None:
            self._normalized_word_vectors_cache = self.word_vectors / \
                                                  np.linalg.norm(self.word_vectors, axis=1)[:, np.newaxis]

        word_idx = self.vocab.token2idx[word]
        similarity = np.dot(self.word_vectors[word_idx], self._normalized_word_vectors_cache.T)
        neighbors_idx = np.argsort(-similarity)[:top_k]
        return [self.vocab.get_token(idx) for idx in neighbors_idx]

    def save(self, save_path: str):
        save_path = Path(save_path)
        save_path.mkdir(parents=True)
        np.save(save_path / "wv.npy", self.word_vectors)
        np.save(save_path / "cv.npy", self.context_vectors)
        self.vocab.save(save_path / "vocab.txt")

    @classmethod
    def load(cls, save_path: str):
        save_path = Path(save_path)
        word_vectors = np.load(save_path / "wv.npy")
        context_vectors = np.load(save_path / "cv.npy")
        vocab = Vocabulary.load(save_path / "vocab.txt")
        vec_size = word_vectors.shape[1]

        self = cls(vocab, vec_size)
        self.word_vectors = word_vectors
        self.context_vectors = context_vectors

        return self
