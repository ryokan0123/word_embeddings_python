from .skip_gram import SkipGram
from .functions import sigmoid
from .negative_sampler import NegativeSampler
from .data_processing.vocabulary import Vocabulary


class SkipGramWithNegativeSampling(SkipGram):

    def __init__(self,
                 vocab: Vocabulary,
                 vec_size: int,
                 negative_sampler: NegativeSampler = None):
        super().__init__(vocab=vocab,
                         vec_size=vec_size)
        self.negative_sampler = negative_sampler

    def fit(self, w: int, c: int, alpha: float):
        """
        Perform fitting for a given word-context pair
        """

        w_vec = self.word_vectors[w]
        c_vec = self.context_vectors[c]
        dot_score = w_vec.dot(c_vec)

        grad = alpha * (1 - sigmoid(dot_score))

        self.word_vectors[w] += grad * c_vec
        self.context_vectors[c] += grad * w_vec

        # perform negative sampling
        for neg_c in self.negative_sampler.sample():
            neg_c_vec = self.context_vectors[neg_c]
            neg_dot_score = -w_vec.dot(neg_c_vec)

            neg_grad = alpha * (1 - sigmoid(neg_dot_score))

            self.word_vectors[w] -= neg_grad * neg_c_vec
            self.context_vectors[neg_c] -= neg_grad * w_vec
