from typing import List
import numpy as np


class NegativeSampler:

    def __init__(self,
                 vocab_size: int,
                 num_samples: int):
        self.vocab_size = vocab_size
        self.num_samples = num_samples

    def sample(self):
        raise NotImplementedError


class UniformNegativeSampler(NegativeSampler):

    def sample(self):
        return [np.random.randint(self.vocab_size) for _ in range(self.num_samples)]


class FreqBasedNegativeSampler(NegativeSampler):

    def __init__(self,
                 vocab_freqs: List[int],
                 num_samples: int = 10,
                 power_factor: float = 0.75,
                 max_table_size: int = int(1e8)):
        super().__init__(vocab_size=len(vocab_freqs), num_samples=num_samples)
        self._unigram_table = self._construct_unigram_table(vocab_freqs,
                                                            power_factor,
                                                            max_table_size)
        self._table_size = self._unigram_table.shape[0]

    def _construct_unigram_table(self,
                                 vocab_freqs: List[int],
                                 power_factor: float,
                                 max_table_size: int) -> np.ndarray:
        vocab_freqs = np.array(vocab_freqs) ** power_factor
        freq_sum = vocab_freqs.sum()
        table_size = min(int(freq_sum), max_table_size)
        # import sys; sys.exit()

        unigram_table = np.zeros(table_size, dtype=int)
        start_idx = 0
        accum_freq = 0
        for idx, freq in enumerate(vocab_freqs):
            accum_freq += freq
            end_idx = int((accum_freq / freq_sum) * table_size)
            unigram_table[start_idx:end_idx] = idx
            start_idx = end_idx
        return unigram_table

    def sample(self) -> List[int]:
        random_indices = np.random.random(size=self.num_samples)
        sampled_token_idx = [self._unigram_table[int(idx * self._table_size)] for idx in random_indices]
        return sampled_token_idx
