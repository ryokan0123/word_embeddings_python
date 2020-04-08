from typing import List, Dict
import random


def extract_word_context_pair(tokens: List,
                              window_size: int):
    for target_idx in range(len(tokens)):
        start = max(0, target_idx - window_size)
        end = min(len(tokens), target_idx + window_size + 1)
        context_words = tokens[start:target_idx] + tokens[target_idx + 1:end]
        for c in context_words:
            yield tokens[target_idx], c


class SubSampling:

    def __init__(self,
                 counter: Dict[str, int],
                 sampling_rate: float = 0.001):
        total_count = sum(counter.values())
        self.sampling_rate_vocab = {w: ((sampling_rate * total_count / c) ** 0.5) for w, c in counter.items()}

    def is_valid_token(self, word: str):
        if self.sampling_rate_vocab[word] < random.random():
            return False
        return True
