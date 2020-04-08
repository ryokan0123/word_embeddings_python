from typing import List, Dict
from collections import Counter


class Vocabulary:
    unk = '@UNKNOWN@'

    def __init__(self):
        self.token2idx = {}
        self.idx2token = []
        self.counter = Counter()

    def get_idx(self, token: str) -> int:
        if token not in self.token2idx:
            return 0
        else:
            return self.token2idx[token]

    def get_token(self, idx: int) -> str:
        return self.idx2token[idx]

    @classmethod
    def from_tokens_list(cls, tokens_list: List[List[str]], min_count: int = 1):
        self = cls()
        for tokens in tokens_list:
            self.counter.update(tokens)
        self.idx2token = [cls.unk] + [token for token, freq in self.counter.items() if freq >= min_count]
        self.token2idx = {token: idx for idx, token in enumerate(self.idx2token)}
        return self

    def save(self, save_path: str):
        with open(save_path, 'w') as f:
            for token in self.idx2token:
                f.write(f'{token}\n')

    @classmethod
    def load(cls, save_path: str):
        self = cls()
        with open(save_path, 'r') as f:
            idx2token = [line.strip() for line in f]
        self.idx2token = idx2token
        self.token2idx = {token: idx for idx, token in enumerate(self.idx2token)}
        return self

    def __len__(self):
        return len(self.idx2token)
