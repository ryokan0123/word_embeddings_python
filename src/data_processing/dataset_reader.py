from typing import List


class DatasetReader:

    @staticmethod
    def tokenize(sentence: str) -> List[str]:
        return sentence.split()

    def read(self, file_path: str) -> List[List[str]]:
        sentence_list = []
        with open(file_path, 'r') as f:
            for line in f:
                tokenized = self.tokenize(line.strip())
                sentence_list.append(tokenized)
        return sentence_list
