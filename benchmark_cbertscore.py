import sys
from unittest.mock import MagicMock

# Mock bert_lib before it's imported by bert_score_lib
mock_bert_lib = MagicMock()
sys.modules['cbertscore.bert_lib'] = mock_bert_lib

import time
import numpy as np
import collections
from cbertscore import bert_score_lib

class MockTokenizer:
    def tokenize(self, text):
        return text.split()

class MockBertModel:
    def __init__(self):
        self.tokenizer = MockTokenizer()

    def get_activations(self, sentences):
        n = len(sentences)
        # Fixed random seed for consistency
        np.random.seed(42)
        tokens = [["[CLS]"] + s.split() + ["[SEP]"] for s in sentences]
        layers = []
        for i in range(n):
            layer_dict = {}
            seq_len = len(tokens[i])
            for ln in range(2): # 2 layers
                layer_dict[ln] = [np.random.rand(768).astype(np.float32) for _ in range(seq_len)]
            layers.append(layer_dict)
        return tokens, layers

def benchmark(label="Default"):
    n_examples = 100
    candidates = ["this is a dummy candidate sentence number " + str(i) for i in range(n_examples)]
    references = ["this is a dummy reference sentence number " + str(i) for i in range(n_examples)]

    mock_model = MockBertModel()

    # 1. Standard Case
    scorer = bert_score_lib.CBertScorer(bert_model=mock_model)
    # Warmup
    _ = scorer.score(candidates[:10], references[:10])

    start_time = time.time()
    results = scorer.score(candidates, references)
    end_time = time.time()
    std_time = end_time - start_time
    print(f"[{label}] Standard - Time: {std_time:.4f}s")

    # 2. Medical Tokens Case
    special_words = ["dummy", "sentence"]
    scorer_med = bert_score_lib.CBertScorer(bert_model=mock_model, special_words=special_words)
    # Warmup
    _ = scorer_med.score(candidates[:10], references[:10])

    start_time = time.time()
    results_med = scorer_med.score(candidates, references)
    end_time = time.time()
    med_time = end_time - start_time
    print(f"[{label}] Medical  - Time: {med_time:.4f}s")

    return results, results_med

if __name__ == "__main__":
    benchmark()
