
import time
import numpy as np
import collections

# Mock the modules that are failing to import
import sys
from unittest.mock import MagicMock

# Mock bert_lib because it tries to import 'bert' which is missing
mock_bert_lib = MagicMock()
sys.modules['cbertscore.bert_lib'] = mock_bert_lib

from cbertscore import bert_score_lib

class MockBertModel:
    def __init__(self):
        self.tokenizer = self.MockTokenizer()

    class MockTokenizer:
        def tokenize(self, text):
            # Very simple tokenizer
            return [t for t in text.split() if t not in ["[CLS]", "[SEP]"]]

    def get_activations(self, texts):
        n_total = len(texts)
        tokens = [text.split() for text in texts]
        layers = []
        for i in range(n_total):
            # Mocking one layer (index 0)
            # layer_dict[0] should have activations for all tokens (including CLS and SEP)
            layer_dict = {0: np.random.rand(len(tokens[i]), 768)}
            layers.append(layer_dict)
        return tokens, layers

def benchmark():
    mock_bert = MockBertModel()
    n_examples = 100
    # Sentences with [CLS] and [SEP]
    candidates = [" ".join(["[CLS]"] + ["word"] * 50 + ["[SEP]"]) for _ in range(n_examples)]
    references = [" ".join(["[CLS]"] + ["word"] * 50 + ["[SEP]"]) for _ in range(n_examples)]

    # 1. Performance test with special_words (medical mode)
    special_words = ["word"]
    scorer = bert_score_lib.CBertScorer(bert_model=mock_bert, special_words=special_words)

    start = time.time()
    scores_medical = scorer.score(candidates, references)
    end = time.time()
    print(f"Time with special_words (medical mode): {end - start:.4f}s")

    # Verify we got results for layer 0 and they have the right length
    assert 0 in scores_medical
    assert len(scores_medical[0]) == n_examples

    # 2. Performance test without special_words
    scorer_no_special = bert_score_lib.CBertScorer(bert_model=mock_bert)
    start = time.time()
    scores_standard = scorer_no_special.score(candidates, references)
    end = time.time()
    print(f"Time without special_words: {end - start:.4f}s")
    assert 0 in scores_standard
    assert len(scores_standard[0]) == n_examples

    # 3. Correctness test for alignment (empty sentences)
    # We need to use a single example to make it easier to check
    cand_empty = ["[CLS] [SEP]"] # Empty after removing CLS/SEP
    ref_empty = ["[CLS] ref [SEP]"]

    start = time.time()
    scores_empty = scorer_no_special.score(cand_empty, ref_empty, skip_empty_sentences_after_tokenization=True)
    end = time.time()
    print(f"Time with empty sentence: {end - start:.4f}s")

    # In the fixed version, scores[0] should contain CBERTScore(-1, -1, -1)
    assert 0 in scores_empty
    assert len(scores_empty[0]) == 1
    assert scores_empty[0][0].precision == -1
    assert scores_empty[0][0].recall == -1
    assert scores_empty[0][0].f_score == -1
    print("Alignment check passed!")

    print("All benchmark tests passed!")

if __name__ == "__main__":
    benchmark()
