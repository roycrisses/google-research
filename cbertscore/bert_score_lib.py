# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of BERTScore."""

import collections
from typing import Optional, Sequence

import numpy as np
from cbertscore import bert_lib

# TODO(agnesbi): Change this to a dataclass.
CBERTScore = collections.namedtuple('CBERTScore',
                                    ['precision', 'recall', 'f_score'])


class CBertScorer(object):
  """Class for scoring the meaning-similarity between two translations."""

  def __init__(self, bert_model_dir=None, bert_model=None,
               special_words = None):
    """Initialize a BERT_Score scorer using default BERT model configuration.

    Args:
      bert_model_dir: string specifing a directory containing a BERT model. This
        model will be loaded using default configuration.
      bert_model: A bert_lib.BertModel object. If specified, overrides
        bert_model_dir. One of bert_model_dir or bert_model must be specified.
      special_words: If provided, these are the only words that matter.
    """
    assert bert_model_dir or bert_model

    if bert_model is not None:
      self.bert_model = bert_model
    else:
      self.bert_model = bert_lib.BertModel(model_dir=bert_model_dir)
    self.medical_tokens = []
    if special_words:
      for w in special_words:
        self.medical_tokens.extend(self.bert_model.tokenizer.tokenize(w))
    self.medical_tokens = set(self.medical_tokens)

  def score(self,
            candidates,
            references,
            skip_empty_sentences_after_tokenization = False):
    """Calculates the BERTScore of the given sentences.

    Args:
      candidates ([str]): List of strings representing the hypothesis sentences.
      references ([str]): List of strings representing the reference strings.
      skip_empty_sentences_after_tokenization: If `False`, the function will
        raise an error, if a candidate or reference is empty after tokenization.
        If `True`, instead of raising an error, the BertScore won't be computed
        and (-1, -1, -1) will be returned for this entry.

    Returns:
      list of named tuples, one for each candidate/reference pair, each of which
      has 3 components:
        precision: how much of the candidate's meaning is in the reference
        recall:    how much of the reference's meaning is in the candidate
        f_score:   overall score (harmonic mean of precision & recall)

    Raises:
      AssertionError: If one of the candidate or reference sentence are empty,
        or if they are empty after tokenization.
    """
    candidates, references = list(candidates), list(references)

    assert len(candidates) and len(references), \
      ('You must provide at least one candidate and reference.')
    assert len(candidates) == len(references), \
      ('The number of candidate sentences must equal the number of reference '
       'sentences!')
    assert all(len(c.strip()) for c in candidates), 'Empty candidate sentence.'
    assert all(len(r.strip()) for r in references), 'Empty reference sentence.'

    n_examples = len(candidates)
    if n_examples == 0:
      return collections.defaultdict(list)

    all_text = candidates + references
    print('About to get activations...')
    tokens, layers = self.bert_model.get_activations(all_text)
    print('Got activations.')

    all_cand_tokens = tokens[:n_examples]
    all_ref_tokens = tokens[-n_examples:]

    all_cand_embeddings = layers[:n_examples]
    all_ref_embeddings = layers[-n_examples:]

    # Get the layer numbers once.
    layer_numbers = [int(k) for k in all_ref_embeddings[0].keys()]

    scores = collections.defaultdict(list)

    for example_idx in range(n_examples):
      # Ignore the first and last tokens when cheking similarity because
      # every sentence starts and ends with [CLS] and [SEP] tokens.
      cand_tokens = all_cand_tokens[example_idx][1:-1]
      ref_tokens = all_ref_tokens[example_idx][1:-1]

      # pylint: disable=g-explicit-length-test
      if (skip_empty_sentences_after_tokenization and
          (len(cand_tokens) == 0 or len(ref_tokens) == 0)):
        scores[-1] = [CBERTScore(precision=-1, recall=-1, f_score=-1)]
        continue

      if len(cand_tokens) == 0:
        raise AssertionError(
            'You have an empty hypothesis sentence for the index {}. '
            'candidates[idx] was {}, in bytes: {}, all_cand_tokens[idx] was {} '
            ''.format(example_idx, candidates[example_idx],
                      candidates[example_idx].encode(),
                      all_cand_tokens[example_idx]))
      if len(ref_tokens) == 0:
        raise AssertionError(
            'You have an empty reference sentence for the index {}. '
            'references[idx] was {}, in bytes: {} all_ref_tokens[idx] was {} '
            'and '.format(example_idx, references[example_idx],
                          references[example_idx].encode(),
                          all_ref_tokens[example_idx]))
      # pylint: enable=g-explicit-length-test

      if self.medical_tokens:
        # Vectorized weights for medical tokens.
        w_cand = np.array(
            [1 if t in self.medical_tokens else 0 for t in cand_tokens])
        w_ref = np.array(
            [1 if t in self.medical_tokens else 0 for t in ref_tokens])

      for ln in layer_numbers:
        cand_embeddings = np.vstack(all_cand_embeddings[example_idx][ln])[1:-1]
        ref_embeddings = np.vstack(all_ref_embeddings[example_idx][ln])[1:-1]

        # Calculate cosine similarity by normalizing then dot product of all
        # pairs.
        cand_embeddings /= np.linalg.norm(
            cand_embeddings, axis=1, keepdims=True)
        ref_embeddings /= np.linalg.norm(
            ref_embeddings, axis=1, keepdims=True)

        sim_matrix = np.matmul(cand_embeddings, ref_embeddings.T)

        if self.medical_tokens:
          # Vectorized weighted precision/recall.
          max_sim_cand = np.max(sim_matrix, axis=1)
          idf_sum_cand = np.sum(w_cand)
          if idf_sum_cand == 0:
            precision = np.nan
          else:
            precision = np.sum(w_cand * max_sim_cand) / idf_sum_cand

          max_sim_ref = np.max(sim_matrix, axis=0)
          idf_sum_ref = np.sum(w_ref)
          if idf_sum_ref == 0:
            recall = np.nan
          else:
            recall = np.sum(w_ref * max_sim_ref) / idf_sum_ref
        else:
          precision = np.mean(np.max(sim_matrix, axis=1))
          recall = np.mean(np.max(sim_matrix, axis=0))

        if precision == 0 and recall == 0:
          f_score = 0
        else:
          f_score = 2 * precision * recall / (precision + recall)

        scores[ln].append(
            CBERTScore(precision=precision, recall=recall, f_score=f_score))

    return scores
