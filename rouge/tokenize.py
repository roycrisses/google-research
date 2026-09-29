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

"""A library for tokenizing text."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import six


# Pre-compile regexes that are used often
NON_ALPHANUM_PATTERN = r"[^a-z0-9]+"
NON_ALPHANUM_RE = re.compile(NON_ALPHANUM_PATTERN)
SPACES_PATTERN = r"\s+"
SPACES_RE = re.compile(SPACES_PATTERN)
VALID_TOKEN_PATTERN = r"^[a-z0-9]+$"
VALID_TOKEN_RE = re.compile(VALID_TOKEN_PATTERN)
_ALPHA_NUM_RE = re.compile(r"[a-z0-9]+")


def tokenize(text, stemmer):
  """Tokenize input text into a list of tokens.

  This approach aims to replicate the approach taken by Chin-Yew Lin in
  the original ROUGE implementation.

  Args:
    text: A text blob to tokenize.
    stemmer: An optional stemmer.

  Returns:
    A list of string tokens extracted from input text.
  """

  # Bolt optimization: Extract lower-case alphanumeric tokens directly in one
  # regex call, avoiding intermediate string creations (sub and split) and
  # skipping redundant VALID_TOKEN_RE checks when no stemmer is used.
  tokens = _ALPHA_NUM_RE.findall(six.ensure_str(text.lower()))

  if stemmer:
    # Bolt optimization: Cache stemming results on the stemmer instance to
    # avoid redundant PorterStemmer operations on duplicate words.
    cache = getattr(stemmer, "_stem_cache", None)
    if cache is None:
      cache = {}
      try:
        stemmer._stem_cache = cache
      except AttributeError:
        cache = None

    if cache is not None:
      stemmed_tokens = []
      for x in tokens:
        if len(x) > 3:
          stemmed = cache.get(x)
          if stemmed is None:
            stemmed = six.ensure_str(stemmer.stem(x))
            cache[x] = stemmed
          stemmed_tokens.append(stemmed)
        else:
          stemmed_tokens.append(x)
      tokens = stemmed_tokens
    else:
      # Only stem words more than 3 characters long.
      tokens = [six.ensure_str(stemmer.stem(x)) if len(x) > 3 else x
                for x in tokens]

    # Drop any empty or invalid tokens after stemming.
    tokens = [x for x in tokens if VALID_TOKEN_RE.match(x)]

  return tokens
