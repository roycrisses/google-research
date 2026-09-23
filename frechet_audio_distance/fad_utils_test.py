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

"""Tests for Fréchet Audio Distance utils."""

import unittest
import numpy as np
from frechet_audio_distance import fad_utils


class FadUtilsTest(unittest.TestCase):

  def test_normalize_loudness(self):
    samples = np.array([0.0, 0.5, 1.0, -1.0])
    normalized = fad_utils.normalize_loudness(samples)
    np.testing.assert_allclose(normalized, samples)

  def test_frechet_distance_identical(self):
    mu1 = np.array([1.0, 2.0, 3.0])
    sigma1 = np.eye(3)
    dist = fad_utils.frechet_distance(mu1, sigma1, mu1, sigma1)
    self.assertAlmostEqual(dist, 0.0, places=5)

  def test_frechet_distance_known_shift(self):
    mu1 = np.array([0.0, 0.0])
    sigma1 = np.array([[2.0, 0.0], [0.0, 2.0]])
    mu2 = np.array([1.0, 1.0])
    sigma2 = np.array([[2.0, 0.0], [0.0, 2.0]])
    # (0-1)^2 + (0-1)^2 = 2. Tr(S1+S2 - 2*sqrt(S1*S2)) = Tr(2I + 2I - 4I) = 0
    dist = fad_utils.frechet_distance(mu1, sigma1, mu2, sigma2)
    self.assertAlmostEqual(dist, 2.0, places=5)

  def test_stable_trace_sqrt_product_accuracy(self):
    np.random.seed(42)
    n = 10
    a = np.random.randn(n, n)
    sigma_test = a @ a.T + np.eye(n)
    b = np.random.randn(n, n)
    sigma_train = b @ b.T + np.eye(n)

    res = fad_utils._stable_trace_sqrt_product(sigma_test, sigma_train)
    self.assertGreater(res, 0.0)
    self.assertFalse(np.isnan(res))

  def test_stable_trace_sqrt_product_singular_fallback(self):
    # Singular matrix
    sigma_test = np.zeros((3, 3))
    sigma_train = np.eye(3)
    res = fad_utils._stable_trace_sqrt_product(sigma_test, sigma_train)
    self.assertGreaterEqual(res, 0.0)


if __name__ == '__main__':
  unittest.main()
