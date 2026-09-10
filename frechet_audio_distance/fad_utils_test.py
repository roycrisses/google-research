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

"""Tests for frechet_audio_distance.fad_utils."""

from absl.testing import absltest
import numpy as np
from frechet_audio_distance import fad_utils


class FadUtilsTest(absltest.TestCase):

  def test_normalize_loudness(self):
    samples = np.array([0.5, -0.5, 0.25], dtype=np.float32)
    normalized = fad_utils.normalize_loudness(samples)
    np.testing.assert_allclose(normalized, [1.0, -1.0, 0.5])

  def test_frechet_distance_identity(self):
    mu1 = np.zeros(128)
    sigma1 = np.eye(128)
    mu2 = np.zeros(128)
    sigma2 = np.eye(128)
    dist = fad_utils.frechet_distance(mu1, sigma1, mu2, sigma2)
    self.assertAlmostEqual(dist, 0.0, places=5)

  def test_frechet_distance_shift(self):
    mu1 = np.zeros(128)
    sigma1 = np.eye(128)
    mu2 = np.ones(128)
    sigma2 = np.eye(128)
    dist = fad_utils.frechet_distance(mu1, sigma1, mu2, sigma2)
    self.assertAlmostEqual(dist, 128.0, places=5)

  def test_frechet_distance_random_matrices(self):
    dim = 64
    np.random.seed(42)
    A = np.random.randn(dim, dim)
    sigma1 = A @ A.T / dim + np.eye(dim) * 0.1
    B = np.random.randn(dim, dim)
    sigma2 = B @ B.T / dim + np.eye(dim) * 0.1
    mu1 = np.random.randn(dim)
    mu2 = np.random.randn(dim)

    dist = fad_utils.frechet_distance(mu1, sigma1, mu2, sigma2)
    self.assertGreater(dist, 0.0)

  def test_invalid_shapes(self):
    mu1 = np.zeros((128, 1))
    sigma1 = np.eye(128)
    mu2 = np.zeros(128)
    sigma2 = np.eye(128)
    with self.assertRaises(ValueError):
      fad_utils.frechet_distance(mu1, sigma1, mu2, sigma2)


if __name__ == "__main__":
  absltest.main()
