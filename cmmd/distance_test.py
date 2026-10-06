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

"""Tests for cmmd.distance."""

import unittest
from cmmd import distance
import jax.numpy as jnp
import numpy as np


class DistanceTest(unittest.TestCase):

  def test_mmd_identical_inputs(self):
    x = np.ones((100, 128), dtype=np.float32)
    y = np.ones((100, 128), dtype=np.float32)
    dist = distance.mmd(x, y)
    np.testing.assert_allclose(dist, 0.0, atol=1e-3)

  def test_mmd_against_reference(self):
    np.random.seed(42)
    x = np.random.randn(50, 32).astype(np.float32)
    y = np.random.randn(50, 32).astype(np.float32)

    # Reference calculation using double loop / exact RBF kernel
    gamma = 1.0 / (2.0 * 10.0**2)

    def rbf_kernel(a, b):
      sq_dist = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
      return np.exp(-gamma * sq_dist)

    k_xx = np.mean(rbf_kernel(x, x))
    k_xy = np.mean(rbf_kernel(x, y))
    k_yy = np.mean(rbf_kernel(y, y))
    expected_mmd = 1000.0 * (k_xx + k_yy - 2.0 * k_xy)

    actual_mmd = distance.mmd(x, y)
    np.testing.assert_allclose(actual_mmd, expected_mmd, rtol=1e-4, atol=1e-3)


if __name__ == "__main__":
  unittest.main()
