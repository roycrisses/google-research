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

"""Tests for CMMD distance calculation."""

from absl.testing import absltest
from cmmd import distance
import jax
import jax.numpy as jnp


class DistanceTest(absltest.TestCase):

  def test_mmd_zero_for_same_distribution(self):
    x = jnp.zeros((10, 5))
    res = distance.mmd(x, x)
    self.assertAlmostEqual(float(res), 0.0, places=5)

  def test_mmd_positive_for_different_distributions(self):
    x = jnp.zeros((50, 10))
    y = jnp.ones((50, 10))
    res = distance.mmd(x, y)
    self.assertGreater(float(res), 0.0)

  def test_mmd_shapes(self):
    x = jax.random.normal(jax.random.PRNGKey(0), (20, 128))
    y = jax.random.normal(jax.random.PRNGKey(1), (20, 128))
    res = distance.mmd(x, y)
    self.assertEqual(res.shape, ())


if __name__ == '__main__':
  absltest.main()
