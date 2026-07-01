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

import unittest
import jax
import jax.numpy as jnp
import numpy as np
from cmmd import distance

class DistanceTest(unittest.TestCase):

  def test_mmd_correctness(self):
    N, D = 10, 5
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (N, D))
    y = jax.random.normal(key, (N, D)) + 1.0

    # Original implementation logic (for reference)
    def ref_mmd(x, y):
      x_sqnorms = jnp.diag(jnp.matmul(x, x.T))
      y_sqnorms = jnp.diag(jnp.matmul(y, y.T))
      gamma = 1 / (2 * 10**2)
      k_xx = jnp.mean(jnp.exp(-gamma * (-2 * jnp.matmul(x, x.T) +
                                       jnp.expand_dims(x_sqnorms, 1) +
                                       jnp.expand_dims(x_sqnorms, 0))))
      k_xy = jnp.mean(jnp.exp(-gamma * (-2 * jnp.matmul(x, y.T) +
                                       jnp.expand_dims(x_sqnorms, 1) +
                                       jnp.expand_dims(y_sqnorms, 0))))
      k_yy = jnp.mean(jnp.exp(-gamma * (-2 * jnp.matmul(y, y.T) +
                                       jnp.expand_dims(y_sqnorms, 1) +
                                       jnp.expand_dims(y_sqnorms, 0))))
      return 1000 * (k_xx + k_yy - 2 * k_xy)

    expected_val = ref_mmd(x, y)
    actual_val = distance.mmd(x, y)

    np.testing.assert_allclose(actual_val, expected_val, atol=1e-5)

  def test_mmd_zero_distance(self):
    N, D = 10, 5
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (N, D))

    actual_val = distance.mmd(x, x)
    # The MMD distance should be close to 0 when x == y.
    # We use a slightly larger tolerance due to floating point precision.
    self.assertAlmostEqual(actual_val, 0.0, places=4)

if __name__ == '__main__':
  unittest.main()
