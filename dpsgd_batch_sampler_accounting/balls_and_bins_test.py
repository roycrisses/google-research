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
import numpy as np
from scipy import stats, special

from dpsgd_batch_sampler_accounting import balls_and_bins


class BallsAndBinsTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(42)

  def test_max_of_gaussians(self):
    sigma = 0.35
    dim = 100
    mog = balls_and_bins.MaxOfGaussians(sigma, dim)

    x = 0.5
    expected_logcdf = dim * stats.norm.logcdf(x, scale=sigma)
    self.assertAlmostEqual(mog.logcdf(x), expected_logcdf, places=6)

    expected_sf = -np.expm1(dim * np.log1p(-stats.norm.sf(x, scale=sigma)))
    self.assertAlmostEqual(mog.sf(x), expected_sf, places=6)

    q = 0.8
    expected_ppf = stats.norm.ppf(np.exp(np.log(q) / dim), scale=sigma)
    self.assertAlmostEqual(mog.ppf(q), expected_ppf, places=6)

    samples = mog.rvs(1000)
    self.assertEqual(samples.shape, (1000,))

  def test_sample_gaussians_conditioned_on_max_value(self):
    sigma = 0.35
    dim = 50
    max_values = np.array([0.2, 0.5, 0.8])
    samples = balls_and_bins.sample_gaussians_conditioned_on_max_value(
        sigma, dim, max_values
    )
    self.assertEqual(samples.shape, (3, dim))
    for i in range(3):
      self.assertTrue(np.all(samples[i] <= max_values[i]))

  def test_sample_order_statistics_from_uniform(self):
    dim = 100
    orders = [1, 5, 10, 50]
    samples = balls_and_bins.sample_order_statistics_from_uniform(
        dim, orders, size=(100,)
    )
    self.assertEqual(samples.shape, (100, 4))
    # Order statistics must be non-increasing for top ranks.
    self.assertTrue(np.all(samples >= 0.0))
    self.assertTrue(np.all(samples <= 1.0))

  def test_privacy_loss(self):
    accountant = balls_and_bins.BnBAccountant()
    sigma = 0.35
    samples = np.random.randn(100, 1, 50)

    loss_add = accountant.privacy_loss(
        sigma, samples, balls_and_bins.AdjacencyType.ADD
    )
    loss_remove = accountant.privacy_loss(
        sigma, samples, balls_and_bins.AdjacencyType.REMOVE
    )
    self.assertEqual(loss_add.shape, (100,))
    self.assertEqual(loss_remove.shape, (100,))
    np.testing.assert_allclose(loss_add, -loss_remove, rtol=1e-5)

  def test_estimate_order_stats_deltas(self):
    accountant = balls_and_bins.BnBAccountant()
    sigma = 0.35
    epsilons = (0.5, 1.0, 2.0)
    num_steps_per_epoch = 100
    sample_size = 1000
    order_stats_seq = [1, 5, 10, 50]

    estimates = accountant.estimate_order_stats_deltas(
        sigma, epsilons, num_steps_per_epoch, sample_size, order_stats_seq
    )
    self.assertEqual(len(estimates), 3)
    # Delta should decrease as epsilon increases.
    self.assertGreater(estimates[0].mean, estimates[1].mean)
    self.assertGreater(estimates[1].mean, estimates[2].mean)


if __name__ == '__main__':
  unittest.main()
