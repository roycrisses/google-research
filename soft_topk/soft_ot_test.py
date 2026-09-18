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
import torch

from soft_topk.soft_ot import (
    sinkhorn_forward,
    sinkhorn_forward_stablized,
    sinkhorn_backward,
    TopKFunc1,
    TopK_custom,
    TopK_stablized,
)


class SoftOTTest(unittest.TestCase):

  def test_sinkhorn_forward(self):
    bs, n, k_ = 4, 10, 2
    C = torch.rand(bs, n, k_)
    mu = torch.ones([1, n, 1]) / n
    nu = torch.tensor([2.0 / n, (n - 2.0) / n]).view([1, 1, 2])
    epsilon = 0.1
    max_iter = 50

    Gamma = sinkhorn_forward(C, mu, nu, epsilon, max_iter)
    self.assertEqual(Gamma.shape, (bs, n, k_))
    # Marginals sum check
    torch.testing.assert_close(Gamma.sum(dim=-1, keepdim=True), mu.expand(bs, n, 1), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(Gamma.sum(dim=-2, keepdim=True), nu.expand(bs, 1, k_), atol=1e-3, rtol=1e-3)

  def test_sinkhorn_forward_stablized(self):
    bs, n, k_ = 4, 10, 2
    C = torch.rand(bs, n, k_)
    mu = torch.ones([1, n, 1]) / n
    nu = torch.tensor([2.0 / n, (n - 2.0) / n]).view([1, 1, 2])
    epsilon = 0.1
    max_iter = 50

    Gamma_stab = sinkhorn_forward_stablized(C, mu, nu, epsilon, max_iter)
    self.assertEqual(Gamma_stab.shape, (bs, n, k_))

    Gamma_std = sinkhorn_forward(C, mu, nu, epsilon, max_iter)
    torch.testing.assert_close(Gamma_stab, Gamma_std, atol=1e-4, rtol=1e-4)

  def test_sinkhorn_backward(self):
    bs, n, k_ = 4, 10, 2
    C = torch.rand(bs, n, k_)
    mu = torch.ones([1, n, 1]) / n
    nu = torch.tensor([2.0 / n, (n - 2.0) / n]).view([1, 1, 2])
    epsilon = 0.1
    max_iter = 50

    Gamma = sinkhorn_forward(C, mu, nu, epsilon, max_iter)
    grad_output_Gamma = torch.ones_like(Gamma)
    grad_C = sinkhorn_backward(grad_output_Gamma, Gamma, mu, nu, epsilon)
    self.assertEqual(grad_C.shape, (bs, n, k_))
    self.assertFalse(torch.isnan(grad_C).any())

  def test_topk_custom_autograd(self):
    bs, n = 4, 10
    scores = torch.randn(bs, n, requires_grad=True)
    topk_mod = TopK_custom(k=2, epsilon=0.1, max_iter=50)
    out = topk_mod(scores)
    self.assertEqual(out.shape, (bs, n))

    loss = out.sum()
    loss.backward()
    self.assertIsNotNone(scores.grad)
    self.assertEqual(scores.grad.shape, (bs, n))
    self.assertFalse(torch.isnan(scores.grad).any())

  def test_topk_stablized(self):
    bs, n = 4, 10
    scores = torch.randn(bs, n)
    topk_stab = TopK_stablized(k=2, epsilon=0.1, max_iter=50)
    out = topk_stab(scores)
    self.assertEqual(out.shape, (bs, n))


if __name__ == '__main__':
  unittest.main()
