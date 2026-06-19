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

"""
Created on Thu Feb  6 13:02:31 2020
@author: yujia
"""

import numpy as np
import torch
from torch.autograd import Function
import torch.nn.functional as F
# pylint: skip-file


def sinkhorn_forward(C, mu, nu, epsilon, max_iter):
    """standard forward of sinkhorn."""

    bs, _, k_ = C.size()

    # Create v on the same device as C
    v = torch.full([bs, 1, k_], 1.0 / k_, device=C.device, dtype=C.dtype)
    G = torch.exp(-C / epsilon)

    for _ in range(max_iter):
        # Use matmul for efficiency instead of element-wise multiplication + sum
        u = mu / torch.matmul(G, v.transpose(-1, -2))
        v = nu / torch.matmul(u.transpose(-1, -2), G)

    Gamma = u * G * v
    return Gamma


def sinkhorn_forward_stablized(C, mu, nu, epsilon, max_iter):
    """sinkhorn forward in log space."""

    bs, n, k_ = C.size()

    # f and g in log-space, scaled by 1/epsilon
    f_hat = torch.zeros([bs, n, 1], device=C.device, dtype=C.dtype)
    g_hat = torch.zeros([bs, 1, k_], device=C.device, dtype=C.dtype)

    log_mu = torch.log(mu)
    log_nu = torch.log(nu)
    C_eps = -C / epsilon

    for _ in range(max_iter):
        # f_hat = f/epsilon. Optimized log-space updates.
        f_hat = -torch.logsumexp(C_eps + g_hat, dim=-1, keepdim=True) + log_mu
        g_hat = -torch.logsumexp(C_eps + f_hat, dim=-2, keepdim=True) + log_nu

    Gamma = torch.exp(C_eps + f_hat + g_hat)
    return Gamma


def sinkhorn_backward(grad_output_Gamma, Gamma, mu, nu, epsilon):
    """
    Standard backward of sinkhorn using Schur complement.
    Optimized for memory and speed.
    """
    bs, n, k_ = Gamma.size()
    k = k_ - 1

    # Use slices to avoid unnecessary copies
    nu_ = nu[:, :, :k]
    Gamma_ = Gamma[:, :, :k]

    inv_mu = 1.0 / mu.view(1, -1)  # [1, n]

    # Pre-multiply Gamma_ by sqrt(inv_mu) for more efficient matmul if needed,
    # but here we just optimize the existing chain.
    Gamma_t = Gamma_.transpose(-1, -2)
    Kappa = torch.diag_embed(nu_.squeeze(-2)) - torch.matmul(Gamma_t * inv_mu.unsqueeze(-2), Gamma_)

    # Optimization for k=1 (common case in TopK_custom with 2 anchors)
    if k == 1:
        inv_Kappa = 1.0 / Kappa
    else:
        inv_Kappa = torch.linalg.inv(Kappa)

    Gamma_mu = inv_mu.unsqueeze(-1) * Gamma_
    L = torch.matmul(Gamma_mu, inv_Kappa)  # [bs, n, k]

    G1 = grad_output_Gamma * Gamma  # [bs, n, k_]
    g1 = G1.sum(-1, keepdim=True)   # [bs, n, 1]
    g2 = G1.sum(-2, keepdim=True)[:, :, :k].transpose(-1, -2) # [bs, k, 1]

    # Combine terms that are multiplied by Gamma to reduce allocations and memory passes
    # grad_C = (-G1 + G2 + G3) / epsilon
    # G2 = G21 + G22 + G23
    # G3 = G31 + G32

    # We want to compute:
    # M = -grad_output_Gamma + (g1*inv_mu) + (g1_L @ Gamma_mu^T) - pad(g1_L) - (L @ g2) + pad(inv_Kappa @ g2)
    # then grad_C = (M * Gamma) / epsilon

    g1_L = torch.matmul(g1.transpose(-1, -2), L) # [bs, 1, k]
    term_G22 = torch.matmul(g1_L, Gamma_mu.transpose(-1, -2)).transpose(-1, -2) # [bs, n, 1]
    term_G23 = -F.pad(g1_L, (0, 1), mode='constant', value=0) # [bs, 1, k_]

    term_G31 = -torch.matmul(L, g2) # [bs, n, 1]
    term_G32 = F.pad(torch.matmul(inv_Kappa, g2).transpose(-1, -2), (0, 1), mode='constant', value=0) # [bs, 1, k_]

    M = -grad_output_Gamma + (g1 * inv_mu.unsqueeze(-1)) + term_G22 + term_G23 + term_G31 + term_G32
    grad_C = (M * Gamma) / epsilon

    return grad_C


class TopKFunc1(Function):
    @staticmethod
    def forward(ctx, C, mu, nu, epsilon, max_iter):

        with torch.no_grad():
            if epsilon>1e-2:
                Gamma = sinkhorn_forward(C, mu, nu, epsilon, max_iter)
                if bool(torch.any(Gamma!=Gamma)):
                    print('Nan appeared in Gamma, re-computing...')
                    Gamma = sinkhorn_forward_stablized(C, mu, nu, epsilon, max_iter)
            else:
                Gamma = sinkhorn_forward_stablized(C, mu, nu, epsilon, max_iter)
            ctx.save_for_backward(mu, nu, Gamma)
            ctx.epsilon = epsilon

        return Gamma

    @staticmethod
    def backward(ctx, grad_output_Gamma):

        epsilon = ctx.epsilon
        mu, nu, Gamma = ctx.saved_tensors
        # mu [1, n, 1]
        # nu [1, 1, k+1]
        #Gamma [bs, n, k+1]
        with torch.no_grad():
            grad_C = sinkhorn_backward(grad_output_Gamma, Gamma, mu, nu, epsilon)
        return grad_C, None, None, None, None


class TopK_custom(torch.nn.Module):
    def __init__(self, k, epsilon=0.1, max_iter = 200):
        super(TopK_custom, self).__init__()
        self.k = k
        self.epsilon = epsilon
        # Register anchors as a buffer to handle device movement automatically
        self.register_buffer('anchors', torch.tensor([0.0, 1.0]).view(1, 1, 2))
        self.max_iter = max_iter

    def forward(self, scores):
        bs, n = scores.size()
        scores = scores.view([bs, n, 1])

        # Handle -inf values more efficiently
        if torch.any(scores == float('-inf')):
            scores_finite = scores[scores != float('-inf')]
            if scores_finite.numel() > 0:
                max_s = scores_finite.max()
                min_s = scores_finite.min()
                filled_value = min_s - (max_s - min_s)
            else:
                filled_value = scores.new_tensor(0.0)
            scores = scores.masked_fill(scores == float('-inf'), filled_value)

        C = (scores - self.anchors)**2
        C = C / (C.max().detach() + 1e-10) # Avoid division by zero

        # Create mu and nu on the same device as scores
        mu = torch.full([1, n, 1], 1.0 / n, device=scores.device, dtype=scores.dtype)
        nu = torch.tensor([self.k / n, (n - self.k) / n], device=scores.device, dtype=scores.dtype).view(1, 1, 2)

        Gamma = TopKFunc1.apply(C, mu, nu, self.epsilon, self.max_iter)
        A = Gamma[:, :, 0] * n

        return A

############################################################################
############################################################################

class TopK_stablized(torch.nn.Module):
    def __init__(self, k, epsilon=0.1, max_iter = 200):
        super(TopK_stablized, self).__init__()
        self.k = k
        self.epsilon = epsilon
        self.register_buffer('anchors', torch.tensor([0.0, 1.0]).view(1, 2, 1))
        self.max_iter = max_iter

    def forward(self, scores):
        bs, n = scores.size()[:2]
        scores = scores.view([bs, 1, n])

        # Handle -inf values more efficiently
        if torch.any(scores == float('-inf')):
            scores_finite = scores[scores != float('-inf')]
            if scores_finite.numel() > 0:
                max_s = scores_finite.max()
                min_s = scores_finite.min()
                filled_value = min_s - (max_s - min_s)
            else:
                filled_value = scores.new_tensor(0.0)
            scores = scores.masked_fill(scores == float('-inf'), filled_value)

        C = (scores - self.anchors)**2
        C = C / (C.max().detach() + 1e-10)

        f_hat = torch.zeros([bs, 1, n], device=scores.device, dtype=scores.dtype)
        g_hat = torch.zeros([bs, 2, 1], device=scores.device, dtype=scores.dtype)

        log_mu = torch.full([1, 1, n], -np.log(n), device=scores.device, dtype=scores.dtype)
        log_nu = torch.log(torch.tensor([self.k / n, (n - self.k) / n], device=scores.device, dtype=scores.dtype)).view(1, 2, 1)
        C_eps = -C / self.epsilon

        for _ in range(self.max_iter):
            f_hat = -torch.logsumexp(C_eps + g_hat, dim=-2, keepdim=True) + log_mu
            g_hat = -torch.logsumexp(C_eps + f_hat, dim=-1, keepdim=True) + log_nu

        P = torch.exp(C_eps + f_hat + g_hat)
        A = P[:, 0, :] * n
        return A

