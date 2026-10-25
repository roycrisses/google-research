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

# pylint: disable=invalid-name
"""Module to evaluate Chebyshev polynomials."""
from jax import numpy as jnp
from sobolev import chebyshev


def eval_schebyt(n, x, low,
                 high):
  """Evaluate the Sobolev-Chebyshev polynomial of the first kind.

  Args:
    n : positive integer.
      Degree of the polynomial, must be an integer.
    x : array_like,
      Points at which to evaluate the Chebyshev polynomial.
    low: float.
      Lower bound for the interval of orthogonality, which is (low, high).
    high: float.
      Upper bound for the interval of orthogonality, which is (low, high).


  Returns:
    T : ndarray
      Values of the polynomial
  """
  if not isinstance(n, int):
    raise NotImplementedError(
        "This function is currently implemented only for " +
        "integer values of n")
  if n == 0:
    return jnp.ones_like(x)
  elif n == 1:
    return jnp.ones_like(x) - 2 * x / (high + low)

  # Performance optimization: Replace O(n^2) recursive evaluation and repeated
  # Chebyshev polynomial evaluations with a single O(n) bottom-up pass.
  x_arr = jnp.asarray(x)
  s_0 = - (high + low) / (high - low)
  s_x = 2 * x_arr / (high - low) + s_0
  s_0_arr = jnp.asarray(s_0, dtype=s_x.dtype)

  # Iteratively evaluate Chebyshev polynomials T_k at s_0 and s_x up to degree n
  t_s0 = [jnp.ones_like(s_x), s_0_arr]
  for k in range(2, n + 1):
    t_s0.append(2.0 * s_0_arr * t_s0[-1] - t_s0[-2])

  t_sx = [jnp.ones_like(s_x), s_x]
  for k in range(2, n + 1):
    t_sx.append(2.0 * s_x * t_sx[-1] - t_sx[-2])

  # Compute Sobolev-Chebyshev polynomials S_k bottom-up for degrees k <= n
  start_k = 0 if n % 2 == 0 else 1
  if start_k == 0:
    s_prev = jnp.ones_like(x_arr)
  else:
    s_prev = jnp.ones_like(x_arr) - 2 * x_arr / (high + low)

  for k in range(start_k + 2, n + 1, 2):
    if k == 2:
      alpha = 1.0
      xi = 0.0
    else:
      xi = (t_s0[k - 2] / t_s0[k]) * (k / (k - 2))
      gamma_k = -xi / (1.0 + 2.0 * (k - 2) * (k - 2) * k)
      alpha = 1.0 / (1.0 - xi - gamma_k)

    p_k = t_sx[k] / t_s0[k]
    p_k2 = t_sx[k - 2] / t_s0[k - 2]
    beta = 1.0 - alpha + xi * alpha
    s_curr = beta * s_prev + alpha * p_k - xi * alpha * p_k2
    s_prev = s_curr

  return s_prev
