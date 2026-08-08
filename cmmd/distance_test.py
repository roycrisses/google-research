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

"""Tests and benchmarks for distance.py."""

import time
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from cmmd import distance


# Reference/Original MMD implementation using jnp.diag(jnp.matmul(x, x.T))
def _original_mmd(x, y):
  x = jnp.asarray(x)
  y = jnp.asarray(y)

  x_sqnorms = jnp.diag(jnp.matmul(x, x.T))
  y_sqnorms = jnp.diag(jnp.matmul(y, y.T))

  gamma = 1 / (2 * distance._SIGMA**2)
  k_xx = jnp.mean(
      jnp.exp(
          -gamma
          * (
              -2 * jnp.matmul(x, x.T)
              + jnp.expand_dims(x_sqnorms, 1)
              + jnp.expand_dims(x_sqnorms, 0)
          )
      )
  )
  k_xy = jnp.mean(
      jnp.exp(
          -gamma
          * (
              -2 * jnp.matmul(x, y.T)
              + jnp.expand_dims(x_sqnorms, 1)
              + jnp.expand_dims(y_sqnorms, 0)
          )
      )
  )
  k_yy = jnp.mean(
      jnp.exp(
          -gamma
          * (
              -2 * jnp.matmul(y, y.T)
              + jnp.expand_dims(y_sqnorms, 1)
              + jnp.expand_dims(y_sqnorms, 0)
          )
      )
  )

  return distance._SCALE * (k_xx + k_yy - 2 * k_xy)


class DistanceTest(unittest.TestCase):

  def test_correctness_small(self):
    """Verifies that the optimized mmd yields the exact same results as original mmd."""
    np.random.seed(42)
    x = np.random.normal(size=(20, 16))
    y = np.random.normal(size=(20, 16))

    original_val = _original_mmd(x, y)
    optimized_val = distance.mmd(x, y)

    np.testing.assert_allclose(original_val, optimized_val, rtol=1e-5, atol=1e-5)
    print("Correctness test on small input: PASSED")

  def test_benchmark_large(self):
    """Benchmarks the optimized mmd vs original mmd on larger inputs."""
    np.random.seed(42)
    n = 10000
    d = 512
    x = np.random.normal(size=(n, d)).astype(np.float32)
    y = np.random.normal(size=(n, d)).astype(np.float32)

    print(f"\n--- Benchmarking with N = {n}, D = {d} ---")

    # 1. Eager Mode (No JIT)
    print("Running Eager Mode (No JIT)...")
    with jax.disable_jit():
      t0 = time.time()
      orig_val_eager = _original_mmd(x, y).block_until_ready()
      time_orig_eager = time.time() - t0
      print(f"Original MMD eager time: {time_orig_eager:.6f} seconds")

      t0 = time.time()
      opt_val_eager = distance.mmd(x, y).block_until_ready()
      time_opt_eager = time.time() - t0
      print(f"Optimized MMD eager time: {time_opt_eager:.6f} seconds")
      print(f"Eager Mode Speedup: {time_orig_eager / time_opt_eager:.2f}x")

    # 2. JIT Compiled Mode
    print("Running JIT Compiled Mode...")
    # Compile the functions first to avoid including JIT compilation overhead in benchmark
    _original_mmd_jit = jax.jit(_original_mmd)
    _ = jax.block_until_ready(distance.mmd(x, y))
    _ = jax.block_until_ready(_original_mmd_jit(x, y))

    # Time original mmd
    t0 = time.time()
    orig_val_jit = _original_mmd_jit(x, y)
    orig_val_jit = jax.block_until_ready(orig_val_jit)
    time_orig_jit = time.time() - t0
    print(f"Original MMD compiled time: {time_orig_jit:.6f} seconds")

    # Time optimized mmd
    t0 = time.time()
    opt_val_jit = distance.mmd(x, y)
    opt_val_jit = jax.block_until_ready(opt_val_jit)
    time_opt_jit = time.time() - t0
    print(f"Optimized MMD compiled time: {time_opt_jit:.6f} seconds")
    print(f"Compiled Mode Speedup: {time_orig_jit / time_opt_jit:.2f}x")

    # Verify correctness
    np.testing.assert_allclose(orig_val_eager, opt_val_eager, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(orig_val_jit, opt_val_jit, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
  unittest.main()
