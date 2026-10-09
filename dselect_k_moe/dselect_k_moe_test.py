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

"""Tests for dselect_k_moe."""

import unittest
import numpy as np
import tensorflow as tf

from dselect_k_moe.dselect_k_moe import DSelectKGate, EntropyRegularizer, SmoothStep


class SmoothStepTest(unittest.TestCase):

  def test_smooth_step_bounds_and_polynomial(self):
    gamma = 2.0
    ss = SmoothStep(gamma=gamma)

    # Test values
    inputs = tf.constant([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=tf.float32)
    output = ss(inputs).numpy()

    # Lower bound (-gamma/2 = -1.0) -> <= -1.0 should be 0.0
    self.assertAlmostEqual(output[0], 0.0)
    self.assertAlmostEqual(output[1], 0.0)

    # Midpoint (0.0) -> 0.5
    self.assertAlmostEqual(output[3], 0.5)

    # Upper bound (gamma/2 = 1.0) -> >= 1.0 should be 1.0
    self.assertAlmostEqual(output[5], 1.0)
    self.assertAlmostEqual(output[6], 1.0)

    # Polynomial check for x = 0.5 with gamma = 2.0:
    # 3*0.5 / (2*2.0) - 2*(0.5**3) / (2.0**3) + 0.5 = 1.5/4.0 - 0.25/8.0 + 0.5 = 0.375 - 0.03125 + 0.5 = 0.84375
    expected_05 = 3 * 0.5 / (2 * gamma) - 2 * (0.5**3) / (gamma**3) + 0.5
    self.assertAlmostEqual(output[4], expected_05, places=5)


class EntropyRegularizerTest(unittest.TestCase):

  def test_entropy_regularizer(self):
    regularizer = EntropyRegularizer(schedule_fn=lambda x: 0.1)
    probabilities = tf.constant([[0.2, 0.8], [0.5, 0.5]], dtype=tf.float32)

    loss = regularizer(probabilities)
    self.assertGreater(loss.numpy(), 0.0)


class DSelectKGateTest(unittest.TestCase):

  def test_task_only_routing(self):
    num_experts = 4
    num_nonzeros = 2
    gate = DSelectKGate(num_nonzeros=num_nonzeros)

    experts = [tf.random.normal((8, 16)) for _ in range(num_experts)]
    output = gate(experts)

    self.assertEqual(output.shape, (8, 16))

  def test_example_conditioned_routing(self):
    num_experts = 4
    num_nonzeros = 2
    regularizer = EntropyRegularizer()
    gate = DSelectKGate(num_nonzeros=num_nonzeros, entropy_reg=regularizer)

    experts = [tf.random.normal((8, 16)) for _ in range(num_experts)]
    routing_inputs = tf.random.normal((8, 10))

    output = gate((experts, routing_inputs), training=True)
    self.assertEqual(output.shape, (8, 16))
    self.assertGreater(len(gate.losses), 0)

  def test_non_power_of_two_experts(self):
    num_experts = 3  # Non power of 2
    num_nonzeros = 2
    gate = DSelectKGate(num_nonzeros=num_nonzeros)

    experts = [tf.random.normal((4, 8)) for _ in range(num_experts)]
    routing_inputs = tf.random.normal((4, 5))

    output = gate((experts, routing_inputs), training=True)
    self.assertEqual(output.shape, (4, 8))
    self.assertGreater(len(gate.losses), 0)


if __name__ == "__main__":
  unittest.main()
