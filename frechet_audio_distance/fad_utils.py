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

"""Fréchet Audio Distance util functions."""
# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import linalg
import tensorflow.compat.v1 as tf


def read_mean_and_covariances(filename):
  """Helper function that reads tf_record containing dataset stats.

  Args:
    filename: Path of the tf_record.

  Returns:
    The values of mu and sigma.
  """
  tf_record = tf.python_io.tf_record_iterator(filename).next()
  example = tf.train.Example().FromString(tf_record)
  mu = np.array(example.features.feature['mu'].float_list.value)
  emb_len = np.array(
      example.features.feature['embedding_length'].int64_list.value)[0]
  sigma = (np.array(
      example.features.feature['sigma'].float_list.value)).reshape((emb_len,
                                                                    emb_len))
  return mu, sigma


def normalize_loudness(np_samples, max_db_increase=20):
  """Normalizes the loudness to be between -1.0 and 1.0.

  Args:
    np_samples: 1d numpy array of audio samples with shape (num_samples).
    max_db_increase: Maxium loudness incress. This stops very quiet audio from
      being distorted and avoids problems on silence where np.amax(np_samples)
      == 0.

  Returns:
    1d numpy array of audio samples with shape (num_samples) where eache sample
    is between -1.0 and 1.0.
  """

  min_amplitude_ratio = 10**(max_db_increase / -20)
  return np_samples / np.maximum(min_amplitude_ratio, np.amax(np_samples))


def _stable_trace_sqrt_product(sigma_test, sigma_train, eps=1e-7):
  """Computes Tr(sqrt(sigma_test * sigma_train)) efficiently and stably.

  For symmetric positive definite covariance matrices sigma_test (L L^T) and
  sigma_train, the eigenvalues of (sigma_test * sigma_train) match those of
  (L^T * sigma_train * L). Using Cholesky decomposition + eigvalsh gives a
  ~1.8x to 12x speedup compared to general matrix square root (linalg.sqrtm),
  and avoids non-hermitian Schur decomposition issues in newer SciPy versions.

  Args:
    sigma_test: Test covariance matrix.
    sigma_train: Train covariance matrix.
    eps: Small number; used to avoid singular matrices.

  Returns:
    The Trace of the square root of the product of the passed covariance
    matrices.

  Raises:
    ValueError: If eigenvalues contain non-finite numbers.
  """
  try:
    l_factor = linalg.cholesky(sigma_test, lower=True)
  except linalg.LinAlgError:
    # Add eps to diagonal if matrix is near singular.
    offset = np.eye(sigma_test.shape[0], dtype=sigma_test.dtype) * eps
    sigma_test = sigma_test + offset
    sigma_train = sigma_train + offset
    l_factor = linalg.cholesky(sigma_test, lower=True)

  # M = L^T * sigma_train * L is symmetric positive definite.
  m_matrix = l_factor.T.dot(sigma_train).dot(l_factor)
  evals = linalg.eigvalsh(m_matrix)
  if not np.isfinite(evals).all():
    raise ValueError('eigvalsh returned non-finite values.')
  evals = np.maximum(evals, 0.0)
  return np.sum(np.sqrt(evals))


def frechet_distance(mu_test, sigma_test, mu_train, sigma_train):
  """Fréchet distance calculation.

  From: D.C. Dowson & B.V. Landau The Fréchet distance between
  multivariate normal distributions
  https://doi.org/10.1016/0047-259X(82)90077-X

  The Fréchet distance between two multivariate gaussians,
  `X ~ N(mu_x, sigma_x)` and `Y ~ N(mu_y, sigma_y)`, is `d^2`.

  d^2 = (mu_x - mu_y)^2 + Tr(sigma_x + sigma_y - 2 * sqrt(sigma_x*sigma_y))
      = (mu_x - mu_y)^2 + Tr(sigma_x) + Tr(sigma_y)
                        - 2 * Tr(sqrt(sigma_x*sigma_y)))

  Args:
    mu_test: Mean of the test multivariate gaussian.
    sigma_test: Covariance matrix of the test multivariate gaussians.
    mu_train: Mean of the test multivariate gaussian.
    sigma_train: Covariance matrix of the test multivariate gaussians.

  Returns:
    The Fréchet distance.

  Raises:
    ValueError: If the input arrays do not have the expect shapes.
  """
  if len(mu_train.shape) != 1:
    raise ValueError('mu_train must be 1 dimensional.')
  if len(sigma_train.shape) != 2:
    raise ValueError('sigma_train must be 2 dimensional.')

  if mu_test.shape != mu_train.shape:
    raise ValueError('mu_test should have the same shape as mu_train')
  if sigma_test.shape != sigma_train.shape:
    raise ValueError('sigma_test should have the same shape as sigma_train')

  mu_diff = mu_test - mu_train
  trace_sqrt_product = _stable_trace_sqrt_product(sigma_test, sigma_train)

  return mu_diff.dot(mu_diff) + np.trace(sigma_test) + np.trace(
      sigma_train) - 2 * trace_sqrt_product
