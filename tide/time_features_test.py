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

"""Tests for time_features."""

import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tide import time_features


class TimeFeaturesTest(unittest.TestCase):

  def test_get_covariates_no_holiday(self):
    dti = pd.date_range('2021-01-01', periods=24, freq='1h')
    tc = time_features.TimeCovariates(dti, holiday=False)
    df = tc.get_covariates()

    self.assertEqual(df.shape, (24, 7))
    self.assertListEqual(
        list(df.columns), ['moh', 'hod', 'dom', 'dow', 'doy', 'moy', 'woy']
    )
    pd.testing.assert_index_equal(df.index, dti)

  def test_get_covariates_with_holiday(self):
    dti = pd.date_range('2021-01-01', periods=100, freq='1h')
    tc = time_features.TimeCovariates(dti, holiday=True)
    df = tc.get_covariates()

    expected_cols = ['moh', 'hod', 'dom', 'dow', 'doy', 'moy', 'woy'] + [
        f'hol_{i}' for i in range(len(time_features.HOLIDAYS))
    ]
    self.assertEqual(df.shape, (100, 7 + len(time_features.HOLIDAYS)))
    self.assertListEqual(list(df.columns), expected_cols)
    pd.testing.assert_index_equal(df.index, dti)
    self.assertFalse(df.isna().any().any())

  def test_get_holidays_numerical_equivalence_to_reference(self):
    # Verify exact numerical equivalence against row-by-row _distance_to_holiday reference
    dti = pd.date_range('2021-01-01', periods=150, freq='6h')
    dti_series = dti.to_series()

    ref_hol_variates = np.vstack([
        dti_series.apply(time_features._distance_to_holiday(h)).values
        for h in time_features.HOLIDAYS
    ])
    ref_hol_covs = StandardScaler().fit_transform(ref_hol_variates.T).T

    tc = time_features.TimeCovariates(dti, holiday=True)
    fast_hol_covs = tc._get_holidays()

    np.testing.assert_allclose(fast_hol_covs, ref_hol_covs, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
  unittest.main()
