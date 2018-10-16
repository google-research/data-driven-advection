# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests functions in states.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from absl.testing import absltest

from pde_superresolution_2d import states


class StatesTest(absltest.TestCase):
  """Test class for functions defined in states.py."""

  def _assert_states_equal(self, state_a, state_b):
    """Asserts that two states are equal."""
    self.assertEqual(state_a.name, state_b.name)
    self.assertEqual(state_a.derivative_orders, state_b.derivative_orders)
    self.assertEqual(state_a.offset, state_b.offset)

  def test_add_prefix(self):
    """Tests add_prefix function."""
    test_prefix = 'test_'
    initial_key = states.StateKey('prefix', (1, 2, 3), (4, 5))
    expected = states.StateKey('test_prefix', (1, 2, 3), (4, 5))
    result = states.add_prefix(test_prefix, initial_key)
    self._assert_states_equal(result, expected)

  def test_add_prefix_tuple(self):
    """Tests add_prefix_tuple function."""
    test_prefix = 'second_test_'
    initial_key_a = states.StateKey('key_a', (1, 2, 3), (4, 5))
    initial_key_b = states.StateKey('key_b', (3, 1, 2), (0, 0))
    initial_tuple = (initial_key_a, initial_key_b)
    prefixed_key_a = states.StateKey('second_test_key_a', (1, 2, 3), (4, 5))
    prefixed_key_b = states.StateKey('second_test_key_b', (3, 1, 2), (0, 0))
    expected_tuple = (prefixed_key_a, prefixed_key_b)
    result_tuple = states.add_prefix_tuple(test_prefix, initial_tuple)
    for state_key_a, state_key_b in zip(result_tuple, expected_tuple):
      self._assert_states_equal(state_key_a, state_key_b)

  def test_add_prefix_keys(self):
    """Tests add_prefix_keys function."""
    test_prefix = 'third_test_'
    initial_key_a = states.StateKey('key_a', (1, 2, 3), (4, 5))
    initial_key_b = states.StateKey('key_b', (3, 1, 2), (0, 0))
    data_a = np.eye(5)
    data_b = np.ones((5, 5))
    initial_dict = {initial_key_a: data_a, initial_key_b: data_b}
    prefixed_key_a = states.StateKey('third_test_key_a', (1, 2, 3), (4, 5))
    prefixed_key_b = states.StateKey('third_test_key_b', (3, 1, 2), (0, 0))
    expected_dict = {prefixed_key_a: data_a, prefixed_key_b: data_b}
    result_dict = states.add_prefix_keys(test_prefix, initial_dict)
    for key_a, value_a in result_dict.items():
      self.assertIn(key_a, expected_dict)
      np.testing.assert_allclose(value_a, expected_dict[key_a])

  def test_add_time_derivative(self):
    """Tests add_time_derivative function."""
    initial_key = states.StateKey('key_a', (1, 2, 3), (4, 5))
    expected = states.StateKey('key_a', (1, 2, 4), (4, 5))
    self._assert_states_equal(expected, states.add_time_derivative(initial_key))


if __name__ == '__main__':
  absltest.main()
