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

from pde_superresolution_2d.core import states
from absl.testing import absltest


class StatesTest(absltest.TestCase):
  """Test class for functions defined in states.py."""

  def _assert_states_equal(self, state_a, state_b):
    """Asserts that two states are equal."""
    self.assertEqual(state_a.name, state_b.name)
    self.assertEqual(state_a.derivative_orders, state_b.derivative_orders)
    self.assertEqual(state_a.offset, state_b.offset)

  def test_with_prefix(self):
    initial_key = states.StateKey('name', (1, 2, 3), (4, 5))
    expected = states.StateKey('exact_name', (1, 2, 3), (4, 5))
    result = initial_key.exact()
    self._assert_states_equal(result, expected)

  def test_time_derivative(self):
    initial_key = states.StateKey('key_a', (1, 2, 3), (4, 5))
    expected = states.StateKey('key_a', (1, 2, 4), (4, 5))
    self._assert_states_equal(expected, initial_key.time_derivative())


if __name__ == '__main__':
  absltest.main()
