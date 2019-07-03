# python3
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
from datadrivenpdes.core import states
from absl.testing import absltest


class StatesTest(absltest.TestCase):
  """Test class for functions defined in states.py."""

  def test_swap_xy(self):
    concentration = states.StateDefinition(
        'concentration', (), (0, 0, 0), (0, 0))
    self.assertEqual(concentration.swap_xy(), concentration)

    x_quantity = states.StateDefinition(
        'quantity', (states.Dimension.X,), (1, 2, 3), (1, 0))
    y_quantity = states.StateDefinition(
        'quantity', (states.Dimension.Y,), (2, 1, 3), (0, 1))
    self.assertEqual(x_quantity.swap_xy(), y_quantity)
    self.assertEqual(y_quantity.swap_xy(), x_quantity)

  def test_with_prefix(self):
    initial_key = states.StateDefinition('name', (), (1, 2, 3), (4, 5))
    expected = states.StateDefinition('exact_name', (), (1, 2, 3), (4, 5))
    result = initial_key.exact()
    self.assertEqual(result, expected)

  def test_time_derivative(self):
    initial_key = states.StateDefinition('key_a', (), (1, 2, 3), (4, 5))
    expected = states.StateDefinition('key_a', (), (1, 2, 4), (4, 5))
    self.assertEqual(expected, initial_key.time_derivative())


if __name__ == '__main__':
  absltest.main()
