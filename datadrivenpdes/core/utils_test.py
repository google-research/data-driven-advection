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
from datadrivenpdes.core import grids
from datadrivenpdes.core import states
from datadrivenpdes.core import utils
from absl.testing import absltest


class UtilsTest(absltest.TestCase):

  def test_component_name(self):
    concentration = states.StateDefinition(
        'concentration', (), (0, 0, 0), (0, 0))
    grid = grids.Grid.from_period(32, 0.1)
    self.assertEqual(utils.component_name(concentration, grid),
                     'concentration/0_0_0/0_0/32_32')

    x_quantity = states.StateDefinition(
        'quantity', (states.Dimension.X,), (1, 2, 3), (1, 0))
    self.assertEqual(utils.component_name(x_quantity), 'x_quantity/1_2_3/1_0')


if __name__ == '__main__':
  absltest.main()
