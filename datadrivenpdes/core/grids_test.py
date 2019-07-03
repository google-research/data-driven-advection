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
import numpy as np
from datadrivenpdes.core import grids

from absl.testing import absltest


class GridTest(absltest.TestCase):
  """Test class for grid.py."""

  def setUp(self):
    self.length = 2 * np.pi
    self.size = 20
    self.step = self.length / self.size
    self.test_grid = grids.Grid(self.size, self.size, self.step)

  def test_grid_sizes(self):
    expected_length = self.length
    self.assertAlmostEqual(self.test_grid.length_x, expected_length)
    self.assertAlmostEqual(self.test_grid.length_y, expected_length)

  def test_mesh_shifts(self):
    mesh_x, mesh_y = self.test_grid.get_mesh()
    self.assertAlmostEqual(mesh_x[-1, 0], self.length - self.step / 2)
    self.assertAlmostEqual(mesh_y[0, -1], self.length - self.step / 2)
    shifted_x, shifted_y = self.test_grid.get_mesh(shift=(1, 0))
    half_step = self.step / 2.
    np.testing.assert_allclose(shifted_x - mesh_x,
                               half_step * np.ones((self.size, self.size)))
    np.testing.assert_allclose(shifted_y - mesh_y,
                               np.zeros((self.size, self.size)))
    # do the same thing with y shift
    shifted_x, shifted_y = self.test_grid.get_mesh(shift=(0, 1))
    np.testing.assert_allclose(shifted_y - mesh_y,
                               half_step * np.ones((self.size, self.size)))
    np.testing.assert_allclose(shifted_x - mesh_x,
                               np.zeros((self.size, self.size)))
    with self.assertRaises(ValueError):
      self.test_grid.get_mesh((1, 1, 0, 1))

  def test_shape(self):
    expected_shape = (self.size, self.size)
    test_shape = self.test_grid.shape
    self.assertEqual(expected_shape, test_shape)

  def test_config(self):
    restored = grids.Grid.from_config(self.test_grid.to_config())
    self.assertEqual(restored, self.test_grid)


if __name__ == '__main__':
  absltest.main()
