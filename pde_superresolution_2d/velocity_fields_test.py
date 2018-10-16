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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from absl.testing import absltest

from pde_superresolution_2d import grids
from pde_superresolution_2d import velocity_fields


class ConstantVelocityFieldTest(absltest.TestCase):
  """Test classes in velocity_field.py."""

  def setUp(self):
    length = 2 * np.pi
    size = 400
    step = length / size
    self.grid = grids.Grid(size, size, step)
    num_terms = 4
    max_periods = 2
    seed_a = 3
    seed_b = 4
    seed_c = 3
    self.test_velocity_a = velocity_fields.ConstantVelocityField.from_seed(
        num_terms, max_periods, seed_a)
    self.test_velocity_b = velocity_fields.ConstantVelocityField.from_seed(
        num_terms, max_periods, seed_b)
    self.test_velocity_c = velocity_fields.ConstantVelocityField.from_seed(
        num_terms, max_periods, seed_c)

  def test_random_seed_effect(self):
    velocity_a_x = self.test_velocity_a.get_velocity_x(0, self.grid)
    velocity_a_y = self.test_velocity_a.get_velocity_y(0, self.grid)
    velocity_b_x = self.test_velocity_b.get_velocity_x(0, self.grid)
    velocity_b_y = self.test_velocity_b.get_velocity_y(0, self.grid)
    velocity_c_x = self.test_velocity_c.get_velocity_x(0, self.grid)
    velocity_c_y = self.test_velocity_c.get_velocity_y(0, self.grid)
    np.testing.assert_allclose(velocity_a_x, velocity_c_x)
    np.testing.assert_allclose(velocity_a_y, velocity_c_y)
    with self.assertRaises(AssertionError):
      np.testing.assert_allclose(velocity_a_x, velocity_b_x)
    with self.assertRaises(AssertionError):
      np.testing.assert_allclose(velocity_a_y, velocity_b_y)

  def test_shift_values(self):
    x_shift = (1, 0)
    y_shift = (0, 1)
    no_shift_velocity_x = self.test_velocity_a.get_velocity_x(0., self.grid)
    no_shift_velocity_y = self.test_velocity_a.get_velocity_y(0., self.grid)
    x_shift_velocity_x = self.test_velocity_a.get_velocity_x(
        0., self.grid, x_shift)
    x_shift_velocity_y = self.test_velocity_a.get_velocity_y(
        0., self.grid, x_shift)
    y_shift_velocity_x = self.test_velocity_a.get_velocity_x(
        0., self.grid, y_shift)
    y_shift_velocity_y = self.test_velocity_a.get_velocity_y(
        0., self.grid, y_shift)
    x_shift_approx_v_x = 0.5 * (
        np.roll(no_shift_velocity_x, (-1, 0)) + no_shift_velocity_x)
    x_shift_approx_v_y = 0.5 * (
        np.roll(no_shift_velocity_y, (-1, 0)) + no_shift_velocity_y)
    y_shift_approx_v_x = 0.5 * (
        np.roll(no_shift_velocity_x, (0, -1)) + no_shift_velocity_x)
    y_shift_approx_v_y = 0.5 * (
        np.roll(no_shift_velocity_y, (0, -1)) + no_shift_velocity_y)
    np.testing.assert_allclose(x_shift_velocity_x, x_shift_approx_v_x, atol=0.1)
    np.testing.assert_allclose(x_shift_velocity_y, x_shift_approx_v_y, atol=0.1)
    np.testing.assert_allclose(y_shift_velocity_x, y_shift_approx_v_x, atol=0.1)
    np.testing.assert_allclose(y_shift_velocity_y, y_shift_approx_v_y, atol=0.1)

  def test_proto_conversion(self):
    velocity_proto = self.test_velocity_a.to_proto().constant_v_field
    self.assertEqual(velocity_proto.num_terms, self.test_velocity_a.num_terms)
    self.assertEqual(velocity_proto.max_periods,
                     self.test_velocity_a.max_periods)
    np.testing.assert_allclose(np.asarray(velocity_proto.amplitudes),
                               self.test_velocity_a.amplitudes, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(velocity_proto.x_wavenumbers),
                               self.test_velocity_a.x_wavenumbers, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(velocity_proto.y_wavenumbers),
                               self.test_velocity_a.y_wavenumbers, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(velocity_proto.phase_shifts),
                               self.test_velocity_a.phase_shifts, rtol=1e-6)

if __name__ == '__main__':
  absltest.main()
