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
from pde_superresolution_2d.advection import equations as advection_equations
from pde_superresolution_2d.advection import velocity_fields
from pde_superresolution_2d.core import grids
from pde_superresolution_2d.core import tensor_ops
from pde_superresolution_2d.core import states
import tensorflow as tf

from absl.testing import absltest


tf.enable_eager_execution()


class ConstantVelocityFieldTest(absltest.TestCase):
  """Test classes in velocity_field.py."""

  def setUp(self):
    self.grid = grids.Grid.from_period(100, 2 * np.pi)

  def test_random_seed_effect(self):
    vfield_a = velocity_fields.ConstantVelocityField.from_seed(seed=3)
    vfield_b = velocity_fields.ConstantVelocityField.from_seed(seed=4)
    vfield_c = velocity_fields.ConstantVelocityField.from_seed(seed=3)

    velocity_a_x = vfield_a.get_velocity_x(0, self.grid)
    velocity_a_y = vfield_a.get_velocity_y(0, self.grid)
    velocity_b_x = vfield_b.get_velocity_x(0, self.grid)
    velocity_b_y = vfield_b.get_velocity_y(0, self.grid)
    velocity_c_x = vfield_c.get_velocity_x(0, self.grid)
    velocity_c_y = vfield_c.get_velocity_y(0, self.grid)

    np.testing.assert_allclose(velocity_a_x, velocity_c_x)
    np.testing.assert_allclose(velocity_a_y, velocity_c_y)
    with self.assertRaises(AssertionError):
      np.testing.assert_allclose(velocity_a_x, velocity_b_x)
    with self.assertRaises(AssertionError):
      np.testing.assert_allclose(velocity_a_y, velocity_b_y)

  def test_shift_values(self):
    x_shift = (1, 0)
    y_shift = (0, 1)

    vfield = velocity_fields.ConstantVelocityField.from_seed(seed=0)

    no_shift_velocity_x = vfield.get_velocity_x(0., self.grid)
    no_shift_velocity_y = vfield.get_velocity_y(0., self.grid)
    x_shift_velocity_x = vfield.get_velocity_x(0., self.grid, x_shift)
    x_shift_velocity_y = vfield.get_velocity_y(0., self.grid, x_shift)
    y_shift_velocity_x = vfield.get_velocity_x(0., self.grid, y_shift)
    y_shift_velocity_y = vfield.get_velocity_y(0., self.grid, y_shift)
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
    vfield = velocity_fields.ConstantVelocityField.from_seed(seed=0)
    velocity_proto = vfield.to_proto().constant_v_field
    np.testing.assert_allclose(np.asarray(velocity_proto.amplitudes),
                               vfield.amplitudes, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(velocity_proto.x_wavenumbers),
                               vfield.x_wavenumbers, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(velocity_proto.y_wavenumbers),
                               vfield.y_wavenumbers, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(velocity_proto.phase_shifts),
                               vfield.phase_shifts, rtol=1e-6)

  def test_face_average(self):
    # construct a velocity field that covers every branch in the integral
    vfield = velocity_fields.ConstantVelocityField(
        x_wavenumbers=np.array([1, 0, 2, 0]),
        y_wavenumbers=np.array([-1, 3, 0, 0]),
        amplitudes=np.array([1.1, 1.2, 1.3, 1.4]),
        phase_shifts=np.arange(4.0),
    )
    high_res_grid = grids.Grid.from_period(256, length=5)
    low_res_grid = grids.Grid.from_period(32, length=5)

    with self.subTest('point and face-average match at high resolution'):
      kwargs = dict(t=0, grid=high_res_grid)

      vx_point = vfield.get_velocity_x(face_average=False, **kwargs)
      vy_point = vfield.get_velocity_y(face_average=False, **kwargs)

      vx_average = vfield.get_velocity_x(face_average=True, **kwargs)
      vy_average = vfield.get_velocity_y(face_average=True, **kwargs)

      np.testing.assert_allclose(vx_point, vx_average, atol=1e-3)
      np.testing.assert_allclose(vy_point, vy_average, atol=1e-3)

    with self.subTest('average high-res and low-res average match'):
      kwargs = dict(t=0, face_average=True)

      vx_low = vfield.get_velocity_x(grid=low_res_grid, shift=(1, 0), **kwargs)
      vy_low = vfield.get_velocity_y(grid=low_res_grid, shift=(0, 1), **kwargs)

      vx_high = vfield.get_velocity_x(
          grid=high_res_grid, shift=(1, 0), **kwargs)
      vy_high = vfield.get_velocity_y(
          grid=high_res_grid, shift=(0, 1), **kwargs)

      vx_def = states.StateDefinition(
          'v', (states.Dimension.X,), (0, 0, 0), offset=(1, 0))
      vy_def = states.StateDefinition(
          'v', (states.Dimension.Y,), (0, 0, 0), offset=(0, 1))
      vx_high_average = tensor_ops.regrid(
          vx_high, vx_def, high_res_grid, low_res_grid)
      vy_high_average = tensor_ops.regrid(
          vy_high, vy_def, high_res_grid, low_res_grid)

      np.testing.assert_allclose(vx_low, vx_high_average)
      np.testing.assert_allclose(vy_low, vy_high_average)

    with self.subTest('divergence free'):
      divergence = advection_equations.flux_to_time_derivative(
          vx_low, vy_low, grid_step=1.0)
      np.testing.assert_allclose(divergence, np.zeros_like(divergence),
                                 atol=1e-8)

  def test_normalize(self):

    def maximum_velocity(vfield):
      vx = vfield.get_velocity_x(0, self.grid)
      vy = vfield.get_velocity_y(0, self.grid)
      return np.sqrt(vx ** 2 + vy ** 2).max()

    original_vfield = velocity_fields.ConstantVelocityField.from_seed(
        max_periods=2, power_law=-2, seed=0, normalize=False)
    original_max_velocity = maximum_velocity(original_vfield)
    self.assertNotAlmostEqual(original_max_velocity, 1.0, places=2)

    new_max_velocity = maximum_velocity(original_vfield.normalize())
    self.assertAlmostEqual(new_max_velocity, 1.0, places=3)


if __name__ == '__main__':
  absltest.main()
