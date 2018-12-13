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

from typing import List

import numpy as np
import tensorflow as tf

from absl.testing import absltest

from pde_superresolution_2d.advection import equations as advection_equations
from pde_superresolution_2d.advection import velocity_fields
from pde_superresolution_2d.core import equations
from pde_superresolution_2d.core import grids
from pde_superresolution_2d.core import integrator
from pde_superresolution_2d.core import models


class IntegratorTest(absltest.TestCase):
  """Tests Integrator class defined in integrator.py."""

  def setUp(self):
    grid_size = 200
    self.grid = grids.Grid(grid_size, grid_size, 2 * np.pi / grid_size)
    v_field = velocity_fields.ConstantVelocityField.from_seed(12, 6, 100)
    self.finite_vol_eq = advection_equations.FiniteVolumeAdvectionDiffusion(
        v_field, 0.1)
    self.finite_diff_eq = advection_equations.FiniteDifferenceAdvectionDiffusion(
        v_field, 0.1)
    self.base_model = models.RollFiniteDifferenceModel()

  def _integrate(
      self,
      equation: equations.Equation,
      model: models.Model,
      grid: grids.Grid,
      times: np.ndarray
  ) -> List[np.ndarray]:
    solver = integrator.Integrator(equation, model, grid)
    with tf.Graph().as_default():
      init_state = equation.initial_state(
          advection_equations.InitialConditionMethod.GAUSSIAN, grid, seed=1)
      evolved_states = solver.integrate_tf(init_state, times)

      with tf.Session() as sess:
        evolved_states_values = sess.run(evolved_states)
    return evolved_states_values

  def _test_sample_run(self,
                       equation: equations.Equation,
                       model: models.Model,
                       grid: grids.Grid):
    times = np.linspace(0, 50 * equation.get_time_step(grid), 10)
    evolved_states_values = self._integrate(equation, model, grid, times)
    solution_shape = np.shape(evolved_states_values[9][equation.STATE_KEYS[0]])
    self.assertEqual(len(evolved_states_values), 10)
    self.assertEqual(solution_shape, (1,) + self.grid.get_shape())

  def _test_close_time_evolution(self,
                                 equation_a: equations.Equation,
                                 equation_b: equations.Equation,
                                 model: models.Model,
                                 grid: grids.Grid):
    times = np.linspace(0, 0.01, 10)
    evolved_states_values_a = self._integrate(equation_b, model, grid, times)
    evolved_states_values_b = self._integrate(equation_a, model, grid, times)
    for state_key in equation_a.STATE_KEYS:
      np.testing.assert_allclose(
          evolved_states_values_a[-1][state_key],
          evolved_states_values_b[-1][state_key],
          atol=1e-3
      )

  def test_sample_run_finite_volume_equation(self):
    self._test_sample_run(self.finite_vol_eq, self.base_model, self.grid)

  def test_sample_run_finite_difference_equation(self):
    self._test_sample_run(self.finite_diff_eq, self.base_model, self.grid)

  def test_finite_diff_close_to_finite_volume(self):
    self._test_close_time_evolution(self.finite_diff_eq, self.finite_vol_eq,
                                    self.base_model, self.grid)

if __name__ == '__main__':
  absltest.main()
