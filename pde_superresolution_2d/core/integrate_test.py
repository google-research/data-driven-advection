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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from absl.testing import absltest

from pde_superresolution_2d.advection import equations as advection_equations
from pde_superresolution_2d.core import grids
from pde_superresolution_2d.core import integrate
from pde_superresolution_2d.core import models


tf.enable_eager_execution()


class IntegrateTest(absltest.TestCase):

  def setUp(self):
    grid_size = 200
    self.grid = grids.Grid(grid_size, grid_size, 2 * np.pi / grid_size)
    self.finite_vol_eq = (
        advection_equations.FiniteVolumeAdvectionDiffusion(0.05))
    self.finite_diff_eq = (
        advection_equations.FiniteDifferenceAdvectionDiffusion(0.05))
    self.upwind_eq = (
        advection_equations.UpwindAdvectionDiffusion(0.05))
    self.base_vol_model = models.FiniteDifferenceModel(
        self.finite_vol_eq, self.grid)
    self.base_diff_model = models.FiniteDifferenceModel(
        self.finite_diff_eq, self.grid)
    self.base_upwind_model = models.FiniteDifferenceModel(
        self.upwind_eq, self.grid)

  def _integrate(self, equation, model, grid, times):
    params = dict(
        concentration=dict(method='sum_of_gaussians', gaussian_width=0.5),
    )
    init_state = equation.random_state(seed=1, grid=grid, params=params)
    return integrate.integrate_times(model, init_state, times)

  def _test_sample_run(self, equation, model, grid):
    times = np.linspace(0, 50 * equation.get_time_step(grid), 11)
    evolved_states = self._integrate(equation, model, grid, times)
    self.assertEqual(equation.base_keys, set(evolved_states))
    self.assertEqual(evolved_states['concentration'].shape,
                     (11,) + self.grid.shape)

  def test_sample_run_finite_volume_equation(self):
    self._test_sample_run(self.finite_vol_eq, self.base_vol_model, self.grid)

  def test_sample_run_finite_difference_equation(self):
    self._test_sample_run(self.finite_diff_eq, self.base_diff_model, self.grid)

  def test_sample_run_upwind_equation(self):
    self._test_sample_run(self.upwind_eq, self.base_upwind_model, self.grid)

  def test_all_close(self):
    times = np.arange(10) * self.finite_diff_eq.get_time_step(self.grid)
    solution_diff = self._integrate(
        self.finite_diff_eq, self.base_diff_model, self.grid, times)
    solution_vol = self._integrate(
        self.finite_diff_eq, self.base_diff_model, self.grid, times)
    solution_upwind = self._integrate(
        self.upwind_eq, self.base_upwind_model, self.grid, times)
    np.testing.assert_allclose(
        solution_diff['concentration'][-1], solution_vol['concentration'][-1],
        atol=0.001, rtol=0.01,
    )
    np.testing.assert_allclose(
        solution_upwind['concentration'][-1], solution_vol['concentration'][-1],
        atol=0.001, rtol=0.01,
    )

if __name__ == '__main__':
  absltest.main()
