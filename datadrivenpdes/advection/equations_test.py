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
from absl.testing import parameterized
import numpy as np
from datadrivenpdes.advection import equations
from datadrivenpdes.core import equations as core_equations
from datadrivenpdes.core import grids
from datadrivenpdes.core import integrate
from datadrivenpdes.core import models
from datadrivenpdes.core import polynomials
import tensorflow as tf

from absl.testing import absltest


tf.enable_eager_execution()

# TODO(dkochkov) update initialization and bring back boundary test.

ADVECTION_DIFFUSION_EQUATIONS = (
    equations.FiniteDifferenceAdvectionDiffusion(diffusion_coefficient=0.001),
    equations.FiniteVolumeAdvectionDiffusion(diffusion_coefficient=0.001),
    equations.UpwindAdvectionDiffusion(diffusion_coefficient=0.001),
)
ADVECTION_EQUATIONS = (
    equations.FiniteDifferenceAdvection(),
    equations.FiniteVolumeAdvection(),
    equations.UpwindAdvection(),
    equations.VanLeerAdvection(),
)
ALL_EQUATIONS = ADVECTION_DIFFUSION_EQUATIONS + ADVECTION_EQUATIONS


class EquationsTest(parameterized.TestCase):

  def test_random_sum_of_gaussians_shape(self):
    grid = grids.Grid.from_period(100, length=10)
    result = equations.random_sum_of_gaussians(grid)
    self.assertEqual(result.shape, (100, 100))

    result = equations.random_sum_of_gaussians(
        grid, num_terms=3, gaussian_width=0.2)
    self.assertEqual(result.shape, (100, 100))

    result = equations.random_sum_of_gaussians(grid, size=2)
    self.assertEqual(result.shape, (2, 100, 100))

    result = equations.random_sum_of_gaussians(grid, size=(2, 3))
    self.assertEqual(result.shape, (2, 3, 100, 100))

  def test_random_fourier_series_shape(self):
    grid = grids.Grid.from_period(100, length=10)
    result = equations.random_fourier_series(grid)
    self.assertEqual(result.shape, (100, 100))

    result = equations.random_fourier_series(grid, size=2)
    self.assertEqual(result.shape, (2, 100, 100))

    result = equations.random_fourier_series(grid, size=(2, 3))
    self.assertEqual(result.shape, (2, 3, 100, 100))

  def test_random_fourier_series_convergence(self):
    grid = grids.Grid.from_period(100, length=10)
    result_6 = equations.random_fourier_series(
        grid, seed=0, max_periods=6, power_law=-4)
    result_7 = equations.random_fourier_series(
        grid, seed=0, max_periods=7, power_law=-4)
    self.assertGreater(result_6.max(), 0.9)
    self.assertLess(result_6.min(), 0.1)
    np.testing.assert_allclose(result_6, result_7, atol=0.01)

  def test_binarize(self):
    x = np.linspace(0, 1, num=7)
    y = equations.binarize(x, slope=100)
    expected = np.array([0, 0, 0, 0.5, 1, 1, 1])
    np.testing.assert_allclose(y, expected, atol=1e-6)

  @parameterized.parameters(*ALL_EQUATIONS)
  def test_random_state(self, equation):
    grid = grids.Grid(200, 200, 2 * np.pi / 200)

    for size in [(), (1,), (2, 3)]:
      with self.subTest(size):
        init_state = equation.random_state(grid, size=size, seed=1)

        # keys and shapes should match
        self.assertEqual(set(init_state), equation.base_keys)
        for array in init_state.values():
          self.assertEqual(array.shape, size + grid.shape)

        # seed should be deterministic
        init_state2 = equation.random_state(grid, size=size, seed=1)
        for key in init_state:
          np.testing.assert_allclose(init_state[key], init_state2[key])

  @parameterized.parameters(*ALL_EQUATIONS)
  def test_take_time_step(self, equation):
    grid = grids.Grid(20, 20, 1)
    inputs = {k: tf.zeros(grid.shape) for k in equation.key_definitions}
    result = equation.take_time_step(grid, **inputs)
    self.assertEqual(set(result), equation.evolving_keys)

  @parameterized.parameters(*ADVECTION_DIFFUSION_EQUATIONS)
  def test_advection_diffusion_config_conversion(self, equation):
    config = equation.to_config()
    expected = {
        'continuous_equation': 'advection_diffusion',
        'discretization': equation.DISCRETIZATION_NAME,
        'parameters': {
            'diffusion_coefficient': equation.diffusion_coefficient,
            'cfl_safety_factor': equation.cfl_safety_factor,
        },
    }
    self.assertEqual(config, expected)

    equation_from_config = core_equations.equation_from_config(config)
    self.assertEqual(type(equation_from_config), type(equation))
    self.assertEqual(equation_from_config.diffusion_coefficient,
                     equation.diffusion_coefficient)
    self.assertEqual(equation_from_config.cfl_safety_factor,
                     equation.cfl_safety_factor)

  @parameterized.parameters(*ADVECTION_EQUATIONS)
  def test_advection_config_conversion(self, equation):
    config = equation.to_config()
    expected = {
        'continuous_equation': 'advection',
        'discretization': equation.DISCRETIZATION_NAME,
        'parameters': {
            'cfl_safety_factor': equation.cfl_safety_factor,
        },
    }
    self.assertEqual(config, expected)

    equation_from_config = core_equations.equation_from_config(config)
    self.assertEqual(type(equation_from_config), type(equation))
    self.assertEqual(equation_from_config.cfl_safety_factor,
                     equation.cfl_safety_factor)

  @parameterized.parameters(
      # Our upwind schemes should propagate exactly one spatial step per time
      # step if in a constant velocity field with cfl_safety_factor=1.
      # The other schemes will have a small amount of numerical dissipation.
      (equations.UpwindAdvection(cfl_safety_factor=1), 1e-7),
      (equations.UpwindAdvectionDiffusion(
          diffusion_coefficient=0, cfl_safety_factor=1), 1e-7),
      (equations.VanLeerAdvection(cfl_safety_factor=1), 1e-7),
      (equations.VanLeerMono5AdvectionDiffusion(
          diffusion_coefficient=0, cfl_safety_factor=1), 1e-7),
      (equations.FiniteDifferenceAdvection(cfl_safety_factor=0.5), 0.015),
      (equations.FiniteDifferenceAdvectionDiffusion(
          diffusion_coefficient=0, cfl_safety_factor=0.5), 0.015),
      (equations.FiniteVolumeAdvection(cfl_safety_factor=0.5), 0.015),
      (equations.FiniteVolumeAdvectionDiffusion(
          diffusion_coefficient=0, cfl_safety_factor=0.5), 0.015),
      (equations.UpwindAdvection(cfl_safety_factor=0.5), 0.015),
      (equations.VanLeerMono5AdvectionDiffusion(
          diffusion_coefficient=0, cfl_safety_factor=0.5), 0.002),
      (equations.VanLeerAdvection(cfl_safety_factor=0.5), 0.002),
      (equations.VanLeerAdvection(cfl_safety_factor=0.5,
                                  limiter=equations.Limiter.GLOBAL), 0.002),
      (equations.VanLeerAdvection(cfl_safety_factor=0.5,
                                  limiter=equations.Limiter.POSITIVE), 0.002),
      (equations.VanLeerAdvection(cfl_safety_factor=0.5,
                                  limiter=equations.Limiter.NONE), 0.002),
  )
  def test_integration_in_constant_velocity_field(self, equation, atol):
    grid = grids.Grid.from_period(100, length=100)
    model = models.FiniteDifferenceModel(equation, grid)

    initial_concentration = equations.symmetrized_gaussian(
        grid, 50, 50, gaussian_width=20)
    self.assertGreaterEqual(initial_concentration.min(), 0.0)
    self.assertLessEqual(initial_concentration.max(), 1.0 + 1e-7)

    initial_state = {
        'concentration': initial_concentration.astype(np.float32),
        'x_velocity': np.ones(grid.shape, np.float32),
        'y_velocity': np.zeros(grid.shape, np.float32)
    }
    steps = round(1 / equation.cfl_safety_factor) * np.arange(10)
    integrated = integrate.integrate_steps(model, initial_state, steps)
    actual = integrated['concentration'].numpy()
    expected = np.stack(
        [np.roll(initial_concentration, i, axis=0) for i in range(10)])
    np.testing.assert_allclose(actual, expected, atol=atol)
    if equation.METHOD is polynomials.Method.FINITE_VOLUME:
      np.testing.assert_allclose(actual[-1].sum(), actual[0].sum(), rtol=1e-7)
    else:
      np.testing.assert_allclose(actual[-1].sum(), actual[0].sum(), rtol=1e-3)
    if equation.MONOTONIC:
      self.assertGreaterEqual(actual[-1].min(), 0.0)
      self.assertLessEqual(actual[-1].max(), 1.0)


if __name__ == '__main__':
  absltest.main()
