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

from absl.testing import parameterized
import numpy as np
from pde_superresolution_2d import metadata_pb2
from pde_superresolution_2d.advection import equations
from pde_superresolution_2d.core import equations as core_equations
from pde_superresolution_2d.core import grids
from pde_superresolution_2d.core import integrate
from pde_superresolution_2d.core import models
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
    equations.VanLeerMono5Advection(),
)
ALL_EQUATIONS = ADVECTION_DIFFUSION_EQUATIONS + ADVECTION_EQUATIONS


class EquationsTest(parameterized.TestCase):

  def test_equation_map(self):
    self.assertIn('advection_diffusion', core_equations.CONTINUOUS_EQUATIONS)
    self.assertIn('advection', core_equations.CONTINUOUS_EQUATIONS)

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

  @parameterized.parameters(*ALL_EQUATIONS)
  def test_random_state(self, equation):
    grid = grids.Grid(200, 200, 2 * np.pi / 200)

    for size in [(), (1,), (2, 3)]:
      with self.subTest(size):
        init_state = equation.random_state(grid, size=size, seed=1)

        # keys and shapes should match
        self.assertEqual(set(init_state), set(equation.STATE_KEYS))
        for array in init_state.values():
          self.assertEqual(array.shape, size + grid.shape)

        # seed should be deterministic
        init_state2 = equation.random_state(grid, size=size, seed=1)
        for key in init_state:
          np.testing.assert_allclose(init_state[key], init_state2[key])

  @parameterized.parameters(*ALL_EQUATIONS)
  def test_take_time_step(self, equation):
    grid = grids.Grid(20, 20, 1)
    state = {k: tf.zeros(grid.shape) for k in equation.STATE_KEYS}
    inputs = {k: tf.zeros(grid.shape) for k in equation.INPUT_KEYS}
    result = equation.take_time_step(state, inputs, grid)
    expected_keys = set(equation.STATE_KEYS) - set(equation.CONSTANT_KEYS)
    self.assertEqual(set(result), expected_keys)

  @parameterized.parameters(*ADVECTION_DIFFUSION_EQUATIONS)
  def test_advection_diffusion_proto_conversion(self, equation):
    equation_proto = equation.to_proto()
    proto = equation_proto.advection_diffusion
    self.assertEqual(proto.diffusion_coefficient,
                     equation.diffusion_coefficient)
    self.assertEqual(proto.cfl_safety_factor, equation.cfl_safety_factor)
    equation_from_proto = core_equations.equation_from_proto(equation_proto)
    self.assertEqual(type(equation_from_proto), type(equation))
    self.assertEqual(equation_from_proto.diffusion_coefficient,
                     equation.diffusion_coefficient)
    self.assertEqual(equation_from_proto.cfl_safety_factor,
                     equation.cfl_safety_factor)

  @parameterized.parameters(*ADVECTION_EQUATIONS)
  def test_advection_proto_conversion(self, equation):
    equation_proto = equation.to_proto()
    proto = equation_proto.advection
    self.assertEqual(proto.cfl_safety_factor, equation.cfl_safety_factor)
    equation_from_proto = core_equations.equation_from_proto(equation_proto)
    self.assertEqual(type(equation_from_proto), type(equation))
    self.assertEqual(equation_from_proto.cfl_safety_factor,
                     equation.cfl_safety_factor)

  @parameterized.parameters(
      # Our upwind schemes should propagate exactly one spatial step per time
      # step if in a constant velocity field with cfl_safety_factor=1.
      # The other schemes will have a small amount of numerical dissipation.
      (equations.UpwindAdvection(cfl_safety_factor=1), 1e-7),
      (equations.UpwindAdvectionDiffusion(
          diffusion_coefficient=0, cfl_safety_factor=1), 1e-7),
      (equations.VanLeerMono5Advection(cfl_safety_factor=1), 1e-7),
      (equations.VanLeerMono5AdvectionDiffusion(
          diffusion_coefficient=0, cfl_safety_factor=1), 1e-7),
      (equations.FiniteDifferenceAdvection(cfl_safety_factor=0.5), 0.015),
      (equations.FiniteDifferenceAdvectionDiffusion(
          diffusion_coefficient=0, cfl_safety_factor=0.5), 0.015),
      (equations.FiniteVolumeAdvection(cfl_safety_factor=0.5), 0.015),
      (equations.FiniteVolumeAdvectionDiffusion(
          diffusion_coefficient=0, cfl_safety_factor=0.5), 0.015),
      (equations.UpwindAdvection(cfl_safety_factor=0.5), 0.015),
      (equations.VanLeerMono5Advection(cfl_safety_factor=0.5), 0.005),
      (equations.VanLeerMono5AdvectionDiffusion(
          diffusion_coefficient=0, cfl_safety_factor=0.5), 0.005),
  )
  def test_integration_in_constant_velocity_field(self, equation, atol):
    grid = grids.Grid.from_period(100, length=100)
    model = models.FiniteDifferenceModel(equation, grid)

    initial_concentration = equations.symmetrized_gaussian(
        grid, 50, 50, gaussian_width=20)
    self.assertGreaterEqual(initial_concentration.min(), 0.0)
    self.assertLessEqual(initial_concentration.max(), 1.0 + 1e-7)

    vx_key, vy_key = equation.CONSTANT_KEYS
    initial_state = {
        equations.C: initial_concentration.astype(np.float32),
        vx_key: np.ones(grid.shape, np.float32),
        vy_key: np.zeros(grid.shape, np.float32)
    }
    steps = round(1 / equation.cfl_safety_factor) * np.arange(10)
    integrated = integrate.integrate_steps(model, initial_state, steps)
    actual = integrated[equations.C].numpy()
    expected = np.stack(
        [np.roll(initial_concentration, i, axis=0) for i in range(10)])
    np.testing.assert_allclose(actual, expected, atol=atol)
    if equation.METHOD == metadata_pb2.Equation.Discretization.FINITE_VOLUME:
      np.testing.assert_allclose(actual[-1].sum(), actual[0].sum(), rtol=1e-7)
    else:
      np.testing.assert_allclose(actual[-1].sum(), actual[0].sum(), rtol=1e-3)
    if equation.MONOTONIC:
      self.assertGreaterEqual(actual[-1].min(), 0.0)
      self.assertLessEqual(actual[-1].max(), 1.0)


if __name__ == '__main__':
  absltest.main()
