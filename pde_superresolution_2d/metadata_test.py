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
"""Tests the DatasetMeta protocol buffer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pde_superresolution_2d import metadata_pb2
from pde_superresolution_2d.advection import equations as advection_equations
from pde_superresolution_2d.core import equations
from pde_superresolution_2d.core import grids
from pde_superresolution_2d.core import models
from absl.testing import absltest


class MetadataTest(absltest.TestCase):

  def setUp(self):
    grid_size = 200
    self.grid = grids.Grid(grid_size, grid_size, 2 * np.pi / grid_size)
    self.equation = advection_equations.FiniteDifferenceAdvectionDiffusion(
        diffusion_coefficient=0.1)
    self.model = models.FiniteDifferenceModel(self.equation, self.grid)
    grid_proto = self.grid.to_proto()
    equation_proto = self.equation.to_proto()
    model_proto = self.model.to_proto()
    self.metadata = metadata_pb2.Dataset(equation=equation_proto,
                                         model=model_proto,
                                         simulation_grid=grid_proto)

  def test_proto_conversion(self):
    self.assertEqual(
        self.metadata.simulation_grid.size_x, self.grid.size_x)
    self.assertEqual(
        self.metadata.simulation_grid.size_y, self.grid.size_y)
    self.assertAlmostEqual(
        self.metadata.simulation_grid.step, self.grid.step)

  def test_proto_reconstruction(self):
    grid_from_proto = grids.Grid.from_proto(self.metadata.simulation_grid)
    self.assertEqual(grid_from_proto.size_x, self.grid.size_x)
    self.assertEqual(grid_from_proto.size_y, self.grid.size_y)
    self.assertAlmostEqual(grid_from_proto.step, self.grid.step)

    equation_from_proto = equations.equation_from_proto(
        self.metadata.equation)
    self.assertIsInstance(
        equation_from_proto,
        advection_equations.FiniteDifferenceAdvectionDiffusion)
    self.assertEqual(equation_from_proto.diffusion_coefficient, 0.1)

if __name__ == '__main__':
  absltest.main()
