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
"""Grid class holds details of discretization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from typing import Tuple

from pde_superresolution_2d import metadata_pb2


class Grid(object):
  """Grid class provides interface to discretization.

  Grid class keeps track of size of the computational domain, spatial step,
  provides helper functions for initialization and evaluation of objects that
  depend on discretization (e.g. velocity fields).

  Attributes:
    size_x: Number of finite volumes along x direction
    size_y: Number of finite volumes along y direction
    x: Length of the computational domain in arbitrary units
    y: Width of the computational domain in arbitrary units
    step: Spatial size of the finite volume cell
  """

  def __init__(self, size_x: int, size_y: int, step: float):
    """Initializes the grid."""
    self.size_x = size_x
    self.size_y = size_y
    self.length_x = step * size_x
    self.length_y = step * size_y
    self.step = step

  def get_mesh(
      self, shift: Tuple[int, int] = (0, 0)) -> Tuple[np.ndarray, np.ndarray]:
    """Generates grid mesh for function evaluation.

    Generates mesh with option of shifting by half of the unit cell, which is
    needed for evaluation of fluxes on the boundaries of unit cells.

    Args:
      shift: Number of x and y half-step shifts along corresponding axis.

    Returns:
      x and y coordinates on the mesh, each of shape=[size_x, size_y]
      and dtype=float64

    Raises:
      ValueError: the size of the shift does not correspond to 2 dimensions.
    """
    if len(shift) != 2:
      raise ValueError('shift length must be equal to two')
    half_step = self.step / 2.
    shift_x = shift[0] * half_step
    shift_y = shift[1] * half_step
    return np.meshgrid(
        np.linspace(
            shift_x, self.length_x + shift_x, self.size_x, endpoint=False),
        np.linspace(
            shift_y, self.length_y + shift_y, self.size_y, endpoint=False),
        indexing='ij')

  def get_shape(self) -> Tuple[int, int]:
    """Returns the shape of the grid."""
    return (self.size_x, self.size_y)

  def to_proto(self) -> metadata_pb2.Grid:
    """Creates a protocol buffer encoding the grid object.

    Returns:
      Protocol buffer encoding the grid object.
    """
    return metadata_pb2.Grid(size_x=self.size_x,
                             size_y=self.size_y,
                             step=self.step)


def grid_from_proto(proto: metadata_pb2.Grid) -> Grid:
  """Constructs Grid object from proto.

  Args:
    proto: Protocol buffer encoding the grid object.

  Returns:
    Grid object constructed from the grid protocol buffer.
  """
  return Grid(proto.size_x, proto.size_y, proto.step)

