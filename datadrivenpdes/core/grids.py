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
"""Grid class holds details of discretization."""
import typing
from typing import Any, Dict, Mapping, Tuple, Type, TypeVar

import numpy as np


T = TypeVar('T')


class Grid(typing.NamedTuple):
  """Description of a grid.

  Grid class keeps track of size of the computational domain, spatial step,
  provides helper functions for initialization and evaluation of objects that
  depend on discretization (e.g. velocity fields).

  Attributes:
    size_x: Number of finite volumes along x direction.
    size_y: Number of finite volumes along y direction.
    step: Spatial size of the finite volume cell.
  """

  size_x: int
  size_y: int
  step: float

  @classmethod
  def from_period(cls: Type[T], size: int, length: float) -> T:
    """Create a grid from period rather than step-size."""
    step = length / size
    return cls(size, size, step)

  @classmethod
  def from_config(cls: Type[T], config: Mapping[str, Any]) -> T:
    """Construct a grid from a configuration dict."""
    return cls(**config)

  def to_config(self) -> Dict[str, Any]:
    """Create a configuration dict representing this grid."""
    return dict(
        size_x=self.size_x,
        size_y=self.size_y,
        step=self.step
    )

  @property
  def length_x(self) -> float:
    """Length of the computational domain in arbitrary units."""
    return self.step * self.size_x

  @property
  def length_y(self) -> float:
    """Width of the computational domain in arbitrary units."""
    return self.step * self.size_y

  @property
  def shape(self) -> Tuple[int, int]:
    """Returns the shape of the grid."""
    return (self.size_x, self.size_y)

  @property
  def ndim(self) -> int:
    """Number of grid dimensions."""
    # currently always 2.
    return len(self.shape)

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
    shift_x = (1 + shift[0]) * half_step
    shift_y = (1 + shift[1]) * half_step
    return np.meshgrid(
        shift_x + self.step * np.arange(self.size_x),
        shift_y + self.step * np.arange(self.size_y),
        indexing='ij')
