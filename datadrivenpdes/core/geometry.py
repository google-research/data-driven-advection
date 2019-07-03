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
"""Code for working with geometric transformations and symmetries."""
from typing import Dict, List, Mapping

from datadrivenpdes.core import states
from datadrivenpdes.core import tensor_ops
import tensorflow as tf


Dimension = states.Dimension

_POSITION_TO_DIMENSION = {0: Dimension.X, 1: Dimension.Y}


class Transform:
  """A geometric transformation for a dict of state tensors."""

  def forward(self, state: Mapping[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Forward transformation."""
    raise NotImplementedError

  def inverse(self, state: Mapping[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Inverse transformation."""
    raise NotImplementedError


class Identity(Transform):
  """Identity transformation."""

  def forward(self, state):
    return state

  def inverse(self, state):
    return state

  def __repr__(self):
    return '<Identity>'


class Reflection(Transform):
  """Geometric reflection over the listed axes, e.g., x -> -x."""

  def __init__(
      self,
      axes: List[Dimension],
      definitions: Mapping[str, states.StateDefinition],
  ):
    self.axes = set(axes)
    self.definitions = definitions

  def forward(self, state):
    index_x = slice(None, None, -1 if Dimension.X in self.axes else 1)
    index_y = slice(None, None, -1 if Dimension.Y in self.axes else 1)
    result = {}
    for key, tensor in state.items():
      definition = self.definitions[key]

      # If our grid is staggered along a reflected axis, we need a shift to
      # compensate for the staggered grid. For example, consider a grid of 6
      # values, with periodic boundary conditions where we reflect over the line
      # marked with |:
      #
      # Originals:
      #   Offset = 0:   0   1   2   3   4   5
      #  Offset = +1:     a   b   c   d   e   f
      # Axis of reflection:       |
      # Reflections:
      #   Offset = 0:   5   4   3   2   1   0
      #  Offset = +1: f   e   d   c   b   a
      #
      # To make the reflected values with offset=+1 align with offsets used by
      # the original grid definition, they should be rolled by -1, i.e.,
      #   Offset = 0:   5   4   3   2   1   0
      #  Offset = +1:     e   d   c   b   a   f
      shift = tuple(-offset if _POSITION_TO_DIMENSION[i] in self.axes else 0
                    for i, offset in enumerate(definition.offset))

      # TODO(shoyer): consider adding a notion of how quantities transform
      # (e.g., vector vs. pseudo-vector) explicitly into our data model.
      # https://en.wikipedia.org/wiki/Parity_(physics)#Effect_of_spatial_inversion_on_some_variables_of_classical_physics
      num_sign_flips = (
          # sign flips due to derivatives, e.g., d/dx -> -d/dx.
          sum(order if _POSITION_TO_DIMENSION[i] in self.axes else 0
              for i, order in enumerate(definition.derivative_orders[:2]))
          # sign flips for vector quantities, e.g., x_velocity -> -x_velocity.
          + sum(dim in self.axes for dim in definition.tensor_indices)
      )
      sign = -1 if num_sign_flips % 2 else 1

      result[key] = sign * tensor_ops.roll_2d(
          tensor[..., index_x, index_y], shift)

    return result

  def inverse(self, state):
    return self.forward(state)

  def __repr__(self):
    sorted_axis_names = sorted(axis.name for axis in self.axes)
    return '<Reflection [{}]>'.format(', '.join(sorted_axis_names))


class Permutation(Transform):
  """Permute axes, i.e., x -> y, y -> x."""

  def __init__(self, definitions: Mapping[str, states.StateDefinition]):
    reverse_lookup = dict(zip(definitions.values(), definitions.keys()))
    self.permuted_keys = {
        key: reverse_lookup[definitions[key].swap_xy()]
        for key in definitions
    }

  def forward(self, state):
    return {
        self.permuted_keys[k]: tensor_ops.swap_xy(v)
        for k, v in state.items()
    }

  def inverse(self, state):
    return self.forward(state)

  def __repr__(self):
    return '<Permutation [X, Y]>'


class Composition(Transform):
  """Composition of multiple geometric transformations.

  The forward transformation applies the sub-transforms in the given order.
  """

  def __init__(self, transforms: List[Transform]):
    self.transforms = transforms

  def forward(self, state):
    result = state
    for transform in self.transforms:
      result = transform.forward(result)
    return result

  def inverse(self, state):
    result = state
    for transform in reversed(self.transforms):
      result = transform.inverse(result)
    return result

  def __repr__(self):
    return '<Composition {}>'.format(self.transforms)


def symmetries_of_the_square(
    definitions: Mapping[str, states.StateDefinition],
) -> List[Transform]:
  """Returns geometric transforms for all elements of the dihedral group D8."""

  identity = Identity()
  flip_x = Reflection([Dimension.X], definitions)
  flip_y = Reflection([Dimension.Y], definitions)
  flip_xy = Reflection([Dimension.X, Dimension.Y], definitions)
  swap_xy = Permutation(definitions)

  transforms = []
  for reflection in [identity, flip_x, flip_y, flip_xy]:
    for permutation in [identity, swap_xy]:
      transforms.append(Composition([permutation, reflection]))

  return transforms
