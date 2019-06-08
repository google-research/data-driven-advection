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
"""StateDefinition and associated utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum
import typing
from typing import Tuple, Type, TypeVar

from pde_superresolution_2d import metadata_pb2

T = TypeVar('T')


class Prefix(enum.Enum):
  BASELINE = 'baseline_'  # prefix for derivatives evaluated on coarse grid
  MODEL = 'model_'  # prefix for states evaluated by the model
  EXACT = 'exact_'  # prefix for states obtained by coarse-graining


class Dimension(enum.IntEnum):
  """Enum denoting physical dimensions."""

  # NOTE(shoyer): we use this instead of the corresponding proto enum class
  # because proto enums are only integers in Python :(

  # NOTE(shoyer): we need to use IntEnum, since otherwise we get an error from
  # tensorflow/python/data/ops/dataset_ops.py:
  # TypeError: Unsupported return value from function passed to Dataset.map()

  X = 1
  Y = 2
  Z = 3

  @classmethod
  def from_proto(cls: Type[T], proto: metadata_pb2.State.TensorIndex) -> T:
    return cls[metadata_pb2.State.TensorIndex.Name(proto)]

  def to_proto(self) -> metadata_pb2.State.TensorIndex:
    return metadata_pb2.State.TensorIndex.Name(self.value)


class StateDefinition(typing.NamedTuple(
    'StateDefinition', [
        ('name', str),
        ('tensor_indices', Tuple[Dimension, ...]),
        ('derivative_orders', Tuple[int, int, int]),
        ('offset', Tuple[int, int])
    ])):
  """Description of the physical quantity that a state tensor corresponds to.

  Example usage:
    x_velocity = StateDefinition(
        'velocity', (metadata_pb2.State.X,), (0, 0, 0), (0, 0))

  Attributes:
    name: name of the corresponding physical variable.
    tensor_indices: tuple of metadata_pb2.State.TensorIndex enum values
      indicating the physical tensor components that this state corresponds to,
      e.g., () for a scalar tensor, (metadata_pb2.State.X,) for the x-component
      of a vector and (metadata_pb2.State.Y,) for the y-component of a vector.
    derivative_orders: derivative orders with respect to x, y, t respectively.
    offset: number of half-integer shifts on the grid. An offset of (0, 0) is
      centered on the unit-cell.
  """

  @classmethod
  def from_proto(cls: Type[T], proto: metadata_pb2.State) -> T:
    """Construct a state from a proto."""
    name = proto.name
    tensor_indices = tuple(
        Dimension.from_proto(index) for index in proto.tensor_indices
    )
    derivative_orders = (proto.deriv_x, proto.deriv_y, proto.deriv_t)
    offset = (proto.offset_x, proto.offset_y)
    return cls(name, tensor_indices, derivative_orders, offset)

  def to_proto(self) -> metadata_pb2.State:
    """Creates a protocol buffer representing the state component."""
    tensor_indices = [index.to_proto() for index in self.tensor_indices]
    deriv_x, deriv_y, deriv_t = self.derivative_orders
    offset_x, offset_y = self.offset
    state_proto = metadata_pb2.State(
        name=self.name,
        tensor_indices=tensor_indices,
        deriv_x=deriv_x,
        deriv_y=deriv_y,
        deriv_t=deriv_t,
        offset_x=offset_x,
        offset_y=offset_y,
    )
    return state_proto

  def swap_xy(self: T) -> T:
    """Swap x and y dimensions on this state."""

    def _tensor_index_swap(index: Dimension) -> Dimension:
      return {
          Dimension.X: Dimension.Y,
          Dimension.Y: Dimension.X,
      }.get(index, index)

    tensor_indices = tuple(map(_tensor_index_swap, self.tensor_indices))
    derivative_orders = (self.derivative_orders[1],
                         self.derivative_orders[0],
                         self.derivative_orders[2])
    offset = self.offset[::-1]
    return self._replace(tensor_indices=tensor_indices,
                         derivative_orders=derivative_orders,
                         offset=offset)

  def time_derivative(self: T) -> T:
    """Returns a StateDefinition with derivative_order_t incremented by 1."""
    derivatives = list(self.derivative_orders)
    derivatives[-1] += 1
    return self._replace(derivative_orders=tuple(derivatives))

  def with_prefix(self: T, prefix: Prefix) -> T:
    """Returns a StateDefinition with a prefix added to the name field."""
    return self._replace(name=prefix.value + self.name)

  def model(self: T) -> T:
    """Returns a StateDefinition for "model" states."""
    return self.with_prefix(Prefix.MODEL)

  def baseline(self: T) -> T:
    """Returns a StateDefinition for "baseline" states."""
    return self.with_prefix(Prefix.BASELINE)

  def exact(self: T) -> T:
    """Returns a StateDefinition for "exact" states."""
    return self.with_prefix(Prefix.EXACT)
