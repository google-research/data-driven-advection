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
"""StateKey named tuple that provides structure to states and state_derivs.

StateKey serves as a key to dictionaries that hold state tensors and state
derivatives tensors. Fields derivative_orders and offset represent partial
spatial derivatives and half integer grid shifts correspondingly.

  Usage example 1 (2D diffusion equation state):

    foo_concentration  # Tensor holding concentration values
    foo_state_key = StateKey('concentration', (0, 0), (0, 0))
    foo_state = {foo_state_key: foo_concentration}

  Usage example 2 (2D diffusion equation requested space derivatives):

    xx_derivative = StateKey('concentration', (2, 0), (0, 0))
    yy_derivative = StateKey('concentration', (0, 2), (0, 0))
    DiffusionEquation.STATE_DERIVATIVES_KEYS = (xx_derivative, yy_derivative)

  Usage example 3 (2D advection equation state):

    bar_concentration  # Tensor holding concentration values
    bar_x_velocity  # Tensor holding x velocity values
    bar_y_velocity  # Tensor holding y velocity values
    bar_concentration_key = StateKey('concentration', (0, 0), (0, 0))
    bar_x_velocity_key = StateKey('velocity_x', (0, 0), (0, 0))
    bar_y_velocity_key = StateKey('velocity_y', (0, 0), (0, 0))
    bar_state = {bar_concentration_key: bar_concentration,
                 bar_x_velocity_key: bar_x_velocity,
                 bar_y_velocity_key: bar_y_velocity}
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum
from pde_superresolution_2d import metadata_pb2
import typing
from typing import Tuple, Type, TypeVar

T = TypeVar('T')


class Prefix(enum.Enum):
  BASELINE = 'baseline_'  # prefix for derivatives evaluated on coarse grid
  MODEL = 'model_'  # prefix for states evaluated by the model
  EXACT = 'exact_'  # prefix for states obtained by coarse-graining


class Direction(enum.Enum):
  X = 0
  Y = 1
  T = 2


class StateKey(typing.NamedTuple(
    'StateKey', [
        ('name', str),
        ('derivative_orders', Tuple[int, int, int]),
        ('offset', Tuple[int, int])
    ])):
  """Description of the state component.

  StateKey is used as a key in the State==Dict[StateKey, tf.Tensor]. It holds
  information about the component stored in the corresponding tensor. It is also
  used by Equation to request values from the model.

  Attributes:
    name: Name of the component.
    derivative_orders: Derivative orders with respect to x, y, t respectively.
    offset: Number of half-integer shifts on the grid.
  """

  @classmethod
  def from_proto(cls: Type[T], proto: metadata_pb2.State) -> T:
    """Construct a state from a proto."""
    name = proto.name
    derivative_orders = (proto.deriv_x, proto.deriv_y, proto.deriv_t)
    offset = (proto.offset_x, proto.offset_y)
    return cls(name, derivative_orders, offset)

  def to_proto(self) -> metadata_pb2.State:
    """Creates a protocol buffer representing the state component."""
    deriv_x, deriv_y, deriv_t = self.derivative_orders
    offset_x, offset_y = self.offset
    state_proto = metadata_pb2.State(
        name=self.name, deriv_x=deriv_x,
        deriv_y=deriv_y, deriv_t=deriv_t,
        offset_x=offset_x, offset_y=offset_y)
    return state_proto

  def time_derivative(self: T) -> T:
    """Returns a StateKey with derivative_order_t incremented by 1."""
    derivatives = list(self.derivative_orders)
    derivatives[-1] += 1
    return self._replace(derivative_orders=tuple(derivatives))

  def with_prefix(self: T, prefix: Prefix) -> T:
    """Returns a StateKey with a prefix added to the name field."""
    return self._replace(name=prefix.value + self.name)

  def model(self: T) -> T:
    """Returns a StateKey for "model" states."""
    return self.with_prefix(Prefix.MODEL)

  def baseline(self: T) -> T:
    """Returns a StateKey for "baseline" states."""
    return self.with_prefix(Prefix.BASELINE)

  def exact(self: T) -> T:
    """Returns a StateKey for "exact" states."""
    return self.with_prefix(Prefix.EXACT)
