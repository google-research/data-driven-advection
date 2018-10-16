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

from typing import Dict, NamedTuple, Tuple, Any

from pde_superresolution_2d import metadata_pb2

StateKey = NamedTuple('StateKey', [('name', str),  # pylint: disable=invalid-name
                                   ('derivative_orders', Tuple[int, int, int]),
                                   ('offset', Tuple[int, int])])
"""Descriptor of the state component.

StateKey is used as a key in the State==Dict[StateKey, tf.Tensor]. It holds
information about the component stored in the corresponding tensor. It is also
used by Equation to request values from the model.

Attributes:
  name: Name of the component.
  derivative_orders: Derivative orders with respect to x, y, t respectively.
  offset: Number of half-integer shifts on the grid.
"""


def add_prefix(prefix: str, state_key: StateKey) -> StateKey:
  """Returns a StateKey with a prefix added the name field."""
  return state_key._replace(name=prefix + state_key.name)


def add_prefix_tuple(prefix: str,
                     state_keys: Tuple[StateKey, ...]) -> Tuple[StateKey, ...]:
  """Returns a Tuple of StateKeys with a prefix added their name fields."""
  return tuple([add_prefix(prefix, state_key) for state_key in state_keys])


def add_prefix_keys(prefix: str,
                    state: Dict[StateKey, Any]) -> Dict[StateKey, Any]:
  """Returns a state with name field of StateKeys prefixed with prefix."""
  return {add_prefix(prefix, key): value for key, value in state.items()}

C = StateKey('concentration', (0, 0, 0), (0, 0))
C_EDGE_X = StateKey('concentration', (0, 0, 0), (1, 0))
C_EDGE_Y = StateKey('concentration', (0, 0, 0), (0, 1))
C_X = StateKey('concentration', (1, 0, 0), (0, 0))
C_Y = StateKey('concentration', (0, 1, 0), (0, 0))
C_T = StateKey('concentration', (0, 0, 1), (0, 0))
C_X_EDGE_X = StateKey('concentration', (1, 0, 0), (1, 0))
C_Y_EDGE_Y = StateKey('concentration', (0, 1, 0), (0, 1))
C_XX = StateKey('concentration', (2, 0, 0), (0, 0))
C_XY = StateKey('concentration', (1, 1, 0), (0, 0))
C_YY = StateKey('concentration', (0, 2, 0), (0, 0))

BASELINE_PREFIX = 'baseline_'  # prefix for derivatives evaluated on coarse grid
MODEL_PREFIX = 'model_'  # prefix for states evaluated by the model
EXACT_PREFFIX = 'exact_'  # prefix for states obtained by coarse-graining

B_C = add_prefix(BASELINE_PREFIX, C)
B_C_EDGE_X = add_prefix(BASELINE_PREFIX, C_EDGE_X)
B_C_EDGE_Y = add_prefix(BASELINE_PREFIX, C_EDGE_Y)
B_C_X = add_prefix(BASELINE_PREFIX, C_X)
B_C_Y = add_prefix(BASELINE_PREFIX, C_Y)
B_C_T = add_prefix(BASELINE_PREFIX, C_T)
B_C_XX = add_prefix(BASELINE_PREFIX, C_XX)
B_C_XY = add_prefix(BASELINE_PREFIX, C_XY)
B_C_YY = add_prefix(BASELINE_PREFIX, C_YY)

VX = StateKey('velocity_x', (0, 0, 0), (0, 0))
VY = StateKey('velocity_y', (0, 0, 0), (0, 0))
VX_T = StateKey('velocity_x', (0, 0, 1), (0, 0))
VY_T = StateKey('velocity_y', (0, 0, 1), (0, 0))


def state_key_to_proto(state_key: StateKey):
  """Creates a protocol buffer representing the state component."""
  deriv_x, deriv_y, deriv_t = state_key.derivative_orders
  offset_x, offset_y = state_key.offset
  state_proto = metadata_pb2.State(
      name=state_key.name, deriv_x=deriv_x,
      deriv_y=deriv_y, deriv_t=deriv_t,
      offset_x=offset_x, offset_y=offset_y)
  return state_proto


def state_key_from_proto(state_proto: metadata_pb2.State) -> StateKey:
  """Creates a protocol buffer representing the state component."""
  deriv_x = state_proto.deriv_x
  deriv_y = state_proto.deriv_y
  deriv_t = state_proto.deriv_t
  offset_x = state_proto.offset_x
  offset_y = state_proto.offset_y
  name = state_proto.name
  return StateKey(str(name), (deriv_x, deriv_y, deriv_t), (offset_x, offset_y))


def add_time_derivative(state_key: StateKey) -> StateKey:
  """Returns a StateKey with derivative_order_t incremented by 1."""
  derivatives = list(state_key.derivative_orders)
  derivatives[-1] += 1
  return state_key._replace(derivative_orders=tuple(derivatives))


def add_time_derivative_tuple(
    state_keys: Tuple[StateKey, ...]) -> Tuple[StateKey, ...]:
  """Returns a Tuple of StateKey with derivative_order_t incremented by 1."""
  return tuple([add_time_derivative(state_key) for state_key in state_keys])
