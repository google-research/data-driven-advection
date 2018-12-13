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
"""Equation classes describe differential equations.

Equation class encapsulate the relation between the spatial state derivatives
and time derivatives for different PDE. State derivatives can be used
combined differently to yield various update schemes. (e.g. finite differences
vs finite volumes).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum
from pde_superresolution_2d.core import grids
from pde_superresolution_2d.core import states
import tensorflow as tf
from typing import Dict, Tuple, Any


class Equation(object):
  """"Base class for PDEs.

  Defines method time_derivative that constructs time derivative of the
  current state using state derivatives provided by the model. The aim is
  to be able to use multiple models and the same integrator for uniform
  performance comparison and experimentation.

  Attributes:
    STATE_KEYS: A tuple of StateKeys that represent the content of a state.
    CONSTANT_KEYS: A tuple of StateKeys found in STATE_KEYs that don't change
      over time.
    SPATIAL_DERIVATIVES_KEYS: A tuple of StateKeys that specifies requested
      spatial derivatives.
  """

  STATE_KEYS = ...  # type: Tuple[states.StateKey, ...]
  CONSTANT_KEYS = ...  # type: Tuple[states.StateKey, ...]
  SPATIAL_DERIVATIVES_KEYS = ...  # type: Tuple[states.StateKey, ...]

  def time_derivative(
      self,
      state: Dict[states.StateKey, tf.Tensor],
      t: float,
      grid: grids.Grid,
      spatial_derivatives: Dict[states.StateKey, tf.Tensor]
  ) -> Dict[states.StateKey, tf.Tensor]:
    """Returns time derivative of the given state.

    Computes time derivatives of the state described by PDE using
    provided spatial derivatives.

    Args:
      state: Current state of the solution at time t.
      t: Time at which to evaluate time derivatives.
      grid: Grid object holding discretization parameters.
      spatial_derivatives: Requested spatial derivatives.

    Returns:
      Time derivative of the state.
    """
    raise NotImplementedError

  def initial_state(
      self,
      init_type: enum.Enum,
      grid: grids.Grid,
      batch_size: int = 1,
      **kwargs: Any
  ) -> Dict[states.StateKey, tf.Tensor]:
    """Returns a state with fully parametrized initial conditions.

    Generates initial conditions of `init_type`. All parameters of
    initialization will be overridden with values from `kwargs`.
    (e.g. position of the gaussian for InitialConditionMethods.GAUSSIAN is
    given by x_position and y_position arguments.) The intended use of this
    method is for testing evaluation on particular initial values. To generate
    random ensemble of initial conditions use initial_random_state.

    Args:
      init_type: Initialization method enum.
      grid: Grid object holding discretization parameters.
      batch_size: Size of the batch dimension.
      **kwargs: Arguments to be passed to initialization methods.

    Returns:
      State with initial values.
    """
    raise NotImplementedError

  def initial_random_state(
      self,
      seed: int,
      init_type: enum.Enum,
      grid: grids.Grid,
      batch_size: int = 1,
      **kwargs: Any
  ) -> Dict[states.StateKey, tf.Tensor]:
    """Returns a state with random initial conditions.

    Generates an initial state with values from an ensemble defined by
    `init_type`. Parameters of the distribution can be modified by providing
    override values in `kwargs`. Intended for generation of training data and
    large validation datasets.

    Args:
      seed: Random seed to use for random number generator.
      init_type: Initialization method enum.
      grid: Grid object holding discretization parameters.
      batch_size: Size of the batch dimension.
      **kwargs: Arguments to be passed to initialization methods.

    Returns:
      State with initial values.
    """
    raise NotImplementedError

  def get_time_step(self, grid: grids.Grid) -> float:
    """Returns appropriate time step for time marching the equation on grid.

    Equation should implement custom logic for choosing the appropriate dt.

    Args:
      grid: Grid object holding discretization parameters.

    Returns:
      The value of an appropriate time step.
    """
    raise NotImplementedError

  def to_tensor(self, state: Dict[states.StateKey, tf.Tensor]) -> tf.Tensor:
    """Compresses a state to a single tensor.

    Args:
      state: State to be converted to a tensor.

    Returns:
      A tensor holding information about the state.
    """
    raise NotImplementedError

  def to_state(self, tensor: tf.Tensor) -> Dict[states.StateKey, tf.Tensor]:
    """Decompresses a tensor into a state.

    Args:
      tensor: A tensor representing the state.

    Returns:
      A state holding information stored in tensor.
    """
    raise NotImplementedError

  def to_proto(self):
    """Creates a protocol buffer holding parameters of the equation."""
    raise NotImplementedError


def check_keys(dictionary: Dict[states.StateKey, Any],
               expected_keys: Tuple[states.StateKey, ...]):
  """Checks if the dictionary keys match the expected keys.

  Args:
    dictionary: Dictionary to check against expected_keys.
    expected_keys: Key we expect to find in the dictionary.

  Raises:
    ValueError: Keys do not match.
  """
  if set(dictionary.keys()) != set(expected_keys):
    raise ValueError('Keys do not match, got {} and {}'.format(
        set(dictionary.keys()), set(expected_keys)))
