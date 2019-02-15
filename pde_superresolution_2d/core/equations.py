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

import collections
import operator

import numpy as np
from pde_superresolution_2d import metadata_pb2
from pde_superresolution_2d.core import grids
from pde_superresolution_2d.core import states
import tensorflow as tf
from typing import Any, Dict, Iterator, Tuple, Type, Union

from google3.net.proto2.python.public import message


KeyedTensors = Dict[states.StateKey, tf.Tensor]
Shape = Union[int, Tuple[int]]


CONTINUOUS_EQUATIONS = {}


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
    INPUT_KEYS: A tuple of StateKeys that specifies requested inputs (i.e.,
      spatial derivatives) for use in the time_derivative() or take_time_step()
      methods.
    DISCRETIZATION_NAME: Name of the discretization method.
    METHOD: Discretization method type (finite difference or finite volume).
    MONOTONIC: Are dynamics guaranteed to be monotonic?
  """

  DISCRETIZATION_NAME = ...   # type: str
  METHOD = ...  # type: metadata_pb2.Equation.Discretization.Method
  MONOTONIC = ...  # type: bool

  STATE_KEYS = ...  # type: Tuple[states.StateKey, ...]
  CONSTANT_KEYS = {}  # type: Tuple[states.StateKey, ...]
  INPUT_KEYS = ...  # type: Tuple[states.StateKey, ...]

  def time_derivative(
      self,
      state: KeyedTensors,
      inputs: KeyedTensors,
      grid: grids.Grid,
      time: float = 0.0,
  ) -> KeyedTensors:
    """Returns time derivative of the given state.

    Computes time derivatives of the state described by PDE using
    provided spatial derivatives.

    Args:
      state: tensors corresponding to each key in STATE_KEYS, indicating the
        current state.
      inputs: tensors corresponding to each key in INPUT_KEYS.
      grid: description of discretization parameters.
      time: time at which to evaluate time derivatives.

    Returns:
      Time derivative for each non-constant term in the state.
    """
    raise NotImplementedError

  def take_time_step(
      self,
      state: KeyedTensors,
      inputs: KeyedTensors,
      grid: grids.Grid,
      time: float = 0.0,
  ) -> KeyedTensors:
    """Take single time-step.

    The time step will be of size self.get_time_step().

    The default implementation is an (explicit) forward Euler method.

    Args:
      state: tensors corresponding to each key in STATE_KEYS, indicating the
        current state.
      inputs: tensors corresponding to each key in INPUT_KEYS.
      grid: description of discretization parameters.
      time: time at which the step is taken.

    Returns:
      Updated values for each non-constant term in the state.
    """
    evolving_state = {k: v for k, v in state.items()
                      if k not in self.CONSTANT_KEYS}
    time_derivs = self.time_derivative(state, inputs, grid, time)
    dt = self.get_time_step(grid)
    new_state = {k: u + dt * time_derivs[k.time_derivative()]
                 for k, u in evolving_state.items()}
    return new_state

  def random_state(
      self,
      grid: grids.Grid,
      params: Dict[str, Dict[str, Any]] = None,
      size: Shape = (),
      seed: int = None,
      dtype: Any = np.float32,
  ) -> Dict[states.StateKey, np.ndarray]:
    """Returns a state with fully parametrized initial conditions.

    Generates initial conditions of `init_type`. All parameters of
    initialization will be overridden with values from `kwargs`.
    (e.g. position of the gaussian for InitialConditionMethods.GAUSSIAN is
    given by x_position and y_position arguments.) The intended use of this
    method is for testing evaluation on particular initial values. To generate
    random ensemble of initial conditions use initial_random_state.

    Args:
      grid: Grid object holding discretization parameters.
      params: initialization parameters.
      size: size of the batch dimension.
      seed: random seed.
      dtype: dtype of the resulting numpy arrays.

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

  def to_proto(self) -> metadata_pb2.Equation:
    """Creates a protocol buffer holding parameters of the equation."""
    raise NotImplementedError

  @classmethod
  def from_proto(cls, proto: message.Message) -> 'Equation':
    """Create this equation from an equation-specific protocol buffer."""
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


def register_continuous_equation(key: str):
  """Register a class with a continuous equation."""
  def decorator(cls: Type[Equation]):
    CONTINUOUS_EQUATIONS[key] = cls
    return cls
  return decorator


def _breadth_first_subclasses(base: Type[Equation]) -> Iterator[Type[Equation]]:
  """Yields all subclasses of a given class in breadth-first order."""
  # https://stackoverflow.com/questions/3862310
  subclasses = collections.deque([base])
  while subclasses:
    subclass = subclasses.popleft()
    yield subclass
    subclasses.extend(subclass.__subclasses__())


def matching_equation_type(
    continuous_equation_type: Type[Equation],
    discretization: str,
) -> Type[Equation]:
  """Find the equation with the matching discretization."""
  for subclass in _breadth_first_subclasses(continuous_equation_type):
    if subclass.DISCRETIZATION_NAME == discretization:
      return subclass

  raise ValueError('equation {} and discretization {} not found'
                   .format(continuous_equation_type, discretization))


def equation_from_proto(
    proto: metadata_pb2.Equation,
    discretization: str = None,
) -> Equation:
  """Constructs an equation from the Equation protocol buffer.

  Args:
    proto: Equation protocol buffer encoding the Equation.
    discretization: Override the discretization scheme for the equation. Needed
      for testing different implementation in training and evaluation.

  Returns:
    Equation object.

  Raises:
    ValueError: Provided protocol buffer was not recognized, check proto names.
  """
  if discretization is None:
    discretization = proto.discretization.name

  continuous_equation = proto.WhichOneof('continuous_equation')
  equation_proto = getattr(proto, continuous_equation)

  base_equation_type = CONTINUOUS_EQUATIONS[continuous_equation]
  equation_type = matching_equation_type(base_equation_type, discretization)
  return equation_type.from_proto(equation_proto)

