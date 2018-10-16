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
import numpy as np
import tensorflow as tf
from typing import Dict, Tuple, Any

from pde_superresolution_2d import grids
from pde_superresolution_2d import metadata_pb2
from pde_superresolution_2d import states
from pde_superresolution_2d import utils
from pde_superresolution_2d import velocity_fields


class InitialConditionMethod(enum.Enum):
  """Enumeration of initialization methods."""
  GAUSSIAN = 'GAUSSIAN'
  FOURIER = 'FOURIER'
  GAUSS_AND_FOURIER = 'GAUSS_AND_FOURIER'
  PLACEHOLDER = 'PLACEHOLDER'


class Equation(object):
  """"Base class for PDEs.

  Defines method time_derivative that constructs time derivative of the
  current state using state derivatives provided by the model. The aim is
  to be able to use multiple models and the same integrator for uniform
  performance comparison and experimentation.

  Attributes:
    State_KEYS: A tuple of StateKeys that represent the content of a state.
    SPATIAL_DERIVATIVES_KEYS: A tuple of StateKeys that specifies requested
      spatial derivatives.
  """

  STATE_KEYS = ...  # type: Tuple[states.StateKey, ...]
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
      init_type: InitialConditionMethod,
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
      init_type: Initialization method, must be one of the
          InitialConditionMethods.
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
      init_type: InitialConditionMethod,
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
      init_type: Initialization method, must be one of the
          InitialConditionMethods.
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


class AdvectionDiffusion(Equation):
  """Base class for advection diffusion equations.

  Advection diffusion equation defines the state and common methods. Specific
  implementation must provide time_derivative() and to_proto() methods.
  """
  STATE_KEYS = (states.C,)

  def __init__(self,
               velocity_field: velocity_fields.VelocityField,
               diffusion_const: float):
    self.velocity_field = velocity_field
    self.diffusion_const = diffusion_const

  def initial_state(
      self,
      init_type: InitialConditionMethod,
      grid: grids.Grid,
      batch_size: int = 1,
      **kwargs: Any
  ) -> Dict[states.StateKey, tf.Tensor]:
    """Returns state with parametrized advection-diffusion initial conditions.

    See base class.

    Args:
      init_type: Initialization method, must be one of the
          InitialConditionMethods.
      grid: Grid object holding discretization parameters.
      batch_size: Size of the batch dimension.
      **kwargs: Parameters of initialization.

    Returns:
      State with specified initial values.

    Raises:
      ValueError: Provided initialization method is not supported.
    """
    state = {}
    if init_type == InitialConditionMethod.PLACEHOLDER:
      state_shape = (batch_size, grid.size_x, grid.size_y)
      state[states.C] = tf.placeholder(tf.float64, shape=state_shape)
    elif init_type == InitialConditionMethod.GAUSSIAN:
      state[states.C] = generate_gaussian(
          batch_size, grid, **kwargs)
    elif init_type == InitialConditionMethod.FOURIER:
      state[states.C] = generate_fourier_terms(
          batch_size, grid, **kwargs)
    else:
      raise ValueError('Given initialization method is not supported')
    return state

  def initial_random_state(
      self,
      seed: int,
      init_type: InitialConditionMethod,
      grid: grids.Grid,
      batch_size: int = 1,
      **kwargs: Any
  ) -> Dict[states.StateKey, tf.Tensor]:
    """Returns a state with random initial conditions for advection diffusion.

    See base class.

    Args:
      seed: Random seed to use for random number generator.
      init_type: Initialization method, must be one of the
          InitialConditionMethods.
      grid: Grid object holding discretization parameters.
      batch_size: Size of the batch dimension.
      **kwargs: Parameters of initialization.

    Returns:
      State with random initial values.

    Raises:
      ValueError: Provided initialization method is not supported.
    """
    rnd_gen = np.random.RandomState(seed=seed)

    state = {}
    if init_type == InitialConditionMethod.GAUSSIAN:
      state[states.C] = generate_random_gaussians(
          batch_size, rnd_gen, grid, **kwargs)
    elif init_type == InitialConditionMethod.FOURIER:
      state[states.C] = generate_random_fourier_terms(
          batch_size, rnd_gen, grid, **kwargs)
    elif init_type == InitialConditionMethod.GAUSS_AND_FOURIER:
      if seed % 2 == 0:
        state[states.C] = generate_random_fourier_terms(
            batch_size, rnd_gen, grid, **kwargs)
      else:
        state[states.C] = generate_random_gaussians(
            batch_size, rnd_gen, grid, **kwargs)
    else:
      raise ValueError('Given initialization method is not supported')
    return state

  def get_time_step(self, grid: grids.Grid) -> float:
    """Returns appropriate time step for time marching the equation on grid.

    Stability condition on time step is V*dt < `grid.step`.

    Args:
      grid: Grid object holding discretization parameters.

    Returns:
      The value of an appropriate time step.
    """
    return 0.01 * grid.step

  def to_tensor(self, state: Dict[states.StateKey, tf.Tensor]) -> tf.Tensor:
    """Compresses a state or a time derivative of a state to a single tensor.

    Args:
      state: State to be converted to a tensor.

    Returns:
      A tensor holding information about the state.

    Raises:
      ValueError: If state is not a state or a state derivative.
    """
    if set(state.keys()) == set((states.C,)):
      return state[states.C]
    elif set(state.keys()) == set((states.C_T,)):
      return state[states.C_T]
    else:
      raise ValueError('Input state is not convertible to a tensor')

  def to_state(self, tensor: tf.Tensor) -> Dict[states.StateKey, tf.Tensor]:
    """Decompresses a tensor into a state of zeroth time derivative.

    Args:
      tensor: A tensor representing the state.

    Returns:
      A state holding information stored in tensor.
    """
    return {states.C: tensor}

  @classmethod
  def from_proto(
      cls,
      proto: metadata_pb2.AdvectionDiffusionEquation
  ) -> Equation:
    """Creates a class instance from protocol buffer.

    Args:
      proto: Protocol buffer holding AdvectionDiffusion parameters.

    Returns:
      AdvectionDiffusion object initialized from the protobuf.
    """
    v_field_proto = proto.velocity_field
    diffusion_const = proto.diffusion_const
    velocity_field = velocity_fields.velocity_field_from_proto(v_field_proto)  # pytype: disable=wrong-arg-types
    return cls(velocity_field, diffusion_const)


class ConvectionDiffusion(Equation):
  """Base class for convection diffusion equations.

  Convection diffusion equation defines the state and common methods. Specific
  implementation must provide time_derivative() and to_proto() methods. Some
  implementations can focus on constant velocity field, which effectively
  represent the advection-diffusion equation, those should be named accordingly.
  """
  STATE_KEYS = (states.C, states.VX, states.VY)

  def __init__(self, diffusion_const: float):
    self.diffusion_const = diffusion_const

  def initial_state(
      self,
      init_type: InitialConditionMethod,
      grid: grids.Grid,
      batch_size: int = 1,
      **kwargs: Any
  ) -> Dict[states.StateKey, tf.Tensor]:
    """Returns a state with fully specified initial conditions.

    Args:
      init_type: Initialization method, must be one of the
          InitialConditionMethods.
      grid: Grid object holding discretization parameters.
      batch_size: Size of the batch dimension.
      **kwargs: Parameters of initialization.

    Returns:
      State with specified initial values.

    Raises:
      ValueError: Provided initialization method is not supported.
    """
    num_velocity_terms = kwargs.get('num_velocity_terms', 4)
    max_velocity_periods = kwargs.get('max_velocity_periods', 3)
    velocity_field_random_seed = kwargs.get('velocity_field_random_seed', 1)

    v_field = velocity_fields.ConstantVelocityField.from_seed(
        num_velocity_terms, max_velocity_periods, velocity_field_random_seed)

    vx_arg = {'t': 0., 'grid': grid, 'shift': (1, 0)}  # evaluated on x edge
    vy_arg = {'t': 0., 'grid': grid, 'shift': (0, 1)}  # evaluated on y edge
    velocity_tile_param = (batch_size, 1, 1)

    state = {}
    if init_type == InitialConditionMethod.PLACEHOLDER:
      state_shape = (batch_size, grid.size_x, grid.size_y)
      state[states.C] = tf.placeholder(tf.float64, shape=state_shape)
      state[states.VX] = tf.placeholder(tf.float64, shape=state_shape)
      state[states.VY] = tf.placeholder(tf.float64, shape=state_shape)
    elif init_type == InitialConditionMethod.GAUSSIAN:
      state[states.C] = generate_gaussian(
          batch_size, grid, **kwargs)
      state[states.VX] = np.tile(v_field.get_velocity_x(
          **vx_arg), velocity_tile_param)
      state[states.VY] = np.tile(v_field.get_velocity_y(
          **vy_arg), velocity_tile_param)
    elif init_type == InitialConditionMethod.FOURIER:
      state[states.C] = generate_fourier_terms(
          batch_size, grid, **kwargs)
      state[states.VX] = np.tile(v_field.get_velocity_x(
          **vx_arg), velocity_tile_param)
      state[states.VY] = np.tile(v_field.get_velocity_y(
          **vy_arg), velocity_tile_param)
    else:
      raise ValueError('Given initialization method is not supported')
    return state

  def initial_random_state(
      self,
      seed: int,
      init_type: InitialConditionMethod,
      grid: grids.Grid,
      batch_size: int = 1,
      **kwargs: Any
  ) -> Dict[states.StateKey, tf.Tensor]:
    """Returns a state with random initial conditions for a given ensemble.

    Args:
      seed: Random seed to use for random number generator.
      init_type: Initialization method, must be one of the
          InitialConditionMethods.
      grid: Grid object holding discretization parameters.
      batch_size: Size of the batch dimension.
      **kwargs: Parameters of initialization.

    Returns:
      State with random initial values.

    Raises:
      ValueError: Provided initialization method is not supported.
    """
    rnd_gen = np.random.RandomState(seed=seed)

    num_velocity_terms = kwargs.get('num_velocity_terms', 4)
    max_velocity_periods = kwargs.get('max_velocity_periods', 3)
    v_field = velocity_fields.ConstantVelocityField(
        rnd_gen, num_velocity_terms, max_velocity_periods)

    vx_arg = {'t': 0., 'grid': grid, 'shift': (1, 0)}  # evaluated on x edge
    vy_arg = {'t': 0., 'grid': grid, 'shift': (0, 1)}  # evaluated on y edge

    state = {}
    state[states.VX] = np.tile(v_field.get_velocity_x(
        **vx_arg), (batch_size, 1, 1))
    state[states.VY] = np.tile(v_field.get_velocity_y(
        **vy_arg), (batch_size, 1, 1))
    if init_type == InitialConditionMethod.GAUSSIAN:
      state[states.C] = generate_random_gaussians(
          batch_size, rnd_gen, grid, **kwargs)
    elif init_type == InitialConditionMethod.FOURIER:
      state[states.C] = generate_random_fourier_terms(
          batch_size, rnd_gen, grid, **kwargs)
    elif init_type == InitialConditionMethod.GAUSS_AND_FOURIER:
      if seed % 2 == 0:
        state[states.C] = generate_random_fourier_terms(
            batch_size, rnd_gen, grid, **kwargs)
      else:
        state[states.C] = generate_random_gaussians(
            batch_size, rnd_gen, grid, **kwargs)
    else:
      raise ValueError('Given initialization method is not supported')
    return state

  def get_time_step(self, grid: grids.Grid) -> float:
    """Returns appropriate time step for time marching the equation on grid.

    Stability condition on time step is V*dt < `grid.step`.

    Args:
      grid: Grid object holding discretization parameters.

    Returns:
      The value of an appropriate time step.
    """
    return 0.01 * grid.step

  def to_tensor(self, state: Dict[states.StateKey, tf.Tensor]) -> tf.Tensor:
    """Compresses a state or a time derivative of a state to a single tensor.

    Args:
      state: State to be converted to a tensor.

    Returns:
      A tensor holding information about the state.

    Raises:
      ValueError: If state is not a state or a state derivative.
    """
    if set(state.keys()) == set((states.C, states.VX, states.VY)):
      return tf.stack(
          [state[states.C], state[states.VX], state[states.VY]], axis=3)
    elif set(state.keys()) == set((states.C_T, states.VX_T, states.VY_T)):
      return tf.stack(
          [state[states.C_T], state[states.VX_T], state[states.VY_T]], axis=3)
    else:
      raise ValueError('Input state is not convertible to a tensor')

  def to_state(self, tensor: tf.Tensor) -> Dict[states.StateKey, tf.Tensor]:
    """Decompresses a tensor into a state of zeroth time derivative.

    Args:
      tensor: A tensor representing the state.

    Returns:
      A state holding information stored in tensor.
    """
    c, vx, vy = tf.unstack(tensor, axis=3)
    return {states.C: c, states.VX: vx, states.VY: vy}

  @classmethod
  def from_proto(
      cls,
      proto: metadata_pb2.ConvectionDiffusionEquation
  ) -> Equation:
    """Creates a class instance from protocol buffer.

    Args:
      proto: Protocol buffer holding ConvectionDiffusion parameters.

    Returns:
      ConvectionDiffusion object initialized from the protobuf.
    """
    diffusion_const = proto.diffusion_const
    return cls(diffusion_const)


class ConstantVelocityConvectionDiffusion(ConvectionDiffusion):
  """Class implementing advection-diffusion with in-state velocity field."""

  def __init__(self, velocity_field: Any, diffusion_const: float):
    """Constructor that implements advection-diffusion interface."""
    del velocity_field  # Not used by equation that stores V in state.
    super(ConstantVelocityConvectionDiffusion, self).__init__(diffusion_const)

  @classmethod
  def from_proto(
      cls,
      proto: metadata_pb2.ConvectionDiffusionEquation
  ) -> Equation:
    """Creates an instance from a protocol buffer.

    Args:
      proto: Protocol buffer holding ConstantVelocityConvectionDiffusion data.

    Returns:
      ConstantVelocityConvectionDiffusion object initialized from the protobuf.
    """
    diffusion_const = proto.diffusion_const
    return cls(None, diffusion_const)


class FiniteDifferenceAdvectionDiffusion(AdvectionDiffusion):
  """"Advection diffusion differential equation.

  Implements Equation interface for advection diffusion equation using finite
  difference scheme. The state for this equation consists of concentration.
  It requires first and second spatial derivatives to produce time derivative.
  """

  SPATIAL_DERIVATIVES_KEYS = (states.C_X, states.C_Y, states.C_XX, states.C_YY)

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
    check_keys(spatial_derivatives, self.SPATIAL_DERIVATIVES_KEYS)
    check_keys(state, self.STATE_KEYS)
    state_time_derivative = {}
    v_x = self.velocity_field.get_velocity_x(t, grid)
    v_y = self.velocity_field.get_velocity_y(t, grid)

    c_x = spatial_derivatives[states.C_X]
    c_y = spatial_derivatives[states.C_Y]
    c_xx = spatial_derivatives[states.C_XX]
    c_yy = spatial_derivatives[states.C_YY]

    state_time_derivative[states.C_T] = tf.add_n([
        -1 * v_x * c_x,
        -1 * v_y * c_y,
        self.diffusion_const * c_xx,
        self.diffusion_const * c_yy,
    ])
    return state_time_derivative

  def to_proto(self) -> metadata_pb2.Equation:
    """Creates a protocol buffer holding parameters of the equation."""
    return metadata_pb2.Equation(
        advection_diffusion=dict(
            diffusion_const=self.diffusion_const,
            velocity_field=self.velocity_field.to_proto(),
        ),
        scheme=metadata_pb2.Equation.FINITE_DIFF
    )


class FiniteVolumeAdvectionDiffusion(AdvectionDiffusion):
  """Implements finite volume scheme."""

  SPATIAL_DERIVATIVES_KEYS = (states.C_EDGE_X, states.C_EDGE_Y,
                              states.C_X_EDGE_X, states.C_Y_EDGE_Y)

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
    check_keys(spatial_derivatives, self.SPATIAL_DERIVATIVES_KEYS)
    check_keys(state, self.STATE_KEYS)
    state_time_derivative = {}
    v_x = self.velocity_field.get_velocity_x(t, grid, shift=(1, 0))
    v_y = self.velocity_field.get_velocity_y(t, grid, shift=(0, 1))

    c_edge_x = spatial_derivatives[states.C_EDGE_X]
    c_edge_y = spatial_derivatives[states.C_EDGE_Y]
    c_x_edge_x = spatial_derivatives[states.C_X_EDGE_X]
    c_y_edge_y = spatial_derivatives[states.C_Y_EDGE_Y]

    # common division by the grid.step is done after aggregation of all fluxes.
    flux_x_out = -1 * v_x * c_edge_x
    flux_x_in = (utils.roll_2d(c_edge_x, (1, 0)) *
                 utils.roll_2d(v_x, (1, 0), (0, 1)))

    flux_y_out = -1 * v_y * c_edge_y
    flux_y_in = (utils.roll_2d(c_edge_y, (0, 1)) *
                 utils.roll_2d(v_y, (0, 1), (0, 1)))

    diff_flux_x_in = -1 * utils.roll_2d(c_x_edge_x, (1, 0))
    diff_flux_x_out = c_x_edge_x
    diff_flux_y_in = -1 * utils.roll_2d(c_y_edge_y, (0, 1))
    diff_flux_y_out = c_y_edge_y

    advective_flux = tf.add_n([
        flux_x_in,
        flux_x_out,
        flux_y_in,
        flux_y_out
    ])

    diffusive_flux = self.diffusion_const * tf.add_n([
        diff_flux_x_in,
        diff_flux_x_out,
        diff_flux_y_in,
        diff_flux_y_out
    ])

    total_flux = (advective_flux + diffusive_flux) / grid.step
    state_time_derivative[states.C_T] = total_flux
    return state_time_derivative

  def to_proto(self) -> metadata_pb2.Equation:
    """Creates a protocol buffer holding parameters of the equation."""
    return metadata_pb2.Equation(
        advection_diffusion=dict(
            diffusion_const=self.diffusion_const,
            velocity_field=self.velocity_field.to_proto(),
        ),
        scheme=metadata_pb2.Equation.FINITE_VOLUME
    )


class FiniteVolumeUpwindAdvectionDiffusion(AdvectionDiffusion):
  """Implements upwind finite volume scheme for advection diffusion equation."""

  SPATIAL_DERIVATIVES_KEYS = (states.C_X_EDGE_X, states.C_Y_EDGE_Y)

  def time_derivative(
      self,
      state: Dict[states.StateKey, tf.Tensor],
      t: float,
      grid: grids.Grid,
      spatial_derivatives: Dict[states.StateKey, tf.Tensor]
  ) -> Dict[states.StateKey, tf.Tensor]:
    """Returns time derivative of the given state.

    Computes time derivatives of the state described by advection diffusion
    equation using given spatial derivatives. Upwind scheme computes fluxes
    based on the direction of the velocity field, which prevents overestimation
    of the flux which can lead to negative values.

    Args:
      state: Current state of the solution at time t.
      t: Time at which to evaluate time derivatives.
      grid: Grid object holding discretization parameters.
      spatial_derivatives: Requested spatial derivatives.

    Returns:
      Time derivative of the state.
    """
    check_keys(spatial_derivatives, self.SPATIAL_DERIVATIVES_KEYS)
    check_keys(state, self.STATE_KEYS)
    state_time_derivative = {}

    v_x_right = self.velocity_field.get_velocity_x(t, grid, shift=(1, 0))
    v_x_left = self.velocity_field.get_velocity_x(t, grid, shift=(-1, 0))

    v_y_top = self.velocity_field.get_velocity_y(t, grid, shift=(0, 1))
    v_y_bottom = self.velocity_field.get_velocity_y(t, grid, shift=(0, -1))

    c = state[states.C]
    c_left = utils.roll_2d(c, (1, 0))
    c_right = utils.roll_2d(c, (-1, 0))
    c_top = utils.roll_2d(c, (0, -1))
    c_bottom = utils.roll_2d(c, (0, 1))

    c_x_edge_x = spatial_derivatives[states.C_X_EDGE_X]
    c_y_edge_y = spatial_derivatives[states.C_Y_EDGE_Y]

    # common division by the grid.step is done after aggregation of all fluxes.
    flux_x_right_out = -1 * c * tf.maximum(v_x_right, 0.)
    flux_x_right_in = c_right * tf.maximum(-v_x_right, 0.)

    flux_x_left_in = c_left * tf.maximum(v_x_left, 0.)
    flux_x_left_out = -1 * c * tf.maximum(-v_x_left, 0.)

    flux_y_top_out = -1 * c * tf.maximum(v_y_top, 0.)
    flux_y_top_in = c_top * tf.maximum(-v_y_top, 0.)

    flux_y_bottom_out = -1 * c * tf.maximum(-v_y_bottom, 0.)
    flux_y_bottom_in = c_bottom * tf.maximum(v_y_bottom, 0.)

    diff_flux_x_left = -1 * utils.roll_2d(c_x_edge_x, (1, 0))
    diff_flux_x_right = c_x_edge_x
    diff_flux_y_bottom = -1 * utils.roll_2d(c_y_edge_y, (0, 1))
    diff_flux_y_top = c_y_edge_y

    advective_flux = tf.add_n([
        flux_x_right_out,
        flux_x_right_in,
        flux_x_left_out,
        flux_x_left_in,
        flux_y_top_out,
        flux_y_top_in,
        flux_y_bottom_out,
        flux_y_bottom_in
    ])

    diffusive_flux = self.diffusion_const * tf.add_n([
        diff_flux_x_left,
        diff_flux_x_right,
        diff_flux_y_top,
        diff_flux_y_bottom
    ])

    total_flux = (advective_flux + diffusive_flux) / grid.step
    state_time_derivative[states.C_T] = total_flux
    return state_time_derivative

  def to_proto(self) -> metadata_pb2.Equation:
    """Creates a protocol buffer holding parameters of the equation."""
    return metadata_pb2.Equation(
        advection_diffusion=dict(
            diffusion_const=self.diffusion_const,
            velocity_field=self.velocity_field.to_proto(),
        ),
        scheme=metadata_pb2.Equation.UPWIND
    )


class InStateVelocityUpwindAdvectionDiffusion(
    ConstantVelocityConvectionDiffusion):
  """Implements upwind finite volume scheme for advection diffusion equation.

  While being interpreted as convection diffusion equation, this class does not
  implement time evolution of the velocity field. Used for solving
  advection-diffusion equation with variable velocity field.
  """

  SPATIAL_DERIVATIVES_KEYS = (states.C_X_EDGE_X, states.C_Y_EDGE_Y)

  def time_derivative(
      self,
      state: Dict[states.StateKey, tf.Tensor],
      t: float,
      grid: grids.Grid,
      spatial_derivatives: Dict[states.StateKey, tf.Tensor]
  ) -> Dict[states.StateKey, tf.Tensor]:
    """Returns time derivative of the given state.

    Computes time derivatives of the state described by advection diffusion
    equation using given spatial derivatives. Upwind scheme computes fluxes
    based on the direction of the velocity field, which prevents overestimation
    of the flux which can lead to negative values.

    Args:
      state: Current state of the solution at time t.
      t: Time at which to evaluate time derivatives.
      grid: Grid object holding discretization parameters.
      spatial_derivatives: Requested spatial derivatives.

    Returns:
      Time derivative of the state.
    """
    check_keys(spatial_derivatives, self.SPATIAL_DERIVATIVES_KEYS)
    check_keys(state, self.STATE_KEYS)
    state_time_derivative = {}

    v_x_right = state[states.VX]
    v_x_left = utils.roll_2d(v_x_right, (1, 0))

    v_y_top = state[states.VY]
    v_y_bottom = utils.roll_2d(v_y_top, (0, 1))

    c = state[states.C]
    c_left = utils.roll_2d(c, (1, 0))
    c_right = utils.roll_2d(c, (-1, 0))
    c_top = utils.roll_2d(c, (0, -1))
    c_bottom = utils.roll_2d(c, (0, 1))

    c_x_edge_x = spatial_derivatives[states.C_X_EDGE_X]
    c_y_edge_y = spatial_derivatives[states.C_Y_EDGE_Y]

    flux_x_right_out = -1 * c * tf.maximum(v_x_right, 0.)
    flux_x_right_in = c_right * tf.maximum(-v_x_right, 0.)

    flux_x_left_in = c_left * tf.maximum(v_x_left, 0.)
    flux_x_left_out = -1 * c * tf.maximum(-v_x_left, 0.)

    flux_y_top_out = -1 * c * tf.maximum(v_y_top, 0.)
    flux_y_top_in = c_top * tf.maximum(-v_y_top, 0.)

    flux_y_bottom_out = -1 * c * tf.maximum(-v_y_bottom, 0.)
    flux_y_bottom_in = c_bottom * tf.maximum(v_y_bottom, 0.)

    diff_flux_x_left = -1 * utils.roll_2d(c_x_edge_x, (1, 0))
    diff_flux_x_right = c_x_edge_x
    diff_flux_y_bottom = -1 * utils.roll_2d(c_y_edge_y, (0, 1))
    diff_flux_y_top = c_y_edge_y

    advective_flux = tf.add_n([
        flux_x_right_out,
        flux_x_right_in,
        flux_x_left_out,
        flux_x_left_in,
        flux_y_top_out,
        flux_y_top_in,
        flux_y_bottom_out,
        flux_y_bottom_in
    ])

    diffusive_flux = self.diffusion_const * tf.add_n([
        diff_flux_x_left,
        diff_flux_x_right,
        diff_flux_y_top,
        diff_flux_y_bottom
    ])

    total_flux = (advective_flux + diffusive_flux) / grid.step
    state_time_derivative[states.C_T] = total_flux
    state_time_derivative[states.VX_T] = tf.zeros_like(v_x_left)
    state_time_derivative[states.VY_T] = tf.zeros_like(v_y_top)
    return state_time_derivative

  def to_proto(self) -> metadata_pb2.Equation:
    """Creates a protocol buffer holding parameters of the equation."""
    return metadata_pb2.Equation(
        in_state_velocity_advection_diffusion=dict(
            diffusion_const=self.diffusion_const,
        ),
        scheme=metadata_pb2.Equation.UPWIND
    )


class InStateVelocityFiniteVolumeAdvectionDiffusion(
    ConstantVelocityConvectionDiffusion):
  """Implements finite volume scheme for advection diffusion equation.

  This class correctly inherits ConvectionDiffusion base class, as it allows
  to train on an ensemble of velocity fields. This is achieved by keeping
  the velocity field as a part of the state. Time derivatives of the velocity
  fields are set to 0.
  """

  SPATIAL_DERIVATIVES_KEYS = (states.C_EDGE_X, states.C_EDGE_Y,
                              states.C_X_EDGE_X, states.C_Y_EDGE_Y)

  def time_derivative(
      self,
      state: Dict[states.StateKey, tf.Tensor],
      t: float,
      grid: grids.Grid,
      spatial_derivatives: Dict[states.StateKey, tf.Tensor]
  ) -> Dict[states.StateKey, tf.Tensor]:
    """Returns time derivative of the given state.

    Computes time derivatives of the state described by advection diffusion
    equation using given spatial derivatives. Upwind scheme computes fluxes
    based on the direction of the velocity field, which prevents overestimation
    of the flux which can lead to negative values. Time derivatives of velocity
    components are set to zero, as advection equation does not evolve them.

    Args:
      state: Current state of the solution at time t.
      t: Time at which to evaluate time derivatives.
      grid: Grid object holding discretization parameters.
      spatial_derivatives: Requested spatial derivatives.

    Returns:
      Time derivative of the state.
    """
    check_keys(spatial_derivatives, self.SPATIAL_DERIVATIVES_KEYS)
    check_keys(state, self.STATE_KEYS)
    state_time_derivative = {}

    v_x = state[states.VX]
    v_y = state[states.VY]
    c_edge_x = spatial_derivatives[states.C_EDGE_X]
    c_edge_y = spatial_derivatives[states.C_EDGE_Y]
    c_x_edge_x = spatial_derivatives[states.C_X_EDGE_X]
    c_y_edge_y = spatial_derivatives[states.C_Y_EDGE_Y]

    flux_x_out = -1 * v_x * c_edge_x
    flux_x_in = (utils.roll_2d(c_edge_x, (1, 0)) *
                 utils.roll_2d(v_x, (1, 0)))

    flux_y_out = -1 * v_y * c_edge_y
    flux_y_in = (utils.roll_2d(c_edge_y, (0, 1)) *
                 utils.roll_2d(v_y, (0, 1)))

    diff_flux_x_in = -1 * utils.roll_2d(c_x_edge_x, (1, 0))
    diff_flux_x_out = c_x_edge_x
    diff_flux_y_in = -1 * utils.roll_2d(c_y_edge_y, (0, 1))
    diff_flux_y_out = c_y_edge_y

    advective_flux = tf.add_n([
        flux_x_in,
        flux_x_out,
        flux_y_in,
        flux_y_out
    ])

    diffusive_flux = self.diffusion_const * tf.add_n([
        diff_flux_x_in,
        diff_flux_x_out,
        diff_flux_y_in,
        diff_flux_y_out
    ])

    total_flux = (advective_flux + diffusive_flux) / grid.step
    state_time_derivative[states.C_T] = total_flux
    state_time_derivative[states.VX_T] = tf.zeros_like(v_x)
    state_time_derivative[states.VY_T] = tf.zeros_like(v_y)
    return state_time_derivative

  def to_proto(self) -> metadata_pb2.Equation:
    """Creates a protocol buffer holding parameters of the equation."""
    return metadata_pb2.Equation(
        in_state_velocity_advection_diffusion=dict(
            diffusion_const=self.diffusion_const,
        ),
        scheme=metadata_pb2.Equation.FINITE_VOLUME
    )


def _fourier_terms(
    rnd_gen: np.random.RandomState,
    grid: grids.Grid,
    num_fourier_terms: int,
    max_fourier_periods: int,
    lower_bound: float = 1e-3
) -> np.ndarray:
  """Generates a normalized distribution of truncated Fourier harmonics."""
  x, y = grid.get_mesh()
  distribution = np.zeros(grid.get_shape())
  for _ in range(num_fourier_terms):
    amplitude = rnd_gen.random_sample()
    phase = rnd_gen.random_sample()
    kx = (rnd_gen.randint(-max_fourier_periods, max_fourier_periods + 1) *
          2 * np.pi / grid.length_x)
    ky = (rnd_gen.randint(-max_fourier_periods, max_fourier_periods + 1) *
          2 * np.pi / grid.length_y)
    distribution += amplitude * np.sin(kx * x + ky * y + phase)

  min_value = np.min(distribution)
  distribution += lower_bound - min_value
  max_amplitude = np.max(distribution)
  distribution /= max_amplitude
  return distribution


def generate_random_fourier_terms(
    batch_size: int,
    rnd_gen: np.random.RandomState,
    grid: grids.Grid,
    num_fourier_terms: int = 5,
    max_fourier_periods: int = 3,
    lower_bound: float = 1e-3,
    **kwargs: Any
) -> np.ndarray:
  """Generates a batch of random initial conditions with Fourier modes.

  Generates random Fourier terms truncated at max_fourier_periods.
  Distribution is periodic in x and y directions. A constant is added to make
  distribution strictly positive (no negative mass). The result is normalized
  to be in range [lower_bound, 1.].

  Args:
    batch_size: Size of the batch dimension.
    rnd_gen: Random number generator to be used for generation of components.
    grid: Grid on which initial conditions are evaluated.
    num_fourier_terms: Number of harmonic terms to generate.
    max_fourier_periods: Number of full periods that can appear across the grid.
    lower_bound: Lower bound for concentration, must be positive.
    **kwargs: Parameters that are passed to other initialization methods.

  Returns:
    Harmonic initial conditions array of shape=[grid.size_x, grid.size_y]
    and dtype=float64.

  Raises:
    ValueError: lower bound can't be negative.
  """
  del kwargs  # not used by generate_random_fourier_terms
  random_fourier_modes = [
      _fourier_terms(rnd_gen, grid, num_fourier_terms,
                     max_fourier_periods, lower_bound)
      for _ in range(batch_size)
  ]
  return np.stack(random_fourier_modes)


def generate_fourier_terms(
    batch_size: int,
    grid: grids.Grid,
    num_fourier_terms: int = 5,
    max_fourier_periods: int = 3,
    lower_bound: float = 1e-3,
    seed: int = 1,
    **kwargs: Any
) -> np.ndarray:
  """Generates a batch of initial conditions with Fourier modes.

  Generates an ensemble of Fourier terms truncated at max_fourier_periods.
  Distribution is periodic in x and y directions. A constant is added to make
  distribution strictly positive (no negative mass). The result is normalized
  to be in range [lower_bound, 1.].

  Args:
    batch_size: Size of the batch dimension.
    grid: Grid on which initial conditions are evaluated.
    num_fourier_terms: Number of harmonic terms to generate.
    max_fourier_periods: Number of full periods that can appear across the grid.
    lower_bound: Lower bound for concentration, must be positive.
    seed: Seed to initialize random number generator.
    **kwargs: Parameters that are passed to other initialization methods.


  Returns:
    Harmonic initial conditions array of shape=[grid.size_x, grid.size_y]
    and dtype=float64.

  Raises:
    ValueError: lower bound can't be negative.
  """
  del kwargs  # not used by generate_fourier_terms
  rnd_gen = np.random.RandomState(seed=seed)
  return generate_random_fourier_terms(
      batch_size, rnd_gen, grid, num_fourier_terms,
      max_fourier_periods, lower_bound)


def _gaussian(
    x: float,
    y: float,
    x_position: float,
    y_position: float,
    gaussian_width: float) -> np.ndarray:
  """Generates a gaussian distribution at given location with given width."""
  return np.exp(-((x - x_position) / gaussian_width) ** 2 +
                -((y - y_position) / gaussian_width) ** 2)


def _symmetrized_gaussian(
    grid: grids.Grid,
    x_position: float,
    y_position: float,
    gaussian_width: float = None
) -> np.ndarray:
  """Generates a gaussian distribution at given location with periodic BC.

  Args:
    grid: Grid on which initial conditions are evaluated.
    x_position: X position of the center of the gaussian.
    y_position: Y position of the center of the gaussian.
    gaussian_width: Width of the gaussian in units of x.

  Returns:
    Gaussian distribution on mesh of shape=[grid.size_x, grid.size_y].
  """
  x, y = grid.get_mesh()
  distribution = np.zeros(grid.get_shape())
  for i in range(-1, 2):
    for j in range(-1, 2):
      distribution += _gaussian(x + np.max(x) * i, y + np.max(y) * j,
                                x_position, y_position, gaussian_width)
  return distribution


def generate_gaussian(
    batch_size: int,
    grid: grids.Grid,
    x_position: float = np.pi,
    y_position: float = np.pi,
    gaussian_width: float = 0.1,
    **kwargs: Any
) -> np.ndarray:
  """Generates initial conditions with specified gaussian distribution.

  Args:
    batch_size: Size of the batch dimension.
    grid: Grid on which initial conditions are evaluated.
    x_position: X coordinate of the center of the gaussian.
    y_position: Y coordinate of the center of the gaussian.
    gaussian_width: Width of the gaussian measured in units of x.
    **kwargs: Parameters that are passed to other initialization methods.

  Returns:
    Initial conditions array of shape=[grid.size_x, grid.size_y]
    and dtype=float64.
  """
  del kwargs  # not used by generate_gaussian
  batch = [
      _symmetrized_gaussian(grid, x_position, y_position, gaussian_width)
      for _ in range(batch_size)
  ]
  return np.stack(batch)


def _symmetrized_gaussians(
    rnd_gen: np.random.RandomState,
    grid: grids.Grid,
    num_gaussian_terms: int,
    gaussian_width: float
) -> np.ndarray:
  """Generates a composition of randomly distributed gaussian distributions.

  Args:
    rnd_gen: Random number generator for position of gaussians.
    grid: Grid on which initial conditions are evaluated.
    num_gaussian_terms: Number of gaussian terms to generate.
    gaussian_width: Width of the gaussian in units of x.

  Returns:
    Composition of randomly distributed gaussian distributions with max value
    normalized to 1.
  """
  x, y = grid.get_mesh()
  distribution = np.zeros(grid.get_shape())
  for _ in range(num_gaussian_terms):
    x_pos = rnd_gen.random_sample() * np.max(x)
    y_pos = rnd_gen.random_sample() * np.max(y)
    distribution += _symmetrized_gaussian(grid, x_pos, y_pos, gaussian_width)
  max_value = np.max(distribution)
  distribution /= max_value
  return distribution


def generate_random_gaussians(
    batch_size: int,
    rnd_gen: np.random.RandomState,
    grid: grids.Grid,
    num_gaussian_terms: int = 1,
    gaussian_width: float = 0.1,
    **kwargs: Any
) -> np.ndarray:
  """Generates a batch of initial conditions with random gaussians.

  Generates a batch of randomly placed guassian distribution that are
  symmetrized with respect to boundary conditions. This avoids sharp accidental
  sharp gradients.

  Args:
    batch_size: Size of the batch dimension.
    rnd_gen: Random number generator for position of gaussians.
    grid: Grid on which initial conditions are evaluated.
    num_gaussian_terms: Number of gaussian terms to generate.
    gaussian_width: Width of the gaussian in units of x.
    **kwargs: Parameters that are passed to other initialization methods.

  Returns:
    Initial conditions with randomly placed gaussians.
  """
  del kwargs  # not used by generate_random_gaussians
  batch = [
      _symmetrized_gaussians(rnd_gen, grid, num_gaussian_terms, gaussian_width)
      for _ in range(batch_size)
  ]
  return np.stack(batch)


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


EQUATION_TYPES = {
    ('advection_diffusion', metadata_pb2.Equation.FINITE_DIFF):
        FiniteDifferenceAdvectionDiffusion,
    ('advection_diffusion', metadata_pb2.Equation.FINITE_VOLUME):
        FiniteVolumeAdvectionDiffusion,
    ('advection_diffusion', metadata_pb2.Equation.UPWIND):
        FiniteVolumeUpwindAdvectionDiffusion,
    ('in_state_velocity_advection_diffusion',
     metadata_pb2.Equation.FINITE_VOLUME):
        InStateVelocityFiniteVolumeAdvectionDiffusion,
    ('in_state_velocity_advection_diffusion',
     metadata_pb2.Equation.UPWIND):
        InStateVelocityUpwindAdvectionDiffusion
}


def equation_from_proto(proto: metadata_pb2.Equation, scheme: str = None):
  """Constructs an equation from the Equation protocol buffer.

  Args:
    proto: Equation protocol buffer encoding the Equation.
    scheme: Override to the scheme of the equation. Needed for testing
        different implementation in training and evaluation.

  Returns:
    Equation object.

  Raises:
    ValueError: Provided protocol buffer was not recognized, check proto names.
  """
  if scheme is None:
    scheme = proto.scheme
  else:
    scheme = metadata_pb2.Equation.DiscretizationScheme.Value(scheme)
  equation_and_scheme = (proto.WhichOneof('continuous_equation'), scheme)
  if equation_and_scheme in EQUATION_TYPES:
    eq_proto = getattr(proto, proto.WhichOneof('continuous_equation'))
    return EQUATION_TYPES[equation_and_scheme].from_proto(eq_proto)

  raise ValueError('Equation protocol buffer is not recognized')
