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
"""Advection diffussion equations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np
from pde_superresolution_2d import metadata_pb2
from pde_superresolution_2d.advection import velocity_fields
from pde_superresolution_2d.core import equations
from pde_superresolution_2d.core import grids
from pde_superresolution_2d.core import states
from pde_superresolution_2d.core import tensor_ops
import tensorflow as tf
from typing import Any, Dict, Tuple, Union

C = states.StateKey('concentration', (0, 0, 0), (0, 0))
C_EDGE_X = states.StateKey('concentration', (0, 0, 0), (1, 0))
C_EDGE_Y = states.StateKey('concentration', (0, 0, 0), (0, 1))
C_X = states.StateKey('concentration', (1, 0, 0), (0, 0))
C_Y = states.StateKey('concentration', (0, 1, 0), (0, 0))
C_T = states.StateKey('concentration', (0, 0, 1), (0, 0))
C_X_EDGE_X = states.StateKey('concentration', (1, 0, 0), (1, 0))
C_Y_EDGE_Y = states.StateKey('concentration', (0, 1, 0), (0, 1))
C_XX = states.StateKey('concentration', (2, 0, 0), (0, 0))
C_YY = states.StateKey('concentration', (0, 2, 0), (0, 0))

VX = states.StateKey('velocity_x', (0, 0, 0), (0, 0))
VY = states.StateKey('velocity_y', (0, 0, 0), (0, 0))
VX_EDGE_X = states.StateKey('velocity_x', (0, 0, 0), (1, 0))
VY_EDGE_Y = states.StateKey('velocity_y', (0, 0, 0), (0, 1))

KeyedTensors = Dict[states.StateKey, tf.Tensor]
ArrayLike = Union[np.ndarray, np.generic, float]
Shape = Union[int, Tuple[int]]

# numpy.random.RandomState uses uint32 for seeds
MAX_SEED_PLUS_ONE = 2**32

# We really are using native Python strings everywhere (no serialization to
# bytes/unicode is involved)
# pylint: disable=g-ambiguous-str-annotation


class _AdvectionDiffusionBase(equations.Equation):
  """Shared base class for advection and advection-diffusion equations."""

  @property
  def diffusion_coefficient(self) -> float:
    raise NotImplementedError

  @property
  def cfl_safety_factor(self) -> float:
    raise NotImplementedError

  def random_state(
      self,
      grid: grids.Grid,
      params: Dict[str, Dict[str, Any]] = None,
      size: Shape = (),
      seed: int = None,
      dtype: Any = np.float32,
  ) -> Dict[states.StateKey, np.ndarray]:
    """Returns a state with random initial conditions for a given ensemble.

    Args:
      grid: Grid object holding discretization parameters.
      params: optional dict with keys 'concentration' and 'velocity' specifying
        concentration and velocity parameters passed on to the
        random_concentration() and random_velocities() methods.
      size: leading "batch" dimensions to include on output tensors.
      seed: random seed to use for random number generator.
      dtype: dtype for generated tensors.

    Returns:
      State with random initial values.

    Raises:
      ValueError: if 'seed' is specified both directly and inside the params
        dict for concentration or velocities.
    """
    if params is None:
      params = {}
    params = dict(params)
    params.setdefault('concentration', {})
    params.setdefault('velocity', {})

    if seed is not None:
      if 'seed' in params['concentration'] or 'seed' in params['velocity']:
        raise ValueError('cannot set seed if concentration or velocity seeds '
                         'also provided.')

      random = np.random.RandomState(seed)
      params['concentration']['seed'] = random.randint(MAX_SEED_PLUS_ONE)
      params['velocity']['seed'] = random.randint(MAX_SEED_PLUS_ONE)

    state = {}
    state[C] = self.random_concentration(
        grid, size, **params['concentration'])

    vx_key, vy_key = self.CONSTANT_KEYS
    state[vx_key], state[vy_key] = self.random_velocities(
        grid, size, **params['velocity'])

    state = {k: v.astype(dtype) for k, v in state.items()}

    return state

  def random_concentration(self,
                           grid: grids.Grid,
                           size: Shape = (),
                           method: str = 'sum_of_gaussians',
                           **kwargs: Any) -> np.ndarray:
    """Create a random concentration."""
    if method == 'sum_of_gaussians':
      return random_sum_of_gaussians(grid=grid, size=size, **kwargs)
    elif method == 'fourier_series':
      return random_fourier_series(grid=grid, size=size, **kwargs)
    else:
      raise ValueError('initialization method not supported: {}'.format(method))

  def random_velocities(self, grid: grids.Grid, size: Shape = (),
                        **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Create random X- and Y-velocities."""

    v_field = velocity_fields.ConstantVelocityField.from_seed(**kwargs)

    vx_key, vy_key = self.CONSTANT_KEYS
    assert vx_key.name == 'velocity_x'
    assert vy_key.name == 'velocity_y'

    if isinstance(size, int):
      size = (size,)
    state_shape = size + (grid.size_x, grid.size_y)

    # finite differences represents the solution at points;
    # finite volumes represents the solution over unit-cells.
    cell_average = (
        self.METHOD != metadata_pb2.Equation.Discretization.FINITE_DIFFERENCE)
    vx = v_field.get_velocity_x(
        t=0, grid=grid, shift=vx_key.offset, cell_average=cell_average)
    vy = v_field.get_velocity_y(
        t=0, grid=grid, shift=vy_key.offset, cell_average=cell_average)
    return np.broadcast_to(vx, state_shape), np.broadcast_to(vy, state_shape)

  def get_time_step(self, grid: grids.Grid, max_velocity: float = 1.0) -> float:
    """Returns appropriate time step for a forward time-step.

    Stability conditions on the time step:
    - For advection: dt <= v_max * dx.
    - For diffusion: dt <= dx**2 / (2**N * D)

    where N is the number of spatial dimensions.

    Both conditions need to be satisfied for stable numerical integration. Note
    that if diffusion dominates, explicit time stepping is a really bad idea.

    References:
      CFL condition (for advection):
        https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition
      FTCS scheme (for diffusion):
        https://en.wikipedia.org/wiki/FTCS_scheme
        http://web.cecs.pdx.edu/~gerry/class/ME448/notes/pdf/FTCS_slides_2up.pdf

    Args:
      grid: Grid object holding discretization parameters.
      max_velocity: maximum velocity of the field.

    Returns:
      The value of an appropriate time step.
    """
    D = self.diffusion_coefficient  # pylint: disable=invalid-name
    N = 2  # pylint: disable=invalid-name
    dx = grid.step
    advect_limit = max_velocity * dx
    max_step = min(advect_limit, dx**2 / (2**N * D)) if D else advect_limit
    return self.cfl_safety_factor * max_step


def max_stable_diffusion(grid: grids.Grid, max_velocity: float = 1.0) -> float:
  """How much diffusion can be added without decreasing the time step?"""
  N = 2  # pylint: disable=invalid-name
  dx = grid.step
  advect_limit = max_velocity * dx
  return (dx ** 2 / 2 ** N) / advect_limit


def upwind_numerical_diffusion(
    grid: grids.Grid,
    cfl: float = 0.9,
    velocity: float = 1.0,
) -> float:
  """How much numerical diffusion get added by a first-order upwind scheme?"""
  # Reference: J. Zhuang et al.: The importance of vertical resolution in the
  # free troposphere (Equation A11)
  return (1 - cfl) * velocity * grid.step / 2


@equations.register_continuous_equation('advection_diffusion')
class AdvectionDiffusion(_AdvectionDiffusionBase):
  """Base class for advection diffusion equations.

  This base class defines the state and common methods.

  Subclasses must implement the time_derivative() method.

  Attributes:
    diffusion_coefficient: diffusion coefficient.
    cfl_safety_factor: safety factor by which to reduce the time step from the
      maximum stable time-step, according to the CFL condition.
  """

  def __init__(
      self,
      diffusion_coefficient: float,
      cfl_safety_factor: float = 0.9,
  ):
    self._diffusion_coefficient = diffusion_coefficient
    self._cfl_safety_factor = cfl_safety_factor

  @property
  def diffusion_coefficient(self) -> float:
    return self._diffusion_coefficient

  @property
  def cfl_safety_factor(self) -> float:
    return self._cfl_safety_factor

  @classmethod
  def from_proto(cls, proto: metadata_pb2.AdvectionDiffusionEquation
                ) -> equations.Equation:
    """Construct an equation from a protocol buffer."""
    return cls(proto.diffusion_coefficient, proto.cfl_safety_factor)

  def to_proto(self) -> metadata_pb2.Equation:
    """Creates a protocol buffer holding parameters of the equation."""
    return metadata_pb2.Equation(
        discretization=dict(
            name=self.DISCRETIZATION_NAME,
            method=self.METHOD,
            monotonic=self.MONOTONIC,
        ),
        advection_diffusion=dict(
            diffusion_coefficient=self.diffusion_coefficient,
            cfl_safety_factor=self.cfl_safety_factor,
        ),
    )


@equations.register_continuous_equation('advection')
class Advection(_AdvectionDiffusionBase):
  """Base class for pure-advection equations."""

  def __init__(self, cfl_safety_factor: float = 0.9):
    self._cfl_safety_factor = cfl_safety_factor

  @property
  def diffusion_coefficient(self) -> float:
    return 0.0

  @property
  def cfl_safety_factor(self) -> float:
    return self._cfl_safety_factor

  @classmethod
  def from_proto(cls, proto: metadata_pb2.AdvectionEquation
                ) -> equations.Equation:
    """Construct an equation from a protocol buffer."""
    return cls(proto.cfl_safety_factor)

  def to_proto(self) -> metadata_pb2.Equation:
    """Creates a protocol buffer holding parameters of the equation."""
    return metadata_pb2.Equation(
        discretization=dict(
            name=self.DISCRETIZATION_NAME,
            method=self.METHOD,
            monotonic=self.MONOTONIC,
        ),
        advection=dict(
            cfl_safety_factor=self.cfl_safety_factor,
        ),
    )


class FiniteDifferenceAdvectionDiffusion(AdvectionDiffusion):
  """"Finite difference advection-diffusion."""
  DISCRETIZATION_NAME = 'finite_difference'
  METHOD = metadata_pb2.Equation.Discretization.FINITE_DIFFERENCE
  MONOTONIC = False

  STATE_KEYS = (C, VX, VY)
  CONSTANT_KEYS = (VX, VY)
  INPUT_KEYS = (VX, VY, C_X, C_Y, C_XX, C_YY)

  def time_derivative(
      self,
      state: KeyedTensors,
      inputs: KeyedTensors,
      grid: grids.Grid,
      time: float = 0.0,
  ) -> KeyedTensors:
    """See base class."""
    del time  # unused
    equations.check_keys(inputs, self.INPUT_KEYS)

    v_x = inputs[VX]
    v_y = inputs[VY]
    c_x = inputs[C_X]
    c_y = inputs[C_Y]
    c_xx = inputs[C_XX]
    c_yy = inputs[C_YY]

    D = self.diffusion_coefficient  # pylint: disable=invalid-name
    c_t = -(v_x * c_x + v_y * c_y) + D * (c_xx + c_yy)
    return {C_T: c_t}


class FiniteDifferenceAdvection(Advection):
  """"Finite difference scheme for advection only."""
  DISCRETIZATION_NAME = 'finite_difference'
  METHOD = metadata_pb2.Equation.Discretization.FINITE_DIFFERENCE
  MONOTONIC = False

  STATE_KEYS = (C, VX, VY)
  CONSTANT_KEYS = (VX, VY)
  INPUT_KEYS = (VX, VY, C_X, C_Y)

  def time_derivative(
      self,
      state: KeyedTensors,
      inputs: KeyedTensors,
      grid: grids.Grid,
      time: float = 0.0,
  ) -> KeyedTensors:
    """See base class."""
    del time  # unused
    equations.check_keys(inputs, self.INPUT_KEYS)
    v_x = inputs[VX]
    v_y = inputs[VY]
    c_x = inputs[C_X]
    c_y = inputs[C_Y]
    c_t = -(v_x * c_x + v_y * c_y)
    return {C_T: c_t}


def _flux_to_time_derivative(flux_x_edge_x, flux_y_edge_y, grid_step):
  """Use continuity to convert from fluxes to a time derivative."""
  # right - left + top - bottom
  numerator = tf.add_n([
      flux_x_edge_x,
      -tensor_ops.roll_2d(flux_x_edge_x, (1, 0)),
      flux_y_edge_y,
      -tensor_ops.roll_2d(flux_y_edge_y, (0, 1)),
  ])
  return -(1 / grid_step) * numerator


class FiniteVolumeAdvectionDiffusion(AdvectionDiffusion):
  """Finite-volume scheme for advection-diffusion."""

  DISCRETIZATION_NAME = 'finite_volume'
  METHOD = metadata_pb2.Equation.Discretization.FINITE_VOLUME
  MONOTONIC = False

  STATE_KEYS = (C, VX_EDGE_X, VY_EDGE_Y)
  CONSTANT_KEYS = (VX_EDGE_X, VY_EDGE_Y)
  INPUT_KEYS = (
      VX_EDGE_X, VY_EDGE_Y, C_EDGE_X, C_EDGE_Y, C_X_EDGE_X, C_Y_EDGE_Y,
  )

  def time_derivative(
      self,
      state: KeyedTensors,
      inputs: KeyedTensors,
      grid: grids.Grid,
      time: float = 0.0,
  ) -> KeyedTensors:
    """See base class."""
    del time  # unused
    equations.check_keys(inputs, self.INPUT_KEYS)

    v_x = inputs[VX_EDGE_X]
    v_y = inputs[VY_EDGE_Y]
    c_edge_x = inputs[C_EDGE_X]
    c_edge_y = inputs[C_EDGE_Y]
    c_x_edge_x = inputs[C_X_EDGE_X]
    c_y_edge_y = inputs[C_Y_EDGE_Y]

    D = self.diffusion_coefficient  # pylint: disable=invalid-name
    flux_x = v_x * c_edge_x - D * c_x_edge_x
    flux_y = v_y * c_edge_y - D * c_y_edge_y
    c_t = _flux_to_time_derivative(flux_x, flux_y, grid.step)
    return {C_T: c_t}


class FiniteVolumeAdvection(Advection):
  """Finite-volume scheme for advection-diffusion."""

  DISCRETIZATION_NAME = 'finite_volume'
  METHOD = metadata_pb2.Equation.Discretization.FINITE_VOLUME
  MONOTONIC = False

  STATE_KEYS = (C, VX_EDGE_X, VY_EDGE_Y)
  CONSTANT_KEYS = (VX_EDGE_X, VY_EDGE_Y)
  INPUT_KEYS = (VX_EDGE_X, VY_EDGE_Y, C_EDGE_X, C_EDGE_Y)

  def time_derivative(
      self,
      state: KeyedTensors,
      inputs: KeyedTensors,
      grid: grids.Grid,
      time: float = 0.0,
  ) -> KeyedTensors:
    """See base class."""
    del time  # unused
    equations.check_keys(inputs, self.INPUT_KEYS)

    v_x = inputs[VX_EDGE_X]
    v_y = inputs[VY_EDGE_Y]
    c_edge_x = inputs[C_EDGE_X]
    c_edge_y = inputs[C_EDGE_Y]

    flux_x = v_x * c_edge_x
    flux_y = v_y * c_edge_y
    c_t = _flux_to_time_derivative(flux_x, flux_y, grid.step)
    return {C_T: c_t}


class UpwindAdvectionDiffusion(AdvectionDiffusion):
  """Upwind finite-volume scheme for advection-diffusion."""

  DISCRETIZATION_NAME = 'upwind'
  METHOD = metadata_pb2.Equation.Discretization.FINITE_VOLUME
  MONOTONIC = True

  STATE_KEYS = (C, VX_EDGE_X, VY_EDGE_Y)
  CONSTANT_KEYS = (VX_EDGE_X, VY_EDGE_Y)
  INPUT_KEYS = (VX_EDGE_X, VY_EDGE_Y, C, C_X_EDGE_X, C_Y_EDGE_Y)

  def time_derivative(
      self,
      state: KeyedTensors,
      inputs: KeyedTensors,
      grid: grids.Grid,
      time: float = 0.0,
  ) -> KeyedTensors:
    """See base class."""
    del time  # unused
    equations.check_keys(inputs, self.INPUT_KEYS)

    v_x = inputs[VX_EDGE_X]
    v_y = inputs[VY_EDGE_Y]
    c = inputs[C]
    c_x_edge_x = inputs[C_X_EDGE_X]
    c_y_edge_y = inputs[C_Y_EDGE_Y]

    c_right = tensor_ops.roll_2d(c, (-1, 0))
    c_top = tensor_ops.roll_2d(c, (0, -1))

    D = self.diffusion_coefficient  # pylint: disable=invalid-name
    flux_x = v_x * tf.where(v_x > 0, c, c_right) - D * c_x_edge_x
    flux_y = v_y * tf.where(v_y > 0, c, c_top) - D * c_y_edge_y
    c_t = _flux_to_time_derivative(flux_x, flux_y, grid.step)
    return {C_T: c_t}


class UpwindAdvection(Advection):
  """Upwind finite-volume scheme for advection only."""

  DISCRETIZATION_NAME = 'upwind'
  METHOD = metadata_pb2.Equation.Discretization.FINITE_VOLUME
  MONOTONIC = True

  STATE_KEYS = (C, VX_EDGE_X, VY_EDGE_Y)
  CONSTANT_KEYS = (VX_EDGE_X, VY_EDGE_Y)
  INPUT_KEYS = (VX_EDGE_X, VY_EDGE_Y, C)

  def time_derivative(
      self,
      state: KeyedTensors,
      inputs: KeyedTensors,
      grid: grids.Grid,
      time: float = 0.0,
  ) -> KeyedTensors:
    """See base class."""
    del time  # unused
    equations.check_keys(inputs, self.INPUT_KEYS)

    v_x = inputs[VX_EDGE_X]
    v_y = inputs[VY_EDGE_Y]
    c = inputs[C]

    c_right = tensor_ops.roll_2d(c, (-1, 0))
    c_top = tensor_ops.roll_2d(c, (0, -1))

    flux_x = v_x * tf.where(v_x > 0, c, c_right)
    flux_y = v_y * tf.where(v_y > 0, c, c_top)
    c_t = _flux_to_time_derivative(flux_x, flux_y, grid.step)
    return {C_T: c_t}


def _minimum(*args):
  return functools.reduce(tf.minimum, args)


def _maximum(*args):
  return functools.reduce(tf.maximum, args)


def _tendency_vanleer_1d(c, v, dx, dt, axis, c_x=None, diffusion_coefficient=0):
  """Calculate the tendency in a single direction."""

  # To match our indexing convention with the paper, note:
  # - concentration c is defined at locations i+1/2
  # - velocity v is defined at locations i+1
  # - c_x is defined at locations i+1

  def roll_minus_one(array):
    return tensor_ops.roll(array, 1, axis)

  def roll_plus_one(array):
    return tensor_ops.roll(array, -1, axis)

  # c is defined at i+1/2
  c_left = roll_minus_one(c)  # i-1/2
  c_right = roll_plus_one(c)  # i+3/2

  delta = c - c_left  # i
  delta_average = 0.5 * (delta + roll_plus_one(delta))  # i+1/2

  # all defined at i+1/2
  c_min = _minimum(c_left, c, c_right)
  c_max = _maximum(c_left, c, c_right)
  mismatch = tf.sign(delta_average) * _minimum(
      abs(delta_average), 2 * (c - c_min), 2 * (c_max - c))

  # all defined at i+1
  # in the paper's notation: (1/2) * (1 -/+ C_{-/+})
  correction = 0.5 - abs(v) * (0.5 * dt / dx)
  flux_minus = v * (c + mismatch * correction)
  flux_plus = v * (c_right - roll_plus_one(mismatch) * correction)
  flux = tf.where(v >= 0, flux_minus, flux_plus)

  if diffusion_coefficient:
    if c_x is None:
      raise ValueError('must provide c_x if diffusion coefficient is set')
    flux -= diffusion_coefficient * c_x

  # defined at i+1/2
  return -(dt / dx) * (flux - roll_minus_one(flux))


class VanLeerMono5AdvectionDiffusion(AdvectionDiffusion):
  """Second order upwind finite-volume scheme for advection-diffusion.

  Adapted from the "mono-5" limiter from
  Lin, S.-J., et al. (1994). "A class of the van Leer-type transport schemes
  and its application to the moisture transport in a general circulation
  model."
  """

  DISCRETIZATION_NAME = 'van_leer_mono5'
  METHOD = metadata_pb2.Equation.Discretization.FINITE_VOLUME
  MONOTONIC = True

  STATE_KEYS = (C, VX_EDGE_X, VY_EDGE_Y)
  CONSTANT_KEYS = (VX_EDGE_X, VY_EDGE_Y)
  INPUT_KEYS = (VX_EDGE_X, VY_EDGE_Y, C, C_X_EDGE_X, C_Y_EDGE_Y)

  def take_time_step(
      self,
      state: KeyedTensors,
      inputs: KeyedTensors,
      grid: grids.Grid,
      time: float = 0.0,
  ) -> KeyedTensors:
    """See base class."""
    del time  # unused
    equations.check_keys(inputs, self.INPUT_KEYS)

    v_x = inputs[VX_EDGE_X]
    v_y = inputs[VY_EDGE_Y]
    c = inputs[C]
    c_x = inputs[C_X_EDGE_X]
    c_y = inputs[C_Y_EDGE_Y]

    dx = dy = grid.step
    dt = self.get_time_step(grid)
    D = self.diffusion_coefficient  # pylint: disable=invalid-name
    x_axis = -2
    y_axis = -1
    tendency_x_then_y = _tendency_vanleer_1d(
        c + 0.5 * _tendency_vanleer_1d(c, v_x, dx, dt, x_axis, c_x, D),
        v_y, dy, dt, y_axis, c_y, D)
    tendency_y_then_x = _tendency_vanleer_1d(
        c + 0.5 * _tendency_vanleer_1d(c, v_y, dy, dt, y_axis, c_y, D),
        v_x, dx, dt, x_axis, c_x, D)
    c_next = tf.add_n([state[C], tendency_x_then_y, tendency_y_then_x])
    return {C: c_next}


class VanLeerMono5Advection(Advection):
  """Second order upwind finite-volume scheme for advection only.

  Based on the "mono-5" limiter from
  Lin, S.-J., et al. (1994). "A class of the van Leer-type transport schemes
  and its application to the moisture transport in a general circulation
  model."
  """

  DISCRETIZATION_NAME = 'van_leer_mono5'
  METHOD = metadata_pb2.Equation.Discretization.FINITE_VOLUME
  MONOTONIC = True

  STATE_KEYS = (C, VX_EDGE_X, VY_EDGE_Y)
  CONSTANT_KEYS = (VX_EDGE_X, VY_EDGE_Y)
  INPUT_KEYS = (VX_EDGE_X, VY_EDGE_Y, C)

  def take_time_step(
      self,
      state: KeyedTensors,
      inputs: KeyedTensors,
      grid: grids.Grid,
      time: float = 0.0,
  ) -> KeyedTensors:
    """See base class."""
    del time  # unused
    equations.check_keys(inputs, self.INPUT_KEYS)

    v_x = inputs[VX_EDGE_X]
    v_y = inputs[VY_EDGE_Y]
    c = inputs[C]

    dx = dy = grid.step
    dt = self.get_time_step(grid)
    x_axis = -2
    y_axis = -1
    tendency_x_then_y = _tendency_vanleer_1d(
        c + 0.5 * _tendency_vanleer_1d(c, v_x, dx, dt, axis=x_axis),
        v_y, dy, dt, axis=y_axis)
    tendency_y_then_x = _tendency_vanleer_1d(
        c + 0.5 * _tendency_vanleer_1d(c, v_y, dy, dt, axis=y_axis),
        v_x, dx, dt, axis=x_axis)
    c_next = tf.add_n([state[C], tendency_x_then_y, tendency_y_then_x])
    return {C: c_next}


def random_fourier_series(
    grid: grids.Grid,
    size: Shape = (),
    seed: int = None,
    num_terms: int = 5,
    max_periods: int = 4,
    normalize: bool = True,
    lower_bound: float = 1e-3,
    upper_bound: float = 1.0,
) -> np.ndarray:
  """Generates a distribution of truncated Fourier harmonics.

  Args:
    grid: grid on which initial conditions are evaluated.
    size: leading "batch" dimensions to include on output tensors.
    seed: random seed to use for random number generator.
    num_terms: number of random fourier terms.
    max_periods: maximum number of x/y periods for fourier terms.
    normalize: if true, normalize the values to fall within the range
      [lower_bound, upper_bound].
    lower_bound: lower bound for values.
    upper_bound: upper bound for values.

  Returns:
    Float64 array of shape size+grid.shape.
  """
  if isinstance(size, int):
    size = (size,)

  random = np.random.RandomState(seed=seed)

  # dimensions [..., 1, 1, term]
  event_shape = size + (1, 1, num_terms)
  amplitude = random.random_sample(size=event_shape)
  kx = random.randint(-max_periods, max_periods + 1, size=event_shape)
  ky = random.randint(-max_periods, max_periods + 1, size=event_shape)
  phase = random.random_sample(size=event_shape)

  # dimensions [..., x, y, 1]
  x, y = grid.get_mesh()
  x = x.reshape((1,) * len(size) + x.shape + (1,))
  y = y.reshape((1,) * len(size) + y.shape + (1,))

  # dimensions [..., x, y, term]
  terms = amplitude * np.sin(
      2 * np.pi * (kx * x / grid.length_x + ky * y / grid.length_y + phase))
  # dimensions [..., x, y]
  distribution = np.sum(terms, axis=-1)

  if normalize:
    distribution_range = (
        distribution.max(axis=(-1, -2), keepdims=True) - distribution.min(
            axis=(-1, -2), keepdims=True))
    scale = (upper_bound - lower_bound) / distribution_range
    distribution = scale * distribution + lower_bound

  return distribution


def _gaussian(
    x: ArrayLike,
    y: ArrayLike,
    x_position: ArrayLike,
    y_position: ArrayLike,
    gaussian_width: ArrayLike,
) -> np.ndarray:
  """Generates a gaussian distribution at given location with given width."""
  position_shape = (
      getattr(x_position, 'shape', ()) + (1,) * len(getattr(x, 'shape', ())))
  x_position = np.reshape(x_position, position_shape)
  y_position = np.reshape(y_position, position_shape)
  return np.exp(-((x - x_position) / gaussian_width)**2 + -(
      (y - y_position) / gaussian_width)**2)


def symmetrized_gaussian(
    grid: grids.Grid,
    x_position: ArrayLike,
    y_position: ArrayLike,
    gaussian_width: ArrayLike,
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
  distribution = 0  # type: np.ndarray
  for i in range(-1, 2):
    for j in range(-1, 2):
      distribution += _gaussian(x + grid.length_x * i, y + grid.length_y * j,
                                x_position, y_position, gaussian_width)
  return distribution


def random_sum_of_gaussians(
    grid: grids.Grid,
    size: Shape = (),
    seed: int = None,
    num_terms: int = 1,
    gaussian_width: float = 0.5,
    normalize: bool = True,
) -> np.ndarray:
  """Generate a random sum of gaussians on a grid.

  Args:
    grid: grid on which initial conditions are evaluated.
    size: leading "batch" dimensions to include on output tensors.
    seed: random seed to use for random number generator.
    gaussian_width: width of the gaussian measured in units of x.
    normalize: if true, normalize the maximum value to one.

  Returns:
    Float64 array of shape size+get.shape.
  """
  random = np.random.RandomState(seed)
  if isinstance(size, int):
    size = (size,)
  size = (num_terms,) + size
  x_pos = random.random_sample(size) * grid.length_x
  y_pos = random.random_sample(size) * grid.length_y
  distribution = symmetrized_gaussian(grid, x_pos, y_pos, gaussian_width)
  distribution = distribution.sum(axis=0)
  if normalize:
    distribution /= distribution.max(axis=(-2, -1), keepdims=True)
  return distribution
