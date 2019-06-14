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
"""Advection diffussion equations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum
import functools
import operator
from typing import Any, Dict, Tuple, Union

import numpy as np
from pde_superresolution_2d.advection import velocity_fields
from pde_superresolution_2d.core import equations
from pde_superresolution_2d.core import grids
from pde_superresolution_2d.core import polynomials
from pde_superresolution_2d.core import states
from pde_superresolution_2d.core import tensor_ops
import tensorflow as tf


StateDef = states.StateDefinition

X = states.Dimension.X
Y = states.Dimension.Y

NO_DERIVATIVES = (0, 0, 0)
D_X = (1, 0, 0)
D_Y = (0, 1, 0)
D_XX = (2, 0, 0)
D_YY = (0, 2, 0)

NO_OFFSET = (0, 0)
X_PLUS_HALF = (1, 0)
Y_PLUS_HALF = (0, 1)

ArrayLike = Union[np.ndarray, np.generic, float]
Shape = Union[int, Tuple[int]]

# numpy.random.RandomState uses uint32 for seeds
MAX_SEED_PLUS_ONE = 2**32


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
  ) -> Dict[str, np.ndarray]:
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
    state['concentration'] = self.random_concentration(
        grid, size, **params['concentration'])

    state['x_velocity'], state['y_velocity'] = self.random_velocities(
        grid, size, **params['velocity'])

    state = {k: v.astype(dtype) for k, v in state.items()}

    return state

  def random_concentration(self,
                           grid: grids.Grid,
                           size: Shape = (),
                           method: str = 'sum_of_gaussians',
                           normalize: bool = True,
                           binarize_center: float = 0.5,
                           binarize_slope: float = 1.0,
                           **kwargs: Any) -> np.ndarray:
    """Create a random concentration."""
    if method == 'sum_of_gaussians':
      concentration = random_sum_of_gaussians(
          grid=grid, size=size, normalize=normalize, **kwargs)
    elif method == 'fourier_series':
      concentration = random_fourier_series(
          grid=grid, size=size, normalize=normalize, **kwargs)
    else:
      raise ValueError('initialization method not supported: {}'.format(method))

    concentration = binarize(concentration, binarize_center, binarize_slope)

    return concentration

  def random_velocities(self, grid: grids.Grid, size: Shape = (),
                        **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Create random X- and Y-velocities."""

    v_field = velocity_fields.ConstantVelocityField.from_seed(**kwargs)

    if isinstance(size, int):
      size = (size,)
    state_shape = size + (grid.size_x, grid.size_y)

    # finite differences represents velocities at points;
    # finite volumes represents velocities over the faces of unit-cells.
    face_average = self.METHOD is not polynomials.Method.FINITE_DIFFERENCE
    shift_vx = self.key_definitions['x_velocity'].offset
    shift_vy = self.key_definitions['y_velocity'].offset
    vx = v_field.get_velocity_x(
        t=0, grid=grid, shift=shift_vx, face_average=face_average)
    vy = v_field.get_velocity_y(
        t=0, grid=grid, shift=shift_vy, face_average=face_average)
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


class AdvectionDiffusion(_AdvectionDiffusionBase):
  """Base class for advection diffusion equations.

  This base class defines the state and common methods.

  Subclasses must implement the time_derivative() method.

  Attributes:
    diffusion_coefficient: diffusion coefficient.
    cfl_safety_factor: safety factor by which to reduce the time step from the
      maximum stable time-step, according to the CFL condition.
  """

  CONTINUOUS_EQUATION_NAME = 'advection_diffusion'

  def __init__(
      self,
      diffusion_coefficient: float,
      cfl_safety_factor: float = 0.9,
  ):
    self._diffusion_coefficient = diffusion_coefficient
    self._cfl_safety_factor = cfl_safety_factor
    super(AdvectionDiffusion, self).__init__()

  @property
  def diffusion_coefficient(self) -> float:
    return self._diffusion_coefficient

  @property
  def cfl_safety_factor(self) -> float:
    return self._cfl_safety_factor

  def get_parameters(self) -> Dict[str, Any]:
    return dict(
        diffusion_coefficient=self.diffusion_coefficient,
        cfl_safety_factor=self.cfl_safety_factor,
    )


class Advection(_AdvectionDiffusionBase):
  """Base class for pure-advection equations."""

  CONTINUOUS_EQUATION_NAME = 'advection'

  def __init__(self, cfl_safety_factor: float = 0.9):
    self._cfl_safety_factor = cfl_safety_factor
    super(Advection, self).__init__()

  @property
  def diffusion_coefficient(self) -> float:
    return 0.0

  @property
  def cfl_safety_factor(self) -> float:
    return self._cfl_safety_factor

  def get_parameters(self) -> Dict[str, Any]:
    return dict(
        cfl_safety_factor=self.cfl_safety_factor,
    )


class FiniteDifferenceAdvectionDiffusion(AdvectionDiffusion):
  """Finite difference advection-diffusion."""
  DISCRETIZATION_NAME = 'finite_difference'
  METHOD = polynomials.Method.FINITE_DIFFERENCE
  MONOTONIC = False

  def __init__(self, *args, **kwargs):
    self.key_definitions = {
        'concentration':
            StateDef('concentration', (), NO_DERIVATIVES, NO_OFFSET),
        'x_velocity': StateDef('velocity', (X,), NO_DERIVATIVES, NO_OFFSET),
        'y_velocity': StateDef('velocity', (Y,), NO_DERIVATIVES, NO_OFFSET),
        'concentration_x': StateDef('concentration', (), D_X, NO_OFFSET),
        'concentration_y': StateDef('concentration', (), D_Y, NO_OFFSET),
        'concentration_xx': StateDef('concentration', (), D_XX, NO_OFFSET),
        'concentration_yy': StateDef('concentration', (), D_YY, NO_OFFSET),
    }
    self.evolving_keys = {'concentration'}
    self.constant_keys = {'x_velocity', 'y_velocity'}
    super(FiniteDifferenceAdvectionDiffusion, self).__init__(*args, **kwargs)

  def time_derivative(
      self, grid, concentration, x_velocity, y_velocity, concentration_x,
      concentration_y, concentration_xx, concentration_yy):
    """See base class."""
    del grid, concentration  # unused
    D = self.diffusion_coefficient  # pylint: disable=invalid-name
    c_t = (
        -(x_velocity * concentration_x + y_velocity * concentration_y)
        + D * (concentration_xx + concentration_yy)
    )
    return {'concentration': c_t}


class FiniteDifferenceAdvection(Advection):
  """"Finite difference scheme for advection only."""
  DISCRETIZATION_NAME = 'finite_difference'
  METHOD = polynomials.Method.FINITE_DIFFERENCE
  MONOTONIC = False

  def __init__(self, *args, **kwargs):
    self.key_definitions = {
        'concentration':
            StateDef('concentration', (), NO_DERIVATIVES, NO_OFFSET),
        'x_velocity': StateDef('velocity', (X,), NO_DERIVATIVES, NO_OFFSET),
        'y_velocity': StateDef('velocity', (Y,), NO_DERIVATIVES, NO_OFFSET),
        'concentration_x': StateDef('concentration', (), D_X, NO_OFFSET),
        'concentration_y': StateDef('concentration', (), D_Y, NO_OFFSET),
    }
    self.evolving_keys = {'concentration'}
    self.constant_keys = {'x_velocity', 'y_velocity'}
    super(FiniteDifferenceAdvection, self).__init__(*args, **kwargs)

  def time_derivative(
      self, grid, concentration, x_velocity, y_velocity,
      concentration_x, concentration_y):
    """See base class."""
    del grid, concentration  # unused
    c_t = -(x_velocity * concentration_x + y_velocity * concentration_y)
    return {'concentration': c_t}


def flux_to_time_derivative(x_flux_edge_x, y_flux_edge_y, grid_step):
  """Use continuity to convert from fluxes to a time derivative."""
  # right - left + top - bottom
  numerator = tf.add_n([
      x_flux_edge_x,
      -tensor_ops.roll_2d(x_flux_edge_x, (1, 0)),
      y_flux_edge_y,
      -tensor_ops.roll_2d(y_flux_edge_y, (0, 1)),
  ])
  return -(1 / grid_step) * numerator


class FiniteVolumeAdvectionDiffusion(AdvectionDiffusion):
  """Finite-volume scheme for advection-diffusion."""

  DISCRETIZATION_NAME = 'finite_volume'
  METHOD = polynomials.Method.FINITE_VOLUME
  MONOTONIC = False

  def __init__(self, *args, **kwargs):
    self.key_definitions = {
        'concentration':
            StateDef('concentration', (), NO_DERIVATIVES, NO_OFFSET),
        'x_velocity': StateDef('velocity', (X,), NO_DERIVATIVES, X_PLUS_HALF),
        'y_velocity': StateDef('velocity', (Y,), NO_DERIVATIVES, Y_PLUS_HALF),
        'concentration_edge_x':
            StateDef('concentration', (), NO_DERIVATIVES, X_PLUS_HALF),
        'concentration_edge_y':
            StateDef('concentration', (), NO_DERIVATIVES, Y_PLUS_HALF),
        'concentration_x_edge_x':
            StateDef('concentration', (), D_X, X_PLUS_HALF),
        'concentration_y_edge_y':
            StateDef('concentration', (), D_Y, Y_PLUS_HALF),
    }
    self.evolving_keys = {'concentration'}
    self.constant_keys = {'x_velocity', 'y_velocity'}
    super(FiniteVolumeAdvectionDiffusion, self).__init__(*args, **kwargs)

  def time_derivative(
      self, grid, concentration, x_velocity, y_velocity, concentration_edge_x,
      concentration_edge_y, concentration_x_edge_x, concentration_y_edge_y):
    """See base class."""
    del concentration  # unused
    D = self.diffusion_coefficient  # pylint: disable=invalid-name
    x_flux = x_velocity * concentration_edge_x - D * concentration_x_edge_x
    y_flux = y_velocity * concentration_edge_y - D * concentration_y_edge_y
    c_t = flux_to_time_derivative(x_flux, y_flux, grid.step)
    return {'concentration': c_t}


class FiniteVolumeAdvection(Advection):
  """Finite-volume scheme for advection-diffusion."""

  DISCRETIZATION_NAME = 'finite_volume'
  METHOD = polynomials.Method.FINITE_VOLUME
  MONOTONIC = False

  def __init__(self, *args, **kwargs):
    self.key_definitions = {
        'concentration':
            StateDef('concentration', (), NO_DERIVATIVES, NO_OFFSET),
        'x_velocity': StateDef('velocity', (X,), NO_DERIVATIVES, X_PLUS_HALF),
        'y_velocity': StateDef('velocity', (Y,), NO_DERIVATIVES, Y_PLUS_HALF),
        'concentration_edge_x':
            StateDef('concentration', (), NO_DERIVATIVES, X_PLUS_HALF),
        'concentration_edge_y':
            StateDef('concentration', (), NO_DERIVATIVES, Y_PLUS_HALF),
    }
    self.evolving_keys = {'concentration'}
    self.constant_keys = {'x_velocity', 'y_velocity'}
    super(FiniteVolumeAdvection, self).__init__(*args, **kwargs)

  def time_derivative(
      self, grid, concentration, x_velocity, y_velocity, concentration_edge_x,
      concentration_edge_y):
    """See base class."""
    del concentration  # unused
    x_flux = x_velocity * concentration_edge_x
    y_flux = y_velocity * concentration_edge_y
    c_t = flux_to_time_derivative(x_flux, y_flux, grid.step)
    return {'concentration': c_t}


class UpwindAdvectionDiffusion(AdvectionDiffusion):
  """Upwind finite-volume scheme for advection-diffusion."""

  DISCRETIZATION_NAME = 'upwind'
  METHOD = polynomials.Method.FINITE_VOLUME
  MONOTONIC = True

  def __init__(self, *args, **kwargs):
    self.key_definitions = {
        'concentration':
            StateDef('concentration', (), NO_DERIVATIVES, NO_OFFSET),
        'x_velocity': StateDef('velocity', (X,), NO_DERIVATIVES, X_PLUS_HALF),
        'y_velocity': StateDef('velocity', (Y,), NO_DERIVATIVES, Y_PLUS_HALF),
        'concentration_x_edge_x':
            StateDef('concentration', (), D_X, X_PLUS_HALF),
        'concentration_y_edge_y':
            StateDef('concentration', (), D_Y, Y_PLUS_HALF),
    }
    self.evolving_keys = {'concentration'}
    self.constant_keys = {'x_velocity', 'y_velocity'}
    super(UpwindAdvectionDiffusion, self).__init__(*args, **kwargs)

  def time_derivative(
      self, grid, concentration, x_velocity, y_velocity, concentration_x_edge_x,
      concentration_y_edge_y):
    """See base class."""
    c = concentration
    c_right = tensor_ops.roll_2d(c, (-1, 0))
    c_top = tensor_ops.roll_2d(c, (0, -1))

    D = self.diffusion_coefficient  # pylint: disable=invalid-name
    x_flux = (x_velocity * tf.where(x_velocity > 0, c, c_right)
              - D * concentration_x_edge_x)
    y_flux = (y_velocity * tf.where(y_velocity > 0, c, c_top)
              - D * concentration_y_edge_y)
    c_t = flux_to_time_derivative(x_flux, y_flux, grid.step)
    return {'concentration': c_t}


class UpwindAdvection(Advection):
  """Upwind finite-volume scheme for advection only."""

  DISCRETIZATION_NAME = 'upwind'
  METHOD = polynomials.Method.FINITE_VOLUME
  MONOTONIC = True

  def __init__(self, *args, **kwargs):
    self.key_definitions = {
        'concentration':
            StateDef('concentration', (), NO_DERIVATIVES, NO_OFFSET),
        'x_velocity': StateDef('velocity', (X,), NO_DERIVATIVES, X_PLUS_HALF),
        'y_velocity': StateDef('velocity', (Y,), NO_DERIVATIVES, Y_PLUS_HALF),
    }
    self.evolving_keys = {'concentration'}
    self.constant_keys = {'x_velocity', 'y_velocity'}
    super(UpwindAdvection, self).__init__(*args, **kwargs)

  def time_derivative(self, grid, concentration, x_velocity, y_velocity):
    """See base class."""
    c = concentration
    c_right = tensor_ops.roll_2d(c, (-1, 0))
    c_top = tensor_ops.roll_2d(c, (0, -1))

    x_flux = x_velocity * tf.where(x_velocity > 0, c, c_right)
    y_flux = y_velocity * tf.where(y_velocity > 0, c, c_top)
    c_t = flux_to_time_derivative(x_flux, y_flux, grid.step)
    return {'concentration': c_t}


def _minimum(*args):
  return functools.reduce(tf.minimum, args)


def _maximum(*args):
  return functools.reduce(tf.maximum, args)


class Limiter(enum.Enum):
  """Enum representing flux limiter options.

  For monotonic constraints on VanLeerAdvection.
  """
  LOCAL = 1
  GLOBAL = 2
  POSITIVE = 3
  NONE = 4


def _tendency_vanleer_1d(c, v, dx, dt, axis, c_x=None, diffusion_coefficient=0,
                         limiter=Limiter.LOCAL):
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
  if limiter is Limiter.LOCAL:
    # local bounds (from neighbouring values) for concentrations
    c_min = _minimum(c_left, c, c_right)
    c_max = _maximum(c_left, c, c_right)
    mismatch = tf.sign(delta_average) * _minimum(
        abs(delta_average), 2 * (c - c_min), 2 * (c_max - c))

  elif limiter is Limiter.GLOBAL:
    # pre-defined global bounds for concentraions, less diffusive than 'mono5'
    c_min = 0.0
    c_max = 1.0
    mismatch = tf.sign(delta_average) * _minimum(
        abs(delta_average),
        2 * tf.maximum(c - c_min, 0),
        2 * tf.maximum(c_max - c, 0))

  elif limiter is Limiter.POSITIVE:
    # only positive definite constraint (i.e. one-sided global limiter)
    mismatch = tf.sign(delta_average) * tf.minimum(
        abs(delta_average), 2 * c)

  elif limiter is Limiter.NONE:
    # vanilla second-order discretization, no monotonic constraints
    mismatch = delta_average

  else:
    raise ValueError('unimplemented limiter')

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
  METHOD = polynomials.Method.FINITE_VOLUME
  MONOTONIC = True

  def __init__(self, *args, **kwargs):
    self.key_definitions = {
        'concentration':
            StateDef('concentration', (), NO_DERIVATIVES, NO_OFFSET),
        'x_velocity': StateDef('velocity', (X,), NO_DERIVATIVES, X_PLUS_HALF),
        'y_velocity': StateDef('velocity', (Y,), NO_DERIVATIVES, Y_PLUS_HALF),
        'concentration_x_edge_x':
            StateDef('concentration', (), D_X, X_PLUS_HALF),
        'concentration_y_edge_y':
            StateDef('concentration', (), D_Y, Y_PLUS_HALF),
    }
    self.evolving_keys = {'concentration'}
    self.constant_keys = {'x_velocity', 'y_velocity'}
    super(VanLeerMono5AdvectionDiffusion, self).__init__(*args, **kwargs)

  def take_time_step(
      self, grid, concentration, x_velocity, y_velocity, concentration_x_edge_x,
      concentration_y_edge_y):
    """See base class."""
    dx = dy = grid.step
    dt = self.get_time_step(grid)
    D = self.diffusion_coefficient  # pylint: disable=invalid-name
    x_axis = -2
    y_axis = -1
    c = concentration
    tendency_x_then_y = _tendency_vanleer_1d(
        c + 0.5 * _tendency_vanleer_1d(
            c, x_velocity, dx, dt, x_axis, concentration_x_edge_x, D),
        y_velocity, dy, dt, y_axis, concentration_y_edge_y, D)
    tendency_y_then_x = _tendency_vanleer_1d(
        c + 0.5 * _tendency_vanleer_1d(
            c, y_velocity, dy, dt, y_axis, concentration_y_edge_y, D),
        x_velocity, dx, dt, x_axis, concentration_x_edge_x, D)
    c_next = tf.add_n([c, tendency_x_then_y, tendency_y_then_x])
    return {'concentration': c_next}


class VanLeerAdvection(Advection):
  """Second order upwind finite-volume scheme for advection only.

  Based on various flux limiters from
  Lin, S.-J., et al. (1994). "A class of the van Leer-type transport schemes
  and its application to the moisture transport in a general circulation
  model."
  """

  DISCRETIZATION_NAME = 'van_leer'
  METHOD = polynomials.Method.FINITE_VOLUME
  MONOTONIC = True

  def __init__(self, *args, **kwargs):
    self.key_definitions = {
        'concentration':
            StateDef('concentration', (), NO_DERIVATIVES, NO_OFFSET),
        'x_velocity': StateDef('velocity', (X,), NO_DERIVATIVES, X_PLUS_HALF),
        'y_velocity': StateDef('velocity', (Y,), NO_DERIVATIVES, Y_PLUS_HALF),
    }
    self.evolving_keys = {'concentration'}
    self.constant_keys = {'x_velocity', 'y_velocity'}

    limiter = kwargs.pop('limiter', Limiter.LOCAL)
    # catch invalid limiter early, not until time stepping
    if not isinstance(limiter, Limiter):
      raise ValueError('must use the Limiter enum')
    self.limiter = limiter

    super(VanLeerAdvection, self).__init__(*args, **kwargs)

  def take_time_step(self, grid, concentration, x_velocity, y_velocity):
    """See base class."""
    dx = dy = grid.step
    dt = self.get_time_step(grid)
    x_axis = -2
    y_axis = -1
    c = concentration

    _tendency_1d = functools.partial(_tendency_vanleer_1d, limiter=self.limiter)
    tendency_x_then_y = _tendency_1d(
        c + 0.5 * _tendency_1d(c, x_velocity, dx, dt, axis=x_axis),
        y_velocity, dy, dt, axis=y_axis)
    tendency_y_then_x = _tendency_1d(
        c + 0.5 * _tendency_1d(c, y_velocity, dy, dt, axis=y_axis),
        x_velocity, dx, dt, axis=x_axis)
    c_next = tf.add_n([c, tendency_x_then_y, tendency_y_then_x])
    return {'concentration': c_next}


def random_fourier_series(
    grid: grids.Grid,
    size: Shape = (),
    seed: int = None,
    max_periods: int = 4,
    power_law: float = -1.5,
    power_law_threshold: float = 1,
    normalize: bool = True,
    lower_bound: float = 1e-3,
    upper_bound: float = 1.0,
) -> np.ndarray:
  """Generates a distribution of truncated Fourier harmonics.

  Args:
    grid: grid on which initial conditions are evaluated.
    size: leading "batch" dimensions to include on output tensors.
    seed: random seed to use for random number generator.
    max_periods: maximum number of x/y periods for fourier terms.
    power_law: power law for the amplitude of fourier components.
    power_law_threshold: threshold wavenumber at which to start applying the
      the power law. Lower wave numbers will have a constant amplitude.
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

  ks = np.arange(-max_periods, max_periods + 1)
  k_x, k_y = [k.ravel() for k in np.meshgrid(ks, ks, indexing='ij')]
  k_mag = (k_x ** 2 + k_y ** 2) ** 0.5
  scale = np.where(k_mag >= power_law_threshold,
                   (k_mag / power_law_threshold) ** float(power_law),
                   1.0)

  x, y = grid.get_mesh()
  grid_shape = (1,) * len(size) + grid.shape
  # dimensions [..., x, y]
  relative_x = x.reshape(grid_shape) / grid.length_x
  relative_y = y.reshape(grid_shape) / grid.length_y

  # loop in ascending order of wavenumber: this reduces memory cost (compared
  # to allocating a single array for all wavenumers) and ensures that changing
  # max_periods doesn't change random number generation for lower order terms.
  distribution = np.zeros(size + grid.shape)
  for _, current_k_x, current_k_y, current_scale in sorted(
      zip(k_mag, k_x, k_y, scale), key=operator.itemgetter(0)):
    # dimensions [..., x, y]
    event_shape = size + (1, 1)
    amplitude = current_scale * random.random_sample(size=event_shape)
    random_cycles = random.random_sample(size=event_shape)
    cycles = (current_k_x * relative_x
              + current_k_y * relative_y
              + random_cycles)
    distribution += amplitude * np.sin(2 * np.pi * cycles)

  if normalize:
    distribution_min = distribution.min(axis=(-1, -2), keepdims=True)
    distribution_range = (
        distribution.max(axis=(-1, -2), keepdims=True) - distribution_min)
    scale = (upper_bound - lower_bound) / distribution_range
    distribution = scale * (distribution - distribution_min) + lower_bound

  return distribution


def _logit(p):
  p = np.clip(p, 1e-8, 1 - 1e-8)
  return np.log(p / (1 - p))


def _sigmoid(x):
  return 1 / (1 + np.exp(-x))


def binarize(
    p: np.ndarray, center: float = 0.5, slope: float = 1.0,
) -> np.ndarray:
  """Smoothly binarize a numpy array via logit and sigmoid transformations.

  Args:
    p: numpy array with values between 0 and 1.
    center: value in p at which to center the binarization.
    slope: slope of binarization. A slope of 1 means no binarization.

  Returns:
    A binarized function.
  """
  if slope <= 0:
    raise ValueError('slope must be positive')
  if not 0 < center < 1:
    raise ValueError('center must be between 0 and 1')
  return _sigmoid(slope * (_logit(p) - _logit(center)))


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
    num_terms: number of gaussians to include.
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
