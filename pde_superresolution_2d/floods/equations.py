"""Implementation of inundation model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pde_superresolution_2d import metadata_pb2
from pde_superresolution_2d.advection import equations as advection_equations
from pde_superresolution_2d.advection import velocity_fields
from pde_superresolution_2d.core import equations
from pde_superresolution_2d.core import grids
from pde_superresolution_2d.core import states
from pde_superresolution_2d.core import tensor_ops
import tensorflow as tf
from typing import Any, Dict, Sequence, Text, Tuple, Type, TypeVar, Union

T = TypeVar('T')

G = 9.8
MANNING_COEFF_FLOODPLAIN = 0.05
MANNING_COEFF_RIVER = 0.02
EPSILON = 1e-4

H = states.StateKey('water_depth', (0, 0, 0), (0, 0))
H_EDGE_X = states.StateKey('water_depth', (0, 0, 0), (1, 0))
H_EDGE_Y = states.StateKey('water_depth', (0, 0, 0), (0, 1))
H_X = states.StateKey('water_depth', (1, 0, 0), (0, 0))
H_Y = states.StateKey('water_depth', (0, 1, 0), (0, 0))
H_T = states.StateKey('water_depth', (0, 0, 1), (0, 0))
H_X_EDGE_X = states.StateKey('water_depth', (1, 0, 0), (1, 0))
H_Y_EDGE_Y = states.StateKey('water_depth', (0, 1, 0), (0, 1))

QX = states.StateKey('flux_x', (0, 0, 0), (0, 0))
QY = states.StateKey('flux_y', (0, 0, 0), (0, 0))
QX_X = states.StateKey('flux_x', (1, 0, 0), (0, 0))
QY_Y = states.StateKey('flux_y', (0, 1, 0), (0, 0))
QX_T = states.StateKey('flux_x', (0, 0, 1), (0, 0))
QY_T = states.StateKey('flux_y', (0, 0, 1), (0, 0))

QX_EDGE_X = states.StateKey('flux_x', (0, 0, 0), (1, 0))
QY_EDGE_Y = states.StateKey('flux_y', (0, 0, 0), (0, 1))
QX_T_EDGE_X = states.StateKey('flux_x', (0, 0, 1), (1, 0))
QY_T_EDGE_Y = states.StateKey('flux_y', (0, 0, 1), (0, 1))

Z = states.StateKey('elevation', (0, 0, 0), (0, 0))
Z_X = states.StateKey('elevation', (1, 0, 0), (0, 0))
Z_Y = states.StateKey('elevation', (0, 1, 0), (0, 0))
Z_T = states.StateKey('elevation', (0, 0, 1), (0, 0))
Z_X_EDGE_X = states.StateKey('elevation', (1, 0, 0), (1, 0))
Z_Y_EDGE_Y = states.StateKey('elevation', (0, 1, 0), (0, 1))

N = states.StateKey('manning_squared', (0, 0, 0), (0, 0))

KeyedTensors = Dict[states.StateKey, tf.Tensor]
Shape = Union[int, Tuple[int]]

# numpy.random.RandomState uses uint32 for seeds
MAX_SEED_PLUS_ONE = 2**32


# This is a hack. The scale is a way to get around the fact that velocity_field
# wave numbers are required to be integers, and we're interested in features
# larger than 2*pi meters.
class ScalarField(object):

  def __init__(self, scale: float, *args, **kwargs):
    self.scale = scale
    self.velocity_field = velocity_fields.ConstantVelocityField.from_seed(
        *args, **kwargs)

  def evaluate(self, grid: grids.Grid):
    scaled_grid = grids.Grid(grid.size_x, grid.size_y, grid.step / self.scale)
    return self.velocity_field.get_velocity_x(0, scaled_grid)


def name_to_key(name: Text, keys: Sequence[states.StateKey]) -> states.StateKey:
  """Returns the first key matching a given name."""
  for k in keys:
    if k.name == name:
      return k
  raise ValueError('No key with name "%s" in %s' % (name, keys))


@equations.register_continuous_equation('saint_venant')
class SaintVenant(equations.Equation):
  """Base class Saint-Venant equations.

  This base class defines the state and common methods.

  Subclasses must implement the time_derivative() or take_time_step() method.
  """

  def __init__(self, time_step_factor: float = 0.01):
    self.time_step_factor = time_step_factor
    super(SaintVenant, self).__init__()

  def random_state(
      self,
      grid: grids.Grid,
      params: Dict[Text, Dict[Text, Any]] = None,
      batch_size: Shape = (),
      seed: int = None,
      dtype: Any = np.float32,
  ) -> Dict[states.StateKey, np.ndarray]:
    if params is None:
      params = {}
    params = dict(params)
    params.setdefault('water_depth', {})
    params.setdefault('elevation', {})

    if seed is not None:
      if 'seed' in params['water_depth'] or 'seed' in params['elevation']:
        raise ValueError('cannot set seed if water_depth seed or elevation '
                         'seed also provided.')

      random = np.random.RandomState(seed)
      params['water_depth']['seed'] = random.randint(MAX_SEED_PLUS_ONE)
      params['elevation']['seed'] = random.randint(MAX_SEED_PLUS_ONE)

    qx_key = name_to_key('flux_x', self.STATE_KEYS)
    qy_key = name_to_key('flux_y', self.STATE_KEYS)

    state = {}
    state[Z] = self.random_elevation_map(grid, batch_size,
                                         **params['elevation'])
    # TODO(geraschenko): manning_coeff will be variable in general.
    state[N] = MANNING_COEFF_FLOODPLAIN**2 * np.ones(state[Z].shape)
    # TODO(geraschenko): add boundary flux components. For now, we force
    # periodic boundary conditions because it's easy to avoid explosions that
    # way.

    state[H] = self.random_water_depth(grid, batch_size,
                                       **params['water_depth'])
    state[qx_key] = 0.0 * state[H]
    state[qy_key] = 0.0 * state[H]

    state = {k: v.astype(dtype) for k, v in state.items()}

    return state

  def random_water_depth(self,
                         grid: grids.Grid,
                         batch_size: Shape = (),
                         **kwargs: Any) -> np.ndarray:
    # TODO(geraschenko): remove dependence on advection_equations.
    # TODO(geraschenko): add non-gaussian initial states.
    return advection_equations.random_sum_of_gaussians(
        grid=grid, size=batch_size, **kwargs)

  def random_elevation_map(self,
                           grid: grids.Grid,
                           batch_size: Shape = (),
                           **kwargs: Any) -> np.ndarray:
    scale = kwargs.pop('scale', 1.0)
    elevation_map = ScalarField(scale, **kwargs)
    return np.tile(elevation_map.evaluate(grid), batch_size + (1, 1))

  def get_time_step(self, grid: grids.Grid) -> float:
    return self.time_step_factor * grid.step

  @classmethod
  def from_proto(cls: Type[T], proto: metadata_pb2.SaintVenantEquation) -> T:
    return cls(time_step_factor=proto.time_step_factor)

  def to_proto(self):
    return metadata_pb2.Equation(
        discretization=dict(
            name=self.DISCRETIZATION_NAME,
            method=self.METHOD,
            monotonic=self.MONOTONIC,
        ),
        saint_venant=dict(time_step_factor=self.time_step_factor),
    )


class FiniteDifferenceSaintVenant(SaintVenant):
  """A finite difference implementation of shallow water equations.

  This closely follows the FluxWithInertia class at
  http://google3/intelligence/flood_forecasting/inundation_model/flux_with_inertia.py?l=21&rcl=205509355
  and specifically equation (8) of
  http://catalytics.asia/wp-content/themes/catalytics/flood/Bates%20et%20al%202010%20new%20lisflood.pdf.
  """
  DISCRETIZATION_NAME = 'finite_difference'
  METHOD = metadata_pb2.Equation.Discretization.FINITE_DIFFERENCE
  MONOTONIC = False

  STATE_KEYS = (Z, N, H, QX, QY)
  CONSTANT_KEYS = (Z, N)
  INPUT_KEYS = (Z, N, H, QX, QY, H_X, H_Y, QX_X, QY_Y, Z_X, Z_Y)

  def time_derivative(
      self,
      state: KeyedTensors,
      inputs: KeyedTensors,
      grid: grids.Grid,
      time: float = 0.0,
  ) -> KeyedTensors:
    del time  # unused
    equations.check_keys(state, self.STATE_KEYS)
    equations.check_keys(inputs, self.INPUT_KEYS)
    state_time_derivative = {}

    def flux_derivative(dh_key, dz_key, q_key):
      """Computes d(flux)/dt using equation (8) from the paper."""
      water_depth = inputs[H]
      d_water_depth = inputs[dh_key]
      elevation = inputs[Z]
      d_elevation = inputs[dz_key]
      water_slope = d_water_depth + d_elevation
      flux = inputs[q_key]

      max_water_height = water_depth + elevation + tf.abs(
          0.5 * grid.step * water_slope)
      max_elevation = elevation + tf.abs(0.5 * grid.step * d_elevation)
      flow_depth = max_water_height - max_elevation
      # flow_depth is clipped below to EPSILON because we divide by it later.
      flow_depth = tf.clip_by_value(flow_depth, EPSILON, flow_depth)

      # Note: the paper subtly assumes that flux is positive. To allow for
      # negative flux, the Q^2 in the friction term must be replaced by Q * |Q|.
      q_dot = -G * (
          flow_depth * water_slope + flux * tf.abs(flux) * inputs[N] /
          (flow_depth**(7 / 3)))

      # Wherever there's no water (i.e. flow_depth is near zero), there should
      # be no flux, but because flow_depth appears in a denominator above, there
      # could be numerical explosions. In FluxWithInertia, the flux is
      # explicitly zeroed out in low-water regions
      # (http://google3/intelligence/flood_forecasting/inundation_model/flux_with_inertia.py?l=110&rcl=226158852).
      # Here we have to accomplish this by setting the derivative to something
      # that will zero out the flux.
      damping_q_dot = -0.8 * flux / self.get_time_step(grid)
      q_dot = tf.where(flow_depth < 10 * EPSILON, damping_q_dot, q_dot)
      return q_dot

    state_time_derivative[QX_T] = flux_derivative(H_X, Z_X, QX)
    state_time_derivative[QY_T] = flux_derivative(H_Y, Z_Y, QY)

    x_flux_x = inputs[QX_X]
    y_flux_y = inputs[QY_Y]

    # TODO(geraschenko): We probably want to do some kind of "flux limiting"
    # here to avoid negative amounts of water. Note that just adjusting H_T to
    # avoid negative water violates conservation of water.
    h_t = -(x_flux_x + y_flux_y)
    state_time_derivative[H_T] = h_t
    return state_time_derivative


class FiniteVolumeSaintVenant(SaintVenant):
  """A finite volume implementation of shallow water equations.

  This closely follows the FluxWithInertia class at
  http://google3/intelligence/flood_forecasting/inundation_model/flux_with_inertia.py?l=21&rcl=205509355
  and specifically equations (8) and (11) of
  http://catalytics.asia/wp-content/themes/catalytics/flood/Bates%20et%20al%202010%20new%20lisflood.pdf
  for time_derivative and take_time_step, respectively.
  """
  DISCRETIZATION_NAME = 'finite_volume'
  METHOD = metadata_pb2.Equation.Discretization.FINITE_VOLUME
  MONOTONIC = False

  STATE_KEYS = (Z, N, H, QX_EDGE_X, QY_EDGE_Y)
  CONSTANT_KEYS = (Z, N)
  INPUT_KEYS = (Z, N, H, QX_EDGE_X, QY_EDGE_Y, H_EDGE_X, H_EDGE_Y, H_X_EDGE_X,
                H_Y_EDGE_Y, Z_X_EDGE_X, Z_Y_EDGE_Y)

  def __init__(self,
               use_implicit_step_scheme: bool = True,
               alpha: float = 0.6,
               **kwargs):
    self.use_implicit_step_scheme = use_implicit_step_scheme
    self.alpha = alpha
    super(FiniteVolumeSaintVenant, self).__init__(**kwargs)

  def time_derivative(self,
                      state: KeyedTensors,
                      inputs: KeyedTensors,
                      grid: grids.Grid,
                      time: float = 0.0) -> KeyedTensors:
    del time  # unused
    equations.check_keys(state, self.STATE_KEYS)
    equations.check_keys(inputs, self.INPUT_KEYS)
    state_time_derivative = {}

    def flux_derivative(h_key, dh_key, dz_key, q_key):
      """Computes d(flux)/dt using equation (8) from the paper."""
      water_depth = inputs[h_key]
      d_water_depth = inputs[dh_key]
      elevation = inputs[Z]
      elevation = 0.5 * (
          elevation + tensor_ops.roll_2d(elevation, h_key.offset))
      d_elevation = inputs[dz_key]
      water_slope = d_water_depth + d_elevation
      flux = inputs[q_key]

      max_water_height = water_depth + elevation + tf.abs(
          0.5 * grid.step * water_slope)
      max_elevation = elevation + tf.abs(0.5 * grid.step * d_elevation)
      flow_depth = max_water_height - max_elevation
      # flow_depth is clipped below to EPSILON because we divide by it later.
      flow_depth = tf.clip_by_value(flow_depth, EPSILON, flow_depth)

      # Note: the paper subtly assumes that flux is positive. To allow for
      # negative flux, the Q^2 in the friction term must be replaced by Q * |Q|.
      q_dot = -G * (
          flow_depth * water_slope + flux * tf.abs(flux) * inputs[N] /
          (flow_depth**(7 / 3)))

      # Wherever there's no water (i.e. flow_depth is near zero), there should
      # be no flux, but because flow_depth appears in a denominator above, there
      # could be numerical explosions. In FluxWithInertia, the flux is
      # explicitly zeroed out in low-water regions
      # (http://google3/intelligence/flood_forecasting/inundation_model/flux_with_inertia.py?l=110&rcl=226158852).
      # Here we have to accomplish this by setting the derivative to something
      # that will zero out the flux.
      damping_q_dot = -0.8 * flux / self.get_time_step(grid)
      q_dot = tf.where(flow_depth < 10 * EPSILON, damping_q_dot, q_dot)
      return q_dot

    state_time_derivative[QX_T_EDGE_X] = flux_derivative(
        H_EDGE_X, H_X_EDGE_X, Z_X_EDGE_X, QX_EDGE_X)
    state_time_derivative[QY_T_EDGE_Y] = flux_derivative(
        H_EDGE_Y, H_Y_EDGE_Y, Z_Y_EDGE_Y, QY_EDGE_Y)

    # TODO(geraschenko): We probably want to do some kind of "flux limiting"
    # here to avoid negative amounts of water. Note that just adjusting H_T to
    # avoid negative water violates conservation of water.
    x_flux_x = inputs[QX_EDGE_X] - tensor_ops.roll_2d(inputs[QX_EDGE_X], (1, 0))
    y_flux_y = inputs[QY_EDGE_Y] - tensor_ops.roll_2d(inputs[QY_EDGE_Y], (0, 1))
    h_t = -(x_flux_x + y_flux_y)
    state_time_derivative[H_T] = h_t
    return state_time_derivative

  def max_delta_t(self, state: KeyedTensors, grid: grids.Grid) -> float:
    """Determines maximum time step based on equation (14) of the paper."""
    # TODO(geraschenko): Use this method once integration supports dynamic time
    # stepping.
    max_depth = tf.reduce_max(state[H])
    delta_t = self.alpha * grid.step / (EPSILON + G * max_depth)**0.5
    return tf.clip_by_value(delta_t, 0.01, 24 * 60 * 60)

  def take_time_step(
      self,
      state: KeyedTensors,
      inputs: KeyedTensors,
      grid: grids.Grid,
      time: float = 0.0,
  ) -> KeyedTensors:
    if not self.use_implicit_step_scheme:
      return super(FiniteVolumeSaintVenant, self).take_time_step(
          state, inputs, grid, time)

    del time  # unused
    equations.check_keys(state, self.STATE_KEYS)
    equations.check_keys(inputs, self.INPUT_KEYS)
    evolved_state = {}

    dt = self.get_time_step(grid)

    def evolved_flux(h_key, dh_key, dz_key, q_key):
      """Computes update to flux using equation (11) from the paper."""
      water_depth = inputs[h_key]
      d_water_depth = inputs[dh_key]
      elevation = state[Z]
      elevation = 0.5 * (
          elevation + tensor_ops.roll_2d(elevation, -np.array(h_key.offset)))
      d_elevation = inputs[dz_key]
      water_slope = d_water_depth + d_elevation
      flux = state[q_key]

      max_water_height = water_depth + elevation + tf.abs(
          0.5 * grid.step * water_slope)
      max_elevation = elevation + tf.abs(0.5 * grid.step * d_elevation)
      flow_depth = max_water_height - max_elevation

      # flow_depth is clipped below to EPSILON because we divide by it later.
      flow_depth = tf.clip_by_value(flow_depth, EPSILON, flow_depth)

      # Note: the paper subtly assumes that flux is positive. To allow for
      # negative flux, we must take the absolute value of the flux in the
      # denominator.
      evolved_flux = (flux - G * flow_depth * water_slope * dt) / (
          1 + dt * G * state[N] * tf.abs(flux) / (flow_depth**(7 / 3)))
      # TODO(geraschenko): Is this clipping necessary? It's done in
      # FluxWithInertia
      # (http://google3/intelligence/flood_forecasting/inundation_model/flux_with_inertia.py?l=110&rcl=226158852)
      # but when flow_depth is small, the denominator above is large, forcing
      # the flux to zero anyway. Perhaps it's to account for the fact that
      # flow_depth is bounded *below* by EPSILON?
      evolved_flux = tf.where(flow_depth < 2 * EPSILON,
                              tf.zeros_like(evolved_flux), evolved_flux)
      return evolved_flux

    evolved_state[QX_EDGE_X] = evolved_flux(H_EDGE_X, H_X_EDGE_X, Z_X_EDGE_X,
                                            QX_EDGE_X)
    evolved_state[QY_EDGE_Y] = evolved_flux(H_EDGE_Y, H_Y_EDGE_Y, Z_Y_EDGE_Y,
                                            QY_EDGE_Y)

    # TODO(geraschenko): We probably want to do some kind of "flux limiting"
    # here to avoid negative amounts of water. Note that just adjusting H to
    # avoid negative water violates conservation of water.
    x_flux = evolved_state[QX_EDGE_X]
    y_flux = evolved_state[QY_EDGE_Y]
    x_flux_x = x_flux - tensor_ops.roll_2d(x_flux, (1, 0))
    y_flux_y = y_flux - tensor_ops.roll_2d(y_flux, (0, 1))
    h_t = -(x_flux_x + y_flux_y)
    evolved_state[H] = state[H] + h_t * dt

    return evolved_state
