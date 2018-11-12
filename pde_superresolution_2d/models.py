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
"""Models evaluate spatial state derivatives.

Models encapsulate the machinery that provides all spatial state derivatives
to the governing equation. They can employ different techniques to produce
state derivatives, such as finite difference methods or neural networks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from typing import Callable, Dict, Tuple

from pde_superresolution_2d import grids
from pde_superresolution_2d import layers
from pde_superresolution_2d import metadata_pb2
from pde_superresolution_2d import states
from pde_superresolution_2d import utils
from pde_superresolution_2d import velocity_fields


class Model(object):
  """Base class for models.

  Implements method to compute state derivatives based on the current state.
  """

  def state_derivatives(
      self,
      state: Dict[states.StateKey, tf.Tensor],
      t: float,
      grid: grids.Grid,
      requested_derivatives: Tuple[states.StateKey, ...]
  ) -> Dict[states.StateKey, tf.Tensor]:
    """Returns a dictionary that holds requested state derivatives.

    Args:
      state: Current state of the solution.
      t: Time at which derivatives are evaluated.
      grid: Grid object holding discretization parameters.
      requested_derivatives: Enumeration of requested spatial state derivatives
          on the grid with requested offsets specified in the StateKeys.

    Returns:
      A dictionary that holds requested state derivatives.
    """
    raise NotImplementedError

  def to_proto(self) -> metadata_pb2.Model:
    """Creates a protocol buffer holding parameters of the model."""
    raise NotImplementedError

  @classmethod
  def from_proto(
      cls,
      proto: metadata_pb2.Model
  ) -> 'Model':
    """Creates a class instance from protocol buffer."""
    raise NotImplementedError

  @classmethod
  def from_hparams(
      cls,
      hparams: tf.contrib.training.HParams,
      train_metadata: metadata_pb2.Dataset
  ) -> 'Model':
    """Creates an instance of a class from a protocol buffer.

    Args:
      hparams: HParams holding architecture parameters.
      train_metadata: Training metadata, used to infer grid and velocities.

    Returns:
      Model object.

    Raises:
      ValueError: Model is not compatible with the equation in the dataset.
    """
    raise NotImplementedError


class RollFiniteDifferenceModel(Model):
  """Model implementing finite difference differentiation using roll method.

  This model uses tf.manip.roll to perform finite difference calculations. It
  assumes periodic boundary conditions. Currently supports evaluation of
  first and second derivatives on centered grid and edge terms by linear
  interpolation.
  """

  def state_derivatives(
      self,
      state: Dict[states.StateKey, tf.Tensor],
      t: float,
      grid: grids.Grid,
      requested_derivatives: Tuple[states.StateKey, ...]
  ) -> Dict[states.StateKey, tf.Tensor]:
    """Returns a dictionary that holds requested state derivatives.

    Args:
      state: Current state of the solution.
      t: Time at which derivatives are evaluated.
      grid: Grid object holding discretization parameters.
      requested_derivatives: StateKeys of requested spatial derivatives.

    Returns:
      A dictionary that holds requested state derivatives implemented by this
      model.

    Raises:
      ValueError: Requested spatial derivatives contain unsupported derivatives.
    """
    del t  # unused t
    spatial_derivatives = {}
    for key in requested_derivatives:
      parent_key = states.StateKey(key.name, (0, 0, 0), (0, 0))
      if parent_key not in state:
        raise ValueError('requested spatial derivatives are not supported')
      key_component = state[parent_key]
      if key.offset == (0, 0):
        if key.derivative_orders == (1, 0, 0):
          partial_x = (utils.roll_2d(key_component, (-1, 0)) -
                       utils.roll_2d(key_component, (1, 0))) / (2. * grid.step)
          spatial_derivatives[key] = partial_x
        if key.derivative_orders == (0, 1, 0):
          partial_y = (utils.roll_2d(key_component, (0, -1)) -
                       utils.roll_2d(key_component, (0, 1))) / (2. * grid.step)
          spatial_derivatives[key] = partial_y
        if key.derivative_orders == (2, 0, 0):
          partial_xx = (utils.roll_2d(key_component, (-1, 0)) +
                        utils.roll_2d(key_component, (1, 0)) -
                        2 * key_component) / (grid.step ** 2)
          spatial_derivatives[key] = partial_xx
        if key.derivative_orders == (0, 2, 0):
          partial_yy = (utils.roll_2d(key_component, (0, -1)) +
                        utils.roll_2d(key_component, (0, 1)) -
                        2 * key_component) / (grid.step ** 2)
          spatial_derivatives[key] = partial_yy
      if key.offset == (1, 0):
        if key.derivative_orders == (0, 0, 0):
          x_edge = (key_component +
                    utils.roll_2d(key_component, (-1, 0))) / 2
          spatial_derivatives[key] = x_edge
        elif key.derivative_orders == (1, 0, 0):
          partial_x_edge_x = (utils.roll_2d(key_component, (-1, 0)) -
                              key_component) / grid.step
          spatial_derivatives[key] = partial_x_edge_x
      if key.offset == (0, 1):
        if key.derivative_orders == (0, 0, 0):
          y_edge = (key_component +
                    utils.roll_2d(key_component, (0, -1))) / 2
          spatial_derivatives[key] = y_edge
        elif key.derivative_orders == (0, 1, 0):
          partial_y_edge_y = (utils.roll_2d(key_component, (0, -1)) -
                              key_component) / grid.step
          spatial_derivatives[key] = partial_y_edge_y
      if key in state and key not in spatial_derivatives:
        spatial_derivatives[key] = state[key]  # provide unchanged values

    if set(spatial_derivatives.keys()) != set(requested_derivatives):
      raise ValueError('requested spatial derivatives are not supported')
    return spatial_derivatives

  def to_proto(self) -> metadata_pb2.Model:
    """Creates a protocol buffer holding parameters of the model."""
    return metadata_pb2.Model(
        roll_finite_difference=metadata_pb2.RollFiniteDifferenceSolver())  # pytype: disable=wrong-arg-types

  @classmethod
  def from_proto(
      cls,
      proto: metadata_pb2.Model
  ) -> Model:
    """Creates an instance of a class from a protocol buffer."""
    del proto  # not used by RollFiniteDifferenceModel
    return cls()

  @classmethod
  def from_hparams(
      cls,
      hparams: tf.contrib.training.HParams,
      train_metadata: metadata_pb2.Dataset
  ) -> Model:
    """Creates an instance of a class from a protocol buffer.

    Args:
      hparams: HParams holding architecture parameters.
      train_metadata: Training metadata, used to infer grid and velocities.

    Returns:
      Model object.

    Raises:
      ValueError: Model is not compatible with the equation in the dataset.
    """
    del train_metadata  # unused by RollFiniteDifferenceModel
    del hparams  # unused by RollFiniteDifferenceModel
    return cls()


class StencilNet(Model):
  """Model that uses neural network to predict spatial derivatives of the state.

  This model estimates all requested spatial derivatives using periodic
  convolutions. It predicts coefficients in stencil basis based on all input
  states, which must contain at least states.C.
  """

  def __init__(self,
               num_layers: int,
               kernel_size: int,
               num_filters: int,
               stencil_size: int,
               input_shift: float,
               input_variance: float,
               activation: Callable = tf.nn.relu):
    """Class constructor."""
    self.num_layers = num_layers
    self.num_filters = num_filters
    self.kernel_size = kernel_size
    self.stencil_size = stencil_size
    self.input_shift = input_shift
    self.input_variance = input_variance
    self.activation = activation

  def _build_cnn_network(
      self,
      state: Dict[states.StateKey, tf.Tensor],
      component_key: states.StateKey
  ) -> tf.Tensor:
    """Builds convolutional network with periodic boundary conditions.

    Uses all components of the state as inputs for prediction.

    Args:
      state: State that is used as an input to the network.
      component_key: StateKey representing the component that is evaluated by
          the neural network. Used to retrieve weights for parameter sharing.

    Returns:
      Tensor representing the prediction of the model.

    Raises:
      ValueError: Concentration is not found in state components.
    """
    if states.C not in state.keys():
      raise ValueError('Concentration is not found in state components.')

    with tf.variable_scope('StencilNet'):
      with tf.variable_scope(utils.component_name(component_key),
                             reuse=tf.AUTO_REUSE):
        normalized_concentration = (
            (state[states.C] - self.input_shift) / self.input_variance)
        remaining_states = [v for k, v in state.items() if k != states.C]
        current_layer = tf.stack(
            [normalized_concentration] + remaining_states, axis=3)

        num_stencil_channels = self.stencil_size ** 2
        for i in range(self.num_layers - 1):
          with tf.variable_scope('conv2d_{}'.format(i)):
            current_layer = layers.nn_conv2d_periodic(
                current_layer, self.kernel_size, self.num_filters,
                use_bias=True, activation=self.activation)
        with tf.variable_scope('output_layer'):
          coefficient_predictions = layers.nn_conv2d_periodic(
              current_layer, self.kernel_size, num_stencil_channels,
              use_bias=True, activation=tf.identity)
        stencil_tensor = utils.generate_stencil_shift_tensors(
            state[states.C], self.stencil_size, self.stencil_size)
        result = tf.einsum('ijkl,ijkl->ijk',
                           coefficient_predictions, stencil_tensor)
    return result

  def state_derivatives(
      self,
      state: Dict[states.StateKey, tf.Tensor],
      t: float,
      grid: grids.Grid,
      requested_derivatives: Tuple[states.StateKey, ...]
  ) -> Dict[states.StateKey, tf.Tensor]:
    """Returns a dictionary that holds requested state derivatives.

    Args:
      state: Current state of the solution.
      t: Time at which derivatives are evaluated.
      grid: Grid object holding discretization parameters.
      requested_derivatives: StateKeys of requested spatial derivatives.

    Returns:
      A dictionary that holds requested state derivatives implemented by this
      model.
    """
    del t  # unused t
    spatial_derivatives = {}
    for key in requested_derivatives:
      spatial_derivatives[key] = self._build_cnn_network(state, key)
    return spatial_derivatives

  @classmethod
  def from_hparams(
      cls,
      hparams: tf.contrib.training.HParams,
      train_metadata: metadata_pb2.Dataset
  ) -> Model:
    """Creates an instance of a class from a protocol buffer.

    Args:
      hparams: HParams holding architecture parameters.
      train_metadata: Training metadata, used to infer grid and velocities.

    Returns:
      Model object.

    Raises:
      ValueError: Model is not compatible with the equation in the dataset.
    """
    num_layers = hparams.num_layers
    kernel_size = hparams.kernel_size
    num_filters = hparams.num_filters
    stencil_size = hparams.stencil_size
    mean = train_metadata.input_mean
    variance = train_metadata.input_variance
    return cls(num_layers, kernel_size, num_filters,
               stencil_size, mean, variance)


class StencilVNet(Model):
  """Model that uses neural network to predict spatial derivatives of the state.

  This model estimate all requested spatial derivatives using periodic
  convolutions. It predict coefficients in stencil basis based on the input
  and the underlying velocity_field, that must be provided at construction time.
  """

  def __init__(self,
               velocity_field: velocity_fields.VelocityField,
               grid: grids.Grid,
               num_layers: int,
               kernel_size: int,
               num_filters: int,
               stencil_size: int,
               input_shift: float,
               input_variance: float,
               activation: Callable = tf.nn.relu):
    """Class constructor."""
    self.velocity_field = velocity_field
    self.grid = grid
    self.num_layers = num_layers
    self.num_filters = num_filters
    self.kernel_size = kernel_size
    self.activation = activation
    self.stencil_size = stencil_size
    self.input_shift = input_shift
    self.input_variance = input_variance

  def _build_cnn_network(
      self,
      state: Dict[states.StateKey, tf.Tensor],
      component_key: states.StateKey
  ) -> tf.Tensor:
    """Builds convolutional network with periodic boundary conditions.

    Uses concentration component of the state as the input for prediction.

    Args:
      state: State that is used as an input to the network.
      component_key: StateKey representing the component that is evaluated by
          the neural network. Used to retrieve weights for parameter sharing.

    Returns:
      Tensor representing the prediction of the model.

    Raises:
      ValueError: Concentration is not found in state components.
    """
    if states.C not in state.keys():
      raise ValueError('Concentration is not found in state components.')

    with tf.variable_scope('StencilVNet'):
      with tf.variable_scope(utils.component_name(component_key),
                             reuse=tf.AUTO_REUSE):
        velocity_x = tf.expand_dims(tf.convert_to_tensor(
            self.velocity_field.get_velocity_x(0., self.grid)), axis=0)
        velocity_y = tf.expand_dims(tf.convert_to_tensor(
            self.velocity_field.get_velocity_y(0., self.grid)), axis=0)
        current_layer = state[states.C] - self.input_shift
        current_layer = current_layer / self.input_variance
        current_layer = tf.stack(
            [current_layer, velocity_x, velocity_y], axis=3)
        num_stencil_channels = self.stencil_size ** 2
        for i in range(self.num_layers - 1):
          with tf.variable_scope('conv2d_{}'.format(i)):
            current_layer = layers.nn_conv2d_periodic(
                current_layer, self.kernel_size, self.num_filters, True,
                self.activation)
        with tf.variable_scope('output_layer'):
          coefficient_predictions = layers.nn_conv2d_periodic(
              current_layer, self.kernel_size, num_stencil_channels, True,
              tf.identity)
        stencil_tensor = utils.generate_stencil_shift_tensors(
            state[states.C], self.stencil_size, self.stencil_size)
        result = tf.einsum('ijkl,ijkl->ijk',
                           coefficient_predictions, stencil_tensor)
    return result

  def state_derivatives(
      self,
      state: Dict[states.StateKey, tf.Tensor],
      t: float,
      grid: grids.Grid,
      requested_derivatives: Tuple[states.StateKey, ...]
  ) -> Dict[states.StateKey, tf.Tensor]:
    """Returns a dictionary that holds requested state derivatives.

    Args:
      state: Current state of the solution.
      t: Time at which derivatives are evaluated.
      grid: Grid object holding discretization parameters.
      requested_derivatives: StateKeys of requested spatial derivatives.

    Returns:
      A dictionary that holds requested state derivatives implemented by this
      model.

    Raises:
      ValueError: Requested spatial derivatives contain unsupported derivatives.
    """
    del t  # unused t
    spatial_derivatives = {}
    for key in requested_derivatives:
      spatial_derivatives[key] = self._build_cnn_network(state, key)
    return spatial_derivatives

  @classmethod
  def from_hparams(
      cls,
      hparams: tf.contrib.training.HParams,
      train_metadata: metadata_pb2.Dataset
  ) -> Model:
    """Creates an instance of a class from a protocol buffer.

    Args:
      hparams: HParams holding architecture parameters.
      train_metadata: Training metadata, used to infer grid and velocities.

    Returns:
      Model object.

    Raises:
      ValueError: Model is not compatible with the equation in the dataset.
    """
    equation_name = train_metadata.equation.WhichOneof('continuous_equation')
    equation_proto = getattr(train_metadata.equation, equation_name)
    if not hasattr(equation_proto, 'velocity_field'):
      raise ValueError('Model is not compatible with the dataset equation.')

    velocity_field_proto = equation_proto.velocity_field
    velocity_field = velocity_fields.velocity_field_from_proto(
        velocity_field_proto)

    grid = grids.grid_from_proto(train_metadata.low_resolution_grid)  # pytype: disable=wrong-arg-types
    mean = train_metadata.input_mean
    variance = train_metadata.input_variance

    num_layers = hparams.num_layers
    kernel_size = hparams.kernel_size
    num_filters = hparams.num_filters
    stencil_size = hparams.stencil_size

    return cls(velocity_field, grid, num_layers, kernel_size,
               num_filters, stencil_size, mean, variance)


MODEL_TYPES = {
    'roll_finite_difference': RollFiniteDifferenceModel,
    'stencil_net': StencilNet,
    'stencil_velocity_net': StencilVNet
}


def model_from_proto(proto: metadata_pb2.Model) -> Model:
  """Constructs a Model from the Model protocol buffer.

  Args:
    proto: Model protocol buffer encoding the Model.

  Returns:
    Model object.

  Raises:
    ValueError: Provided protocol buffer was not recognized, check proto names.
  """
  if proto.WhichOneof('model') in MODEL_TYPES.keys():
    return MODEL_TYPES[proto.WhichOneof('model')].from_proto(proto)
  raise ValueError('Model protocol buffer is not recognized')


def build_model(
    hparams: tf.contrib.training.HParams = None,
    train_metadata: metadata_pb2.Dataset = None,
    model_type_name: str = ''
) -> Model:
  """Constructs a trainable model with parameters from hparams and metadata.

  Args:
    hparams: HParams holding architecture parameters.
    train_metadata: Training metadata, used to infer grid and velocities.
    model_type_name: Name of the model to instantiate, overrides hparams.

  Returns:
    Corresponding model.

  Raises:
    ValueError: Model with provided name is not registered.
  """
  if hparams is None:
    model_type = model_type_name
  else:
    model_type = hparams.model_type

  if model_type not in MODEL_TYPES:
    raise ValueError('Model {} is not registered.'.format(model_type))
  return MODEL_TYPES[model_type].from_hparams(hparams, train_metadata)
