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
"""Models evaluate spatial state derivatives.

Models encapsulate the machinery that provides all spatial state derivatives
to the governing equation. They can employ different techniques to produce
state derivatives, such as finite difference methods or neural networks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from typing import (
    Any, Dict, List, Optional, Mapping, Set, TypeVar, Tuple, Union,
)

import numpy as np
from pde_superresolution_2d import metadata_pb2
from pde_superresolution_2d.core import equations
from pde_superresolution_2d.core import geometry
from pde_superresolution_2d.core import grids
from pde_superresolution_2d.core import polynomials
from pde_superresolution_2d.core import readers
from pde_superresolution_2d.core import states
from pde_superresolution_2d.core import tensor_ops
import tensorflow as tf


nest = tf.contrib.framework.nest


T = TypeVar('T')


def sorted_values(x: Dict[Any, T]) -> List[T]:
  """Returns the sorted values of a dictionary."""
  return [x[k] for k in sorted(x)]


def stack_dict(state: Dict[Any, tf.Tensor]) -> tf.Tensor:
  """Stack a dict of tensors along its last axis."""
  return tf.stack(sorted_values(state), axis=-1)


class TimeStepModel(tf.keras.Model):
  """Model that predicts the state at the next time-step."""

  def __init__(
      self,
      equation: equations.Equation,
      grid: grids.Grid,
      num_time_steps: int = 1,
      name: str = 'time_step_model',
  ):
    """Initialize a time-step model."""
    super(TimeStepModel, self).__init__(name=name)

    if num_time_steps < 1:
      raise ValueError('must use at least one time step')

    self.equation = equation
    self.grid = grid
    self.num_time_steps = num_time_steps

    # used by keras
    self.output_names = sorted(equation.evolving_keys)

  def load_data(
      self,
      metadata: metadata_pb2.Dataset,
      prefix: states.Prefix = states.Prefix.EXACT,
  ) -> tf.data.Dataset:
    """Load data into a tf.data.Dataset for inferrence or training."""

    def replace_state_keys_with_names(state):
      return {k: state[equation.key_definitions[k].with_prefix(prefix)]
              for k in equation.base_keys}

    equation = readers.get_equation(metadata)
    grid = readers.get_output_grid(metadata)
    keys = [equation.key_definitions[k].with_prefix(prefix)
            for k in equation.base_keys]
    dataset = readers.initialize_dataset(metadata, [keys], [grid])
    dataset = dataset.map(replace_state_keys_with_names)
    return dataset

  def call(self, inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Predict the evolved state.

    Args:
      inputs: dict of tensors with dimensions [batch, x, y].

    Returns:
      labels: dict of tensors with dimensions [batch, time, x, y], giving the
        predicted state at steps [1, ..., self.num_time_steps].
    """
    constant_state = {k: v for k, v in inputs.items()
                      if k in self.equation.constant_keys}
    evolving_inputs = {k: v for k, v in inputs.items()
                       if k in self.equation.evolving_keys}

    def advance(evolving_state, _):
      state = dict(evolving_state)
      state.update(constant_state)
      return self.take_time_step(state)

    advanced = tf.scan(
        advance, tf.range(self.num_time_steps), initializer=evolving_inputs)
    advanced = tensor_ops.moveaxis(advanced, source=0, destination=1)
    return advanced

  def time_derivative(
      self, state: Mapping[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Compute the time derivative.

    Args:
      state: current state of the solution.

    Returns:
      Updated values for each non-constant term in the state.
    """
    raise NotImplementedError

  def take_time_step(
      self, state: Mapping[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Take a single time-step.

    Args:
      state: current state of the solution.

    Returns:
      Updated values for each non-constant term in the state.
    """
    raise NotImplementedError

  def to_proto(self) -> metadata_pb2.Model:
    """Creates a protocol buffer holding parameters of the model."""
    raise NotImplementedError


class SpatialDerivativeModel(TimeStepModel):
  """Model that predicts the next time-step implicitly via spatial derivatives.
  """

  def __init__(self, equation, grid, num_time_steps=1,
               name='spatial_derivative_model'):
    super(SpatialDerivativeModel, self).__init__(
        equation, grid, num_time_steps, name)

  def spatial_derivatives(
      self, state: Mapping[str, tf.Tensor],
  ) -> Dict[str, tf.Tensor]:
    """Predict all needed spatial derivatives."""
    raise NotImplementedError

  def time_derivative(
      self, state: Mapping[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """See base class."""
    inputs = self.spatial_derivatives(state)
    outputs = self.equation.time_derivative(self.grid, **inputs)
    return outputs

  def take_time_step(
      self, state: Mapping[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """See base class."""
    inputs = self.spatial_derivatives(state)
    outputs = self.equation.take_time_step(self.grid, **inputs)
    return outputs


class FiniteDifferenceModel(SpatialDerivativeModel):
  """Baseline model with fixed order finite-differences or finite-volumes.

  This model doesn't need to be trained.
  """

  def __init__(
      self, equation, grid, accuracy_order=1, num_time_steps=1,
      name='finite_difference_model',
  ):
    super(FiniteDifferenceModel, self).__init__(
        equation, grid, num_time_steps, name)

    self.accuracy_order = accuracy_order
    self.parents = {}  # type: Dict[str, str]
    self.coefficients = {}  # type: Dict[str, Optional[np.ndarray]]
    self.stencils = {}  # type: Dict[str, List[np.ndarray]]

    for key in self.equation.all_keys:
      parent = equation.find_base_key(key)

      key_def = equation.key_definitions[key]
      parent_def = equation.key_definitions[parent]

      stencils = []
      for parent_offset, argument_offset, derivative_order in zip(
          parent_def.offset, key_def.offset, key_def.derivative_orders):
        stencil = polynomials.regular_stencil_1d(
            abs(parent_offset - argument_offset),
            derivative_order,
            accuracy_order,
            grid.step)
        stencils.append(stencil)

      if all(stencil.size == 1 for stencil in stencils):
        # sentinel value indicating that we should just reuse the parent tensor
        # rather than extracting patches and applying coefficients
        coefficients = None
      else:
        coefficients_2d = polynomials.coefficients(
            stencils, equation.METHOD, key_def.derivative_orders,
            accuracy_order, grid.step)
        coefficients = tf.convert_to_tensor(coefficients_2d.ravel(), tf.float32)

      self.parents[key] = parent
      self.stencils[key] = stencils
      self.coefficients[key] = coefficients

  def spatial_derivatives(
      self, inputs: Mapping[str, tf.Tensor], request: Set[str] = None,
  ) -> Dict[str, tf.Tensor]:
    """See base class."""
    if request is None:
      request = self.equation.all_keys

    result = {}
    for key in request:
      coefficients = self.coefficients[key]

      source = inputs[self.parents[key]]
      if coefficients is None:
        result[key] = source
      else:
        sizes = [stencil.size for stencil in self.stencils[key]]

        key_def = self.equation.key_definitions[key]
        parent_def = self.equation.key_definitions[self.parents[key]]
        shifts = [k - p for p, k in zip(parent_def.offset, key_def.offset)]

        patches = tensor_ops.extract_patches_2d(source, sizes, shifts)
        result[key] = tf.tensordot(coefficients, patches, axes=[-1, -1])
        assert result[key].shape[-2:] == source.shape[-2:], (
            result[key], source)

    return result

  def to_proto(self) -> metadata_pb2.Model:
    params = dict(accuracy_order=self.accuracy_order)
    return metadata_pb2.Model(finite_difference=params)


def _round_down_to_odd(x):
  return x if x % 2 else x - 1


def _round_down_to_even(x):
  return x - 1 if x % 2 else x


def build_stencils(
    key: states.StateDefinition,
    parent: states.StateDefinition,
    max_stencil_size: int,
    grid_step: float
) -> List[np.ndarray]:
  """Create stencils for use with learned coefficients."""
  stencils = []

  for parent_offset, key_offset in zip(parent.offset, key.offset):

    if parent_offset == key_offset:
      size = _round_down_to_odd(max_stencil_size)
    else:
      size = _round_down_to_even(max_stencil_size)

    # examples:
    # stencil_size=5 -> [-2, -1, 0, 1, 2]
    # stencil_size=4 -> [-2, -1, 0, 1]
    int_range = np.arange(size) - size // 2

    stencil = grid_step * (0.5 * abs(key_offset - parent_offset) + int_range)

    stencils.append(stencil)

  # we should only be using zero-centered stencils
  if not all(np.allclose(stencil.sum(), 0) for stencil in stencils):
    raise ValueError('stencils are not zero-centered for {} -> {}: {}'
                     .format(parent, key, stencils))

  return stencils


ConstraintLayer = Union[
    polynomials.PolynomialAccuracy, polynomials.PolynomialBias]


class FixedCoefficientsLayer(tf.keras.layers.Layer):
  """Layer representing fixed learned coefficients for a single derivative."""

  def __init__(
      self,
      constraint_layer: ConstraintLayer,
      stencils: List[np.ndarray],
      shifts: List[int],
      input_key: Optional[str] = None,
  ):
    self.constraint_layer = constraint_layer
    self.stencils = stencils
    self.shifts = shifts
    self.input_key = input_key
    super(FixedCoefficientsLayer, self).__init__()

  def build(self, input_shape):
    shape = [self.constraint_layer.input_size]
    self.kernel = self.add_weight('kernel', shape=shape)

  def compute_output_shape(self, input_shape):
    return input_shape[:-1]

  def call(self, inputs):
    coefficients = self.constraint_layer(self.kernel)
    sizes = [stencil.size for stencil in self.stencils]
    patches = tensor_ops.extract_patches_2d(inputs, sizes, self.shifts)
    return tf.einsum('s,bxys->bxy', coefficients, patches)


class VaryingCoefficientsLayer(tf.keras.layers.Layer):
  """Layer representing varying coefficients for a single derivative."""

  def __init__(
      self,
      constraint_layer: ConstraintLayer,
      stencils: List[np.ndarray],
      shifts: List[int],
      input_key: Optional[str] = None,
  ):
    self.constraint_layer = constraint_layer
    self.stencils = stencils
    self.shifts = shifts
    self.input_key = input_key
    self.kernel_size = constraint_layer.input_size
    super(VaryingCoefficientsLayer, self).__init__(trainable=False)

  def compute_output_shape(self, input_shape):
    return input_shape[:-1]

  def call(self, inputs):
    (kernel, source) = inputs
    coefficients = self.constraint_layer(kernel)
    sizes = [stencil.size for stencil in self.stencils]
    patches = tensor_ops.extract_patches_2d(source, sizes, self.shifts)
    return tf.einsum('bxys,bxys->bxy', coefficients, patches)


def normalize_learned_and_fixed_keys(
    learned_keys: Optional[Set[str]],
    fixed_keys: Optional[Set[str]],
    equation: equations.Equation,
) -> Tuple[Set[str], Set[str]]:
  """Normalize learned and fixed equation inputs."""
  if learned_keys is None and fixed_keys is None:
    fixed_keys = equation.base_keys
    learned_keys = equation.derived_keys

  elif fixed_keys is None:
    learned_keys = set(learned_keys)
    fixed_keys = equation.all_keys - learned_keys

  elif learned_keys is None:
    fixed_keys = set(fixed_keys)
    learned_keys = equation.all_keys - fixed_keys

  else:
    learned_keys = set(learned_keys)
    fixed_keys = set(fixed_keys)

    if learned_keys.intersection(fixed_keys):
      raise ValueError('learned and fixed inputs must be disjoint sets: '
                       '{} vs {}'.format(learned_keys, fixed_keys))

    missing_inputs = equation.all_keys - learned_keys - fixed_keys
    if missing_inputs:
      raise ValueError(
          'inputs {} not inclued in learned or fixed inputs: {} vs {}'
          .format(missing_inputs, learned_keys, fixed_keys))

  return learned_keys, fixed_keys


def build_output_layers(
    equation, grid, learned_keys,
    stencil_size=5,
    initial_accuracy_order=1,
    constrained_accuracy_order=1,
    layer_cls=FixedCoefficientsLayer,
) -> Dict[str, ConstraintLayer]:
  """Build a map of output layers for spatial derivative models."""
  layers = {}
  for key in learned_keys:
    parent = equation.find_base_key(key)
    key_def = equation.key_definitions[key]
    parent_def = equation.key_definitions[parent]

    stencils = build_stencils(key_def, parent_def, stencil_size, grid.step)
    shifts = [k - p for p, k in zip(parent_def.offset, key_def.offset)]
    constraint_layer = polynomials.constraint_layer(
        stencils, equation.METHOD, key_def.derivative_orders[:2],
        constrained_accuracy_order, initial_accuracy_order, grid.step,
    )
    layers[key] = layer_cls(
        constraint_layer, stencils, shifts, input_key=parent)
  return layers


class LinearModel(SpatialDerivativeModel):
  """Learn constant linear filters for spatial derivatives."""

  def __init__(self, equation, grid, stencil_size=5, initial_accuracy_order=1,
               constrained_accuracy_order=1, learned_keys=None,
               fixed_keys=None, num_time_steps=1, name='linear_model'):
    super(LinearModel, self).__init__(
        equation, grid, num_time_steps, name)
    self.learned_keys, self.fixed_keys = (
        normalize_learned_and_fixed_keys(learned_keys, fixed_keys, equation))
    self.output_layers = build_output_layers(
        equation, grid, self.learned_keys, stencil_size, initial_accuracy_order,
        constrained_accuracy_order, layer_cls=FixedCoefficientsLayer)
    self.fd_model = FiniteDifferenceModel(
        equation, grid, initial_accuracy_order)

  def spatial_derivatives(self, inputs):
    """See base class."""
    result = {}

    for key in self.learned_keys:
      layer = self.output_layers[key]
      input_tensor = inputs[layer.input_key]
      result[key] = layer(input_tensor)

    if self.fixed_keys:
      result.update(
          self.fd_model.spatial_derivatives(inputs, self.fixed_keys)
      )

    return result


class Conv2DPeriodic(tf.keras.layers.Layer):
  """Conv2D layer with periodic boundary conditions."""

  def __init__(self, filters, kernel_size, **kwargs):
    # Let Conv2D handle argument normalization, e.g., kernel_size -> tuple
    self._layer = tf.keras.layers.Conv2D(
        filters, kernel_size, padding='valid', **kwargs)
    self.filters = self._layer.filters
    self.kernel_size = self._layer.kernel_size

    if any(size % 2 == 0 for size in self.kernel_size):
      raise ValueError('kernel size for conv2d is not odd: {}'
                       .format(self.kernel_size))

    super(Conv2DPeriodic, self).__init__()

  def build(self, input_shape):
    self._layer.build(input_shape)
    super(Conv2DPeriodic, self).build(input_shape)

  def compute_output_shape(self, input_shape):
    return input_shape[:-1] + (self.filters,)

  def call(self, inputs):
    padded = tensor_ops.pad_periodic_2d(inputs, self.kernel_size)
    result = self._layer(padded)
    assert result.shape[1:3] == inputs.shape[1:3], (result, inputs)
    return result


def conv2d_stack(num_outputs, num_layers=5, filters=32, kernel_size=5,
                 activation='relu', **kwargs):
  """Create a sequence of Conv2DPeriodic layers."""
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Lambda(stack_dict))
  for _ in range(num_layers - 1):
    layer = Conv2DPeriodic(
        filters, kernel_size, activation=activation, **kwargs)
    model.add(layer)
  model.add(Conv2DPeriodic(num_outputs, kernel_size, **kwargs))
  return model


class PseudoLinearModel(SpatialDerivativeModel):
  """Learn pseudo-linear filters for spatial derivatives."""

  def __init__(self, equation, grid, stencil_size=5, initial_accuracy_order=1,
               constrained_accuracy_order=1, learned_keys=None,
               fixed_keys=None, core_model_func=conv2d_stack,
               num_time_steps=1, geometric_transforms=None,
               predict_permutations=True, name='pseudo_linear_model',
               **kwargs):
    super(PseudoLinearModel, self).__init__(
        equation, grid, num_time_steps, name)

    self.learned_keys, self.fixed_keys = (
        normalize_learned_and_fixed_keys(learned_keys, fixed_keys, equation))
    self.output_layers = build_output_layers(
        equation, grid, self.learned_keys, stencil_size, initial_accuracy_order,
        constrained_accuracy_order, layer_cls=VaryingCoefficientsLayer)
    self.fd_model = FiniteDifferenceModel(
        equation, grid, initial_accuracy_order)

    if not predict_permutations:
      # NOTE(shoyer): this only makes sense if geometric_transforms includes
      # permutations. Otherwise you won't be predicting every needed tensor.
      modeled = set()
      for key in sorted(self.output_layers):
        value = equation.key_definitions[key]
        swapped = value.swap_xy()
        if swapped in modeled:
          del self.output_layers[key]
        modeled.add(value)

    if geometric_transforms is None:
      geometric_transforms = [geometry.Identity()]
    self.geometric_transforms = geometric_transforms

    num_outputs = sum(
        layer.kernel_size for layer in self.output_layers.values()
    )
    self.core_model = core_model_func(num_outputs, **kwargs)

  def _apply_model(self, inputs):
    net = self.core_model(inputs)

    size_splits = [
        self.output_layers[key].kernel_size for key in self.output_layers
    ]
    heads = tf.split(net, size_splits, axis=-1)

    result = {}
    for (key, layer), head in zip(self.output_layers.items(), heads):
      input_tensor = inputs[layer.input_key]
      result[key] = layer([head, input_tensor])
    return result

  def spatial_derivatives(self, inputs):
    """See base class."""

    # averaging over all possible orientations gives us a result that is
    # guaranteed to be rotation invariant
    result_list = collections.defaultdict(list)
    for transform in self.geometric_transforms:
      output = transform.inverse(self._apply_model(transform.forward(inputs)))
      for k, v in output.items():
        result_list[k].append(v)
    result = {k: tf.add_n(v) / len(v) if len(v) > 1 else v[0]
              for k, v in result_list.items()}

    if self.fixed_keys:
      result.update(
          self.fd_model.spatial_derivatives(inputs, self.fixed_keys)
      )

    return result


class NonlinearModel(SpatialDerivativeModel):
  """Learn spatial derivatives directly."""

  def __init__(self, equation, grid, core_model_func=conv2d_stack,
               learned_keys=None, fixed_keys=None, num_time_steps=1,
               finite_diff_accuracy_order=1, name='nonlinear_model', **kwargs):
    super(NonlinearModel, self).__init__(equation, grid, num_time_steps, name)
    self.learned_keys, self.fixed_keys = (
        normalize_learned_and_fixed_keys(learned_keys, fixed_keys, equation))
    self.core_model = core_model_func(
        num_outputs=len(self.learned_keys), **kwargs)
    self.fd_model = FiniteDifferenceModel(
        equation, grid, finite_diff_accuracy_order)

  def spatial_derivatives(
      self, inputs: Mapping[str, tf.Tensor],
  ) -> Dict[str, tf.Tensor]:
    """See base class."""
    net = self.core_model(inputs)
    heads = tf.unstack(net, axis=-1)
    result = dict(zip(self.learned_keys, heads))

    if self.fixed_keys:
      result.update(
          self.fd_model.spatial_derivatives(inputs, self.fixed_keys)
      )
    return result


class DirectModel(TimeStepModel):
  """Learn time-evolution directly, ignoring the equation."""

  def __init__(self, equation, grid, core_model_func=conv2d_stack,
               num_time_steps=1, finite_diff_accuracy_order=1,
               name='direct_model', **kwargs):
    super(DirectModel, self).__init__(equation, grid, num_time_steps, name)
    self.keys = equation.evolving_keys
    self.core_model = core_model_func(num_outputs=len(self.keys), **kwargs)

  def take_time_step(self, inputs):
    """See base class."""
    net = self.core_model(inputs)
    heads = tf.unstack(net, axis=-1)
    return dict(zip(self.keys, heads))
