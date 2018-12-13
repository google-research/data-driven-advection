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
"""Miscellaneous graph building utilities that use modules in the project."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from typing import Dict, Tuple

from pde_superresolution_2d.core import equations
from pde_superresolution_2d.core import grids
from pde_superresolution_2d.core import integrator
from pde_superresolution_2d.core import models
from pde_superresolution_2d.core import states


def time_integrate_graph(
    input_state: Dict[states.StateKey, tf.Tensor],
    equation: equations.Equation,
    model: models.Model,
    grid: grids.Grid,
    times: np.ndarray,
    unpack: bool = True
) -> Tuple[Dict[states.StateKey, tf.Tensor]]:
  """Generates computational graph that integrates equation in time.

  Creates an instance of an integrator that interfaces given equation, model and
  the grid to perform time integration.

  Args:
    input_state: State representing the initial configuration of the system
    equation: Equation used to perform time integration.
    model: Model evaluating time derivatives.
    grid: Grid specifying discretization of the system.
    times: Time slices at which the solution is needed. First value must
        correspond to the time at which the input_state is given.
    unpack: Boolean specifying whether to split integrated values into a tuple
        of individual states.

  Returns:
    A tuple of tensors, every entry in the tuple corresponds to a time slice
    specified in `times`.
  """
  equation_integrator = integrator.Integrator(equation, model, grid)
  if unpack:
    return equation_integrator.integrate_tf(input_state, times)
  else:
    return equation_integrator.integrate_tf_stacked(input_state, times)


def spatial_derivative_graph(
    input_state: Dict[states.StateKey, tf.Tensor],
    request: Tuple[states.StateKey],
    model: models.Model,
    grid: grids.Grid
) -> Dict[states.StateKey, tf.Tensor]:
  """Generates computational graph that evaluates spatial derivatives.

  Computes spatial derivatives of the equation that is time independent.
  Uses 0 as time input.

  Args:
    input_state: State representing the initial configuration of the system
    request: Tuple of statekeys to be computed.
    model: Model evaluating time derivatives.
    grid: Grid specifying discretization of the system.

  Returns:
    A tuple of states representing representing spatial derivatives.

  # TODO(dkochkov) Remove explicit time dependence when moved to state.
  """
  return model.state_derivatives(input_state, 0., grid, request)


def time_derivative_graph(
    input_state: Dict[states.StateKey, tf.Tensor],
    equation: equations.Equation,
    model: models.Model,
    grid: grids.Grid
) -> Dict[states.StateKey, tf.Tensor]:
  """Generates computational graph that evaluates time derivative.

  Computes time derivative of a time independent equation. Uses 0 as time input.

  Args:
    input_state: State representing the initial configuration of the system
    equation: Equation used to perform time integration.
    model: Model evaluating time derivatives.
    grid: Grid specifying discretization of the system.

  Returns:
    A state representing time derivative of the input_state.
  """
  request = equation.SPATIAL_DERIVATIVES_KEYS
  spatial_derivatives = model.state_derivatives(input_state, 0., grid, request)
  time_derivative = equation.time_derivative(
      input_state, 0., grid, spatial_derivatives)
  return time_derivative


def resample_state_graph(
    input_state: Dict[states.StateKey, tf.Tensor],
    high_res_grid: grids.Grid,
    low_res_grid: grids.Grid
) -> Dict[states.StateKey, tf.Tensor]:
  """Generates computational graph that downsamples the solution.

  Args:
    input_state: State representing the initial configuration of the system
    high_res_grid: Grid defining the high resolution.
    low_res_grid: Grid defining the target low resolution.

  Returns:
    A tensor representing the input state at lower resolution.

  Raises:
    ValueError: Grids sizes are not compatible for downsampling.
  """
  current_size_x, current_size_y = high_res_grid.get_shape()
  new_size_x, new_size_y = low_res_grid.get_shape()
  if current_size_x % new_size_x != 0 or current_size_y % new_size_y != 0:
    raise ValueError('grids are not compatible')
  x_factor = current_size_x // new_size_x
  y_factor = current_size_y // new_size_y
  single_cell_weight = 1 / (x_factor * y_factor)

  kernel_shape = [x_factor, y_factor, 1, 1]
  strides = [1, x_factor, y_factor, 1]
  kernel = tf.ones(shape=kernel_shape, dtype=tf.float64) * single_cell_weight
  resampled_state = {}
  for key in input_state.keys():
    resampled_field = tf.nn.conv2d(tf.expand_dims(input_state[key], axis=3),
                                   kernel, strides=strides, padding='VALID')
    resampled_state[key] = tf.squeeze(resampled_field, axis=3)
  return resampled_state
