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
"""Integrator module performs time integrations of a given equation.

Integrator mediates interactions between equation, model and the grid
to perform time integration.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from typing import Dict, Tuple

from pde_superresolution_2d import equations
from pde_superresolution_2d import grids
from pde_superresolution_2d import models
from pde_superresolution_2d import states


class Integrator(object):
  """Integrator class provides methods that numerically integrate PDEs.

  Integrator performs numerical integration in time. It serves as a mediator
  interfacing equation, model and grid classes to perform time evolution.
  Integrator class should be instantiated for every new combination
  of equation, model and grid (components can be used in multiple integrators).

  Usage example:
      times = np.linspace(0, 10, 100)
      equation = Equation(...)
      grid = Grid(...)
      model = Model(...)
      integrator = Integrator(equation, model, grid)

      state = equation.initial_random_state(...)
      integration_tensor = integrator.integrate_tf(state, times)
      with tf.Session() as sess:
        result = sess.run(integration_tensor)
  """

  def __init__(self, equation: equations.Equation, model: models.Model,
               grid: grids.Grid):
    """Initializes the integrator class.

    Args:
      equation: Equation to be integrated.
      model: Model used to compute spatial derivatives.
      grid: Grid on which the equation is solved.
    """
    self.equation = equation
    self.model = model
    self.grid = grid

  def _time_derivative_tf(self, state: Dict[states.StateKey, tf.Tensor],
                          t: float) -> Dict[states.StateKey, tf.Tensor]:
    """Returns time derivative of the current state at time t.

    A wrapper function of Equation.time_derivative() to interface it with
    tf.contrib.integrate.odeint_fixed.

    Args:
      state: State of the solution at time t.
      t: Time at which the derivative is evaluated.

    Returns:
      Time derivative of the state at time t.
    """
    request = self.equation.SPATIAL_DERIVATIVES_KEYS
    spatial_derivs = self.model.state_derivatives(state, t, self.grid, request)
    return self.equation.time_derivative(state, t, self.grid, spatial_derivs)

  def integrate_tf_stacked(
      self,
      state: Dict[states.StateKey, tf.Tensor],
      time_targets: np.ndarray,
  ) -> tf.Tensor:
    """Returns time evolved state with time_targets added as 0th dimension.

    Use with caution, as the result of this method violates an implicit contract
    of state tensor dimensions being [batch, x, y]. The use case should be
    limited to scenarios where batch dimension (==1) is squeezed after
    integration.

    Args:
      state: Starting value of the state.
      time_targets: Time values at which the solution is recorded.

    Returns:
      Time evolved states at times specified in time_targets. 0th dimension has
      the same size as time_targets.
    """

    def wrapper_func(y0: tf.Tensor, t: float) -> tf.Tensor:
      temp_state = self.equation.to_state(y0)
      state_time_derivative = self._time_derivative_tf(temp_state, t)
      time_derivative = self.equation.to_tensor(state_time_derivative)
      return time_derivative

    evolving_component = self.equation.to_tensor(state)
    solutions = tf.contrib.integrate.odeint_fixed(
        wrapper_func,
        evolving_component,
        t=time_targets,
        dt=self.equation.get_time_step(self.grid),
        method='midpoint'
    )
    return solutions

  def integrate_tf(
      self,
      state: Dict[states.StateKey, tf.Tensor],
      time_targets: np.ndarray,
  ) -> Tuple[Dict[states.StateKey, tf.Tensor]]:
    """Returns time evolved states at time_targets.

    Args:
      state: Starting value of the state.
      time_targets: Time values at which the solution is recorded.

    Returns:
      Time evolved states at times specified in `time_targets`. Has the same
      length as `time_targets`.
    """

    solutions = self.integrate_tf_stacked(state, time_targets)
    split_time_tensors = tf.split(solutions, len(time_targets), axis=0)
    split_time_states = tuple(self.equation.to_state(tf.squeeze(tensor, 0))
                              for tensor in split_time_tensors)
    return split_time_states
