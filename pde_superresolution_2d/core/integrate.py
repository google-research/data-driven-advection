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
"""Integrate models over time."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pde_superresolution_2d.core import models
from pde_superresolution_2d.core import states
from pde_superresolution_2d.core import tensor_ops
import tensorflow as tf
from typing import Dict, Union

nest = tf.contrib.framework.nest


KeyedTensors = Dict[states.StateKey, tf.Tensor]

# Note: Python's type system allows supplying substituting integers for floats
ArrayLike = Union[np.ndarray, np.generic, float]


def integrate_steps(
    model: models.TimeStepModel,
    state: KeyedTensors,
    steps: ArrayLike,
    initial_time: float = 0.0,
    axis: int = 0,
) -> KeyedTensors:
  """Integrate some fixed number of time steps.

  Args:
    model: model to integrate.
    state: starting value of the state.
    steps: number of time steps at which the solution is saved.
    initial_time: initial time for time integration.
    axis: axis in result tensors along which the integrated solution is
      stacked.

  Returns:
    Time evolved states at the times specified in `times`. Each tensor has the
    same shape as the inputs, with an additional dimension inserted to store
    values at each requested time.
  """
  state = nest.map_structure(tf.convert_to_tensor, state)
  constant_state = {k: v for k, v in state.items()
                    if k in model.equation.CONSTANT_KEYS}
  evolving_state = {k: v for k, v in state.items()
                    if k not in model.equation.CONSTANT_KEYS}

  def advance_one_step(evolving_state, time):
    del time  # unused
    inputs = dict(evolving_state)
    inputs.update(constant_state)
    outputs = model.take_time_step(inputs)
    return outputs

  def advance_until_saved_step(evolving_state, start_stop):
    dt = model.equation.get_time_step(model.grid)
    steps = tf.range(*start_stop)
    times = dt * tf.cast(steps, tf.float32) + initial_time
    result = tf.foldl(advance_one_step, times, initializer=evolving_state)
    return result

  starts = np.concatenate([[0], steps[:-1]])
  integrated = tf.scan(advance_until_saved_step, [starts, steps],
                       initializer=evolving_state)

  integrated_constants = nest.map_structure(
      lambda x: tf.broadcast_to(x, [len(steps)] + x.shape.as_list()),
      constant_state)
  integrated.update(integrated_constants)

  return tensor_ops.moveaxis(integrated, 0, axis)


def integrate_times(
    model: models.TimeStepModel,
    state: KeyedTensors,
    times: ArrayLike,
    initial_time: float = 0.0,
    axis: int = 0,
) -> KeyedTensors:
  """Returns time evolved states at the requested times.

  TODO(shoyer): consider adding optional interpolation. Currently we require
  that the requested times are *exact* multiples of the time step.

  Args:
    model: model to integrate.
    state: starting value of the state.
    times: time values at which the integrated solution is recorded.
    initial_time: initial time for time integration.
    axis: axis in result tensors along which the integrated solution is
      stacked.

  Returns:
    Time evolved states at the times specified in `times`. Each tensor has the
    same shape as the inputs, with an additional dimension inserted to store
    values at each requested time.
  """
  dt = model.equation.get_time_step(model.grid)
  approx_steps = (times - initial_time) / dt
  steps = np.around(approx_steps).astype(int)

  if not np.allclose(approx_steps, steps, atol=1e-8):
    raise ValueError('evaluation times {} are not an integer multiple of the '
                     'time step {}: {}'.format(times, dt, approx_steps))

  return integrate_steps(model, state, steps, initial_time, axis)
