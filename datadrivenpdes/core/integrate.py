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
"""Integrate models over time."""
from typing import Dict, Union

import numpy as np
from datadrivenpdes.core import models
from datadrivenpdes.core import tensor_ops
import tensorflow as tf

nest = tf.contrib.framework.nest
xla = tf.contrib.compiler.xla

KeyedTensors = Dict[str, tf.Tensor]

# Note: Python's type system allows supplying substituting integers for floats
ArrayLike = Union[np.ndarray, np.generic, float]


def _xla_decorator(func):
  def wrapper(*args):
    return xla.compile(func, args)
  return wrapper


def integrate_steps(
    model: models.TimeStepModel,
    state: KeyedTensors,
    steps: ArrayLike,
    initial_time: float = 0.0,
    axis: int = 0,
    xla_compile: bool = False,
) -> KeyedTensors:
  """Integrate some fixed number of time steps.

  Args:
    model: model to integrate.
    state: starting value of the state.
    steps: number of time steps at which the solution is saved.
    initial_time: initial time for time integration.
    axis: axis in result tensors along which the integrated solution is
      stacked.
    xla_compile: whether to compile with XLA or not.

  Returns:
    Time evolved states at the times specified in `times`. Each tensor has the
    same shape as the inputs, with an additional dimension inserted to store
    values at each requested time.
  """
  # TODO(shoyer): explicitly include time?
  del initial_time  # unused

  state = nest.map_structure(tf.convert_to_tensor, state)
  steps = tf.convert_to_tensor(steps, dtype=tf.int32)
  constant_state = {k: v for k, v in state.items()
                    if k in model.equation.constant_keys}
  evolving_state = {k: v for k, v in state.items()
                    if k in model.equation.evolving_keys}

  @tf.function
  def advance_until_saved_step(state, start_stop):
    """Integrate until the next step at which to save results."""
    start, stop = start_stop
    # can't use range() in a for loop with XLA:
    # https://github.com/tensorflow/tensorflow/issues/30182
    i = start
    while i < stop:
      state = model.take_time_step({**state, **constant_state})
      i += 1
    return state

  if xla_compile:
    advance_until_saved_step = _xla_decorator(advance_until_saved_step)

  starts = tf.concat([[0], steps[:-1]], axis=0)
  integrated = tf.scan(advance_until_saved_step, [starts, steps],
                       initializer=evolving_state)

  integrated_constants = nest.map_structure(
      lambda x: tf.broadcast_to(x, steps.shape.as_list() + x.shape.as_list()),
      constant_state)
  integrated.update(integrated_constants)

  return tensor_ops.moveaxis(integrated, 0, axis)


def integrate_times(
    model: models.TimeStepModel,
    state: KeyedTensors,
    times: ArrayLike,
    initial_time: float = 0.0,
    axis: int = 0,
    xla_compile: bool = False,
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
    xla_compile: whether to compile with XLA or not.

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

  return integrate_steps(model, state, steps, initial_time, axis, xla_compile)
