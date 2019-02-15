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
"""Functions for evaluation of various state quantities and saving data.

This module provides functions for evaluation of common quantities and
IO functions for writing dataset and metadata.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import operator

import numpy as np
from pde_superresolution_2d import metadata_pb2
from pde_superresolution_2d.core import equations
from pde_superresolution_2d.core import grids
from pde_superresolution_2d.core import integrate
from pde_superresolution_2d.core import models
from pde_superresolution_2d.core import states
from pde_superresolution_2d.core import utils
from pde_superresolution_2d.core import tensor_ops
import tensorflow as tf
from typing import Dict, List, Tuple, TypeVar

nest = tf.contrib.framework.nest


KeyedTensors = Dict[states.StateKey, tf.Tensor]
StateGridTensors = Dict[Tuple[states.StateKey, grids.Grid], tf.Tensor]
StatisticsDict = Dict[Tuple[states.StateKey, grids.Grid], Tuple[float, float]]


def convert_to_tf_example(state: StateGridTensors) -> bytes:
  """Generates serialized tensorflow example holding dataset_states.

  Generates tf.Example from the states in dataset_states. The key is generated
  from the StateKey and corresponding grid. Example is then serialized.

  Args:
    state: A dict of states to be serialized.

  Returns:
    tf.Example serialized to a string.

  Raises:
    ValueError: The number of states doesn't equal to the number of grids.
  """
  def _floats_feature(value: np.ndarray) -> tf.train.Feature:
    """Helper function to convert list of floats to a tf.train.Feature."""
    return tf.train.Feature(
        float_list=tf.train.FloatList(value=np.ravel(value)))

  feature = {}
  for (state_key, grid), tensor in state.items():
    feature[utils.component_name(state_key, grid)] = _floats_feature(tensor)
  example = tf.train.Example(features=tf.train.Features(feature=feature))
  return example.SerializeToString()


def unstack(state: KeyedTensors, num: int) -> List[KeyedTensors]:
  """Unstack keyed tensors into a list."""
  unstacked = tf.contrib.framework.nest.map_structure(
      lambda x: tf.unstack(x, num), state)
  result = [{} for _ in range(num)]
  for k, tensor_list in unstacked.items():
    for i, tensor in enumerate(tensor_list):
      result[i][k] = tensor
  return result


def integrate_state(
    state: KeyedTensors,
    model: models.TimeStepModel,
    times: np.ndarray,
    example_time_steps: np.ndarray,
) -> List[KeyedTensors]:
  """Integrate the given state to produce training examples.

  Each result in the output is a short "movie" showing the evolution of the
  state over a series of contiguous time steps, suitable for input as a training
  example.

  Args:
    state: initial conditions.
    model: Model evaluating time derivatives.
    times: time points at which to provide examples. First value must
        correspond to the time at which the state is given.
    example_time_steps: time-steps to take within each example.

  Returns:
    List of state dictionaries of tensors. Each tensor has dimensions
    [time, x, y], with the number of time steps equal to example_time_steps.
  """
  # TODO(shoyer): instead of integrating twice, we could integrate just once and
  # reshape the data. That would require a bit more book-keeping.
  integrated_once = integrate.integrate_times(model, state, times, axis=0)
  integrated_twice = tf.map_fn(
      lambda x: integrate.integrate_steps(model, x, example_time_steps),
      integrated_once,
  )
  return unstack(integrated_twice, len(times))


K = TypeVar('K')
V = TypeVar('V')


def merge(entries: List[Tuple[K, V]]) -> Dict[K, V]:
  """Merge a list of key-value pairs into a dict, raising if keys are reused."""
  result = {}
  for k, v in entries:
    if k in result:
      raise ValueError('duplicate entries with key {}'.format(k))
    result[k] = v
  return result


class Builder(object):
  """Defines interface for building datasets.

  Defines an interface that datasets must implement to be compatible with the
  dataset generation and pipeline.
  """

  def __init__(
      self,
      equation: equations.Equation,
      simulation_grid: grids.Grid,
      output_grid: grids.Grid,
      times: np.ndarray,
      example_time_steps: int,
  ):
    """Builder constructor."""
    self.equation = equation
    self.simulation_grid = simulation_grid
    self.output_grid = output_grid
    self.times = times
    self.example_time_steps = example_time_steps

  @property
  def model(self):
    return models.FiniteDifferenceModel(self.equation, self.simulation_grid)

  @property
  def coarse_model(self):
    return models.FiniteDifferenceModel(self.equation, self.output_grid)

  def integrate(self, state: KeyedTensors) -> List[KeyedTensors]:
    assert tf.executing_eagerly()
    grid_ratio = int(round(self.output_grid.step / self.simulation_grid.step))
    example_time_steps = grid_ratio * np.arange(self.example_time_steps)
    return integrate_state(state, self.model, self.times, example_time_steps)

  def postprocess(self, state: KeyedTensors) -> StateGridTensors:
    """Post-process integrated states."""
    raise NotImplementedError

  def convert_to_tf_example(self, state: StateGridTensors) -> bytes:
    """Converts input_data to serialized tf.Example protos."""
    assert tf.executing_eagerly()
    return convert_to_tf_example(state)

  def save_metadata(
      self,
      statistics_dict: StatisticsDict,
      records_path: str,
      metadata_path: str,
      num_shards: int,
      extra_fields: dict,  # pylint: disable=g-bare-generic
  ) -> None:
    """Saves dataset metadata to `path`."""
    # pytype disabled due to b/120915509
    # pytype: disable=wrong-arg-types
    data_components = []
    for (key, grid), (mean, variance) in statistics_dict.items():
      data_components.append(
          metadata_pb2.Dataset.DataComponent(
              grid=grid.to_proto(),
              state_key=key.to_proto(),
              mean=mean,
              variance=variance,
          )
      )

    # TODO(shoyer): use official APIs for this
    file_names = [
        records_path +
        '-{0:0{width}}-of-{1:0{width}}'.format(shard, num_shards, width=5)
        for shard in range(num_shards)]

    metadata = metadata_pb2.Dataset(
        components=data_components,
        file_names=file_names,
        equation=self.equation.to_proto(),
        model=self.model.to_proto(),
        output_grid=self.output_grid.to_proto(),
        simulation_grid=self.simulation_grid.to_proto(),
        times=self.times.tolist(),
        example_time_steps=self.example_time_steps,
        num_shards=num_shards,
        **extra_fields
    )
    # pytype: enable=wrong-arg-types
    utils.save_proto(metadata, metadata_path)


class TimeDerivatives(Builder):
  """Provides functions to build dataset of states and time derivatives."""

  def postprocess(self, state: KeyedTensors) -> StateGridTensors:
    """Resample states to low resolution."""
    result = []

    # solution
    coarse_state = tensor_ops.resample_mean(
        state, self.simulation_grid, self.output_grid)
    for k, v in coarse_state.items():
      result.append(((k.exact(), self.output_grid), v))

    # time derivatives
    time_derivative = self.model.time_derivative(state)

    coarse_time_derivative = tensor_ops.resample_mean(
        time_derivative, self.simulation_grid, self.output_grid)
    for k, v in coarse_time_derivative.items():
      result.append(((k.exact(), self.output_grid), v))

    baseline_time_derivative = self.coarse_model.time_derivative(coarse_state)
    for k, v in baseline_time_derivative.items():
      result.append(((k.baseline(), self.output_grid), v))

    return merge(result)


class AllDerivatives(Builder):
  """Provides functions to build dataset of states and all derivatives."""

  def postprocess(self, state: KeyedTensors) -> StateGridTensors:
    """Post-process integrated states."""
    result = []

    # solution
    coarse_state = tensor_ops.resample_mean(
        state, self.simulation_grid, self.output_grid)
    for k, v in coarse_state.items():
      result.append(((k.exact(), self.output_grid), v))

    # spatial derivatives
    spatial_derivatives = self.model.spatial_derivatives(state)
    coarse_spatial_derivatives = tensor_ops.resample_mean(
        spatial_derivatives, self.simulation_grid, self.output_grid)
    for k, v in coarse_spatial_derivatives.items():
      if k not in state:
        result.append(((k.exact(), self.output_grid), v))

    baseline_spatial_derivatives = self.coarse_model.spatial_derivatives(
        coarse_state)
    for k, v in baseline_spatial_derivatives.items():
      if k not in state:
        result.append(((k.baseline(), self.output_grid), v))

    # time derivatives
    time_derivative = self.model.time_derivative(state)

    coarse_time_derivative = tensor_ops.resample_mean(
        time_derivative, self.simulation_grid, self.output_grid)
    for k, v in coarse_time_derivative.items():
      result.append(((k.exact(), self.output_grid), v))

    baseline_time_derivative = self.coarse_model.time_derivative(coarse_state)
    for k, v in baseline_time_derivative.items():
      result.append(((k.baseline(), self.output_grid), v))

    return merge(result)


class HighResolution(Builder):
  """Provides functions to build dataset of states at high resolution."""

  def postprocess(self, state: KeyedTensors) -> StateGridTensors:
    return {(k.exact(), self.simulation_grid): v for k, v in state.items()}


class TimeEvolution(Builder):
  """Save the results of time-evolution."""

  def postprocess(self, state: KeyedTensors) -> StateGridTensors:
    """Resample states to low resolution."""
    result = []

    # exact solution
    coarse_state = tensor_ops.resample_mean(
        state, self.simulation_grid, self.output_grid)
    for k, v in coarse_state.items():
      result.append(((k.exact(), self.output_grid), v))

    # baseline solution
    initial_coarse_state = nest.map_structure(
        operator.itemgetter(0), coarse_state)
    integrated_baseline = integrate.integrate_steps(
        self.coarse_model, initial_coarse_state,
        np.arange(self.example_time_steps))
    for k, v in integrated_baseline.items():
      result.append(((k.baseline(), self.output_grid), v))

    return merge(result)


DATASET_TYPES = {
    'time_derivatives': TimeDerivatives,
    'all_derivatives': AllDerivatives,
    'high_resolution': HighResolution,
    'time_evolution': TimeEvolution,
}
