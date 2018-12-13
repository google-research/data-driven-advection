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

import apache_beam as beam
import numpy as np
import tensorflow as tf
from typing import Callable, Dict, List, Tuple

from pde_superresolution_2d.advection import equations as advection_equations
from pde_superresolution_2d.core import equations
from pde_superresolution_2d.core import graph_builders
from pde_superresolution_2d.core import grids
from pde_superresolution_2d.core import models
from pde_superresolution_2d.core import states
from pde_superresolution_2d.core import utils
from pde_superresolution_2d import metadata_pb2


class TensorFlowDoFn(beam.DoFn):
  """Class implementing a beam transformation described as tensorflow graph.

  This class serves as a wrapper to interface execution of TF graphs in beam.
  It uses the provided build_graph_function to define the graph and streams the
  input PCollection through the graph.inputs returning graph.outputs.
  The intent of this design is to avoid duplicate description of classes
  implementing similar "process" and "start_bundle" methods.
  """

  def __init__(self, build_graph_function: Callable):
    """Constructor of the class.

    Args:
      build_graph_function: Function that generates the graph defining the
          transformation on PCollection. Must return graph input and graph
          output tensors.
    """
    self.build_graph_function = build_graph_function

  def start_bundle(self):
    """Constructs the processing graph of the transformation."""
    self.graph = tf.Graph()
    with self.graph.as_default():
      self.inputs, self.outputs = self.build_graph_function()
    self.sess = tf.Session(graph=self.graph)

  def process(
      self,
      element: Tuple[Dict[states.StateKey, np.ndarray], ...]
  ) -> List[Tuple[Dict[states.StateKey, np.ndarray], ...]]:
    """Method that is applied to the input PCollection, graph execution."""
    result = []
    for state in element:
      feed_dict = {self.inputs[key]: state[key] for key in state.keys()}
      result.append(self.sess.run(self.outputs, feed_dict=feed_dict))
    return result


def make_components_metadata(
    data_states: Tuple[Tuple[states.StateKey, ...], ...],
    data_grids: Tuple[grids.Grid, ...]
) -> List[metadata_pb2.Dataset.DataComponent]:
  """Creates a list of data components protocol buffers.

  Args:
    data_states: Tuple of states.keys() that are present in the dataset.
    data_grids: Grid objects specifying the discretization (shape) of the data.

  Returns:
    List of DataComponent protocol buffers build from data_keys and data_grids.
  """
  data_components_list = []
  for data_state, grid in zip(data_states, data_grids):
    for state_key in data_state:
      grid_proto = grid.to_proto()
      state_proto = states.state_key_to_proto(state_key)
      component = metadata_pb2.Dataset.DataComponent(grid=grid_proto,
                                                     state_key=state_proto)  # pytype: disable=wrong-arg-types
      data_components_list.append(component)
  return data_components_list


class MeanVarianceCombineFn(beam.CombineFn):
  """Class implementing a beam transformation that combines dataset statistics.

  Implements methods required by beam.CombineFn interface to be used in a
  pipeline. Called during the dataset construction process to provide mean and
  variance of the primary inputs in the dataset.
  """

  def __init__(self, extract_values: Callable):
    """Constructs a class with given method for evaluation of mean and variance.

    Args:
      extract_values: Function that extracts primary input state as 1D ndarray.
    """
    self.extract_values = extract_values

  def create_accumulator(self) -> Tuple[float, float, int]:
    return 0.0, 0.0, 0

  def add_input(
      self,
      accumulator: Tuple[float, float, int],
      input: np.ndarray  # pylint: disable=redefined-builtin
  ) -> Tuple[float, float, int]:
    """Includes input components to the running mean and added_variance.

    Implementation below follows Welford's algorithm described in more detail at
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    in the Online algorithm section. `added_variance` corresponds to M2.

    Args:
      accumulator: Accumulator of mean, aggregated variance and count.
      input: New set of values to be added to the accumulator.

    Returns:
      Updated accumulator that includes values from the input.
    """
    mean, added_variance, count = accumulator
    input_values = self.extract_values(input)
    for value in np.nditer(input_values):
      count += 1
      new_mean = mean + (value - mean) / count
      new_added_variance = added_variance + (value - mean) * (value - new_mean)
      mean = new_mean
      added_variance = new_added_variance
    return new_mean, new_added_variance, count

  def merge_accumulators(
      self,
      accumulators: List[Tuple[float, float, int]]
  ) -> Tuple[float, float, int]:
    """Merges accumulators to estimate the combined mean and added_variance."""
    means, added_variances, counts = zip(*accumulators)
    total_count = np.sum(counts)
    added_mean = np.sum([means[i] * counts[i] for i in range(len(counts))])
    new_mean = added_mean / total_count

    new_added_variance = np.sum(
        [added_variances[i] + counts[i] * (means[i] - new_mean)**2
         for i in range(len(counts))])
    return new_mean, new_added_variance, total_count

  def extract_output(
      self,
      accumulator: Tuple[float, float, int]) -> Tuple[float, float]:
    """Extracts mean and variance."""
    mean, added_variance, count = accumulator
    if count > 1:
      return mean, added_variance / (count - 1)
    else:
      return mean, 0


def generate_metadata(
    base_equation: equations.Equation,
    base_model: models.Model,
    high_resolution_grid: grids.Grid,
    low_resolution_grid: grids.Grid,
    data_states: Tuple[Tuple[states.StateKey, ...], ...],
    data_grids: Tuple[grids.Grid, ...],
    max_time: float,
    num_time_slices: int,
    input_mean: float,
    input_variance: float,
    initialization_seed_offset: int,
    num_samples: int,
    records_path: str,
    num_shards: int
) -> metadata_pb2.Dataset:
  """Generates a protocol buffer holding datasets metadata.

  Args:
    base_equation: Equation used to generate the data.
    base_model: Model used to compute spatial derivatives.
    high_resolution_grid: High resolution grid on which the equation was solved.
    low_resolution_grid: Low resolution grid resulting from resampling.
    data_states: Tuple of states.keys() that appear in the dataset.
    data_grids: Corresponding grids on which the states are computed.
        Must have the same length as data_states.
    max_time: Time for how long the equation is integrated.
    num_time_slices: How many samples are drawn from a single integration run.
    input_mean: Mean value of the primary input.
    input_variance: Variance of the primary input.
    initialization_seed_offset: Integer seed offset of random initialization.
    num_samples: Integer number of initial conditions in the dataset.
    records_path: Path where the tfrecords will be saved.
    num_shards: Number of shards in the dataset.

  Returns:
    Protocol buffer holding metadata sufficient to reconstruct the system and
    verify / infer shapes and components of the data.

  Raises:
    ValueError: Number of data_states doesn't match the number of data_grids.
  """
  if len(data_states) != len(data_grids):
    raise ValueError('data_states must have the same length as data_grids.')
  file_names = [
      records_path +
      '-{0:0{width}}-of-{1:0{width}}'.format(shard, num_shards, width=5)
      for shard in range(num_shards)]
  high_res_grid_proto = high_resolution_grid.to_proto()
  low_res_grid_proto = low_resolution_grid.to_proto()
  equation_proto = base_equation.to_proto()
  model_proto = base_model.to_proto()
  data_components = utils.make_components_metadata(data_states, data_grids)
  metadata = metadata_pb2.Dataset(
      components=data_components,
      file_names=file_names,
      equation=equation_proto,
      model=model_proto,
      low_resolution_grid=low_res_grid_proto,
      high_resolution_grid=high_res_grid_proto,
      max_time=max_time,
      num_time_slices=num_time_slices,
      input_mean=input_mean,
      input_variance=input_variance,
      initialization_seed_offset=initialization_seed_offset,
      num_samples=num_samples,
      num_shards=num_shards
  )
  return metadata


def convert_to_tf_examples(
    dataset_states: Tuple[Dict[states.StateKey, np.ndarray], ...],
    dataset_grids: Tuple[grids.Grid, ...]
) -> bytes:
  """Generates serialized tensorflow example holding dataset_states.

  Generates tf.Example from the states in dataset_states. The key is generated
  from the StateKey and corresponding grid. Example is then serialized.

  Args:
    dataset_states: A tuple of states to be serialized in the example. Must have
        the same length as dataset_grids.
    dataset_grids: Grids describing the resolution of states in dataset_states.

  Returns:
    tf.Example serialized to a string.

  Raises:
    ValueError: The number of states doesn't equal to the number of grids.
  """
  def _floats_feature(value: np.ndarray) -> tf.train.Feature:
    """Helper function to convert list of floats to a tf.train.Feature."""
    return tf.train.Feature(
        float_list=tf.train.FloatList(value=value.reshape(-1)))

  feature = {}
  for state, grid in zip(dataset_states, dataset_grids):
    for key in state.keys():
      feature[utils.component_name(key, grid)] = _floats_feature(state[key])
  example = tf.train.Example(features=tf.train.Features(feature=feature))
  return example.SerializeToString()


def integrate_states_transform(
    equation: equations.Equation,
    model: models.Model,
    grid: grids.Grid,
    times: np.ndarray
) -> beam.DoFn:
  """Generates a beam do function that integrates PCollection of states.

  Args:
    equation: Equation to be integrated.
    model: Model evaluating time derivatives.
    grid: Grid specifying discretization of the system.
    times: Time slices at which the solution is needed. First value must
        correspond to the time at which the input_state is given.

  Returns:
    Class implementing beam function that performs corresponding time
    integration of the input PCollection.
  """
  def time_integration_graph(
  ) -> Tuple[Dict[states.StateKey, tf.placeholder],
             Tuple[Dict[states.StateKey, tf.Tensor], ...]]:
    """Function that generate computational graph for time integration.

    This method will be called inside of a beam.DoFn.start_bundle method. It
    implicitly passes the equation, model, grid, times parameters to the
    transformation.

    Returns:
      Tuple of input placeholder for the transformation and the tuple of output
      states corresponding to the solution at time slices specified in `times`.
    """
    placeholder_init = advection_equations.InitialConditionMethod.PLACEHOLDER
    input_state = equation.initial_state(placeholder_init, grid)
    return input_state, graph_builders.time_integrate_graph(
        input_state, equation, model, grid, times)

  integrate_do_fn = TensorFlowDoFn(time_integration_graph)
  return integrate_do_fn


class Dataset(object):
  """Defines interface for building datasets.

  Defines an interface that datasets must implement to be compatible with the
  dataset generation and pipeline.
  """

  def preprocess_states_transform(self) -> beam.DoFn:
    """Generates a beam transform that preprocesses a PCollection of states."""
    raise NotImplementedError

  def integrate_states_transform(self) -> beam.DoFn:
    """Generates a beam transform that integrates a PCollection of states."""
    raise NotImplementedError

  def compute_input_statistics_transform(self) -> beam.CombineFn:
    """Generates a beam transform that computes input statistics."""
    raise NotImplementedError

  def convert_to_tf_examples(
      self,
      input_data: Tuple[Dict[states.StateKey, np.ndarray]]
  ) -> bytes:
    """Converts input_data to tf.Examples and serializes it."""
    raise NotImplementedError

  def generate_metadata(
      self,
      input_mean: float,
      input_variance: float
  ) -> metadata_pb2.Dataset:
    """Generates dataset metadata protocol buffer."""
    raise NotImplementedError

  def save_metadata(
      self,
      output_path: str,
      mean_and_variance: Tuple[float, float]):
    """Saves dataset metadata to `output_path`."""
    mean, variance = mean_and_variance
    utils.save_proto(self.generate_metadata(mean, variance), output_path)


class TimeDerivativeDataset(Dataset):
  """Provides functions to build dataset of states and time derivatives.

  Implements methods for construction of the beam pipeline for preprocessing of
  integrated states into a dataset of coarse states and coarse time derivatives
  obtained by resampling the high resolution time derivatives. Also provides a
  method to write metadata.

  Attributes:
    DATASET_STATES_KEYS: StateKeys that appear in the dataset.
  """

  def __init__(
      self,
      equation: equations.Equation,
      model: models.Model,
      high_res_grid: grids.Grid,
      low_res_grid: grids.Grid,
      max_time: float,
      num_time_slices: int,
      initialization_seed_offset: int,
      num_samples: int,
      records_path: str,
      num_shards: int
  ):
    """Dataset constructor."""
    self.records_path = records_path
    self.num_shards = num_shards
    self.equation = equation
    self.model = model
    self.high_res_grid = high_res_grid
    self.low_res_grid = low_res_grid
    self.max_time = max_time
    self.num_time_slices = num_time_slices
    self.num_samples = num_samples
    self.initialization_seed_offset = initialization_seed_offset
    self.times = np.linspace(0, self.max_time, self.num_time_slices)

    state_keys = equation.STATE_KEYS
    time_derivative_state_keys = states.add_time_derivative_tuple(state_keys)
    self.dataset_states_keys = (
        state_keys,
        time_derivative_state_keys,
        states.add_prefix_tuple(states.BASELINE_PREFIX,
                                time_derivative_state_keys)
    )
    self.dataset_grids = (low_res_grid,) * 3  # state, 2 coarse time_derivatives

  def preprocess_states_transform(self) -> beam.DoFn:
    """Generates a beam function that preprocesses a PCollection of states."""

    def preprocess_graph():
      """Function that build a computational graph that performs preprocessing.

      This function is then passed to the BeamExecute class, where it constructs
      the processing graph in the start_bundle method. Instance parameters that
      this function uses must be pickle-able.

      Returns:
        Beam class that computes all of the component of the dataset from
        individual states.
      """
      placeholder_init = advection_equations.InitialConditionMethod.PLACEHOLDER
      state = self.equation.initial_state(placeholder_init, self.high_res_grid)

      time_derivative = graph_builders.time_derivative_graph(
          state, self.equation, self.model, self.high_res_grid)

      coarse_state = graph_builders.resample_state_graph(
          state, self.high_res_grid, self.low_res_grid)
      coarse_time_derivative = graph_builders.resample_state_graph(
          time_derivative, self.high_res_grid, self.low_res_grid)
      baseline_time_derivative = graph_builders.time_derivative_graph(
          coarse_state, self.equation, self.model, self.low_res_grid)
      baseline_time_derivative = states.add_prefix_keys(
          states.BASELINE_PREFIX, baseline_time_derivative)
      outputs = (coarse_state, coarse_time_derivative, baseline_time_derivative)
      return state, outputs

    process_do_fn = TensorFlowDoFn(preprocess_graph)
    return process_do_fn

  def integrate_states_transform(self) -> beam.DoFn:
    """Generates a beam transform that integrates a PCollection of states."""
    return integrate_states_transform(self.equation, self.model,
                                      self.high_res_grid, self.times)

  def compute_input_statistics_transform(self) -> beam.CombineFn:
    """Generates a beam transform that computes input statistics."""

    def extract_values(
        input_states: Tuple[Dict[states.StateKey, np.ndarray]]
    ) -> np.ndarray:
      coarse_state_index = 0
      return input_states[coarse_state_index][advection_equations.C].flatten()

    compute_input_statistics_fn = MeanVarianceCombineFn(extract_values)
    return compute_input_statistics_fn

  def convert_to_tf_examples(
      self,
      input_data: Tuple[Dict[states.StateKey, np.ndarray]]
  ) -> bytes:
    """Converts input_data to tf.Examples and serializes it."""
    return convert_to_tf_examples(input_data, self.dataset_grids)

  def generate_metadata(
      self,
      input_mean: float,
      input_variance: float
  ) -> metadata_pb2.Dataset:
    """Generates dataset metadata protocol buffer."""
    return generate_metadata(
        self.equation, self.model, self.high_res_grid, self.low_res_grid,
        self.dataset_states_keys, self.dataset_grids, self.max_time,
        self.num_time_slices, input_mean, input_variance,
        self.initialization_seed_offset, self.num_samples,
        self.records_path, self.num_shards
    )


class AllDerivativeDataset(Dataset):
  """Provides functions to build dataset of states and time derivatives.

  Implements methods for construction of the beam pipeline for preprocessing of
  integrated states into a dataset of coarse states and coarse time derivatives
  obtained by resampling the high resolution time derivatives. Also provides a
  method to write metadata.

  Attributes:
    DATASET_STATES_KEYS: StateKeys that appear in the dataset.
  """

  def __init__(
      self,
      equation: equations.Equation,
      model: models.Model,
      high_res_grid: grids.Grid,
      low_res_grid: grids.Grid,
      max_time: float,
      num_time_slices: int,
      initialization_seed_offset: int,
      num_samples: int,
      records_path: str,
      num_shards: int
  ):
    """Dataset constructor."""
    self.records_path = records_path
    self.num_shards = num_shards
    self.equation = equation
    self.model = model
    self.high_res_grid = high_res_grid
    self.low_res_grid = low_res_grid
    self.max_time = max_time
    self.num_time_slices = num_time_slices
    self.initialization_seed_offset = initialization_seed_offset
    self.num_samples = num_samples
    self.times = np.linspace(0, self.max_time, self.num_time_slices)

    state_keys = equation.STATE_KEYS
    time_derivative_state_keys = states.add_time_derivative_tuple(state_keys)
    spatial_derivatives = equation.SPATIAL_DERIVATIVES_KEYS
    self.dataset_states_keys = (
        state_keys,
        time_derivative_state_keys,
        states.add_prefix_tuple(states.BASELINE_PREFIX,
                                time_derivative_state_keys),
        spatial_derivatives,
        states.add_prefix_tuple(states.BASELINE_PREFIX, spatial_derivatives)
    )
    self.dataset_grids = (low_res_grid,) * 5

  def preprocess_states_transform(self) -> beam.DoFn:
    """Generates a beam function that preprocesses a PCollection of states."""

    def preprocess_graph():
      """Function that build a computational graph that performs preprocessing.

      This function is then passed to the BeamExecute class, where it constructs
      the processing graph in the start_bundle method. Instance parameters that
      this function uses must be pickle-able.

      Returns:
        Beam class that computes all of the component of the dataset from
        individual states.
      """
      placeholder_init = advection_equations.InitialConditionMethod.PLACEHOLDER
      state = self.equation.initial_state(placeholder_init, self.high_res_grid)
      derivatives_request = self.equation.SPATIAL_DERIVATIVES_KEYS

      spatial_derivatives = graph_builders.spatial_derivative_graph(
          state, derivatives_request, self.model, self.high_res_grid)

      time_derivative = graph_builders.time_derivative_graph(
          state, self.equation, self.model, self.high_res_grid)

      coarse_state = graph_builders.resample_state_graph(
          state, self.high_res_grid, self.low_res_grid)

      coarse_time_derivative = graph_builders.resample_state_graph(
          time_derivative, self.high_res_grid, self.low_res_grid)
      baseline_time_derivative = graph_builders.time_derivative_graph(
          coarse_state, self.equation, self.model, self.low_res_grid)
      baseline_time_derivative = states.add_prefix_keys(
          states.BASELINE_PREFIX, baseline_time_derivative)

      coarse_spatial_derivatives = graph_builders.resample_state_graph(
          spatial_derivatives, self.high_res_grid, self.low_res_grid)
      baseline_spatial_derivatives = graph_builders.spatial_derivative_graph(
          coarse_state, derivatives_request, self.model, self.low_res_grid)
      baseline_spatial_derivatives = states.add_prefix_keys(
          states.BASELINE_PREFIX, baseline_spatial_derivatives)
      outputs = (
          coarse_state,
          coarse_time_derivative,
          baseline_time_derivative,
          coarse_spatial_derivatives,
          baseline_spatial_derivatives
      )
      return state, outputs

    process_do_fn = TensorFlowDoFn(preprocess_graph)
    return process_do_fn

  def integrate_states_transform(self) -> beam.DoFn:
    """Generates a beam transform that integrates a PCollection of states."""
    return integrate_states_transform(self.equation, self.model,
                                      self.high_res_grid, self.times)

  def compute_input_statistics_transform(self) -> beam.CombineFn:
    """Generates a beam transform that computes input statistics."""

    def extract_values(input_states: Tuple[Dict[states.StateKey, np.ndarray]]):
      coarse_state_index = 0
      return input_states[coarse_state_index][advection_equations.C].flatten()

    compute_input_statistics_fn = MeanVarianceCombineFn(extract_values)
    return compute_input_statistics_fn

  def convert_to_tf_examples(
      self,
      input_data: Tuple[Dict[states.StateKey, np.ndarray]]
  ) -> bytes:
    """Converts input_data to tf.Examples and serializes it."""
    return convert_to_tf_examples(input_data, self.dataset_grids)

  def generate_metadata(
      self,
      input_mean: float,
      input_variance: float
  ) -> metadata_pb2.Dataset:
    """Generates dataset metadata protocol buffer."""
    return generate_metadata(
        self.equation, self.model, self.high_res_grid, self.low_res_grid,
        self.dataset_states_keys, self.dataset_grids, self.max_time,
        self.num_time_slices, input_mean, input_variance,
        self.initialization_seed_offset, self.num_samples,
        self.records_path, self.num_shards
    )


class HighResolutionDataset(Dataset):
  """Provides functions to build dataset of states at high resolution.

  Implements methods for construction of the beam pipeline for integration of
  states and saving them to TFRecords.
  """

  def __init__(
      self,
      equation: equations.Equation,
      model: models.Model,
      high_res_grid: grids.Grid,
      low_res_grid: grids.Grid,
      max_time: float,
      num_time_slices: int,
      initialization_seed_offset: int,
      num_samples: int,
      records_path: str,
      num_shards: int
  ):
    """Dataset constructor."""
    self.records_path = records_path
    self.num_shards = num_shards
    self.equation = equation
    self.model = model
    self.high_res_grid = high_res_grid
    self.low_res_grid = low_res_grid
    self.max_time = max_time
    self.num_time_slices = num_time_slices
    self.initialization_seed_offset = initialization_seed_offset
    self.num_samples = num_samples
    self.times = np.linspace(0, self.max_time, self.num_time_slices)

    state_keys = equation.STATE_KEYS
    self.dataset_states_keys = (state_keys,)
    self.dataset_grids = (high_res_grid,)

  def preprocess_states_transform(self) -> beam.DoFn:
    """Generates a beam DoFn that only unpacks the tuple of values."""

    class UnpackTransform(beam.DoFn):
      """Class implementing beam transform that unpacks integrated states."""

      def process(self, element):
        result = [(state,) for state in element]
        return result

    return UnpackTransform()

  def integrate_states_transform(self) -> beam.DoFn:
    """Generates a beam transform that integrates a PCollection of states."""
    return integrate_states_transform(self.equation, self.model,
                                      self.high_res_grid, self.times)

  def compute_input_statistics_transform(self) -> beam.CombineFn:
    """Generates a beam transform that computes input statistics."""

    def extract_values(input_states: Tuple[Dict[states.StateKey, np.ndarray]]):
      coarse_state_index = 0
      return input_states[coarse_state_index][advection_equations.C].flatten()

    compute_input_statistics_fn = MeanVarianceCombineFn(extract_values)
    return compute_input_statistics_fn

  def convert_to_tf_examples(
      self,
      input_data: Tuple[Dict[states.StateKey, np.ndarray]]
  ) -> bytes:
    """Converts input_data to tf.Examples and serializes it."""
    return convert_to_tf_examples(input_data, self.dataset_grids)

  def generate_metadata(
      self,
      input_mean: float,
      input_variance: float
  ) -> metadata_pb2.Dataset:
    """Generates dataset metadata protocol buffer."""
    return generate_metadata(
        self.equation, self.model, self.high_res_grid, self.low_res_grid,
        self.dataset_states_keys, self.dataset_grids, self.max_time,
        self.num_time_slices, input_mean, input_variance,
        self.initialization_seed_offset, self.num_samples,
        self.records_path, self.num_shards
    )


DATASET_TYPES = {
    'time_derivatives': TimeDerivativeDataset,
    'all_derivatives': AllDerivativeDataset,
    'high_resolution': HighResolutionDataset
}
