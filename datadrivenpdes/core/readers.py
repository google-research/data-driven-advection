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
"""DatasetReader provides interface for retrieval of states from the datasets.

This module is a wrapper around tf.data.TFRecordDataset that provides methods
for parsing tfrecords directly into states and state derivatives.
"""
import json
from typing import Any, Dict, Iterable, Mapping, List, Sequence, Tuple

from datadrivenpdes.core import equations
from datadrivenpdes.core import grids
from datadrivenpdes.core import states
from datadrivenpdes.core import utils
import tensorflow as tf
from tensorflow.io import gfile


def initialize_dataset(
    metadata: Mapping[str, Any],
    requested_data_keys: Sequence[Sequence[states.StateDefinition]],
    requested_data_grids: Sequence[grids.Grid],
) -> tf.data.Dataset:
  """Returns a tf.data.Dataset, setup to provide requested states.

  Args:
    metadata: metadata dict created by builders.Builders.save_metadata().
    requested_data_keys: State keys of the requested data.
    requested_data_grids: Grids corresponding to requested_data_keys. Must
        have the same length as requested_data_keys.

  Returns:
    TFRecordDataset setup to generate requested states from the dataset.
  """
  train_files = metadata['file_names']
  train_files = [str(train_file) for train_file in train_files]
  data_keys = data_component_keys(metadata['components'])
  features = _generate_features(data_keys, metadata['example_num_time_steps'])
  _assert_compatible(requested_data_keys, requested_data_grids, features)
  train_dataset = tf.data.TFRecordDataset(
      train_files,
      buffer_size=int(1e7),
      num_parallel_reads=16,
  )

  def parse_function(example_proto):
    """Parsing function that converts example proto to a tuple of states."""
    parsed_features = tf.io.parse_single_example(example_proto, features)
    # TODO(dkochkov) Consider switching logic to use parse_example.
    output_states = []
    for state_keys, grid in zip(requested_data_keys, requested_data_grids):
      state = {state_key: parsed_features[utils.component_name(state_key, grid)]
               for state_key in state_keys}
      output_states.append(state)
    return output_states

  train_dataset = train_dataset.map(parse_function)
  return train_dataset


def load_metadata(metadata_path: str) -> Dict[str, Any]:
  """Read saved configuration metadata from a file on disk."""
  with gfile.GFile(metadata_path) as f:
    return json.loads(f.read())


def data_component_keys(
    components: Iterable[Mapping[str, Any]]
) -> List[Tuple[states.StateDefinition, grids.Grid]]:
  """Parses data components from the metadata.

  Args:
    components: Dataset protocol buffer holding the metadata for the dataset.

  Returns:
    Components in the dataset organized as tuple of StateDefinition, Grid pairs.
  """
  data_components = []
  for component in components:
    state = states.StateDefinition.from_config(component['state_definition'])
    grid = grids.Grid.from_config(component['grid'])
    data_components.append((state, grid))
  return data_components


def _generate_features(
    data_components: List[Tuple[states.StateDefinition, grids.Grid]],
    example_num_time_steps: int,
) -> Dict[str, tf.io.FixedLenFeature]:
  """Generates features dictionary to be used to parse tfrecord files."""
  features = {}
  for state_key, grid in data_components:
    shape = (example_num_time_steps,) + grid.shape
    string_key = utils.component_name(state_key, grid)
    features[string_key] = tf.io.FixedLenFeature(shape, tf.float32)
  return features


def _assert_compatible(
    requested_data_keys: Sequence[Sequence[states.StateDefinition]],
    requested_data_grids: Sequence[grids.Grid],
    available_features: Dict[str, tf.io.FixedLenFeature]):
  """Checks that the requested data is available in the dataset.

  Args:
    requested_data_keys: StateDefinitions of the requested states grouped in
      tuples.
    requested_data_grids: Grids corresponding to the data_keys.
    available_features: Dictionary of features available in the dataset.

  Raises:
    ValueError: Requested data is not present in the dataset.
  """
  for state_keys, grid in zip(requested_data_keys, requested_data_grids):
    for state_key in state_keys:
      name = utils.component_name(state_key, grid)
      if name not in available_features:
        raise ValueError('requested data {} is not present in the dataset: {}'
                         .format(name, list(available_features)))


def get_output_grid(metadata: Mapping[str, Any]) -> grids.Grid:
  """Reconstructs the low resolution grid from metadata."""
  return grids.Grid.from_config(metadata['output_grid'])


def get_simulation_grid(metadata: Mapping[str, Any]) -> grids.Grid:
  """Reconstructs the high resolution grid from metadata."""
  return grids.Grid.from_config(metadata['simulation_grid'])


def get_equation(metadata: Mapping[str, Any]) -> equations.Equation:
  return equations.equation_from_config(metadata['equation'])
