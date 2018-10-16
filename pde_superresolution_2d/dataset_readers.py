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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from typing import Dict, Tuple

from google.protobuf import text_format
from tensorflow import gfile
from pde_superresolution_2d import equations
from pde_superresolution_2d import grids
from pde_superresolution_2d import metadata_pb2
from pde_superresolution_2d import models
from pde_superresolution_2d import states
from pde_superresolution_2d import utils


def initialize_dataset(
    metadata: metadata_pb2.Dataset,
    requested_data_keys: Tuple[Tuple[states.StateKey, ...], ...],
    requested_data_grids: Tuple[grids.Grid, ...]
) -> tf.data.Dataset:
  """Returns a tf.data.Dataset, setup to provide requested states.

  Args:
    metadata: Dataset message containing metadata.
    requested_data_keys: State keys of the requested data.
    requested_data_grids: Grids corresponding to requested_data_keys. Must
        have the same length as requested_data_keys.

  Returns:
    TFRecordDataset setup to generate requested states from the dataset.
  """
  train_files = metadata.file_names
  train_files = [str(train_file) for train_file in train_files]
  data_components = _parse_components(metadata)
  features = _generate_features(data_components)
  _assert_compatible(requested_data_keys, requested_data_grids, features)
  train_dataset = tf.data.TFRecordDataset(train_files)

  def parse_function(example_proto):
    """Parsing function that converts example proto to a tuple of states."""
    parsed_features = tf.parse_single_example(example_proto, features)
    # TODO(dkochkov) Consider switching logic to use parse_example.
    output_states = []
    for state_keys, grid in zip(requested_data_keys, requested_data_grids):
      state = {}
      for state_key in state_keys:
        string_key = utils.component_name(state_key, grid)
        value = tf.reshape(parsed_features[string_key], grid.get_shape())
        state[state_key] = tf.to_double(value)
      output_states.append(state)
    return tuple(output_states)

  train_dataset = train_dataset.map(parse_function)
  return train_dataset


def load_metadata(metadata_path: str) -> metadata_pb2.Dataset:
  """Loads metadata from a file.

  Args:
    metadata_path: Full path to the metadata file.

  Returns:
    Dataset message generated from the file.
  """
  with gfile.Open(metadata_path) as reader:
    proto = text_format.Parse(reader.read(), metadata_pb2.Dataset())
  return proto


def _parse_components(
    metadata: metadata_pb2.Dataset
) -> Tuple[Tuple[states.StateKey, grids.Grid], ...]:
  """Parses data components from the metadata.

  Args:
    metadata: Dataset protocol buffer holding the metadata for the dataset.

  Returns:
    Components in the dataset organized as tuple of StateKey, Grid pairs.
  """
  data_components_proto = metadata.components
  data_components = []
  for component in data_components_proto:
    data_components.append((states.state_key_from_proto(component.state_key),  # pytype: disable=wrong-arg-types
                            grids.grid_from_proto(component.grid)))  # pytype: disable=wrong-arg-types
  return tuple(data_components)


def _generate_features(
    data_components: Tuple[Tuple[states.StateKey, grids.Grid], ...]
) -> Dict[str, tf.FixedLenFeature]:
  """Generates features dictionary to be used to parse tfrecord files.

  Args:
    data_components: Components in the dataset. Must come in the same order
        as they were serialized by the dataset_builder.

  Returns:
    Dictionary of features that maps names to TensorFlow fixed length feature.
  """
  features = {}
  for state_key, grid in data_components:
    feature_size = np.prod(grid.get_shape())
    string_key = utils.component_name(state_key, grid)
    features[string_key] = tf.FixedLenFeature([feature_size], tf.float32)
  return features


def _assert_compatible(
    requested_data_keys: Tuple[Tuple[states.StateKey, ...], ...],
    requested_data_grids: Tuple[grids.Grid, ...],
    available_features: Dict[str, tf.FixedLenFeature]):
  """Checks that the requested data is available in the dataset.

  Args:
    requested_data_keys: StateKeys of the requested states grouped in tuples.
    requested_data_grids: Grids corresponding to the data_keys.
    available_features: Dictionary of features available in the dataset.

  Raises:
    ValueError: Requested data is not present in the dataset.
  """
  for state_keys, grid in zip(requested_data_keys, requested_data_grids):
    for state_key in state_keys:
      if utils.component_name(state_key, grid) not in available_features:
        raise ValueError('requested data is not present in the dataset')


def get_low_res_grid(metadata: metadata_pb2.Dataset) -> grids.Grid:
  """Reconstructs the low resolution grid from metadata."""
  return grids.grid_from_proto(metadata.low_resolution_grid)  # pytype: disable=wrong-arg-types


def get_high_res_grid(metadata: metadata_pb2.Dataset) -> grids.Grid:
  """Reconstructs the high resolution grid from metadata."""
  return grids.grid_from_proto(metadata.high_resolution_grid)  # pytype: disable=wrong-arg-types


def get_baseline_model(metadata: metadata_pb2.Dataset) -> models.Model:
  """Reconstructs the model used to generate dataset."""
  return models.model_from_proto(metadata.model)  # pytype: disable=wrong-arg-types


def get_equation(metadata: metadata_pb2.Dataset) -> equations.Equation:
  """Reconstructs the equation used to generate the dataset."""
  return equations.equation_from_proto(metadata.equation)  # pytype: disable=wrong-arg-types
