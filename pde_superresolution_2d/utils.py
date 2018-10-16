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
"""Miscellaneous utility functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import gfile
from typing import Any, List, Tuple

from google.protobuf import text_format
from pde_superresolution_2d import grids
from pde_superresolution_2d import metadata_pb2
from pde_superresolution_2d import states


def roll_2d(
    tensor: tf.Tensor,
    shifts: Tuple[int, int],
    axes: Tuple[int, int] = (1, 2)
) -> tf.Tensor:
  """Implements roll operation using concatenation for (1, 2) and (0, 1) axes.

  Args:
    tensor: Tensor to roll.
    shifts: Number of integer rotations to perform along corresponding axes.
    axes: Axes around which the tensor is rolled, must be one of (1, 2) (0, 1).

  Returns:
    A tensor rolled `shifts` steps along specified axes correspondingly.

  Raises:
    ValueError: Roll along these axes is currently not supported.
  """
  shift_x = -shifts[0]
  shift_y = -shifts[1]
  if axes == (1, 2):
    x_roll_result = tf.concat([tensor[:, shift_x:, :],
                               tensor[:, :shift_x, :]], 1)
    y_roll_result = tf.concat([x_roll_result[:, :, shift_y:],
                               x_roll_result[:, :, :shift_y]], 2)
  elif axes == (0, 1):
    x_roll_result = tf.concat([tensor[shift_x:, ...], tensor[:shift_x, ...]], 0)
    y_roll_result = tf.concat([x_roll_result[:, shift_y:, ...],
                               x_roll_result[:, :shift_y, ...]], 1)
  else:
    raise ValueError('Roll along these axes is currently not supported')
  return y_roll_result


def generate_stencil_shift_tensors(input_tensor, size_x, size_y) -> tf.Tensor:
  """Generates a tensor composed of stencil shifts stacked along 3rd axis.

  Args:
    input_tensor: 3D tensor representing original distribution.
    size_x: Size of the stencil along x direction.
    size_y: Size of the stencil along y direction.

  Returns:
    4D Tensor composed of stencil shifts stacked along the last axis.
  """
  stencil_tensors = []
  for shift_x in range(-size_x // 2 + 1, size_x // 2 + 1):
    for shift_y in range(-size_y // 2 + 1, size_y // 2 + 1):
      stencil_tensors.append(roll_2d(input_tensor, (shift_x, shift_y)))
  return tf.stack(stencil_tensors, axis=3)


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


def component_name(
    state_key: states.StateKey,
    grid: grids.Grid = None
) -> str:
  """Generates string keys from StateKey and Grid combination.

  Generates a string-key by combining parameters of the StateKey and Grid.

  Args:
    state_key: StateKey describing the quantity, derivatives and offsets.
    grid: Grid object specifying the grid on which the state is evaluated.

  Returns:
    String to be used as a key in the tf.Example protocol buffer.
  """
  if grid is not None:
    grid_sizes = [grid.size_x, grid.size_y]
  else:
    grid_sizes = []

  components = (
      [state_key.name] +
      list(state_key.derivative_orders) +
      list(state_key.offset) +
      grid_sizes
  )
  return '_'.join(map(str, components))


def save_proto(proto: Any, output_path: str):
  """Saves a `proto` protocol buffer to the `output_path`.

  Args:
    proto: Protocol buffer to be written to `output_path`.
    output_path: Path where to save `proto` protocol buffer.
  """
  with gfile.Open(output_path, 'w') as f:
    f.write(text_format.MessageToString(proto))


def load_proto(proto_path: str, pb_class: Any) -> Any:
  """Loads protocol buffer from a file.

  Args:
    proto_path: Full path to the file containing protocol buffer data.
    pb_class: Message class to be parsed.

  Returns:
    Dataset message generated from the file.
  """
  with gfile.Open(proto_path) as reader:
    proto = text_format.Parse(reader.read(), pb_class)
  return proto
