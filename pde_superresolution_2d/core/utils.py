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
"""Miscellaneous utility functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Optional

import numpy as np
from pde_superresolution_2d.core import grids
from pde_superresolution_2d.core import states
from tensorflow.io import gfile

from google.protobuf import text_format


def component_name(
    state_def: states.StateDefinition,
    grid: Optional[grids.Grid] = None,
) -> str:
  """Generates string keys from StateDefinition and Grid combination.

  Generates a string-key by combining parameters of the StateDefinition and
  Grid.

  Args:
    state_def: StateDefinition describing the quantity, derivatives and offsets.
    grid: Grid object specifying the grid on which the state is evaluated.

  Returns:
    String to be used as a key in the tf.Example protocol buffer.
  """
  if state_def.tensor_indices:
    name = (''.join(index.name.lower() for index in state_def.tensor_indices)
            + '_' + state_def.name)
  else:
    name = state_def.name

  underscore_join = lambda x: '_'.join(map(str, x))
  components = [
      name,
      underscore_join(state_def.derivative_orders),
      underscore_join(state_def.offset),
  ]
  if grid is not None:
    components.append(underscore_join([grid.size_x, grid.size_y]))
  return '/'.join(map(str, components))


def save_proto(proto: Any, output_path: str):
  """Saves a `proto` protocol buffer to the `output_path`.

  Args:
    proto: Protocol buffer to be written to `output_path`.
    output_path: Path where to save `proto` protocol buffer.
  """
  with gfile.GFile(output_path, 'w') as f:
    f.write(text_format.MessageToString(proto))


def load_proto(proto_path: str, pb_class: Any) -> Any:
  """Loads protocol buffer from a file.

  Args:
    proto_path: Full path to the file containing protocol buffer data.
    pb_class: Message class to be parsed.

  Returns:
    Dataset message generated from the file.
  """
  with gfile.GFile(proto_path) as reader:
    proto = text_format.Parse(reader.read(), pb_class)
  return proto


def integer_ratio(multiplied, base, epsilon=1e-6) -> int:
  """Calculate the integer ratio between two numbers, or raise ValueError."""
  factor = int(round(multiplied / base))
  if not np.isclose(factor, multiplied / base, rtol=epsilon):
    raise ValueError(
        '{} is not an integer multiple of {}'
        .format(multiplied, base))
  return factor
