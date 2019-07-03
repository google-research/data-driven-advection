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
from typing import Optional

import numpy as np
from datadrivenpdes.core import grids
from datadrivenpdes.core import states


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


def integer_ratio(multiplied, base, epsilon=1e-6) -> int:
  """Calculate the integer ratio between two numbers, or raise ValueError."""
  factor = int(round(multiplied / base))
  if not np.isclose(factor, multiplied / base, rtol=epsilon):
    raise ValueError(
        '{} is not an integer multiple of {}'
        .format(multiplied, base))
  return factor
