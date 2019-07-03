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
"""Beam utilities."""
import apache_beam as beam
import numpy as np
from typing import List, Tuple


class MeanVarianceCombineFn(beam.CombineFn):
  """Class implementing a beam transformation that computes dataset statistics.

  Implements methods required by beam.CombineFn interface to be used in a
  pipeline. Called during the dataset construction process to provide mean and
  variance of the primary inputs in the dataset.
  """

  def create_accumulator(self) -> Tuple[float, float, int]:
    return 0.0, 0.0, 0

  def add_input(
      self,
      accumulator: Tuple[float, float, int],
      values: np.ndarray  # pylint: disable=redefined-builtin
  ) -> Tuple[float, float, int]:
    """Includes input components to the running mean and added_variance.

    Implementation below follows Welford's algorithm described in more detail at
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    in the Online algorithm section. `added_variance` corresponds to M2.

    Args:
      accumulator: Accumulator of mean, aggregated variance and count.
      values: New set of values to be added to the accumulator.

    Returns:
      Updated accumulator that includes values from the input.
    """
    mean, added_variance, count = accumulator
    for value in np.nditer(values):
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
