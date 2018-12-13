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
"""Tests for utils.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
from pde_superresolution_2d.core import utils
import tensorflow as tf
from typing import Tuple

from absl.testing import absltest


def manip_roll_2d(
    tensor: tf.Tensor,
    shifts: Tuple[int, int],
    axes: Tuple[int, int] = (1, 2)
) -> tf.Tensor:
  """Rolls tensor along given axes using the old version based on tf.manip.roll.

  Args:
    tensor: Tensor to roll.
    shifts: Number of integer rotations to perform along corresponding axes.
    axes: Axes around which the rolling if performed.

  Returns:
    A tensor rolled shifts steps along specified axes correspondingly.
  """
  return tf.manip.roll(tensor, shift=shifts, axis=axes)


class UtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('even_shifts', (4, 8)),
      ('odd_shifts', (11, 9)),
      ('even_odd_shifts', (10, 11)),
      ('odd_even_shifts', (13, 6)))
  def test_concatenation_roll(self, shifts):
    """Roll using concatenation should give the same result as tf.manip.roll."""
    with tf.Graph().as_default():
      batch_input = tf.random_uniform(shape=(2, 50, 50))
      single_input = tf.random_uniform(shape=(25, 25))
      batch_axes = (1, 2)
      axes = (0, 1)
      batch_manip = manip_roll_2d(batch_input, shifts, batch_axes)
      batch_concat = utils.roll_2d(batch_input, shifts, batch_axes)

      single_manip = manip_roll_2d(single_input, shifts, axes)
      single_concat = utils.roll_2d(single_input, shifts, axes)

      with tf.Session() as sess:
        batch_results = sess.run([batch_manip, batch_concat])
        single_results = sess.run([single_manip, single_concat])
        np.testing.assert_allclose(batch_results[0], batch_results[1])
        np.testing.assert_allclose(single_results[0], single_results[1])


if __name__ == '__main__':
  absltest.main()
