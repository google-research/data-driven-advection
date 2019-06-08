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
"""Tests for tensor_ops.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
from pde_superresolution_2d.core import grids
from pde_superresolution_2d.core import states
from pde_superresolution_2d.core import tensor_ops
import tensorflow as tf

from absl.testing import absltest

# Use eager mode by default
tf.enable_eager_execution()


def tf_roll_2d(tensor, shifts, axes=(-2, -1)):
  return tf.roll(tensor, shift=shifts, axis=axes)


class TensorOpsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('even_shifts', (4, 8)),
      ('odd_shifts', (11, 9)),
      ('even_odd_shifts', (10, 11)),
      ('odd_even_shifts', (13, 6)))
  def test_roll_consistency(self, shifts):
    batch_input = tf.random_uniform(shape=(2, 50, 50))
    single_input = tf.random_uniform(shape=(25, 25))

    batch_manip = tf_roll_2d(batch_input, shifts)
    batch_concat = tensor_ops.roll_2d(batch_input, shifts)
    np.testing.assert_allclose(batch_manip, batch_concat)

    single_manip = tf_roll_2d(single_input, shifts)
    single_concat = tensor_ops.roll_2d(single_input, shifts)
    np.testing.assert_allclose(single_manip, single_concat)

  def test_pad_periodic(self):
    inputs = tf.range(4)

    actual = tensor_ops.pad_periodic(inputs, [(0, 0)])
    np.testing.assert_array_equal(actual, inputs)

    actual = tensor_ops.pad_periodic(inputs, [(0, 1)])
    expected = np.array([0, 1, 2, 3, 0])
    np.testing.assert_array_equal(actual, expected)

    actual = tensor_ops.pad_periodic(inputs, [(1, 0)])
    expected = np.array([3, 0, 1, 2, 3])
    np.testing.assert_array_equal(actual, expected)

    actual = tensor_ops.pad_periodic(inputs, [(2, 2)])
    expected = np.array([2, 3, 0, 1, 2, 3, 0, 1])
    np.testing.assert_array_equal(actual, expected)

  def test_paddings_for_conv2d(self):
    # without padding: [1, 2, 3] -> [[1, 2], [2, 3]] -> [1.5, 2.5]
    # shift of +0.5 -> padding at end -> [1.5, 2.5, 2]
    actual = tensor_ops.paddings_for_conv2d([2, 1], [1, 0])
    self.assertEqual(actual, [(0, 0), (0, 1), (0, 0), (0, 0)])
    # shift of -0.5 -> padding at start -> [2, 1.5, 2.5]
    actual = tensor_ops.paddings_for_conv2d([2, 1], [-1, 0])
    self.assertEqual(actual, [(0, 0), (1, 0), (0, 0), (0, 0)])

  def test_stack_all_contiguous_slices(self):
    actual = tensor_ops.stack_all_contiguous_slices(tf.range(6), slice_size=4)
    expected = np.stack([np.arange(4), np.arange(1, 5), np.arange(2, 6)])
    np.testing.assert_array_equal(actual, expected)

  def test_regrid(self):
    tensor = tf.convert_to_tensor(
        [[1, 2, 3, 4], [5, 6, 7, 8]], dtype=tf.float32)
    source = grids.Grid(2, 4, step=1)
    destination = grids.Grid(1, 2, step=2)

    definition = states.StateDefinition('centered', (), (0, 0, 0), (0, 0))
    actual = tensor_ops.regrid(tensor, definition, source, destination)
    np.testing.assert_array_equal(actual, [[3.5, 5.5]])

    definition = states.StateDefinition('x_offset', (), (0, 0, 0), (1, 0))
    actual = tensor_ops.regrid(tensor, definition, source, destination)
    np.testing.assert_array_equal(actual, [[5.5, 7.5]])

    definition = states.StateDefinition('y_offset', (), (0, 0, 0), (0, 1))
    actual = tensor_ops.regrid(tensor, definition, source, destination)
    np.testing.assert_array_equal(actual, [[4, 6]])

    definition = states.StateDefinition('xy_offset', (), (0, 0, 0), (1, 1))
    actual = tensor_ops.regrid(tensor, definition, source, destination)
    np.testing.assert_array_equal(actual, [[6, 8]])


if __name__ == '__main__':
  absltest.main()
