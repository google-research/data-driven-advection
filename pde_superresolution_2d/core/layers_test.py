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
"""Sanity tests for layers.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from absl.testing import absltest

from pde_superresolution_2d.core import layers


class LayersTest(absltest.TestCase):
  """Tests for functions in layers.py."""

  def test_periodic_padding(self):
    """Test that periodic padding expands the shape and pads correct values."""
    even_test = np.reshape(np.arange(1, 17), (1, 4, 4, 1))
    odd_test = np.reshape(np.arange(1, 26), (1, 5, 5, 1))

    with tf.Graph().as_default():
      even_padding_even_kern = layers.pad_2d_periodic(even_test, kernel_size=2)
      even_padding_odd_kern = layers.pad_2d_periodic(even_test, kernel_size=3)
      odd_padding_even_kern = layers.pad_2d_periodic(odd_test, kernel_size=2)
      odd_padding_odd_kern = layers.pad_2d_periodic(odd_test, kernel_size=3)

      with tf.train.MonitoredSession() as sess:
        even_even_result = sess.run(even_padding_even_kern)
        even_odd_result = sess.run(even_padding_odd_kern)
        odd_even_result = sess.run(odd_padding_even_kern)
        odd_odd_result = sess.run(odd_padding_odd_kern)

    self.assertEqual(np.shape(even_even_result), (1, 5, 5, 1))
    self.assertEqual(np.shape(even_odd_result), (1, 6, 6, 1))
    self.assertEqual(np.shape(odd_even_result), (1, 6, 6, 1))
    self.assertEqual(np.shape(odd_odd_result), (1, 7, 7, 1))

    np.testing.assert_allclose(even_even_result[0, 0, :, 0],
                               even_even_result[0, 4, :, 0])
    np.testing.assert_allclose(even_odd_result[0, 0, :, 0],
                               even_odd_result[0, 4, :, 0])

    np.testing.assert_allclose(odd_even_result[0, 0, :, 0],
                               odd_even_result[0, 5, :, 0])
    np.testing.assert_allclose(odd_odd_result[0, 0, :, 0],
                               odd_odd_result[0, 5, :, 0])

  def test_nn_conv2d_periodic(self):
    """Test that nn_conv2d_periodic is commutative with translations."""
    test_input = np.random.random(size=(1, 10, 10, 2))

    with tf.Graph().as_default():
      with tf.variable_scope('even_nn_conv2d_periodic', reuse=tf.AUTO_REUSE):
        even_conv = layers.nn_conv2d_periodic(
            test_input, kernel_size=2, num_filters=4)
        even_roll = tf.manip.roll(test_input, shift=(4, 5), axis=(1, 2))
        even_roll_conv = tf.manip.roll(even_conv, shift=(4, 5), axis=(1, 2))
        even_conv_roll = layers.nn_conv2d_periodic(
            even_roll, kernel_size=2, num_filters=4)

      with tf.variable_scope('odd_nn_conv2d_periodic', reuse=tf.AUTO_REUSE):
        odd_conv = layers.nn_conv2d_periodic(
            test_input, kernel_size=3, num_filters=4)
        odd_roll = tf.manip.roll(test_input, shift=(4, 5), axis=(1, 2))
        odd_roll_conv = tf.manip.roll(odd_conv, shift=(4, 5), axis=(1, 2))
        odd_conv_roll = layers.nn_conv2d_periodic(
            odd_roll, kernel_size=3, num_filters=4)

      with tf.train.MonitoredSession() as sess:
        even_conv_roll_values = sess.run(even_conv_roll)
        even_roll_conv_values = sess.run(even_roll_conv)
        odd_conv_roll_values = sess.run(odd_conv_roll)
        odd_roll_conv_values = sess.run(odd_roll_conv)

    np.testing.assert_allclose(even_conv_roll_values, even_roll_conv_values)
    np.testing.assert_allclose(odd_conv_roll_values, odd_roll_conv_values)

if __name__ == '__main__':
  absltest.main()




