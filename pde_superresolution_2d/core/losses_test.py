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
"""Test of loss helper functions defined in losses.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from absl.testing import absltest
from absl.testing import parameterized

from pde_superresolution_2d.core import losses
from pde_superresolution_2d.core import states


C = states.StateKey('concentration', (0, 0, 0), (0, 0))


class LossesTest(parameterized.TestCase):
  """Tests loss generating functions."""

  @parameterized.named_parameters(
      ('batch_and_sizes_a', (5, 20)),
      ('batch_and_sizes_b', (1, 40)))
  def test_state_mean_loss(self, batch_and_size):
    """Tests state mean loss."""

    batch, size = batch_and_size
    with tf.Graph().as_default():
      input_state = {
          C: 2 * tf.ones((batch, size, size), dtype=tf.float64)
      }

      target_state = {
          C: tf.zeros((batch, size, size), dtype=tf.float64)
      }

      l1_loss = losses.state_mean_loss(input_state, target_state, (C,),
                                       tf.losses.absolute_difference)

      l2_loss = losses.state_mean_loss(input_state, target_state, (C,),
                                       tf.losses.mean_squared_error)

      with tf.Session():
        self.assertAlmostEqual(l1_loss.eval(), 2)
        self.assertAlmostEqual(l2_loss.eval(), 4)

  def test_state_weighted_mean_loss(self):
    """Tests that the weight is added as expected."""

    with tf.Graph().as_default():
      input_state = {
          C: 2 * tf.ones((5, 20, 20), dtype=tf.float64)
      }

      target_state = {
          C: tf.zeros((5, 20, 20), dtype=tf.float64)
      }

      small_ref_state = {
          C: tf.ones((5, 20, 20), dtype=tf.float64)
      }
      large_ref_state = {
          C: 3 * tf.ones((5, 20, 20), dtype=tf.float64)
      }

      l1_loss_s = losses.state_weighted_mean_loss(
          input_state, target_state, small_ref_state,
          (C,), tf.losses.absolute_difference)

      l1_loss_l = losses.state_weighted_mean_loss(
          input_state, target_state, large_ref_state,
          (C,), tf.losses.absolute_difference)

      l2_loss_s = losses.state_weighted_mean_loss(
          input_state, target_state, small_ref_state,
          (C,), tf.losses.mean_squared_error)

      l2_loss_l = losses.state_weighted_mean_loss(
          input_state, target_state, large_ref_state,
          (C,), tf.losses.mean_squared_error)

      with tf.Session():
        # Losses = (2 / (0.1 * 2 + 1)) * 2 and (4 / (0.1 * 4 + 1)) * 4
        self.assertAlmostEqual(l1_loss_s.eval(), 20 / 6, places=4)
        self.assertAlmostEqual(l2_loss_s.eval(), 80 / 7, places=4)

        # For baseline with large loss weight should default to 1.
        self.assertAlmostEqual(l1_loss_l.eval(), 2)
        self.assertAlmostEqual(l2_loss_l.eval(), 4)


if __name__ == '__main__':
  absltest.main()
