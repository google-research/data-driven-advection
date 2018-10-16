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
"""Layers define frequently used modules for models based on neural networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from typing import Callable


def pad_2d_periodic(inputs: tf.Tensor, kernel_size: int) -> tf.Tensor:
  """Pads 4D tensor along axes (1, 2) to produce periodic boundary effect.

  Performs padding in 2 steps, by padding left and right sides first, then
  adding the top and the bottom parts. Uses slice and concat ops. Filter and
  batch dimensions should not be padded.

  Args:
    inputs: Tensor to be padded on the boundary. [btach, x, y, channels]
    kernel_size: Size of the square kernel to be applied to the inputs.

  Returns:
    Tensor padded on the boundary to produce periodic boundary effect.

  Raises:
    ValueError: Input tensor must be 4D.
    ValueError: The size of the kernel spans the length of the dimension.
  """
  inputs = tf.convert_to_tensor(inputs, name='inputs')

  if len(inputs.shape) != 4:
    raise ValueError('inputs must be 4D for periodic padding')

  input_size_x = int(inputs.get_shape()[1])
  input_size_y = int(inputs.get_shape()[2])
  if input_size_x <= kernel_size or input_size_y <= kernel_size:
    raise ValueError('The size of the kernel spans the length of the dimension')

  if kernel_size % 2 == 1:
    pad_index_left = input_size_x - (kernel_size - 1) // 2
    pad_index_top = input_size_y - (kernel_size - 1) // 2
    pad_index_right = (kernel_size - 1) // 2
    pad_index_bottom = (kernel_size - 1) // 2
  else:
    pad_index_left = input_size_x - kernel_size // 2 + 1
    pad_index_top = input_size_y - kernel_size // 2 + 1
    pad_index_right = kernel_size // 2
    pad_index_bottom = kernel_size // 2

  left_pad = inputs[:, pad_index_left:, :, :]
  right_pad = inputs[:, :pad_index_right, :, :]

  horizontal = tf.concat([left_pad, inputs, right_pad], axis=1)

  top_pad = horizontal[:, :, pad_index_top:, :]
  bottom_pad = horizontal[:, :, :pad_index_bottom, :]

  return tf.concat([top_pad, horizontal, bottom_pad], axis=2)


def nn_conv2d_periodic(
    inputs: tf.Tensor,
    kernel_size: int,
    num_filters: int,
    use_bias: bool = True,
    activation: Callable = tf.nn.relu
) -> tf.Tensor:
  """Builds a convolutional layer with periodic padding on top of inputs.

  Args:
    inputs: Input tensor, must be 4D.
    kernel_size: Size of the kernel to construct.
    num_filters: Number of filters to use.
    use_bias: Boolean, indicating whether to add bias term.
    activation: Activation function to use.

  Returns:
    Result of periodic 2d convolution of `inputs` with the kernel.
  """
  # TODO(dkochkov) Pass tf.layers object here instead of using get_variable.
  with tf.name_scope('periodic_padding'):
    inputs = tf.convert_to_tensor(inputs, name='inputs')
    x_paded = pad_2d_periodic(inputs, kernel_size)
  weights = tf.get_variable(
      'W_ij',
      [kernel_size, kernel_size, inputs.get_shape()[3], num_filters],
      tf.float64
  )
  conv = tf.nn.conv2d(x_paded, weights, strides=[1, 1, 1, 1], padding='VALID')
  if use_bias:
    bias = tf.get_variable(
        'b_i',
        [num_filters],
        tf.float64,
        initializer=tf.zeros_initializer()
    )
    return activation(tf.add(conv, bias))
  return activation(conv)
