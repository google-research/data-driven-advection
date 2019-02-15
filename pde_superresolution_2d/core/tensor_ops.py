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
"""Tensor operations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pde_superresolution_2d.core import grids
import tensorflow as tf
from typing import Dict, List, Sequence, Tuple, Union


def auto_nest(func):
  """Automatically support nested tensors in the first argument."""
  def wrapper(tensors, *args, **kwargs):
    return tf.contrib.framework.nest.map_structure(
        lambda x: func(x, *args, **kwargs), tensors)
  return wrapper


def _normalize_axis(axis: int, ndim: int) -> int:
  if not -ndim <= axis < ndim:
    raise ValueError('invalid axis {} for ndim {}'.format(axis, ndim))
  if axis < 0:
    axis += ndim
  return axis


def _roll_once(
    tensor: tf.Tensor,
    shift: int,
    axis: int,
) -> tf.Tensor:
  """Roll along a single dimension like tf.roll()."""
  if not shift:
    return tensor
  axis = _normalize_axis(axis, len(tensor.shape))
  slice_left = (slice(None),) * axis + (slice(-shift, None),)
  slice_right = (slice(None),) * axis + (slice(None, -shift),)
  return tf.concat([tensor[slice_left], tensor[slice_right]], axis=axis)


@auto_nest
def roll(
    tensor: tf.Tensor,
    shift: Union[int, Sequence[int]],
    axis: Union[int, Sequence[int]],
) -> tf.Tensor:
  """Like tf.roll(), but runs on GPU as a well as CPU."""
  if isinstance(axis, int):
    axis = [axis]
  if isinstance(shift, int):
    shift = [shift]
  result = tensor
  for axis_element, shift_element in zip(axis, shift):
    result = _roll_once(result, shift_element, axis_element)
  return result


@auto_nest
def roll_2d(
    tensor: tf.Tensor,
    shifts: Tuple[int, int],
    axes: Tuple[int, int] = (-2, -1)
) -> tf.Tensor:
  """Roll by default along the last two axes."""
  return roll(tensor, shifts, axes)


def _pad_periodic_by_axis(
    tensor: tf.Tensor, padding: Sequence[int], axis: int,
) -> tf.Tensor:
  """Periodic padding along one axis."""
  axis = _normalize_axis(axis, len(tensor.shape))
  if len(padding) != 2:
    raise ValueError('padding must have length 2: {}'.format(padding))
  if any(pad < 0 for pad in padding):
    raise ValueError('padding must be positive: {}'.format(padding))
  pad_left, pad_right = padding

  slice_left = (slice(None),) * axis + (slice(-pad_left, None),)
  slice_right = (slice(None),) * axis + (slice(None, pad_right),)

  if pad_left and pad_right:
    tensors = [tensor[slice_left], tensor, tensor[slice_right]]
    return tf.concat(tensors, axis=axis)
  elif pad_left:
    tensors = [tensor[slice_left], tensor]
    return tf.concat(tensors, axis=axis)
  elif pad_right:
    tensors = [tensor, tensor[slice_right]]
    return tf.concat(tensors, axis=axis)
  else:
    return tensor


@auto_nest
def pad_periodic(
    tensor: tf.Tensor, paddings: Sequence[Sequence[int]],
) -> tf.Tensor:
  """Periodic padding, with an API like tf.pad()."""
  result = tensor
  for axis, padding in enumerate(paddings):
    result = _pad_periodic_by_axis(result, padding, axis)
  return result


def paddings_for_conv2d(
    kernel_size: Sequence[int],
    shifts: Sequence[int] = (0, 0),
) -> List[Tuple[int, int]]:
  """Paddings for a conv2d valid convolution to return the original shape."""
  if len(kernel_size) != 2 or len(shifts) != 2:
    raise ValueError('kernel_size and shifts must have length 2')

  paddings = [(0, 0)]
  for size, shift in zip(kernel_size, shifts):
    pad_left = (size - shift) // 2
    paddings.append((pad_left, size - pad_left - 1))
  paddings += [(0, 0)]
  return paddings


@auto_nest
def pad_periodic_2d(
    tensor: tf.Tensor,
    kernel_size: Sequence[int],
    shifts: Sequence[int] = (0, 0),
) -> tf.Tensor:
  """Pad a tensor in preparation for a conv2d valid convolution."""
  if len(tensor.shape) != 4:
    raise ValueError('tensor has wrong number of dimensions: {}'.format(tensor))

  paddings = paddings_for_conv2d(kernel_size, shifts)
  result = pad_periodic(tensor, paddings)
  return result


@auto_nest
def extract_patches_2d(
    tensor: tf.Tensor,
    kernel_size: Sequence[int],
    shifts: Sequence[int] = (0, 0),
) -> tf.Tensor:
  """Create a tensor of patches for use with finite difference coefficients.

  Args:
    tensor: 2D or 3D tensor representing a grid variable.
    kernel_size: size of the stencil along the x and y directions.
    shifts: shifts of the stencil along the x and y directions.

  Returns:
    4D Tensor composed of stencil shifts stacked along the last axis.
  """
  if len(tensor.shape) == 2:
    added_batch = True
    tensor = tensor[tf.newaxis, ...]
  else:
    added_batch = False

  if len(tensor.shape) != 3:
    raise ValueError('tensor has wrong number of dimensions: {}'.format(tensor))

  paddings = paddings_for_conv2d(kernel_size, shifts)[:-1]
  padded = pad_periodic(tensor, paddings)

  size_x, size_y = kernel_size
  extracted = tf.extract_image_patches(padded[..., tf.newaxis],
                                       ksizes=[1, size_x, size_y, 1],
                                       strides=[1, 1, 1, 1],
                                       rates=[1, 1, 1, 1],
                                       padding='VALID')

  if added_batch:
    result = tf.squeeze(extracted, axis=0)
  else:
    result = extracted

  return result


@auto_nest
def resample_mean(
    tensor: tf.Tensor,
    source_grid: grids.Grid,
    destination_grid: grids.Grid
) -> tf.Tensor:
  """Generates computational graph that downsamples the solution.

  Args:
    tensor: State representing the initial configuration of the system
    source_grid: Grid defining the high resolution.
    destination_grid: Grid defining the target low resolution.

  Returns:
    A tensor representing the input state at lower resolution.

  Raises:
    ValueError: Grids sizes are not compatible for downsampling.
  """
  # TODO(shoyer): move to grids.py?
  current_size_x, current_size_y = source_grid.shape
  new_size_x, new_size_y = destination_grid.shape
  if current_size_x % new_size_x != 0 or current_size_y % new_size_y != 0:
    raise ValueError('grids are not compatible')
  x_factor = current_size_x // new_size_x
  y_factor = current_size_y // new_size_y

  tmp_grid_shape = [new_size_x, x_factor, new_size_y, y_factor]
  tmp_shape = tensor.shape.as_list()[:-2] + tmp_grid_shape
  return tf.reduce_mean(tf.reshape(tensor, tmp_shape), axis=(-3, -1))


@auto_nest
def moveaxis(tensor: tf.Tensor, source: int, destination: int) -> tf.Tensor:
  """TensorFlow version of np.moveaxis."""
  ndim = len(tensor.shape)
  source = _normalize_axis(source, ndim)
  destination = _normalize_axis(destination, ndim)
  order = [n for n in range(ndim) if n != source]
  order.insert(destination, source)
  return tf.transpose(tensor, order)


@auto_nest
def stack_all_contiguous_slices(
    tensor: tf.Tensor, slice_size: int, new_axis: int = 0,
) -> tf.Tensor:
  """Stack all contiguous slices along the first axis of a tensor."""
  size = tensor.shape[0].value
  return tf.stack([tensor[i : i+slice_size]
                   for i in range(size - slice_size + 1)], axis=new_axis)

