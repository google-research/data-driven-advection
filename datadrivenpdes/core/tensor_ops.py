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
"""Tensor operations."""
import collections
import typing
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from datadrivenpdes.core import grids
from datadrivenpdes.core import states
import tensorflow as tf


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
  ndim = len(tensor.shape)

  axis = _normalize_axis(axis, ndim)
  if len(padding) != 2:
    raise ValueError('padding must have length 2: {}'.format(padding))
  if any(pad < 0 for pad in padding):
    raise ValueError('padding must be positive: {}'.format(padding))
  pad_left, pad_right = padding

  slice_left = [slice(None)] * ndim
  slice_left[axis] = slice(-pad_left, None)
  slice_left = tuple(slice_left)
  slice_right = [slice(None)] * ndim
  slice_right[axis] = slice(None, pad_right)
  slice_right = tuple(slice_right)

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
                                       [1, size_x, size_y, 1],
                                       strides=[1, 1, 1, 1],
                                       rates=[1, 1, 1, 1],
                                       padding='VALID')

  if added_batch:
    result = tf.squeeze(extracted, axis=0)
  else:
    result = extracted

  return result


TensorLike = Union[np.ndarray, tf.Tensor]


def regrid_mean(
    tensor: TensorLike,
    factor: int,
    offset: int = 0,
    axis: int = -1,
) -> tf.Tensor:
  """Resample data to a lower-resolution by averaging data point.

  Args:
    tensor: input tensor.
    factor: integer factor by which to reduce the size of the given axis.
    offset: integer offset at which to place blocks.
    axis: integer axis to resample over.

  Returns:
    Array with averaged data along axis.

  Raises:
    ValueError: if the original axis size is not evenly divided by factor.
  """
  if offset:
    # TODO(shoyer): support this with roll()
    raise NotImplementedError('offset not supported yet')

  tensor = tf.convert_to_tensor(tensor)
  shape = tensor.shape.as_list()
  axis = _normalize_axis(axis, len(shape))
  multiple, residual = divmod(shape[axis], factor)
  if residual:
    raise ValueError('resample factor {} must divide size {}'
                     .format(factor, shape[axis]))

  new_shape = shape[:axis] + [multiple, factor] + shape[axis+1:]
  new_shape = [-1 if size is None else size for size in new_shape]

  return tf.reduce_mean(tf.reshape(tensor, new_shape), axis=axis+1)


def regrid_subsample(
    tensor: TensorLike,
    factor: int,
    offset: int = 0,
    axis: int = -1) -> tf.Tensor:
  """Resample data to a lower-resolution by subsampling data-points.

  Args:
    tensor: input tensor.
    factor: integer factor by which to reduce the size of the given axis.
    offset: integer offset at which to subsample.
    axis: integer axis to resample over.

  Returns:
    Array with subsampled data along axis.

  Raises:
    ValueError: if the original axis size is not evenly divided by factor.
  """
  tensor = tf.convert_to_tensor(tensor)
  shape = tensor.shape.as_list()
  axis = _normalize_axis(axis, len(shape))
  residual = shape[axis] % factor
  if residual:
    raise ValueError('resample factor {} must divide size {}'
                     .format(factor, shape[axis]))

  if offset < 0 or offset >= factor:
    raise ValueError('invalid offset {} not in [0, {})'.format(offset, factor))

  indexer = [slice(None)] * len(shape)
  indexer[axis] = slice(offset, None, factor)

  return tensor[tuple(indexer)]


def _regrid_tensor(
    tensor: tf.Tensor,
    definition: states.StateDefinition,
    factors: Tuple[int, ...],
    axes: Tuple[int, ...],
) -> tf.Tensor:
  """Regrid a Tensor along all its axes."""
  result = tensor
  for factor, cell_offset, axis in zip(factors, definition.offset, axes):
    # TODO(shoyer): add a notion of the geometry (cell vs face variables)
    # directly into the data model in StateDefinition?
    if cell_offset == 0:
      result = regrid_mean(result, factor, offset=0, axis=axis)
    elif cell_offset == 1:
      offset = factor - 1
      result = regrid_subsample(result, factor, offset, axis)
    else:
      raise ValueError('unsupported offset: {}'.format(cell_offset))
  return result


# pylint: disable=unused-argument
@typing.overload
def regrid(
    tensor: TensorLike,
    definition: states.StateDefinition,
    source: grids.Grid,
    destination: grids.Grid
) -> tf.Tensor:
  pass


@typing.overload
def regrid(  # pylint: disable=function-redefined
    tensor: Mapping[str, TensorLike],
    definition: Mapping[str, states.StateDefinition],
    source: grids.Grid,
    destination: grids.Grid
) -> Dict[str, tf.Tensor]:
  pass
# pylint: enable=unused-argument


def regrid(inputs, definitions, source, destination):  # pylint: disable=function-redefined
  """Regrid to lower resolution using an appropriate method.

  This function assumes that quantities at staggered grid locations should be
  regridded using subsampling instead of averages. This is appropriate for
  typical finite difference/volume discretizations.

  Args:
    inputs: state(s) to regrid.
    definitions: definition(s) of this tensor quantity.
    source: fine resolution Grid.
    destination: coarse resolution Grid.

  Returns:
    Tensor(s) representing the input state at lower resolution.

  Raises:
    ValueError: Grids sizes are not compatible for downsampling.
  """
  factors, residuals = zip(*[
      divmod(current_size, new_size)
      for current_size, new_size in zip(source.shape, destination.shape)
  ])

  if any(residuals):
    raise ValueError('grids are not aligned: {} vs {}'
                     .format(source, destination))

  axes = tuple(-(n + 1) for n in reversed(range(source.ndim)))

  if isinstance(inputs, collections.Mapping):
    result = {k: _regrid_tensor(inputs[k], definitions[k], factors, axes)
              for k in inputs}
  else:
    result = _regrid_tensor(inputs, definitions, factors, axes)
  return result  # pytype: disable=bad-return-type


def regrid_masked_mean_2d(
    tensor: TensorLike,
    mask: TensorLike,
    source: grids.Grid,
    destination: grids.Grid,
) -> tf.Tensor:
  """Regrid using the mean of unmasked elements.

  The input tensor is regridded over the last two axes. Entirely masked regions
  are set to zero.

  Args:
    tensor: tensor to regrid.
    mask: valid elements to include in the mean.
    source: fine resolution Grid.
    destination: coarse resolution Grid.

  Returns:
    Averaged tensor.
  """
  tensor = tf.convert_to_tensor(tensor)
  mask = tf.convert_to_tensor(mask)

  shape = tensor.shape.as_list()

  new_shape = shape[:-2]
  for current_size, new_size in zip(source.shape, destination.shape):
    factor, residual = divmod(current_size, new_size)
    if residual:
      raise ValueError('grids are not aligned')
    new_shape.extend([new_size, factor])

  new_shape = [-1 if size is None else size for size in new_shape]

  mask = tf.cast(mask, tensor.dtype)
  total = tf.reduce_sum(tf.reshape(tensor * mask, new_shape), axis=(-3, -1))
  count = tf.reduce_sum(tf.reshape(mask, new_shape), axis=(-3, -1))
  return total / tf.maximum(count, 1.0)


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
def swap_xy(tensor: tf.Tensor) -> tf.Tensor:
  """Swap x and y dimensions on a tensor."""
  return moveaxis(tensor, -2, -1)


@auto_nest
def stack_all_contiguous_slices(
    tensor: tf.Tensor, slice_size: int, new_axis: int = 0,
) -> tf.Tensor:
  """Stack all contiguous slices along the first axis of a tensor."""
  size = tensor.shape[0].value
  return tf.stack([tensor[i : i+slice_size]
                   for i in range(size - slice_size + 1)], axis=new_axis)

