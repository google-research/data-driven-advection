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
"""Polynomial based models for finite differences and finite volumes."""
import enum
import functools
from typing import Any, Iterator, Optional, Sequence, Tuple

import numpy as np
import scipy.special
import tensorflow as tf


class Method(enum.Enum):
  """Discretization method."""
  FINITE_DIFFERENCE = 1
  FINITE_VOLUME = 2


def regular_stencil_1d(
    offset: int,
    derivative_order: int,
    accuracy_order: int = 1,
    grid_step: float = 1,
) -> np.ndarray:
  """Return the smallest stencil on which finite differences can be calculated.

  Args:
    offset: half-integer offset between input and output grids.
    derivative_order: integer derivative order to calculate.
    accuracy_order: integer order of polynomial accuracy to enforce. By default,
      only 1st order accuracy is guaranteed.
    grid_step: spacing between grid points/cells.

  Returns:
    1D numpy array giving positions at which to calculate finite differences.
  """
  min_grid_size = derivative_order + accuracy_order

  if offset == 0:
    max_offset = min_grid_size // 2  # 1 -> 0, 2 -> 1, 3 -> 1, 4 -> 2, ...
    grid = np.arange(-max_offset, max_offset + 1) * grid_step
  elif offset == 1:
    max_offset = (min_grid_size + 1) // 2  # 1 -> 1, 2 -> 1, 3 -> 2, 4 -> 2, ...
    grid = (0.5 + np.arange(-max_offset, max_offset)) * grid_step
  else:
    raise ValueError('unexpected offset: {}'.format(offset))  # pylint: disable=g-doc-exception

  return grid


def _kronecker_product(arrays: Sequence[np.ndarray]) -> np.ndarray:
  return functools.reduce(np.kron, arrays)


def _exponents_up_to_degree(degree: int,
                            num_dimensions: int) -> Iterator[Tuple[int]]:
  """Generate all exponents up to given degree.

  Args:
    degree: a non-negative integer representing the maximum degree.
    num_dimensions: a non-negative integer representing the number of
      dimensions.

  Yields:
    An iterator over all tuples of non-negative integers of length
    num_dimensions, whose sum is at most degree. For example, for degree=2 and
    num_dimensions=2, this iterates through [(0, 0), (0, 1), (0, 2), (1, 0),
    (1, 1), (2, 0)].
  """
  if num_dimensions == 0:
    yield tuple()
  else:
    for d in range(degree + 1):
      for exponents in _exponents_up_to_degree(degree - d, num_dimensions - 1):
        yield (d,) + exponents


def constraints(
    stencils: Sequence[np.ndarray],
    method: Method,
    derivative_orders: Sequence[int],
    accuracy_order: int,
    grid_step: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
  """Setup a linear equation A @ c = b for finite difference coefficients.

  Elements are returned in row-major order, e.g., if two stencils of length 2
  are provided: s00, s01, s10, s11.

  Args:
    stencils: list of arrays giving 1D stencils in each direction.
    method: discretization method (i.e., finite volumes or finite differences).
    derivative_orders: integer derivative orders to approximate in each grid
      direction.
    accuracy_order: minimum accuracy orders for the solution in each grid
      direction.
    grid_step: spacing between grid cells. Required if calculating a finite
      volume stencil.

  Returns:
    Tuple of arrays `(A, b)` where `A` is 2D and `b` is 1D providing linear
    constraints. Any vector of finite difference coefficients `c` such that
    `A @ c = b` satisfies the requested accuracy order. The matrix `A` is
    guaranteed not to have more rows than columns.

  Raises:
    ValueError: if the linear constraints are not satisfiable.

  References:
    https://en.wikipedia.org/wiki/Finite_difference_coefficient
    Fornberg, Bengt (1988), "Generation of Finite Difference Formulas on
      Arbitrarily Spaced Grids", Mathematics of Computation, 51 (184): 699-706,
      doi:10.1090/S0025-5718-1988-0935077-0, ISSN 0025-5718.
  """
  # TODO(shoyer): consider supporting arbitrary non-rectangular stencils.
  # TODO(shoyer): consider support different accuracy orders in different
  # directions.
  if accuracy_order < 1:
    raise ValueError('cannot compute constriants with non-positive '
                     'accuracy_order: {}'.format(accuracy_order))

  if len(stencils) != len(derivative_orders):
    raise ValueError('mismatched lengths for stencils and derivative_orders')

  all_constraints = {}

  # See http://g3doc/third_party/py/datadrivenpdes/g3doc/polynomials.md.
  num_dimensions = len(stencils)
  max_degree = accuracy_order + sum(derivative_orders) - 1
  for exponents in _exponents_up_to_degree(max_degree, num_dimensions):

    # build linear constraints for a single polynomial term:
    # \prod_i {x_i}^{m_i}
    lhs_terms = []
    rhs_terms = []

    for exponent, stencil, derivative_order in zip(exponents, stencils,
                                                   derivative_orders):

      if method is Method.FINITE_VOLUME:
        if grid_step is None:
          raise ValueError('grid_step is required for finite volumes')
        # average value of x**m over a centered grid cell
        lhs_terms.append(
            1 / grid_step * ((stencil + grid_step / 2)**(exponent + 1) -
                             (stencil - grid_step / 2)**(exponent + 1)) /
            (exponent + 1))
      elif method is Method.FINITE_DIFFERENCE:
        lhs_terms.append(stencil**exponent)
      else:
        raise ValueError('unexpected method: {}'.format(method))

      if exponent == derivative_order:
        # we get a factor of m! for m-th order derivative in each direction
        rhs_term = scipy.special.factorial(exponent)
      else:
        rhs_term = 0
      rhs_terms.append(rhs_term)

    lhs = tuple(_kronecker_product(lhs_terms))
    rhs = np.prod(rhs_terms)

    if lhs in all_constraints and all_constraints[lhs] != rhs:
      raise ValueError('conflicting constraints')
    all_constraints[lhs] = rhs

  # ensure a deterministic order for the rows (note: could drop this if when we
  # commit to Python 3.6+, due to dictionaries being ordered)
  lhs_rows, rhs_rows = zip(*sorted(all_constraints.items()))
  A = np.array(lhs_rows)  # pylint: disable=invalid-name
  b = np.array(rhs_rows)
  return A, b


def _high_order_coefficients_1d(
    stencil: np.ndarray,
    method: Method,
    derivative_order: int,
    grid_step: float = None,
) -> np.ndarray:
  """Calculate highest-order coefficients in 1D."""
  # Use the highest order accuracy we can ensure in general. (In some cases,
  # e.g., centered finite differences, this solution actually has higher order
  # accuracy.)
  accuracy_order = stencil.size - derivative_order
  A, b = constraints(  # pylint: disable=invalid-name
      [stencil], method, [derivative_order], accuracy_order, grid_step)
  return np.linalg.solve(A, b)


def coefficients(
    stencils: Sequence[np.ndarray],
    method: Method,
    derivative_orders: Sequence[int],
    accuracy_order: Optional[int] = None,
    grid_step: float = None,
) -> np.ndarray:
  """Calculate standard finite difference/volume coefficients.

  These coefficients are constructed by taking an outer product of coefficients
  along each dimension independently. The resulting coefficients have *at least*
  the requested accuracy order.

  Args:
    stencils: sequence of 1d stencils, one per grid dimension.
    method: discretization method (i.e., finite volumes or finite differences).
    derivative_orders: integer derivative orders to approximate, per grid
      dimension.
    accuracy_order: accuracy order for the solution. By default, the highest
      possible accuracy is used in each direction.
    grid_step: spacing between grid cells. Required if calculating a finite
      volume stencil.

  Returns:
    NumPy array with one-dimension per stencil giving first order finite
    difference coefficients on the grid.
  """
  slices = []
  sizes = []
  all_coefficients = []
  for stencil, derivative_order in zip(stencils, derivative_orders):
    if accuracy_order is None:
      excess = 0
    else:
      excess = stencil.size - derivative_order - accuracy_order
    start = excess // 2
    stop = stencil.size - excess // 2
    slice_ = slice(start, stop)
    axis_coefficients = _high_order_coefficients_1d(stencil[slice_], method,
                                                    derivative_order, grid_step)

    slices.append(slice_)
    sizes.append(stencil[slice_].size)
    all_coefficients.append(axis_coefficients)

  result = np.zeros(tuple(stencil.size for stencil in stencils))
  result[tuple(slices)] = _kronecker_product(all_coefficients).reshape(sizes)
  return result


class PolynomialAccuracy(tf.keras.layers.Layer):
  """Layer to enforce polynomial accuracy for finite difference coefficients.

  Attributes:
    input_size: length of input vectors that are transformed into valid finite
      difference coefficients.
    stencil_size: size of the resulting stencil.
    bias: numpy array of shape (grid_size,) to which zero vectors are mapped.
    nullspace: numpy array of shape (input_size, output_size) representing the
      nullspace of the constraint matrix.
  """

  def __init__(
      self,
      stencils: Sequence[np.ndarray],
      method: Method,
      derivative_orders: Sequence[int],
      accuracy_order: int = 1,
      bias_accuracy_order: Optional[int] = 1,
      grid_step: float = None,
      bias: np.ndarray = None,
      dtype: Any = np.float32,
  ):
    """Constructor.

    Args:
      stencils: sequence of 1d stencils, one per grid dimension.
      method: discretization method (i.e., finite volumes or finite
        differences).
      derivative_orders: integer derivative orders to approximate, per grid
        dimension.
      accuracy_order: integer order of polynomial accuracy to enforce.
      bias_accuracy_order: integer order of polynomial accuracy to use for the
        bias term. Only used if bias is not provided.
      grid_step: spacing between grid cells.
      bias: np.ndarray of shape (grid_size,) to which zero-vectors will be
        mapped. Must satisfy polynomial accuracy to the requested order. By
        default, we use standard low-order coefficients for the given grid.
      dtype: dtype to use for computing this layer.
    """
    if grid_step is None:
      raise TypeError('grid_step is required for PolynomialAccuracy')

    A, b = constraints(  # pylint: disable=invalid-name
        stencils, method, derivative_orders, accuracy_order, grid_step)

    if bias is None:
      bias_grid = coefficients(stencils, method, derivative_orders,
                               bias_accuracy_order, grid_step)
      bias = bias_grid.ravel()

    norm = np.linalg.norm(np.dot(A, bias) - b)
    if norm > 1e-8:
      raise ValueError('invalid bias, not in nullspace')  # pylint: disable=g-doc-exception

    # https://en.wikipedia.org/wiki/Kernel_(linear_algebra)#Nonhomogeneous_systems_of_linear_equations
    _, _, v = np.linalg.svd(A)
    input_size = A.shape[1] - A.shape[0]
    if not input_size:
      raise ValueError(  # pylint: disable=g-doc-exception
          'there is only one valid solution accurate to this order')

    # nullspace from the SVD is always normalized such that its singular values
    # are 1 or 0, which means it's actually independent of the grid spacing.
    nullspace = v[-input_size:]

    # ensure the nullspace is scaled comparably to the bias
    # TODO(shoyer): fix this for arbitrary spaced grids
    nullspace /= (grid_step**np.array(derivative_orders)).prod()

    self.input_size = input_size
    self.output_size = b.size
    self.nullspace = tf.convert_to_tensor(nullspace, dtype)
    self.bias = tf.convert_to_tensor(bias, dtype)

    super().__init__(trainable=False, dtype=dtype)

  def compute_output_shape(self, input_shape):
    return input_shape[:-1] + (self.output_size,)

  def call(self, x: tf.Tensor) -> tf.Tensor:
    # TODO(geraschenko): explore projecting out the nullspace from full rank
    # inputs instead.
    return self.bias + tf.tensordot(x, self.nullspace, axes=[-1, 0])


class PolynomialBias(tf.keras.layers.Layer):
  """Layer that just adds a polynomial bias."""

  def __init__(self, *args, **kwargs):
    dtype = kwargs.pop('dtype', np.float32)
    bias = coefficients(*args, **kwargs).ravel()
    self.bias = tf.convert_to_tensor(bias, dtype)
    self.input_size = bias.size
    super().__init__(trainable=False, dtype=dtype)

  def call(self, x: tf.Tensor) -> tf.Tensor:
    return self.bias + x


def constraint_layer(
    stencils: Sequence[np.ndarray],
    method: Method,
    derivative_orders: Sequence[int],
    constrained_accuracy_order: int = 1,
    initial_accuracy_order: Optional[int] = 1,
    grid_step: float = None,
    dtype: Any = np.float32,
) -> tf.keras.layers.Layer:
  """Create a Keras layer for enforcing polynomial accuracy constraints."""
  if constrained_accuracy_order:
    return PolynomialAccuracy(
        stencils,
        method,
        derivative_orders,
        accuracy_order=constrained_accuracy_order,
        bias_accuracy_order=initial_accuracy_order,
        grid_step=grid_step,
        dtype=dtype,
    )
  else:
    if constrained_accuracy_order != 0:
      raise ValueError('invalid constrained_accuracy_order')

    return PolynomialBias(
        stencils,
        method,
        derivative_orders,
        initial_accuracy_order,
        grid_step,
    )
