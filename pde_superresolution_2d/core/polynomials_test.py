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
"""Tests for polynomial finite differences."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest  # pylint: disable=g-bad-import-order
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from pde_superresolution_2d import metadata_pb2
from pde_superresolution_2d.core import polynomials

# this simplifies tests
tf.enable_eager_execution()

FINITE_DIFF = metadata_pb2.Equation.Discretization.FINITE_DIFFERENCE
FINITE_VOL = metadata_pb2.Equation.Discretization.FINITE_VOLUME


class PolynomialsTest(parameterized.TestCase):

  # For test-cases, see
  # https://en.wikipedia.org/wiki/Finite_difference_coefficient
  @parameterized.parameters(
      dict(stencil=[-1, 0, 1], derivative_order=1, expected=[-1 / 2, 0, 1 / 2]),
      dict(stencil=[-1, 0, 1], derivative_order=2, expected=[1, -2, 1]),
      dict(
          stencil=[-2, -1, 0, 1, 2],
          derivative_order=2,
          expected=[-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12]),
      dict(
          stencil=[-2, -1, 0, 1, 2],
          derivative_order=2,
          accuracy_order=1,
          expected=[0, 1, -2, 1, 0]),
      dict(stencil=[0, 1], derivative_order=1, expected=[-1, 1]),
      dict(stencil=[0, 2], derivative_order=1, expected=[-0.5, 0.5]),
      dict(stencil=[0, 0.5], derivative_order=1, expected=[-2, 2]),
      dict(
          stencil=[0, 1, 2, 3, 4],
          derivative_order=4,
          expected=[1, -4, 6, -4, 1]),
  )
  def test_finite_difference_coefficients_1d(self,
                                             stencil,
                                             derivative_order,
                                             expected,
                                             accuracy_order=None):
    result = polynomials.coefficients([np.array(stencil)], FINITE_DIFF,
                                      [derivative_order], accuracy_order)
    np.testing.assert_allclose(result, expected)

  @parameterized.parameters(
      dict(
          stencils=[[-0.5, 0.5], [-0.5, 0.5]],
          derivative_orders=[0, 0],
          expected=[[0.25, 0.25], [0.25, 0.25]]),
      dict(
          stencils=[[-0.5, 0.5], [-0.5, 0.5]],
          derivative_orders=[0, 1],
          expected=[[-0.5, 0.5], [-0.5, 0.5]]),
      dict(
          stencils=[[-0.5, 0.5], [-0.5, 0.5]],
          derivative_orders=[1, 1],
          expected=[[1, -1], [-1, 1]]),
      dict(
          stencils=[[-1, 0, 1], [-0.5, 0.5]],
          derivative_orders=[1, 0],
          expected=[[-0.25, -0.25], [0, 0], [0.25, 0.25]]),
  )
  def test_finite_difference_coefficients_2d(self, stencils, derivative_orders,
                                             expected):
    args = ([np.array(s) for s in stencils], FINITE_DIFF, derivative_orders)
    result = polynomials.coefficients(*args)
    np.testing.assert_allclose(result, expected)

    result = polynomials.coefficients(*args, accuracy_order=1)
    np.testing.assert_allclose(result, expected)

  # based in part on the WENO tutorial
  @parameterized.parameters(
      dict(stencil=[-0.5, 0.5], derivative_order=0, expected=[1 / 2, 1 / 2]),
      dict(
          stencil=[-1.5, -0.5, 0.5, 1.5],
          derivative_order=0,
          accuracy_order=1,
          expected=[0, 1 / 2, 1 / 2, 0]),
      dict(stencil=[-1, 1], derivative_order=0, expected=[1 / 2, 1 / 2]),
      dict(stencil=[-1.5, -0.5], derivative_order=0, expected=[-1 / 2, 3 / 2]),
      dict(
          stencil=[-0.5, 0.5, 1.5],
          derivative_order=0,
          expected=[1 / 3, 5 / 6, -1 / 6]),
      dict(stencil=[-0.5, 0.5], derivative_order=1, expected=[-1, 1]),
      dict(stencil=[-1, 1], derivative_order=1, expected=[-1 / 2, 1 / 2]),
      dict(stencil=[-1, 0, 1], derivative_order=1, expected=[-1 / 2, 0, 1 / 2]),
      dict(stencil=[0.5, 1.5, 2.5], derivative_order=1, expected=[-2, 3, -1]),
      dict(
          stencil=[-1.5, -0.5, 0.5, 1.5],
          derivative_order=1,
          expected=[1 / 12, -5 / 4, 5 / 4, -1 / 12]),
      dict(
          stencil=[-.75, -0.25, 0.25, 0.75],
          derivative_order=1,
          expected=[1 / 6, -5 / 2, 5 / 2, -1 / 6]),
  )
  def test_finite_volume_coefficients_1d(self,
                                         stencil,
                                         derivative_order,
                                         expected,
                                         accuracy_order=None):
    step = stencil[1] - stencil[0]
    result = polynomials.coefficients([np.array(stencil)],
                                      FINITE_VOL, [derivative_order],
                                      accuracy_order,
                                      grid_step=step)
    np.testing.assert_allclose(result, expected)

  @parameterized.parameters(
      dict(
          accuracy_order=1,
          method=FINITE_DIFF,
          expected_a=[[1, 1]],
          expected_b=[1]),
      dict(
          accuracy_order=1,
          method=FINITE_VOL,
          expected_a=[[1, 1]],
          expected_b=[1]),
      dict(
          accuracy_order=2,
          method=FINITE_DIFF,
          expected_a=[[-1 / 2, 1 / 2], [1, 1]],
          expected_b=[0, 1]),
      dict(
          accuracy_order=2,
          method=FINITE_VOL,
          expected_a=[[-1 / 2, 1 / 2], [1, 1]],
          expected_b=[0, 1]),
  )
  def test_constraints_1d(self, accuracy_order, method, expected_a, expected_b):
    a, b = polynomials.constraints([np.array([-0.5, 0.5])],
                                   method,
                                   derivative_orders=[0],
                                   accuracy_order=accuracy_order,
                                   grid_step=1.0)
    np.testing.assert_allclose(a, expected_a)
    np.testing.assert_allclose(b, expected_b)

  def test_constraints_2d(self):
    # these constraints should be under-determined.
    stencils = [np.array([-0.5, 0.5])] * 2
    A, b = polynomials.constraints(  # pylint: disable=invalid-name
        stencils,
        FINITE_DIFF,
        derivative_orders=[0, 0],
        accuracy_order=2)
    # three constraints, for each term in [1, x, y]
    self.assertEqual(A.shape, (3, 4))
    self.assertEqual(b.shape, (3,))
    # explicitly test two valid solutions
    np.testing.assert_allclose(A.dot([1 / 4, 1 / 4, 1 / 4, 1 / 4]), b)
    np.testing.assert_allclose(A.dot([4 / 10, 1 / 10, 1 / 10, 4 / 10]), b)

  @parameterized.parameters(
      dict(stencil=[-2, -1, 0, 1, 2], method=FINITE_DIFF, derivative_order=1),
      dict(stencil=[-2, -1, 0, 1, 2], method=FINITE_DIFF, derivative_order=2),
      dict(
          stencil=[-1.5, -0.5, 0.5, 1.5],
          method=FINITE_DIFF,
          derivative_order=1),
      dict(
          stencil=[-1.5, -0.5, 0.5, 1.5], method=FINITE_VOL,
          derivative_order=1),
  )
  def test_polynomial_accuracy_layer_consistency_1d(self,
                                                    stencil,
                                                    method,
                                                    derivative_order,
                                                    accuracy_order=2):
    kwargs = dict(
        stencils=[np.array(stencil)],
        method=method,
        derivative_orders=[derivative_order],
        accuracy_order=accuracy_order,
        grid_step=1.0,
    )
    A, b = polynomials.constraints(**kwargs)  # pylint: disable=invalid-name
    layer = polynomials.PolynomialAccuracy(**kwargs)

    inputs = np.random.RandomState(0).randn(10, layer.input_size)
    outputs = layer(inputs.astype(np.float32))

    residual = np.einsum('ij,...j->...i', A, outputs) - b
    np.testing.assert_allclose(residual, 0, atol=1e-6)

  @parameterized.parameters(
      dict(
          stencils=[[-1, 0, 1], [-1.5, -0.5, 0.5, 1.5]],
          method=FINITE_VOL,
          derivative_orders=[0, 1]),
      dict(
          stencils=[[-1.5, -0.5, 0.5, 1.5]] * 2,
          method=FINITE_DIFF,
          derivative_orders=[1, 2]),
  )
  def test_polynomial_accuracy_layer_consistency_2d(self,
                                                    stencils,
                                                    method,
                                                    derivative_orders,
                                                    accuracy_order=2):
    kwargs = dict(
        stencils=[np.array(x) for x in stencils],
        method=method,
        derivative_orders=derivative_orders,
        accuracy_order=accuracy_order,
        grid_step=1.0,
    )
    A, b = polynomials.constraints(**kwargs)  # pylint: disable=invalid-name
    layer = polynomials.PolynomialAccuracy(**kwargs)

    inputs = np.random.RandomState(0).randn(10, 10, layer.input_size)
    outputs = layer(inputs.astype(np.float32))

    residual = np.einsum('ij,...j->...i', A, outputs) - b
    np.testing.assert_allclose(residual, 0, atol=1e-5)

  @parameterized.parameters(
      dict(derivative_order=0, offset=0, expected=[0]),
      dict(derivative_order=1, offset=0, expected=[-1, 0, 1]),
      dict(derivative_order=2, offset=0, expected=[-1, 0, 1]),
      dict(derivative_order=3, offset=0, expected=[-2, -1, 0, 1, 2]),
      dict(derivative_order=4, offset=0, expected=[-2, -1, 0, 1, 2]),
      dict(derivative_order=0, offset=1, expected=[-0.5, 0.5]),
      dict(derivative_order=1, offset=1, expected=[-0.5, 0.5]),
      dict(derivative_order=2, offset=1, expected=[-1.5, -0.5, 0.5, 1.5]),
      dict(derivative_order=3, offset=1, expected=[-1.5, -0.5, 0.5, 1.5]),
      dict(
          derivative_order=0,
          accuracy_order=6,
          offset=0,
          expected=[-3, -2, -1, 0, 1, 2, 3]),
      dict(
          derivative_order=0,
          accuracy_order=6,
          offset=1,
          expected=[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]),
  )
  def test_regular_stencil_1d(self,
                              offset,
                              derivative_order,
                              expected,
                              accuracy_order=1):
    actual = polynomials.regular_stencil_1d(offset, derivative_order,
                                            accuracy_order)
    np.testing.assert_allclose(actual, expected)


if __name__ == '__main__':
  absltest.main()
