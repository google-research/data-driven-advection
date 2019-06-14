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
"""Test of models classes in models.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
from pde_superresolution_2d.advection import equations as advection_equations
from pde_superresolution_2d.core import equations
from pde_superresolution_2d.core import geometry
from pde_superresolution_2d.core import grids
from pde_superresolution_2d.core import models
from pde_superresolution_2d.core import polynomials
from pde_superresolution_2d.core import states
from pde_superresolution_2d.core import tensor_ops
import tensorflow as tf

from absl.testing import absltest


# Use eager mode by default
tf.enable_eager_execution()


StateDef = states.StateDefinition

NO_DERIVATIVES = (0, 0, 0)
D_X = (1, 0, 0)
D_Y = (0, 1, 0)
D_XX = (2, 0, 0)
D_YY = (0, 2, 0)

NO_OFFSET = (0, 0)
X_PLUS_HALF = (1, 0)
Y_PLUS_HALF = (0, 1)


class BuildStencilsTest(absltest.TestCase):

  def assert_sequences_allclose(self, a, b):
    self.assertEqual(len(a), len(b))
    for a_entry, b_entry in zip(a, b):
      np.testing.assert_allclose(a_entry, b_entry)

  def test_build_stencils(self):
    sk00 = StateDef(name='foo', tensor_indices=(), derivative_orders=(0, 0, 0),
                    offset=(0, 0))
    sk10 = StateDef(name='foo', tensor_indices=(), derivative_orders=(0, 0, 0),
                    offset=(1, 0))
    self.assert_sequences_allclose([[-1, 0, 1], [-1, 0, 1]],
                                   models.build_stencils(sk00, sk00, 3, 1.0))
    self.assert_sequences_allclose([[-2, 0, 2], [-2, 0, 2]],
                                   models.build_stencils(sk00, sk00, 3, 2.0))
    self.assert_sequences_allclose([[-1, 0, 1], [-1, 0, 1]],
                                   models.build_stencils(sk00, sk00, 4, 1.0))
    self.assert_sequences_allclose([[-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2]],
                                   models.build_stencils(sk00, sk00, 5, 1.0))
    self.assert_sequences_allclose([[-0.5, 0.5], [-1, 0, 1]],
                                   models.build_stencils(sk00, sk10, 3, 1.0))
    self.assert_sequences_allclose([[-0.5, 0.5], [-1, 0, 1]],
                                   models.build_stencils(sk10, sk00, 3, 1.0))


class FiniteDifferenceModelTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(model_cls=models.FiniteDifferenceModel, model_kwargs={}),
      dict(model_cls=models.LinearModel,
           model_kwargs=dict(constrained_accuracy_order=0)),
      dict(model_cls=models.LinearModel,
           model_kwargs=dict(constrained_accuracy_order=1)),
      dict(model_cls=models.PseudoLinearModel,
           model_kwargs=dict(constrained_accuracy_order=0)),
      dict(model_cls=models.PseudoLinearModel,
           model_kwargs=dict(constrained_accuracy_order=1)),
  )
  def test_from_centered(self, model_cls, model_kwargs):

    class Equation(equations.Equation):
      METHOD = polynomials.Method.FINITE_DIFFERENCE

      def __init__(self):
        self.key_definitions = {
            'c': StateDef('concentration', (), NO_DERIVATIVES, NO_OFFSET),
            'c_edge_x':
                StateDef('concentration', (), NO_DERIVATIVES, X_PLUS_HALF),
            'c_edge_y':
                StateDef('concentration', (), NO_DERIVATIVES, Y_PLUS_HALF),
            'c_x': StateDef('concentration', (), D_X, NO_OFFSET),
            'c_y': StateDef('concentration', (), D_Y, NO_OFFSET),
            'c_x_edge_x': StateDef('concentration', (), D_X, X_PLUS_HALF),
            'c_y_edge_y': StateDef('concentration', (), D_Y, Y_PLUS_HALF),
            'c_xx': StateDef('concentration', (), D_XX, NO_OFFSET),
            'c_yy': StateDef('concentration', (), D_YY, NO_OFFSET),
        }
        self.evolving_keys = {'c'}
        self.constant_keys = set()

    grid = grids.Grid.from_period(10, length=1)
    equation = Equation()
    model = model_cls(equation, grid, **model_kwargs)

    inputs = tf.convert_to_tensor(
        np.random.RandomState(0).random_sample((1,) + grid.shape), tf.float32)

    # create variables, then reset them all to zero
    model.spatial_derivatives({'c': inputs})
    for variable in model.variables:
      variable.assign(tf.zeros_like(variable))

    actual_derivatives = model.spatial_derivatives({'c': inputs})

    expected_derivatives = {
        'c': inputs,
        'c_edge_x': (inputs + tensor_ops.roll_2d(inputs, (-1, 0))) / 2,
        'c_edge_y': (inputs + tensor_ops.roll_2d(inputs, (0, -1))) / 2,
        'c_x': (-tensor_ops.roll_2d(inputs, (1, 0))
                + tensor_ops.roll_2d(inputs, (-1, 0))) / (2 * grid.step),
        'c_y': (-tensor_ops.roll_2d(inputs, (0, 1))
                + tensor_ops.roll_2d(inputs, (0, -1))) / (2 * grid.step),
        'c_x_edge_x': (
            -inputs + tensor_ops.roll_2d(inputs, (-1, 0))) / grid.step,
        'c_y_edge_y': (
            -inputs + tensor_ops.roll_2d(inputs, (0, -1))) / grid.step,
        'c_xx': (tensor_ops.roll_2d(inputs, (1, 0))
                 - 2 * inputs
                 + tensor_ops.roll_2d(inputs, (-1, 0))) / grid.step ** 2,
        'c_yy': (tensor_ops.roll_2d(inputs, (0, 1))
                 - 2 * inputs
                 + tensor_ops.roll_2d(inputs, (0, -1))) / grid.step ** 2,
    }

    for key, expected in sorted(expected_derivatives.items()):
      np.testing.assert_allclose(
          actual_derivatives[key], expected,
          atol=1e-5, rtol=1e-5, err_msg=repr(key))

  @parameterized.parameters(
      dict(model_cls=models.FiniteDifferenceModel, model_kwargs={}),
      dict(model_cls=models.LinearModel,
           model_kwargs=dict(constrained_accuracy_order=0)),
      dict(model_cls=models.LinearModel,
           model_kwargs=dict(constrained_accuracy_order=1)),
      dict(model_cls=models.PseudoLinearModel,
           model_kwargs=dict(constrained_accuracy_order=0)),
      dict(model_cls=models.PseudoLinearModel,
           model_kwargs=dict(constrained_accuracy_order=1)),
  )
  def test_from_edge(self, model_cls, model_kwargs):

    class Equation(equations.Equation):
      METHOD = polynomials.Method.FINITE_DIFFERENCE

      def __init__(self):
        self.key_definitions = {
            'c': StateDef('concentration', (), NO_DERIVATIVES, NO_OFFSET),
            'c_edge_x':
                StateDef('concentration', (), NO_DERIVATIVES, X_PLUS_HALF),
            'c_edge_y':
                StateDef('concentration', (), NO_DERIVATIVES, Y_PLUS_HALF),
            'c_x': StateDef('concentration', (), D_X, NO_OFFSET),
            'c_x_edge_x': StateDef('concentration', (), D_X, X_PLUS_HALF),
            'c_xx': StateDef('concentration', (), D_XX, NO_OFFSET),
        }
        self.evolving_keys = {'c_edge_x'}
        self.constant_keys = set()

    grid = grids.Grid.from_period(10, length=1)
    equation = Equation()
    model = model_cls(equation, grid, **model_kwargs)

    inputs = tf.convert_to_tensor(
        np.random.RandomState(0).random_sample((1,) + grid.shape), tf.float32)

    # create variables, then reset them all to zero
    model.spatial_derivatives({'c_edge_x': inputs})
    for variable in model.variables:
      variable.assign(tf.zeros_like(variable))

    actual_derivatives = model.spatial_derivatives({'c_edge_x': inputs})

    expected_derivatives = {
        'c': (tensor_ops.roll_2d(inputs, (1, 0)) + inputs) / 2,
        'c_edge_x': inputs,
        'c_edge_y': (
            inputs
            + tensor_ops.roll_2d(inputs, (0, -1))
            + tensor_ops.roll_2d(inputs, (1, 0))
            + tensor_ops.roll_2d(inputs, (1, -1))
        ) / 4,
        'c_x': (-tensor_ops.roll_2d(inputs, (1, 0)) + inputs) / grid.step,
        'c_x_edge_x': (
            -tensor_ops.roll_2d(inputs, (1, 0))
            + tensor_ops.roll_2d(inputs, (-1, 0))) / (2 * grid.step),
        'c_xx': (1/2 * tensor_ops.roll_2d(inputs, (2, 0))
                 - 1/2 * tensor_ops.roll_2d(inputs, (1, 0))
                 - 1/2 * inputs
                 + 1/2 * tensor_ops.roll_2d(inputs, (-1, 0))) / grid.step ** 2,
    }

    for key, expected in sorted(expected_derivatives.items()):
      np.testing.assert_allclose(
          actual_derivatives[key], expected,
          atol=1e-5, rtol=1e-5, err_msg=repr(key))


class FiniteDifferenceDiffusionEquation(equations.Equation):
  METHOD = polynomials.Method.FINITE_DIFFERENCE

  def __init__(self):
    self.key_definitions = {
        'c': StateDef('concentration', (), NO_DERIVATIVES, NO_OFFSET),
        'c_xx': StateDef('concentration', (), D_XX, NO_OFFSET),
        'c_yy': StateDef('concentration', (), D_YY, NO_OFFSET),
    }
    self.evolving_keys = {'c'}
    self.constant_keys = set()
    super(FiniteDifferenceDiffusionEquation, self).__init__()

  def time_derivative(self, grid, c, c_xx, c_yy):
    del grid, c  # unused
    return {'c': c_xx + c_yy}


class FiniteVolumeDiffusionEquation(equations.Equation):
  METHOD = polynomials.Method.FINITE_DIFFERENCE

  def __init__(self):
    self.key_definitions = {
        'c': StateDef('concentration', (), NO_DERIVATIVES, NO_OFFSET),
        'c_x': StateDef('concentration', (), D_X, X_PLUS_HALF),
        'c_y': StateDef('concentration', (), D_Y, Y_PLUS_HALF),
    }
    self.evolving_keys = {'c'}
    self.constant_keys = set()
    super(FiniteVolumeDiffusionEquation, self).__init__()

  def time_derivative(self, grid, c, c_x, c_y):
    del grid, c  # unused
    c_xx = c_x - tensor_ops.roll_2d(c_x, (1, 0))
    c_yy = c_y - tensor_ops.roll_2d(c_y, (0, 1))
    return {'c': c_xx + c_yy}


class RotationalInvarianceTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(equation=FiniteDifferenceDiffusionEquation(),
           model_cls=models.FiniteDifferenceModel),
      dict(equation=FiniteVolumeDiffusionEquation(),
           model_cls=models.FiniteDifferenceModel),
      dict(equation=advection_equations.FiniteVolumeAdvection(),
           model_cls=models.FiniteDifferenceModel),
      dict(equation=FiniteDifferenceDiffusionEquation(),
           model_cls=models.PseudoLinearModel,
           predict_permutations=True),
      dict(equation=FiniteDifferenceDiffusionEquation(),
           model_cls=models.PseudoLinearModel,
           predict_permutations=False),
      dict(equation=FiniteVolumeDiffusionEquation(),
           model_cls=models.PseudoLinearModel,
           predict_permutations=True),
      dict(equation=FiniteVolumeDiffusionEquation(),
           model_cls=models.PseudoLinearModel,
           predict_permutations=False),
      dict(equation=advection_equations.FiniteVolumeAdvection(),
           model_cls=models.PseudoLinearModel,
           predict_permutations=True),
      dict(equation=advection_equations.FiniteVolumeAdvection(),
           model_cls=models.PseudoLinearModel,
           predict_permutations=False),
  )
  def test_rotation_invariance(self, equation, model_cls, **model_kwargs):
    # even untrained models should be rotationally invariant
    grid = grids.Grid.from_period(4, length=1)
    symmetries = geometry.symmetries_of_the_square(equation.key_definitions)
    if model_cls is models.PseudoLinearModel:
      model_kwargs.update(geometric_transforms=symmetries)
    model = model_cls(equation, grid, num_time_steps=2, **model_kwargs)

    rs = np.random.RandomState(0)
    inputs = {
        k: tf.convert_to_tensor(rs.random_sample((1,) + grid.shape), tf.float32)
        for k in equation.base_keys
    }
    for forward_model in [model.spatial_derivatives, model.time_derivative]:
      with self.subTest(forward_model):
        expected = forward_model(inputs)
        self.assertNotEmpty(expected)

        for transform in symmetries:
          actual = transform.inverse(forward_model(transform.forward(inputs)))
          for k in sorted(expected):
            np.testing.assert_allclose(
                actual[k].numpy(), expected[k].numpy(), atol=1e-5, rtol=1e-5,
                err_msg='{}: {}'.format(k, transform))


if __name__ == '__main__':
  absltest.main()
