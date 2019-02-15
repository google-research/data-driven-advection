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

from absl import flags
from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np
from pde_superresolution_2d import metadata_pb2
from pde_superresolution_2d.core import equations
from pde_superresolution_2d.core import grids
from pde_superresolution_2d.core import models
from pde_superresolution_2d.core import readers
from pde_superresolution_2d.core import states
from pde_superresolution_2d.core import tensor_ops
from pde_superresolution_2d.pipelines import create_training_data
import tensorflow as tf

from absl.testing import absltest

FLAGS = flags.FLAGS

nest = tf.contrib.framework.nest


# Use eager mode by default
tf.enable_eager_execution()


C = states.StateKey('concentration', (0, 0, 0), (0, 0))
C_EDGE_X = states.StateKey('concentration', (0, 0, 0), (1, 0))
C_EDGE_Y = states.StateKey('concentration', (0, 0, 0), (0, 1))
C_X = states.StateKey('concentration', (1, 0, 0), (0, 0))
C_Y = states.StateKey('concentration', (0, 1, 0), (0, 0))
C_X_EDGE_X = states.StateKey('concentration', (1, 0, 0), (1, 0))
C_Y_EDGE_Y = states.StateKey('concentration', (0, 1, 0), (0, 1))
C_XX = states.StateKey('concentration', (2, 0, 0), (0, 0))
C_YY = states.StateKey('concentration', (0, 2, 0), (0, 0))


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
      STATE_KEYS = (C,)
      INPUT_KEYS = (C, C_EDGE_X, C_EDGE_Y, C_X, C_Y, C_X_EDGE_X,
                    C_Y_EDGE_Y, C_XX, C_YY)
      METHOD = metadata_pb2.Equation.Discretization.FINITE_DIFFERENCE

    grid = grids.Grid.from_period(10, length=1)
    equation = Equation()
    model = model_cls(equation, grid, **model_kwargs)

    inputs = tf.convert_to_tensor(
        np.random.RandomState(0).random_sample((1,) + grid.shape), tf.float32)

    # create variables, then reset them all to zero
    model.spatial_derivatives({C: inputs})
    for variable in model.variables:
      variable.assign(tf.zeros_like(variable))

    actual_derivatives = model.spatial_derivatives({C: inputs})

    expected_derivatives = {
        C: inputs,
        C_EDGE_X: (inputs + tensor_ops.roll_2d(inputs, (-1, 0))) / 2,
        C_EDGE_Y: (inputs + tensor_ops.roll_2d(inputs, (0, -1))) / 2,
        C_X: (-tensor_ops.roll_2d(inputs, (1, 0))
              + tensor_ops.roll_2d(inputs, (-1, 0))) / (2 * grid.step),
        C_Y: (-tensor_ops.roll_2d(inputs, (0, 1))
              + tensor_ops.roll_2d(inputs, (0, -1))) / (2 * grid.step),
        C_X_EDGE_X: (-inputs + tensor_ops.roll_2d(inputs, (-1, 0))) / grid.step,
        C_Y_EDGE_Y: (-inputs + tensor_ops.roll_2d(inputs, (0, -1))) / grid.step,
        C_XX: (tensor_ops.roll_2d(inputs, (1, 0))
               - 2 * inputs
               + tensor_ops.roll_2d(inputs, (-1, 0))) / grid.step ** 2,
        C_YY: (tensor_ops.roll_2d(inputs, (0, 1))
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
      STATE_KEYS = (C_EDGE_X,)
      INPUT_KEYS = (C, C_EDGE_X, C_EDGE_Y, C_X, C_X_EDGE_X, C_XX)
      METHOD = metadata_pb2.Equation.Discretization.FINITE_DIFFERENCE

    grid = grids.Grid.from_period(10, length=1)
    equation = Equation()
    model = model_cls(equation, grid, **model_kwargs)

    inputs = tf.convert_to_tensor(
        np.random.RandomState(0).random_sample((1,) + grid.shape), tf.float32)

    # create variables, then reset them all to zero
    model.spatial_derivatives({C_EDGE_X: inputs})
    for variable in model.variables:
      variable.assign(tf.zeros_like(variable))

    actual_derivatives = model.spatial_derivatives({C_EDGE_X: inputs})

    expected_derivatives = {
        C: (tensor_ops.roll_2d(inputs, (1, 0)) + inputs) / 2,
        C_EDGE_X: inputs,
        C_EDGE_Y: (
            inputs
            + tensor_ops.roll_2d(inputs, (0, -1))
            + tensor_ops.roll_2d(inputs, (1, 0))
            + tensor_ops.roll_2d(inputs, (1, -1))
        ) / 4,
        C_X: (-tensor_ops.roll_2d(inputs, (1, 0)) + inputs) / grid.step,
        C_X_EDGE_X: (
            -tensor_ops.roll_2d(inputs, (1, 0))
            + tensor_ops.roll_2d(inputs, (-1, 0))) / (2 * grid.step),
        C_XX: (1/2 * tensor_ops.roll_2d(inputs, (2, 0))
               - 1/2 * tensor_ops.roll_2d(inputs, (1, 0))
               - 1/2 * inputs
               + 1/2 * tensor_ops.roll_2d(inputs, (-1, 0))) / grid.step ** 2,
    }

    for key, expected in sorted(expected_derivatives.items()):
      np.testing.assert_allclose(
          actual_derivatives[key], expected,
          atol=1e-5, rtol=1e-5, err_msg=repr(key))


class IntegrationTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    # create training data
    output_path = FLAGS.test_tmpdir
    output_name = 'temp'
    with flagsaver.flagsaver(
        dataset_path=output_path,
        dataset_name=output_name,
        dataset_type='time_evolution',
        num_shards=1,
        total_time_steps=5,
        example_time_steps=8,
        time_step_interval=1,
        num_seeds=10):
      create_training_data.main([])

    metadata_path = '{}/{}.metadata'.format(output_path, output_name)
    cls.metadata = readers.load_metadata(metadata_path)
    super(IntegrationTest, cls).setUpClass()

  @parameterized.parameters(
      dict(model_cls=models.FiniteDifferenceModel),
      dict(model_cls=models.LinearModel),
      dict(model_cls=models.PseudoLinearModel),
      dict(model_cls=models.NonlinearModel),
      dict(model_cls=models.DirectModel),
  )
  def test_training(self, model_cls):
    # a basic integration test
    equation = readers.get_equation(self.metadata)
    grid = readers.get_output_grid(self.metadata)
    model = model_cls(equation, grid)

    def create_inputs(state):
      inputs = nest.map_structure(lambda x: x[:-1], state)
      labels = state['concentration'][1:]
      return inputs, labels

    training_data = (
        model.load_data(self.metadata)
        .repeat()
        .shuffle(100)
        .map(create_inputs)
    )
    model.compile(optimizer=tf.train.AdamOptimizer(1e-4),
                  loss='mean_squared_error')
    model.fit(training_data, epochs=1, steps_per_epoch=100)
    model.evaluate(training_data, steps=10)

  @parameterized.parameters(
      dict(model_cls=models.LinearModel),
      dict(model_cls=models.PseudoLinearModel),
      dict(model_cls=models.NonlinearModel),
  )
  def test_training_multiple_times(self, model_cls):
    # a basic integration test
    equation = readers.get_equation(self.metadata)
    grid = readers.get_output_grid(self.metadata)
    model = model_cls(equation, grid, num_time_steps=4)

    def create_inputs(state):
      # (batch, x, y)
      inputs = nest.map_structure(lambda x: x[:-model.num_time_steps], state)
      # (batch, time, x, y)
      labels = tensor_ops.stack_all_contiguous_slices(
          state['concentration'][1:], model.num_time_steps, new_axis=1)
      return inputs, labels

    training_data = (
        model.load_data(self.metadata)
        .map(create_inputs)
        .apply(tf.data.experimental.unbatch())
        .shuffle(100)
        .repeat()
        .batch(8, drop_remainder=True)
        .prefetch(1)
    )
    model.compile(optimizer=tf.train.AdamOptimizer(1e-4),
                  loss='mean_squared_error')
    model.fit(training_data, epochs=1, steps_per_epoch=100)
    model.evaluate(training_data, steps=10)

if __name__ == '__main__':
  absltest.main()
