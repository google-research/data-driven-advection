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
"""End-to-end tests for data generation and model training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl.testing import flagsaver
from absl.testing import parameterized
import apache_beam as beam
from pde_superresolution_2d.core import models
from pde_superresolution_2d.core import readers
from pde_superresolution_2d.core import tensor_ops
from pde_superresolution_2d.pipelines import create_training_data
import tensorflow as tf

from absl.testing import absltest

FLAGS = flags.FLAGS

nest = tf.contrib.framework.nest


# Use eager mode by default
tf.enable_eager_execution()


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
        num_seeds=4):
      create_training_data.main([], runner=beam.runners.DirectRunner())

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
        .shuffle(10)
        .map(create_inputs)
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='mean_squared_error')
    model.fit(training_data, epochs=1, steps_per_epoch=5)
    model.evaluate(training_data, steps=2)

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
        .shuffle(10)
        .repeat()
        .batch(4, drop_remainder=True)
        .prefetch(1)
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='mean_squared_error')
    model.fit(training_data, epochs=1, steps_per_epoch=5)
    model.evaluate(training_data, steps=2)


if __name__ == '__main__':
  absltest.main()
