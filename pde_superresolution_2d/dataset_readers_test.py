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
"""Test for dataset readers.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

from absl import flags
from absl.testing import flagsaver
import numpy as np
import tensorflow as tf

from tensorflow import gfile
from absl.testing import absltest
from pde_superresolution_2d import create_training_data
from pde_superresolution_2d import dataset_readers
from pde_superresolution_2d import equations
from pde_superresolution_2d import metadata_pb2
from pde_superresolution_2d import models
from pde_superresolution_2d import states


FLAGS = flags.FLAGS


class WriteReadDataTest(absltest.TestCase):

  def test_shapes_and_exceptions(self):
    """Dataset writer and reader test, checks shapes and exceptions."""
    output_path = FLAGS.test_tmpdir
    output_name = 'temp'
    equation_name = 'advection_diffusion'
    equation_scheme = 'FINITE_VOLUME'
    model_type = 'roll_finite_difference'
    dataset_type = 'all_derivatives'
    high_resolution = 125
    low_resolution = 25
    shards = 2
    batch_size = 3
    diffusion_const = 0.3

    scheme = metadata_pb2.Equation.DiscretizationScheme.Value(equation_scheme)
    expected_equation_type = equations.EQUATION_TYPES[
        (equation_name, scheme)]

    # create a temporary dataset
    with flagsaver.flagsaver(
        dataset_path=output_path,
        dataset_name=output_name,
        equation_name=equation_name,
        equation_scheme=equation_scheme,
        baseline_model_type=model_type,
        high_resolution_points=high_resolution,
        low_resolution_points=low_resolution,
        diffusion_const=diffusion_const,
        dataset_type=dataset_type,
        num_shards=shards,
        max_time=0.04,
        num_time_slices=10,
        num_samples=3
    ):
      create_training_data.main([])

    metadata_path = os.path.join(output_path, output_name + '.metadata')
    self.assertTrue(gfile.Exists(metadata_path))
    dataset_metadata = dataset_readers.load_metadata(metadata_path)
    low_res_grid = dataset_readers.get_low_res_grid(dataset_metadata)
    high_res_grid = dataset_readers.get_high_res_grid(dataset_metadata)
    equation = dataset_readers.get_equation(dataset_metadata)
    baseline_model = dataset_readers.get_baseline_model(dataset_metadata)

    self.assertEqual(low_res_grid.size_x, low_resolution)
    self.assertEqual(low_res_grid.size_y, low_resolution)
    self.assertEqual(high_res_grid.size_x, high_resolution)
    self.assertEqual(high_res_grid.size_y, high_resolution)
    self.assertAlmostEqual(high_res_grid.step, 2 * np.pi / high_resolution)
    self.assertAlmostEqual(equation.diffusion_const, diffusion_const)
    self.assertIsInstance(equation, expected_equation_type)
    self.assertIsInstance(baseline_model, models.MODEL_TYPES[model_type])

    valid_data_keys = ((states.C,), (states.C_EDGE_X, states.C_Y_EDGE_Y))
    invalid_data_keys = ((states.C, states.C_X), (states.C_EDGE_X,))
    valid_data_grids = (low_res_grid, low_res_grid)
    invalid_data_grids = (low_res_grid, high_res_grid)

    with self.assertRaises(ValueError):
      dataset_readers.initialize_dataset(
          dataset_metadata, invalid_data_keys, valid_data_grids)
    with self.assertRaises(ValueError):
      dataset_readers.initialize_dataset(
          dataset_metadata, valid_data_keys, invalid_data_grids)
    with self.assertRaises(ValueError):
      dataset_readers.initialize_dataset(
          dataset_metadata, invalid_data_keys, invalid_data_grids)

    dataset = dataset_readers.initialize_dataset(
        dataset_metadata, valid_data_keys, valid_data_grids)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)

    iterator = tf.data.Iterator.from_structure(
        dataset.output_types, dataset.output_shapes)
    iterator_initializer = iterator.make_initializer(dataset)

    with tf.Session() as sess:
      sess.run(iterator_initializer)
      first_state, second_state = sess.run(iterator.get_next())
    self.assertEqual(set(first_state.keys()), set(valid_data_keys[0]))
    self.assertEqual(set(second_state.keys()), set(valid_data_keys[1]))
    first_state_shape = np.shape(first_state[valid_data_keys[0][0]])
    second_state_shape = np.shape(second_state[valid_data_keys[1][0]])
    expected_shape = (batch_size, low_resolution, low_resolution)
    self.assertEqual(first_state_shape, expected_shape)
    self.assertEqual(second_state_shape, expected_shape)

  def test_statistics(self):
    """Dataset writer and reader test, checks statistics computations."""
    output_path = FLAGS.test_tmpdir
    output_name = 'temp'

    equation_name = 'advection_diffusion'
    equation_scheme = 'FINITE_VOLUME'

    # create a temporary dataset
    with flagsaver.flagsaver(
        dataset_path=output_path,
        dataset_name=output_name,
        equation_name=equation_name,
        equation_scheme=equation_scheme,
        baseline_model_type='roll_finite_difference',
        high_resolution_points=100,
        low_resolution_points=25,
        diffusion_const=0.3,
        dataset_type='all_derivatives',
        num_shards=1,
        max_time=0.04,
        num_time_slices=10,
        num_samples=3
    ):
      create_training_data.main([])

    metadata_path = os.path.join(output_path, output_name + '.metadata')
    dataset_metadata = dataset_readers.load_metadata(metadata_path)
    low_res_grid = dataset_readers.get_low_res_grid(dataset_metadata)

    data_keys = ((states.C,),)
    data_grids = (low_res_grid,)

    dataset = dataset_readers.initialize_dataset(
        dataset_metadata, data_keys, data_grids)
    dataset = dataset.repeat(1)
    dataset = dataset.batch(1)

    iterator = tf.data.Iterator.from_structure(
        dataset.output_types, dataset.output_shapes)
    iterator_initializer = iterator.make_initializer(dataset)
    state = iterator.get_next()

    all_data = []
    with tf.Session() as sess:
      sess.run(iterator_initializer)
      try:
        while True:
          all_data.append(sess.run(state[0][states.C]).flatten())
      except tf.errors.OutOfRangeError:
        all_data = np.concatenate(all_data)

    expected_mean = np.mean(all_data)
    expected_variance = np.std(all_data) ** 2

    metadata_mean = dataset_metadata.input_mean
    metadata_variance = dataset_metadata.input_variance

    np.testing.assert_allclose(metadata_mean, expected_mean, atol=1e-3)
    np.testing.assert_allclose(metadata_variance, expected_variance, atol=1e-3)


if __name__ == '__main__':
  absltest.main()
