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
"""Test for dataset readers.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

from absl import flags
from absl.testing import flagsaver
import apache_beam as beam
import numpy as np

from pde_superresolution_2d.advection import equations as advection_equations
from pde_superresolution_2d.core import readers
from pde_superresolution_2d.pipelines import create_training_data
import tensorflow as tf
from tensorflow.io import gfile
from absl.testing import absltest


# dataset writing needs to be happen in eager mode
tf.enable_eager_execution()

FLAGS = flags.FLAGS


class WriteReadDataTest(absltest.TestCase):

  def test_shapes_and_exceptions(self):
    """Dataset writer and reader test, checks shapes and exceptions."""
    output_path = FLAGS.test_tmpdir
    output_name = 'temp'
    equation_name = 'advection_diffusion'
    discretization = 'finite_volume'
    dataset_type = 'all_derivatives'
    high_resolution = 125
    low_resolution = 25
    shards = 2
    example_time_steps = 3
    batch_size = 4
    diffusion_coefficient = 0.3

    expected_equation = advection_equations.FiniteVolumeAdvectionDiffusion(
        diffusion_coefficient=diffusion_coefficient)

    # create a temporary dataset
    with flagsaver.flagsaver(
        dataset_path=output_path,
        dataset_name=output_name,
        equation_name=equation_name,
        discretization=discretization,
        simulation_grid_resolution=high_resolution,
        output_grid_resolution=low_resolution,
        equation_kwargs=str(dict(diffusion_coefficient=diffusion_coefficient)),
        dataset_type=dataset_type,
        num_shards=shards,
        total_time_steps=10,
        example_time_steps=example_time_steps,
        time_step_interval=5,
        num_seeds=4,
    ):
      create_training_data.main([], runner=beam.runners.DirectRunner())

    metadata_path = os.path.join(output_path, output_name + '.metadata')
    self.assertTrue(gfile.exists(metadata_path))
    dataset_metadata = readers.load_metadata(metadata_path)
    low_res_grid = readers.get_output_grid(dataset_metadata)
    high_res_grid = readers.get_simulation_grid(dataset_metadata)
    equation = readers.get_equation(dataset_metadata)

    self.assertEqual(low_res_grid.size_x, low_resolution)
    self.assertEqual(low_res_grid.size_y, low_resolution)
    self.assertEqual(high_res_grid.size_x, high_resolution)
    self.assertEqual(high_res_grid.size_y, high_resolution)
    self.assertAlmostEqual(high_res_grid.step, 2 * np.pi / high_resolution)
    self.assertAlmostEqual(
        equation.diffusion_coefficient, diffusion_coefficient)
    self.assertIs(type(equation), type(expected_equation))
    self.assertEqual(dataset_metadata.model.WhichOneof('model'),
                     'finite_difference')

    state_keys = expected_equation.key_definitions
    valid_data_keys = ((state_keys['concentration'].exact(),),
                       (state_keys['concentration_edge_x'].exact(),
                        state_keys['concentration_y_edge_y'].exact()))
    invalid_data_keys = ((state_keys['concentration'],
                          state_keys['concentration_edge_x']),
                         (state_keys['concentration_edge_x'],))
    valid_data_grids = (low_res_grid, low_res_grid)
    invalid_data_grids = (low_res_grid, high_res_grid)

    with self.assertRaises(ValueError):
      readers.initialize_dataset(
          dataset_metadata, invalid_data_keys, valid_data_grids)
    with self.assertRaises(ValueError):
      readers.initialize_dataset(
          dataset_metadata, valid_data_keys, invalid_data_grids)
    with self.assertRaises(ValueError):
      readers.initialize_dataset(
          dataset_metadata, invalid_data_keys, invalid_data_grids)

    dataset = readers.initialize_dataset(
        dataset_metadata, valid_data_keys, valid_data_grids)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)

    [(first_state, second_state)] = dataset.take(1)
    self.assertEqual(set(first_state.keys()), set(valid_data_keys[0]))
    self.assertEqual(set(second_state.keys()), set(valid_data_keys[1]))
    first_state_shape = np.shape(first_state[valid_data_keys[0][0]])
    second_state_shape = np.shape(second_state[valid_data_keys[1][0]])
    expected_shape = (
        batch_size, example_time_steps, low_resolution, low_resolution)
    self.assertEqual(first_state_shape, expected_shape)
    self.assertEqual(second_state_shape, expected_shape)

  def test_statistics(self):
    """Dataset writer and reader test, checks statistics computations."""
    output_path = FLAGS.test_tmpdir
    output_name = 'temp'

    equation_name = 'advection_diffusion'
    discretization = 'finite_volume'

    # create a temporary dataset
    with flagsaver.flagsaver(
        dataset_path=output_path,
        dataset_name=output_name,
        equation_name=equation_name,
        discretization=discretization,
        simulation_grid_resolution=256,
        output_grid_resolution=32,
        dataset_type='all_derivatives',
        total_time_steps=10,
        example_time_steps=3,
        time_step_interval=5,
        num_seeds=4,
    ):
      create_training_data.main([], runner=beam.runners.DirectRunner())

    metadata_path = os.path.join(output_path, output_name + '.metadata')
    dataset_metadata = readers.load_metadata(metadata_path)
    low_res_grid = readers.get_output_grid(dataset_metadata)

    equation = advection_equations.FiniteVolumeAdvectionDiffusion(
        diffusion_coefficient=0.1)
    data_key = equation.key_definitions['concentration'].exact()
    dataset = readers.initialize_dataset(
        dataset_metadata, ((data_key,),), (low_res_grid,))
    dataset = dataset.repeat(1)
    dataset = dataset.batch(1)
    all_data = np.concatenate(
        [np.ravel(data[0][data_key]) for data in dataset])

    expected_mean = np.mean(all_data)
    expected_variance = np.var(all_data, ddof=1)

    keys = readers.data_component_keys(dataset_metadata.components)
    components_dict = {k: v for k, v in zip(keys, dataset_metadata.components)}

    component = components_dict[data_key, low_res_grid]
    metadata_mean = component.mean
    metadata_variance = component.variance

    np.testing.assert_allclose(metadata_mean, expected_mean, atol=1e-3)
    np.testing.assert_allclose(metadata_variance, expected_variance, atol=1e-3)


if __name__ == '__main__':
  absltest.main()
