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
"""Run a beam pipeline to generate training data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np

from typing import Dict, Tuple

from apache_beam import runner

from pde_superresolution_2d import dataset_builders
from pde_superresolution_2d import equations
from pde_superresolution_2d import grids
from pde_superresolution_2d import metadata_pb2
from pde_superresolution_2d import models
from pde_superresolution_2d import states
from pde_superresolution_2d import velocity_fields


# files
flags.DEFINE_string(
    'dataset_path', None,
    'Path to the folder where to save the dataset.')
flags.DEFINE_string(
    'dataset_name', None,
    'Name of the dataset without extensions.')

# equation and model types
flags.DEFINE_string(
    'equation_name', 'advection_diffusion',
    'The name of the equation to solve.')
flags.DEFINE_enum(
    'equation_scheme', 'FINITE_VOLUME',
    metadata_pb2.Equation.DiscretizationScheme.keys(),
    'The id of the discretization scheme to use.', case_sensitive=False)
flags.DEFINE_string(
    'baseline_model_type', 'roll_finite_difference',
    'What model to use to generate ground truth.')

# grid and equation parameters
flags.DEFINE_integer(
    'high_resolution_points', 256,
    'Number of cells on high resolution grid along x and y axis.')
flags.DEFINE_integer(
    'low_resolution_points', 32,
    'Number of cells on low resolution grid along x and y axis.')
flags.DEFINE_float(
    'diffusion_const', 0.005,
    'Diffusion constant.')
flags.DEFINE_integer(
    'num_velocity_field_terms', 5,
    'Number of terms in the velocity field.')
flags.DEFINE_integer(
    'max_velocity_field_periods', 3,
    'Maximum spatial frequency of the velocity field along any axis')
flags.DEFINE_integer(
    'velocity_field_random_seed', 1,
    'Random seed used to generate the velocity field when one is required by'
    'the equation and kept constant for all initial conditions.')

# initial conditions parameters
flags.DEFINE_enum_class(
    'initial_conditions_type', 'FOURIER', equations.InitialConditionMethod,
    'Type of initial conditions to use.')
flags.DEFINE_integer(
    'num_gaussian_terms', 4,
    'Number of Gaussian terms to generate for Gaussian initial conditions.')
flags.DEFINE_float(
    'gaussian_width', 0.2,
    'Width of the Gaussian initial conditions as fraction of the system size.')
flags.DEFINE_integer(
    'num_fourier_terms', 4,
    'Number of Fourier to generate for Fourier initial conditions.')
flags.DEFINE_integer(
    'max_fourier_term_periods', 4,
    'Maximum spatial frequency of the initial conditions along any axis.')

# model and dataset  parameters
flags.DEFINE_string(
    'dataset_type', 'time_derivatives',
    'Type of the dataset to be generated.')
flags.DEFINE_integer(
    'num_shards', 1,
    'Number of shards to break the dataset into.')
flags.DEFINE_float(
    'max_time', 6.0,
    'Total time for which to run each integration.')
flags.DEFINE_integer(
    'num_time_slices', 100,
    'Number of time slices to extract from one integration trace.')
flags.DEFINE_integer(
    'num_samples', 1,
    'Number of different initial seeds to integrate.')
flags.DEFINE_integer(
    'initialization_seed_offset', 1000000,
    'Integer seed offset for random number generator. This should be larger '
    'than the largest possible number of evaluation seeds, but smaller '
    'than 2^32 (the size of NumPy\'s random number seed).')

FLAGS = flags.FLAGS


def main(_):
  runner.program_started()

  dataset_path = FLAGS.dataset_path
  dataset_name = FLAGS.dataset_name
  metadata_path = os.path.join(dataset_path, dataset_name + '.metadata')
  records_path = os.path.join(dataset_path, dataset_name + '.tfrecord')
  num_shards = FLAGS.num_shards
  initialization_seed_offset = FLAGS.initialization_seed_offset

  # Grid parameters.
  high_resolution_size = FLAGS.high_resolution_points
  low_resolution_size = FLAGS.low_resolution_points
  high_resolution_step = 2 * np.pi / high_resolution_size
  low_resolution_step = 2 * np.pi / low_resolution_size

  # Initial conditions parameters.
  init_method = FLAGS.initial_conditions_type
  initial_conditions_parameters = {
      'num_velocity_field_terms': FLAGS.num_velocity_field_terms,
      'max_velocity_field_periods': FLAGS.max_velocity_field_periods,
      'num_gaussian_terms': FLAGS.num_gaussian_terms,
      'gaussian_width': FLAGS.gaussian_width,
      'num_fourier_terms': FLAGS.num_fourier_terms,
      'max_fourier_term_periods': FLAGS.max_fourier_term_periods,
      'velocity_field_random_seed': FLAGS.velocity_field_random_seed
  }

  # Parameters of the equation.
  num_velocity_terms = FLAGS.num_velocity_field_terms
  max_velocity_periods = FLAGS.max_velocity_field_periods
  velocity_field_random_seed = FLAGS.velocity_field_random_seed
  diffusion_const = FLAGS.diffusion_const
  discretization_scheme = metadata_pb2.Equation.DiscretizationScheme.Value(
      FLAGS.equation_scheme)
  equation_and_scheme = (FLAGS.equation_name, discretization_scheme)

  # Integration parameters.
  max_time = FLAGS.max_time
  num_time_slices = FLAGS.num_time_slices
  num_samples = FLAGS.num_samples

  # Instantiation of the components of the system.
  equation_class = equations.EQUATION_TYPES[equation_and_scheme]
  dataset_type = dataset_builders.DATASET_TYPES[FLAGS.dataset_type]

  high_res_grid = grids.Grid(
      high_resolution_size, high_resolution_size, high_resolution_step)
  low_res_grid = grids.Grid(
      low_resolution_size, low_resolution_size, low_resolution_step)

  velocity_field = velocity_fields.ConstantVelocityField.from_seed(
      num_velocity_terms, max_velocity_periods, velocity_field_random_seed)

  equation = equation_class(velocity_field, diffusion_const)
  model = models.build_model(model_type_name=FLAGS.baseline_model_type)
  dataset_builder = dataset_type(
      equation, model, high_res_grid, low_res_grid, max_time, num_time_slices,
      initialization_seed_offset, num_samples, records_path, num_shards)

  integrate_map = dataset_builder.integrate_states_transform()
  process_map = dataset_builder.preprocess_states_transform()
  compute_stats_map = dataset_builder.compute_input_statistics_transform()

  seeds = [i + initialization_seed_offset for i in range(num_samples)]

  def generate_initial_states(
      seed: int,
      equation: equations.Equation = equation,
      grid: grids.Grid = high_res_grid
  ) -> Tuple[Dict[states.StateKey, np.ndarray]]:
    """Function to be passed to beam.Map to generates initial states."""
    initial_states = equation.initial_random_state(
        seed, init_method, grid, **initial_conditions_parameters)
    return (initial_states,)

  def convert_to_tf_examples(
      inputs: Tuple[Dict[states.StateKey, np.ndarray]]
  ) -> str:
    """Function to be passed to beam.Map to serialize data examples."""
    return dataset_builder.convert_to_tf_examples(inputs)

  def finalize_metadata(mean_and_variance: Tuple[float, float]):
    """Updates data statistics in the metadata and write it to a file."""
    dataset_builder.save_metadata(metadata_path, mean_and_variance)

  def build_pipeline(root):
    """Builds a pipeline that generates and saves tfrecords and metadata."""
    generate_pipeline = (
        root
        | beam.Create(seeds)
        | beam.Map(generate_initial_states)
        | 'integrate' >> beam.ParDo(integrate_map)
        | 'process' >> beam.ParDo(process_map)
    )

    save_pipeline = (  # pylint: disable=unused-variable
        generate_pipeline
        | beam.Map(convert_to_tf_examples)
        | beam.Reshuffle()
        | beam.io.tfrecordio.WriteToTFRecord(records_path,
                                             num_shards=num_shards)
    )

    statistics_pipeline = (  # pylint: disable=unused-variable
        generate_pipeline
        | beam.CombineGlobally(compute_stats_map)
        | beam.Map(finalize_metadata)
    )

  runner.DirectRunner().run(build_pipeline).wait_until_finish()

if __name__ == '__main__':
  app.run(main)

