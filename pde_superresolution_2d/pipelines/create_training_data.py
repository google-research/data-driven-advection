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

import ast
import os.path

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
from pde_superresolution_2d import metadata_pb2
from pde_superresolution_2d.core import builders
from pde_superresolution_2d.core import equations
from pde_superresolution_2d.core import grids
from pde_superresolution_2d.core import models
from pde_superresolution_2d.pipelines import beamlib
import tensorflow as tf

# Imports to register equations by name in equations.CONTINUOUS_EQUATIONS.
# pylint: disable=unused-import,g-bad-import-order
from pde_superresolution_2d.advection import equations as advection_equations
from pde_superresolution_2d.floods import equations as floods_equations
# pylint: enable=unused-import,g-bad-import-order

from apache_beam import runner

# our beam pipeline requires eager mode
tf.enable_eager_execution()

# files
flags.DEFINE_string('dataset_path', None,
                    'Path to the folder where to save the dataset.')
flags.DEFINE_string('dataset_name', None,
                    'Name of the dataset without extensions.')
flags.DEFINE_integer('num_shards', 1,
                     'Number of shards to break the dataset into.')

# equation parameters
flags.DEFINE_string('equation_name', 'advection_diffusion',
                    'The name of the equation to solve.')
flags.DEFINE_string(
    'discretization', 'finite_volume',
    'Name of the discretization method to use.')
flags.DEFINE_string(
    'equation_kwargs',
    str(dict(diffusion_coefficient=0.005, cfl_safety_factor=0.9)),
    'Keyword arguments to pass to the Equation constructor.')
flags.DEFINE_string(
    'random_state_params', '{}',
    'Keyword arguments to pass as params to Equation.random_state().')

# grid parameters
flags.DEFINE_integer(
    'simulation_grid_resolution', 128,
    'Number of cells on high resolution grid along x and y axis.')
flags.DEFINE_integer(
    'output_grid_resolution', 32,
    'Number of cells on low resolution grid along x and y axis.')
flags.DEFINE_float('grid_length', 2 * np.pi,
                   'Length of one side of the square grid.')

# dataset parameters
flags.DEFINE_enum(
    'dataset_type', 'time_evolution', builders.DATASET_TYPES,
    'Type of the dataset to be generated.')
flags.DEFINE_integer(
    'total_time_steps', 100,
    'Number of saved-resolution time-steps to save from a single seed.')
flags.DEFINE_integer('example_time_steps', 10,
                     'Total number of time steps saved in a single example.')
flags.DEFINE_integer(
    'time_step_interval', 10,
    'Interval between starting time-steps for different examples.')
flags.DEFINE_integer('num_seeds', 1,
                     'Number of different initial seeds to integrate.')
flags.DEFINE_integer(
    'initialization_seed_offset', 1000000,
    'Integer seed offset for random number generator. This should be larger '
    'than the largest possible number of evaluation seeds, but smaller '
    'than 2^32 (the size of NumPy\'s random number seed).')

FLAGS = flags.FLAGS


def main(_):
  runner.program_started()

  # files
  dataset_path = FLAGS.dataset_path
  dataset_name = FLAGS.dataset_name
  metadata_path = os.path.join(dataset_path, dataset_name + '.metadata')
  records_path = os.path.join(dataset_path, dataset_name + '.tfrecord')
  num_shards = FLAGS.num_shards
  initialization_seed_offset = FLAGS.initialization_seed_offset

  # instantiate components of the system.
  equation_class = equations.matching_equation_type(
      equations.CONTINUOUS_EQUATIONS[FLAGS.equation_name], FLAGS.discretization)
  equation = equation_class(**ast.literal_eval(FLAGS.equation_kwargs))

  simulation_grid = grids.Grid.from_period(FLAGS.simulation_grid_resolution,
                                           FLAGS.grid_length)
  output_grid = grids.Grid.from_period(FLAGS.output_grid_resolution,
                                       FLAGS.grid_length)

  # NOTE: we currently only support the case where the advection term in the
  # time-step formula dominates. If you go to too-high resolution or too-high
  # diffusion, this will break. But numerical integration with the forward Euler
  # method is really slow in these cases, anyways.
  time_step = equation.get_time_step(output_grid)
  steps = np.arange(0, FLAGS.total_time_steps, FLAGS.time_step_interval)
  times = time_step * steps

  num_seeds = FLAGS.num_seeds

  builder_type = builders.DATASET_TYPES[FLAGS.dataset_type]
  builder = builder_type(
      equation, simulation_grid, output_grid, times,
      example_time_steps=FLAGS.example_time_steps,
  )

  extra_metadata_fields = dict(
      initialization_seed_offset=FLAGS.initialization_seed_offset,
      num_seeds=FLAGS.num_seeds,
  )

  seeds = [i + initialization_seed_offset for i in range(num_seeds)]
  rs_params = ast.literal_eval(FLAGS.random_state_params)

  def random_state(seed):
    return equation.random_state(simulation_grid, params=rs_params, seed=seed)

  def build_pipeline(root):
    """Builds a pipeline that generates and saves tfrecords and metadata."""
    generate_pipeline = (
        root
        | beam.Create(seeds)
        | 'random_state' >> beam.Map(random_state)
        | 'integrate' >> beam.FlatMap(builder.integrate)
        | 'postprocess' >> beam.Map(builder.postprocess))

    save_pipeline = (  # pylint: disable=unused-variable
        generate_pipeline
        | beam.Reshuffle()
        | beam.Map(builder.convert_to_tf_example)
        | beam.io.tfrecordio.WriteToTFRecord(
            records_path, num_shards=num_shards))

    statistics_pipeline = (  # pylint: disable=unused-variable
        generate_pipeline
        | 'items' >> beam.FlatMap(lambda state: state.items())
        | 'calculate_statistics' >> beam.CombinePerKey(
            beamlib.MeanVarianceCombineFn())
        | 'combine_statistics' >> beam.combiners.ToDict()
        | 'save_metadata' >> beam.Map(
            builder.save_metadata,
            records_path,
            metadata_path,
            num_shards=num_shards,
            extra_fields=extra_metadata_fields))

  runner.DirectRunner().run(build_pipeline).wait_until_finish()


if __name__ == '__main__':
  app.run(main)
