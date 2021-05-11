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
"""Run a beam pipeline to generate training data."""
import ast
import os

from absl import app
from absl import flags

try:
  import apache_beam as beam
  from datadrivenpdes.pipelines import beamlib
except:
  pass

import numpy as np
from datadrivenpdes.core import builders
from datadrivenpdes.core import equations
from datadrivenpdes.core import grids
import tensorflow as tf

# Ensure Equation subclasses are defined so we can look them up by name.
# pylint: disable=unused-import,g-bad-import-order
from datadrivenpdes.advection import equations as advection_equations
# pylint: enable=unused-import,g-bad-import-order

from apache_beam import runners
# our beam pipeline requires eager mode
tf.enable_eager_execution()

# files
flags.DEFINE_string(
    'dataset_path', None,
    'Path to the folder where to save the dataset.')
flags.DEFINE_string(
    'dataset_name', None,
    'Name of the dataset without extensions.')
flags.DEFINE_integer(
    'num_shards', 1,
    'Number of shards to break the dataset into when saved on disk.')

# equation parameters
flags.DEFINE_string(
    'equation_name', 'advection',
    'The name of the equation to solve.')
flags.DEFINE_string(
    'discretization', 'finite_volume',
    'Name of the discretization method to use.')
flags.DEFINE_string(
    'equation_kwargs',
    str(dict()),
    'Keyword arguments to pass to the Equation constructor.')
flags.DEFINE_string(
    'random_state_params', '{}',
    'Keyword arguments to pass as params to Equation.random_state().')

# grid parameters
flags.DEFINE_integer(
    'simulation_grid_size', 128,
    'Number of cells on high resolution grid along x and y axis.')
flags.DEFINE_integer(
    'output_grid_size', 32,
    'Number of cells on low resolution grid along x and y axis.')
flags.DEFINE_float(
    'grid_length', 2 * np.pi,
    'Length of one side of the square grid.')

# dataset parameters
flags.DEFINE_enum(
    'dataset_type', 'time_evolution', builders.DATASET_TYPES,
    'Type of the dataset to be generated.')
flags.DEFINE_integer(
    'num_seeds', 1,
    'Number of different initial seeds to integrate.')
flags.DEFINE_integer(
    'total_time_steps', 100,
    'Number of fine-resolution time-steps for which to integrate from '
    'random initial conditions to produce saved examples.')
flags.DEFINE_integer(
    'time_step_interval', 10,
    'Interval at which to sample from the fine-resolution simulations for '
    'saved examples.')
flags.DEFINE_integer(
    'example_num_time_steps', 10,
    'Total number of coarse-resolution time steps saved each example.')
flags.DEFINE_integer(
    'initialization_seed_offset', 1000000,
    'Integer seed offset for random number generator. This should be larger '
    'than the largest possible number of evaluation seeds, but smaller '
    'than 2^32 (the size of NumPy\'s random number seed).')

FLAGS = flags.FLAGS


def flags_as_dict():
  module = FLAGS.find_module_defining_flag('dataset_path')
  flags_list = FLAGS.flags_by_module_dict()[module]
  return {flag.name: flag.value for flag in flags_list}


def main(_, runner=None):
  if runner is None:
    # must create before flags are used
    runner = runners.DirectRunner()

  # files
  dataset_path = FLAGS.dataset_path
  tf.io.gfile.makedirs(dataset_path)
  dataset_name = FLAGS.dataset_name
  metadata_path = os.path.join(dataset_path, dataset_name + '.metadata.json')
  records_path = os.path.join(dataset_path, dataset_name + '.tfrecord')
  num_shards = FLAGS.num_shards
  initialization_seed_offset = FLAGS.initialization_seed_offset

  # instantiate components of the system.
  equation_class = equations.matching_equation_type(
      FLAGS.equation_name, FLAGS.discretization)
  equation = equation_class(**ast.literal_eval(FLAGS.equation_kwargs))

  simulation_grid = grids.Grid.from_period(FLAGS.simulation_grid_size,
                                           FLAGS.grid_length)
  output_grid = grids.Grid.from_period(FLAGS.output_grid_size,
                                       FLAGS.grid_length)
  initial_condition_steps = np.arange(
      0, FLAGS.total_time_steps, FLAGS.time_step_interval)

  builder_type = builders.DATASET_TYPES[FLAGS.dataset_type]
  builder = builder_type(
      equation, simulation_grid, output_grid, initial_condition_steps,
      example_num_time_steps=FLAGS.example_num_time_steps,
  )

  flags_dict = flags_as_dict()

  seeds = [i + initialization_seed_offset for i in range(FLAGS.num_seeds)]
  rs_params = ast.literal_eval(FLAGS.random_state_params)

  def random_state(seed):
    return equation.random_state(simulation_grid, params=rs_params, seed=seed)

  def build_pipeline(root):
    """Builds a pipeline that generates and saves tfrecords and metadata."""

    # NOTE(shoyer): we use Reshuffle transforms to ensure that Beam doesn't
    # consolidate expensive computations into fused tasks that cannot be
    # parallelized.
    generate_pipeline = (
        root
        | beam.Create(seeds)
        | 'random_state' >> beam.Map(random_state)
        | 'integrate_initial_conditions'
        >> beam.FlatMap(builder.integrate_for_initial_conditions)
        | 'split_integrate_tasks'>>  beam.Reshuffle()
        | 'integrate_each_example' >> beam.Map(builder.integrate_each_example)
        | 'postprocess' >> beam.Map(builder.postprocess)
    )

    save_pipeline = (  # pylint: disable=unused-variable
        generate_pipeline
        | 'split_simulation_and_saving' >> beam.Reshuffle()
        | beam.Map(builder.convert_to_tf_example)
        | beam.io.tfrecordio.WriteToTFRecord(
            records_path, num_shards=num_shards)
    )

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
            flags=flags_dict,
        )
    )

  runner.run(build_pipeline)


if __name__ == '__main__':
  app.run(main)
