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
"""Provides functions for evaluation of the performance of trained models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np
import tensorflow as tf
from typing import Dict, Tuple, Any
import xarray as xr

from pde_superresolution_2d import dataset_readers
from pde_superresolution_2d import equations
from pde_superresolution_2d import graph_builders
from pde_superresolution_2d import models
from pde_superresolution_2d import states
from pde_superresolution_2d import utils


# output sample axis names per entry
OUTPUT_METRICS_FORMAT = ('time', 'x', 'y', 'component')


def states_metrics(
    model_state: Dict[states.StateKey, tf.Tensor],
    baseline_state: Dict[states.StateKey, tf.Tensor],
    target_state: Dict[states.StateKey, tf.Tensor],
    components: Tuple[states.StateKey, ...]
) -> Dict[states.StateKey, Tuple[tf.Tensor, tf.Tensor]]:
  """Generates concatenation metrics of the model, baseline and target states.

  Args:
    model_state: State (state derivative) predicted by the model.
    baseline_state: State (state derivative) predicted by the baseline.
    target_state: Target state (state derivative) obtained from high resolution.
    components: StateKeys to save.

  Returns:
    Dictionary with update_op and value_op tensors corresponding to metrics.
  """
  metrics = {}
  for key in components:
    model_key = states.add_prefix(states.MODEL_PREFIX, key)
    baseline_key = states.add_prefix(states.BASELINE_PREFIX, key)
    metrics[key] = tf.contrib.metrics.streaming_concat(
        tf.squeeze(target_state[key]))
    metrics[model_key] = tf.contrib.metrics.streaming_concat(
        tf.squeeze(model_state[key]))
    metrics[baseline_key] = tf.contrib.metrics.streaming_concat(
        tf.squeeze(baseline_state[key]))
  return metrics


def evaluate_metrics(
    session: tf.Session(),
    metrics: Dict[Any, Tuple[tf.Tensor, tf.Tensor]],
    num_samples: int,
) -> Dict[Any, Any]:
  """Evaluates given metrics and aggregates it into a Tuple of dictionaries.

  Args:
    session: An active session.
    metrics: Tuple of metrics to evaluate.
    num_samples: Number of batches to evaluate the metrics on.

  Returns:
    Aggregated metrics over num_samples.
  """
  value_op, update_op = tf.contrib.metrics.aggregate_metric_map(metrics)
  for _ in range(num_samples):
    session.run(update_op)
  return session.run(value_op)


def integration_metrics(
    seed: int,
    hparams: tf.contrib.training.HParams,
    init_method: equations.InitialConditionMethod,
    initial_conditions_parameters: Dict[str, Any],
    times: np.ndarray,
) -> xr.Dataset:
  """Evaluates model by integrating equation from random initial conditions.

  Args:
    seed: Seed for initialization of the initial conditions.
    hparams: Parameters for evaluation.
    init_method: Initial conditions method.
    initial_conditions_parameters: Dictionary specifying properties of the
        ensemble of initial conditions. Must contain initialization type.
    times: Array specifying the time slices where we want to evaluate the result
        of integration. First value corresponds to initial conditions.

  Returns:
    A tuple of dictionaries holding update and evaluate tensors.

  Raises:
    ValueError: Missing a necessary component in the input.
  """
  hparams = copy.deepcopy(hparams)
  training_meta = dataset_readers.load_metadata(hparams.training_metadata_path)

  high_res_grid = dataset_readers.get_high_res_grid(training_meta)
  low_res_grid = dataset_readers.get_low_res_grid(training_meta)
  equation = dataset_readers.get_equation(training_meta)
  baseline_model = dataset_readers.get_baseline_model(training_meta)
  model = models.build_model(hparams, training_meta)

  if not hparams.model_equation_scheme:
    model_equation = equation
  else:
    model_equation = equations.equation_from_proto(
        training_meta.equation, hparams.model_equation_scheme)  # pytype: disable=wrong-arg-types

  with tf.Graph().as_default():
    high_res_state = equation.initial_random_state(
        seed, init_method, high_res_grid, **initial_conditions_parameters)
    low_res_state = graph_builders.resample_state_graph(
        high_res_state, high_res_grid, low_res_grid)

    high_res_integrated = graph_builders.time_integrate_graph(
        high_res_state, equation, baseline_model, high_res_grid, times, False)
    model_low_res_integrated = graph_builders.time_integrate_graph(
        low_res_state, model_equation, model, low_res_grid, times, False)
    baseline_low_res_integrated = graph_builders.time_integrate_graph(
        low_res_state, equation, baseline_model, low_res_grid, times, False)

    num_samples = len(times)
    num_state_components = len(equation.STATE_KEYS)
    low_res_data_shape = (num_samples, low_res_grid.size_x,
                          low_res_grid.size_y, num_state_components)
    high_res_data_shape = (num_samples, high_res_grid.size_x,
                           high_res_grid.size_y, num_state_components)
    high_res_integrated_states = equation.to_state(
        tf.reshape(high_res_integrated, high_res_data_shape))
    model_low_res_integrated_states = equation.to_state(
        tf.reshape(model_low_res_integrated, low_res_data_shape))
    baseline_low_res_integrated_states = equation.to_state(
        tf.reshape(baseline_low_res_integrated, low_res_data_shape))

    exact_low_res_integrated_states = graph_builders.resample_state_graph(
        high_res_integrated_states, high_res_grid, low_res_grid)

    state_metrics = states_metrics(model_low_res_integrated_states,
                                   baseline_low_res_integrated_states,
                                   exact_low_res_integrated_states,
                                   (states.C,))

    init = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()

    session = tf.Session()
    session.run([init, init_l])

    temp_saver = tf.train.Saver(tf.trainable_variables())
    ckpt_state = tf.train.get_checkpoint_state(hparams.model_dir)
    temp_saver.restore(session, ckpt_state.model_checkpoint_path)

    result = evaluate_metrics(session, state_metrics, 1)
    evaluation_dataset = xr.Dataset(
        {
            utils.component_name(key):
                (OUTPUT_METRICS_FORMAT, np.reshape(result[key],
                                                   low_res_data_shape))
            for key in result.keys()
        })
  return evaluation_dataset


EVALUATION_METHOD = {
    'integration': integration_metrics
}
