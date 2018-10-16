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
"""Utility functions for training neural network models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import google_type_annotations

import copy
import functools
import tensorflow.google as tf
from typing import Any, Dict, Tuple

from pde_superresolution_2d import dataset_readers
from pde_superresolution_2d import equations
from pde_superresolution_2d import graph_builders
from pde_superresolution_2d import losses
from pde_superresolution_2d import metadata_pb2
from pde_superresolution_2d import models
from pde_superresolution_2d import states
from pde_superresolution_2d import utils


def create_hparams(**kwargs: Any) -> tf.contrib.training.HParams:
  """Create default hyper-parameters for training a model.

  Neural network parameters:
    model_type: String, name of the model.
    model_equation_type: String, name of the underlying equation used by model.
    num_layers: Integer number of conv layers to use for coefficient prediction.
    num_filters: Integer number of filters to use per conv layer.
    kernel_size: Integer kernel size.
    stencil_size: Integer size of the stencil used for coefficient based models.
    activation: Activation function used in between hidden layers.

  Training parameters:
    batch_size: Int, size of the batch size.
    learning_rates: List[float], piecewise learning rates values.
    learning_stops: List[int] specifying global steps at which to move on to the
        next learning rate or stop training.
    checkpoint_interval: Integer, number of global steps between consecutive
        checkpoints. Evaluation is performed at every checkpoint.
    eval_steps: Number of samples to use for evaluation.
    opt_beta1: Float, optimizers beta1 parameter.
    opt_beta2: Float, optimizers beta2 parameter.

  Data parameters:
    input_mean: Float, mean input value in the dataset. Used for normalization.
    input_variance: Float, variance of the input value.

  Loss parameters:
    baseline_weighted: Boolean indicating whether loss is to be weighted by the
        error ratio of the baseline and current model.
    epsilon: The regularization of the loss function weighted by the error ratio
        of the baseline model: loss = l_model / (epsilon*l_model + l_baseline).

  Run parameters:
    model_dir: String, path to the directory where to save the trained model.
    enable_multi_eval: Boolean indicating whether to run parallel evaluation on
        the training data. As of 07/2018 only available when running on borg.

  Args:
    **kwargs: default hyper-parameter values to override.

  Returns:
    HParams object with all hyperparameter values.
  """
  hparams = tf.contrib.training.HParams(
      # neural network parameters
      model_type='stencil_velocity_net',
      model_equation_type='',
      num_layers=4,
      num_filters=36,
      kernel_size=3,
      stencil_size=3,
      activation='relu',
      # training parameters
      training_scheme='time_derivative_training',
      batch_size=1,
      learning_rates=[3e-4, 4e-5, 3e-6],
      learning_stops=[5000, 15000, 30000],
      checkpoint_interval=100,
      eval_steps=100,
      opt_beta1=0.9,
      opt_beta2=0.99,
      # loss parameters
      baseline_weighted=False,
      epsilon=0.1,
      # Data parameters
      input_mean=3.36,
      input_variance=0.272,
      training_metadata_path='',
      validation_metadata_path='',
      test_meta='',
      # Run parameters
      model_dir='',
      enable_multi_eval=False
  )
  hparams.override_from_dict(kwargs)
  return hparams


def set_data_dependent_hparams(
    hparams: tf.contrib.training.HParams,
    train_metadata_path: str,
    validation_metadata_path: str):
  """Add data-dependent hyperparameters to hparams.

  Args:
    hparams: Hyper-parameters describing the training setup. Will be modified
        by adding path to the train and validation datasets metadata.
    train_metadata_path: Path to the training dataset metadata.
    validation_metadata_path: Path to the validation dataset metadata.
  """
  hparams.set_hparam('training_metadata_path', train_metadata_path)
  hparams.set_hparam('validation_metadata_path', validation_metadata_path)

  training_meta = dataset_readers.load_metadata(train_metadata_path)
  hparams.set_hparam('input_mean', training_meta.input_mean)
  hparams.set_hparam('input_variance', training_meta.input_variance)


def create_learning_rate(global_step: tf.Tensor,
                         hparams: tf.contrib.training.HParams) -> tf.Tensor:
  """Returns constant or piecewise constant learning rate based of hparams.

  Args:
    global_step: Tensor representing global step.
    hparams: A HParams object holding hyper-parameters.

  Returns:
    Learning rate tensor.
  """
  if len(hparams.learning_rates) > 1:
    learning_rate = tf.train.piecewise_constant(
        global_step, boundaries=hparams.learning_stops[:-1],
        values=hparams.learning_rates)
  else:
    (learning_rate,) = hparams.learning_rates
  return learning_rate


def create_training_step(
    loss: tf.Tensor,
    learning_rate: tf.Tensor,
    hparams: tf.contrib.training.HParams
) -> tf.Tensor:
  """Creates training step by minimizing loss with Adam optimizer.

  Args:
    loss: Scalar loss tensor.
    learning_rate: Learning rate tensor.
    hparams: A HParams object holding hyper-parameters.

  Returns:
    Training step operation.
  """
  global_step = tf.train.get_or_create_global_step()
  optimizer = tf.train.AdamOptimizer(learning_rate, beta2=hparams.opt_beta2)
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # Do we need this?
  with tf.control_dependencies(update_ops):
    train_step = optimizer.minimize(loss, global_step=global_step)
  return train_step


def time_derivative_training_scheme(
    hparams: tf.contrib.training.HParams
) -> tf.estimator.DistributedTrainingSpec:
  """Generates training spec implementing time derivative training.

  Time derivative training is based on the process of minimization of the loss
  norm between the predicted time derivative and the coarse-grained values.

  Args:
    hparams: A HParams object holding model and training parameters.

  Returns:
    Training spec ready for one-off training.
  """
  hparams = copy.deepcopy(hparams)
  training_meta = dataset_readers.load_metadata(hparams.training_meta)
  validation_meta = dataset_readers.load_metadata(hparams.validation_meta)

  low_res_grid = dataset_readers.get_low_res_grid(training_meta)
  equation = dataset_readers.get_equation(training_meta)
  model = models.build_model(hparams, training_meta)

  if not hparams.model_equation_type:
    model_equation = equation
  else:
    model_equation = equations.equation_from_proto(
        training_meta.equation, hparams.model_equation_scheme)  # pytype: disable=wrong-arg-types

  run_config = tf.estimator.RunConfig(
      save_summary_steps=hparams.checkpoint_interval,
      save_checkpoints_steps=hparams.checkpoint_interval)

  # Define dataset keys required for training.
  state_keys = equation.STATE_KEYS
  state_keys_time_derivative = states.add_time_derivative_tuple(state_keys)

  if hparams.baseline_weighted:
    baseline_state_keys_time_derivative = states.add_prefix_tuple(
        states.BASELINE_PREFIX, state_keys_time_derivative)
    requested_data_keys = (state_keys, state_keys_time_derivative,
                           baseline_state_keys_time_derivative)
    requested_data_grids = (low_res_grid, low_res_grid, low_res_grid)
  else:
    requested_data_keys = (state_keys, state_keys_time_derivative)
    requested_data_grids = (low_res_grid, low_res_grid)

  def build_input_pipeline(
      metadata: metadata_pb2.Dataset
  ) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
    """Builds an input pipeline from the dataset, looping over examples.

    Args:
      metadata: Protocol buffer holding datasets metadata.

    Returns:
      Tuple of dictionaries holding features and labels tensors respectively.
    """
    dataset = dataset_readers.initialize_dataset(
        metadata, requested_data_keys, requested_data_grids)
    dataset = dataset.repeat()
    dataset = dataset.batch(hparams.batch_size)

    iterator = tf.data.Iterator.from_structure(
        dataset.output_types, dataset.output_shapes)
    iterator_init = iterator.make_initializer(dataset)

    if hparams.baseline_weighted:
      input_state, target, baseline = iterator.get_next()
      features = {'input': input_state, 'baseline': baseline}
      labels = {'target': target}
    else:
      input_state, target = iterator.get_next()
      features = {'input': input_state}
      labels = {'target': target}

    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator_init)
    return (features, labels)

  def model_fn(
      features: Dict[str, tf.Tensor],
      labels: Dict[str, tf.Tensor],
      mode: tf.estimator.ModeKeys
  ) -> tf.estimator.EstimatorSpec:
    """Builds computational graph for train, eval and predict regimes.

    Args:
      features: States coming from the pipeline.
      labels: Corresponding 'exact' time derivatives of features.
      mode: Mode of execution, must be one of PREDICT, TRAIN, EVAL.

    Returns:
      Estimator spec corresponding to the `mode` execution.
    """
    input_state = features['input']
    target = labels['target']
    if hparams.baseline_weighted:
      baseline = features['baseline']

    global_step = tf.train.get_or_create_global_step()
    model_time_derivative = graph_builders.time_derivative_graph(
        input_state, model_equation, model, low_res_grid)

    # Define prediction mode spec, which generates time derivatives.
    if mode == tf.estimator.ModeKeys.PREDICT:
      prediction = {
          utils.component_name(key, low_res_grid): model_time_derivative[key]
          for key in model_time_derivative.keys()}
      return tf.estimator.EstimatorSpec(
          mode=tf.estimator.ModeKeys.PREDICT, predictions=prediction)

    loss_normalization = 1 / (hparams.input_mean) ** 2
    if hparams.baseline_weighted:
      loss = loss_normalization * losses.state_weighted_mean_loss(
          model_time_derivative, target, baseline, state_keys_time_derivative,
          tf.losses.mean_squared_error, epsilon=hparams.epsilon)
    else:
      loss = loss_normalization * losses.state_mean_loss(
          model_time_derivative, target, state_keys_time_derivative,
          tf.losses.mean_squared_error)

    # Define training mode spec.
    if mode == tf.estimator.ModeKeys.TRAIN:
      learning_rate = create_learning_rate(global_step, hparams)
      train_op = create_training_step(loss, learning_rate, hparams)
      return tf.estimator.EstimatorSpec(
          mode=tf.estimator.ModeKeys.TRAIN, loss=loss, train_op=train_op)

    # Define evaluation mode spec.
    if mode == tf.estimator.ModeKeys.EVAL:
      # TODO(dkochkov) add more metric to monitor the training process.
      loss_metric = {'average_loss': tf.metrics.mean(loss)}
      return tf.estimator.EstimatorSpec(
          mode=tf.estimator.ModeKeys.EVAL, loss=loss,
          eval_metric_ops=loss_metric)

  # Define estimator using the above model_fn
  estimator = tf.estimator.Estimator(model_fn=model_fn,
                                     model_dir=hparams.model_dir,
                                     config=run_config)

  train_input = functools.partial(build_input_pipeline, metadata=training_meta)
  validation_input = functools.partial(build_input_pipeline,
                                       metadata=validation_meta)

  # Define spec for training and evaluation.
  train_spec = tf.estimator.TrainSpec(input_fn=train_input,
                                      max_steps=hparams.learning_stops[-1])

  eval_spec = []
  eval_spec.append(tf.estimator.EvalSpec(name='validation_data',
                                         input_fn=validation_input,
                                         steps=hparams.eval_steps,
                                         throttle_secs=0))

  if hparams.enable_multi_eval:
    eval_on_train_input = functools.partial(build_input_pipeline,
                                            metadata=training_meta)
    eval_spec.append(tf.estimator.EvalSpec(name='training_data',
                                           input_fn=eval_on_train_input,
                                           steps=hparams.eval_steps,
                                           throttle_secs=0))

  return tf.estimator.DistributedTrainingSpec(estimator, train_spec, eval_spec)


TRAINING_METHODS = {
    'time_derivative_training': time_derivative_training_scheme
}
