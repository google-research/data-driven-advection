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
"""Utility functions for building loss tensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import google_type_annotations

import tensorflow as tf
from typing import Callable, Dict, Tuple

from pde_superresolution_2d import states


def state_mean_loss(
    input_state: Dict[states.StateKey, tf.Tensor],
    target_state: Dict[states.StateKey, tf.Tensor],
    components: Tuple[states.StateKey],
    loss_function: Callable
) -> tf.Tensor:
  """Generates loss term using input and target states for given `components`.

  Args:
    input_state: State holding the predicted values.
    target_state: State holding the target values.
    components: Components for which to generate loss.
    loss_function: Loss function to use, must follow tf.losses syntax.

  Returns:
    Scalar loss tensor.
  """
  losses = []
  for component in components:
    losses.append(loss_function(input_state[component], target_state[component],
                                reduction=tf.losses.Reduction.MEAN))
  return tf.add_n(losses)


def state_weighted_mean_loss(
    input_state: Dict[states.StateKey, tf.Tensor],
    target_state: Dict[states.StateKey, tf.Tensor],
    ref_state: Dict[states.StateKey, tf.Tensor],
    components: Tuple[states.StateKey],
    loss_function: Callable = tf.nn.l2_loss,
    epsilon: float = 0.1
) -> tf.Tensor:
  """Generates reference-weighted loss term for given `components`.

  Args:
    input_state: State holding the predicted values.
    target_state: State holding the target values.
    ref_state: State holding the reference predictions.
    components: Components for which the generate loss.
    loss_function: Loss function to use.
    epsilon: Fraction of the models loss to be added to the reference loss.

  Returns:
    Scalar tensor representing the weighted loss.
  """
  losses = []
  for component in components:
    ref_loss = loss_function(ref_state[component], target_state[component])
    model_loss = loss_function(input_state[component], target_state[component])
    weights = tf.maximum(1., model_loss / (epsilon * model_loss + ref_loss))
    losses.append(tf.losses.compute_weighted_loss(model_loss, weights))
  return tf.add_n(losses)
