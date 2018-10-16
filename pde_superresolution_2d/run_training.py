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
"""Binary for running training."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
from absl import app
from absl import flags
from absl import logging
import tensorflow.google as tf

from pde_superresolution_2d import training
from pde_superresolution_2d import utils


flags.DEFINE_string(
    'model_dir', '',
    'Directory where to save the model.')
flags.DEFINE_string(
    'train_path', '',
    'Path to the training dataset metadata.')
flags.DEFINE_string(
    'validation_path', '',
    'Path to the test datasets metadata.')

flags.DEFINE_boolean(
    'enable_multi_eval', False,
    'Enable evaluation on the training dataset.')

flags.DEFINE_string(
    'hparams', '',
    'Additional hyper-parameter values to use, in the form of a '
    'comma-separated list of name=value pairs, e.g., '
    '"num_layers=3,num_filters=64".')


FLAGS = flags.FLAGS


def main(unused_argv):
  hparams = training.create_hparams()
  training.set_data_dependent_hparams(
      hparams, FLAGS.train_path, FLAGS.validation_path)
  hparams.parse(FLAGS.hparams)
  hparams.set_hparam('model_dir', FLAGS.model_dir)
  hparams.set_hparam('enable_multi_eval', FLAGS.enable_multi_eval)

  hparams_path = os.path.join(FLAGS.model_dir, 'hparams.pbtxt')
  utils.save_proto(hparams.to_proto(), hparams_path)

  training_scheme = training.TRAINING_METHODS[hparams.training_scheme]

  logging.info('Starting training loop')
  tf.estimator.parameterized_train_and_evaluate(
      training_scheme, hparams=hparams)


if __name__ == '__main__':
  app.run(main)
