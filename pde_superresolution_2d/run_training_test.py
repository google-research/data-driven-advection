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
"""Integration test for model training."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

from absl import flags
from absl.testing import flagsaver

from absl.testing import absltest

from pde_superresolution_2d import create_training_data
from pde_superresolution_2d import run_training


FLAGS = flags.FLAGS


class CreateTrainingDataTest(absltest.TestCase):

  def test(self):
    dataset_path = self.create_tempdir('data').full_path
    dataset_name = 'temp'

    # create data
    with flagsaver.flagsaver(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        num_shards=1,
        max_time=0.03,
        num_time_slices=10,
        num_samples=2,
    ):
      create_training_data.main([])

    # run training
    model_dir = self.create_tempdir('model').full_path
    train_path = os.path.join(dataset_path, dataset_name + '.metadata')
    with flagsaver.flagsaver(
        model_dir=model_dir,
        train_path=train_path,
        validation_path=train_path,
        hparams='learning_rates=[1e-4],learning_stops=[10]',
    ):
      run_training.main([])


if __name__ == '__main__':
  absltest.main()
