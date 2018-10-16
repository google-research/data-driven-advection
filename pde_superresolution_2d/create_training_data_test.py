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
"""Test for create_training_data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

from absl import flags
from absl.testing import flagsaver

from tensorflow import gfile
from absl.testing import absltest
from pde_superresolution_2d import create_training_data


FLAGS = flags.FLAGS


class CreateTrainingDataTest(absltest.TestCase):

  def test(self):
    output_path = FLAGS.test_tmpdir
    output_name = 'temp'
    shards = 1

    # run the beam job
    with flagsaver.flagsaver(
        dataset_path=output_path,
        dataset_name=output_name,
        num_shards=shards,
        max_time=0.03,
        num_time_slices=10,
        num_samples=2):
      create_training_data.main([])

    # verify that file was written
    data_path = os.path.join(output_path,
                             output_name + '.tfrecord-00000-of-0000%i' % shards)
    metadata_path = os.path.join(output_path, output_name + '.metadata')
    self.assertTrue(gfile.Exists(data_path))
    self.assertTrue(gfile.Exists(metadata_path))


if __name__ == '__main__':
  absltest.main()
