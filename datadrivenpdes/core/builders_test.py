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
from datadrivenpdes.core import builders
import tensorflow as tf

from absl.testing import absltest

# Use eager mode by default
tf.enable_eager_execution()


class BuildersTest(absltest.TestCase):

  def test_unstack_dict(self):
    tensors = {'a': tf.range(3), 'b': tf.range(3, 6)}
    unstacked = builders.unstack_dict(tensors, num=3)
    to_numpy = lambda value: {k: v.numpy() for k, v in value.items()}
    actual = [to_numpy(value) for value in unstacked]
    expected = [{'a': 0, 'b': 3}, {'a': 1, 'b': 4}, {'a': 2, 'b': 5}]
    self.assertEqual(expected, actual)


if __name__ == '__main__':
  absltest.main()
