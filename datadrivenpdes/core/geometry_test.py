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
import collections

import numpy as np
from datadrivenpdes.core import geometry
from datadrivenpdes.core import states
import tensorflow as tf

from absl.testing import absltest

# tests assume eager execution
tf.enable_eager_execution()


X = states.Dimension.X


class GeometryTest(absltest.TestCase):

  def test_reflection(self):
    definitions = {
        'q': states.StateDefinition('q', (), (0, 0, 0), (0, 0)),
        'q_edge_x': states.StateDefinition('q', (), (0, 0, 0), (1, 0)),
        'q_edge_y': states.StateDefinition('q', (), (0, 0, 0), (0, 1)),
        'q_x': states.StateDefinition('q', (), (1, 0, 0), (0, 0)),
        'q_xy': states.StateDefinition('q', (), (1, 1, 0), (0, 0)),
        'q_xx': states.StateDefinition('q', (), (2, 0, 0), (0, 0)),
        'x_velocity': states.StateDefinition('q', (X,), (0, 0, 0), (0, 0)),
    }
    transform = geometry.Reflection([X], definitions)
    self.assertEqual(repr(transform), '<Reflection [X]>')

    inputs = {
        k: tf.convert_to_tensor(np.array([[0, 1, 2, 3]]).T)
        for k in definitions
    }
    result = transform.forward(inputs)
    np.testing.assert_array_equal(
        result['q'].numpy().T, [[3, 2, 1, 0]])
    np.testing.assert_array_equal(
        result['q_edge_x'].numpy().T, [[2, 1, 0, 3]])
    np.testing.assert_array_equal(
        result['q_edge_y'].numpy().T, [[3, 2, 1, 0]])
    np.testing.assert_array_equal(
        result['q_x'].numpy().T, [[-3, -2, -1, 0]])
    np.testing.assert_array_equal(
        result['q_xy'].numpy().T, [[-3, -2, -1, 0]])
    np.testing.assert_array_equal(
        result['q_xx'].numpy().T, [[3, 2, 1, 0]])
    np.testing.assert_array_equal(
        result['x_velocity'].numpy().T, [[-3, -2, -1, 0]])

  def test_permutation(self):
    definitions = {
        'q_edge_x': states.StateDefinition('q', (), (0, 0, 0), (1, 0)),
        'q_edge_y': states.StateDefinition('q', (), (0, 0, 0), (0, 1)),
    }
    transform = geometry.Permutation(definitions)

    rs = np.random.RandomState(0)
    inputs = {'q_edge_x': tf.convert_to_tensor(rs.randn(5, 5))}
    result = transform.forward(inputs)
    self.assertEqual(set(result), {'q_edge_y'})
    np.testing.assert_array_equal(
        result['q_edge_y'].numpy().T, inputs['q_edge_x'].numpy())

  def test_symmetries_of_the_square(self):
    definitions = {
        'q': states.StateDefinition('q', (), (0, 0, 0), (0, 0)),
        'q_x': states.StateDefinition('q', (), (0, 0, 0), (1, 0)),
        'q_y': states.StateDefinition('q', (), (0, 0, 0), (0, 1)),
    }
    transforms = geometry.symmetries_of_the_square(definitions)
    self.assertLen(transforms, 8)

    rs = np.random.RandomState(0)
    state = {
        'q': tf.convert_to_tensor(rs.randn(5, 5)),
        'q_x': tf.convert_to_tensor(rs.randn(5, 5)),
        'q_y': tf.convert_to_tensor(rs.randn(5, 5)),
    }
    for transform in transforms:
      with self.subTest(transform):
        roundtripped = transform.inverse(transform.forward(state))
        np.testing.assert_array_equal(
            state['q'].numpy(), roundtripped['q'].numpy())
        np.testing.assert_array_equal(
            state['q_x'].numpy(), roundtripped['q_x'].numpy())
        np.testing.assert_array_equal(
            state['q_y'].numpy(), roundtripped['q_y'].numpy())

    with self.subTest('uniqueness'):
      rs = np.random.RandomState(0)
      state = {
          'q': tf.convert_to_tensor(rs.randn(5, 5)),
      }
      results_set = {
          tuple(transform.forward(state)['q'].numpy().ravel().tolist())
          for transform in transforms
      }
      self.assertLen(results_set, 8)

    with self.subTest('permutation_counts'):
      rs = np.random.RandomState(0)
      state = {
          'q': tf.convert_to_tensor(rs.randn(5, 5)),
          'q_x': tf.convert_to_tensor(rs.randn(5, 5)),
      }
      counts = collections.defaultdict(int)
      for transform in transforms:
        result = transform.forward(state)
        for k in result:
          counts[k] += 1
      self.assertEqual(counts, {'q': 8, 'q_x': 4, 'q_y': 4})


if __name__ == '__main__':
  absltest.main()
