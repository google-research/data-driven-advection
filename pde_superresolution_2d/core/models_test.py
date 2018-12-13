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
"""Test of models classes in models.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pde_superresolution_2d.advection import equations as advection_equations
from pde_superresolution_2d.advection import velocity_fields
from pde_superresolution_2d.core import grids
from pde_superresolution_2d.core import models
from pde_superresolution_2d.core import states
from pde_superresolution_2d.core import utils
import tensorflow as tf

from absl.testing import absltest


class RollFiniteDifferenceModelTest(absltest.TestCase):
  """Tests for RollFiniteDifference model."""

  def setUp(self):
    length = 2 * np.pi
    size = 200
    step = length / size
    self.grid = grids.Grid(size, size, step)
    self.model = models.RollFiniteDifferenceModel()

  def test_roll_method(self):
    """Tests the roll method used for finite differences."""
    inputs_value = [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]
    roll_x_expected_result = [[[7, 8, 9], [1, 2, 3], [4, 5, 6]]]
    roll_y_expected_result = [[[3, 1, 2], [6, 4, 5], [9, 7, 8]]]
    with tf.Graph().as_default():
      inputs = tf.constant(inputs_value, dtype=tf.float64)
      roll_x_result = utils.roll_2d(inputs, (1, 0))
      roll_y_result = utils.roll_2d(inputs, (0, 1))
      with tf.Session():
        np.testing.assert_allclose(roll_x_result.eval(), roll_x_expected_result)
        np.testing.assert_allclose(roll_y_result.eval(), roll_y_expected_result)

  def test_spatial_derivatives(self):
    """Test that finite difference derivatives are close to true derivatives."""
    derivative_keys = (
        advection_equations.C,
        advection_equations.C_X,
        advection_equations.C_XX,
        advection_equations.C_Y,
        advection_equations.C_YY,
        advection_equations.C_X_EDGE_X,
        advection_equations.C_Y_EDGE_Y
    )

    x, y = self.grid.get_mesh()
    x_shifted_x, x_shifted_y = self.grid.get_mesh((1, 0))
    y_shifted_x, y_shifted_y = self.grid.get_mesh((0, 1))
    amplitudes = (0.15, 0.76, 1.213)
    wavelengths_x = (2, 3, 1)
    wavelengths_y = (3, 1, 2)
    phase_offsets = (0.123, 2.543, 1.734)
    exact_values = {key: 0 for key in derivative_keys}
    for i in range(len(amplitudes)):
      kx = wavelengths_x[i]
      ky = wavelengths_y[i]
      phi = phase_offsets[i]
      ampl = amplitudes[i]
      center_phase = kx * x + ky * y + phi
      x_edge_phase = kx * x_shifted_x + ky * x_shifted_y + phi
      y_edge_phase = kx * y_shifted_x + ky * y_shifted_y + phi

      exact_values[advection_equations.C] += ampl * np.sin(center_phase)
      exact_values[advection_equations.C_X] += ampl * kx * np.cos(center_phase)
      exact_values[advection_equations.C_Y] += ampl * ky * np.cos(center_phase)
      exact_values[advection_equations.C_XX] -= ampl * kx * kx * np.sin(center_phase)
      exact_values[advection_equations.C_YY] -= ampl * ky * ky * np.sin(center_phase)
      exact_values[advection_equations.C_X_EDGE_X] += ampl * kx * np.cos(x_edge_phase)
      exact_values[advection_equations.C_Y_EDGE_Y] += ampl * ky * np.cos(y_edge_phase)

    for key in exact_values.keys():
      exact_values[key] = np.expand_dims(exact_values[key], axis=0)

    distribution = exact_values[advection_equations.C]
    with tf.Graph().as_default():
      state_tensor = tf.constant(distribution, dtype=tf.float64)
      state = {advection_equations.C: state_tensor}
      spatial_derivatives = self.model.state_derivatives(
          state, 0, self.grid, derivative_keys)
      with tf.Session() as sess:
        derivatives_values = sess.run(spatial_derivatives)

    for key in derivatives_values.keys():
      np.testing.assert_allclose(
          derivatives_values[key], exact_values[key], atol=1e-2)

  def test_spatial_derivatives_exceptions(self):
    """Tests that unsupported requests results in exception."""
    with tf.Graph().as_default():
      state_tensor = tf.random_normal(shape=(1, 100, 100), dtype=tf.float64)
      unsupported_key_a = states.StateKey('pressure', (0, 0), (0, 0))
      unsupported_key_b = states.StateKey('concentration', (5, 0), (0, 0))
      unsupported_key_c = states.StateKey('concentration', (1, 0), (1, 0))
      state = {advection_equations.C: state_tensor}
      bad_request_a = (unsupported_key_a,)
      bad_request_b = (unsupported_key_b,)
      bad_request_c = (unsupported_key_c,)
      bad_request_d = (advection_equations.C, unsupported_key_c)
      with self.assertRaises(ValueError):
        self.model.state_derivatives(state, 0, self.grid, bad_request_a)
      with self.assertRaises(ValueError):
        self.model.state_derivatives(state, 0, self.grid, bad_request_b)
      with self.assertRaises(ValueError):
        self.model.state_derivatives(state, 0, self.grid, bad_request_c)
      with self.assertRaises(ValueError):
        self.model.state_derivatives(state, 0, self.grid, bad_request_d)

  def test_proto_conversion(self):
    """Test that the model can be converted to protocol buffer and back."""
    model_proto = self.model.to_proto()
    self.assertEqual(model_proto.WhichOneof('model'), 'roll_finite_difference')
    model_from_proto = models.model_from_proto(model_proto)
    self.assertIsInstance(model_from_proto, models.RollFiniteDifferenceModel)


class StencilNetTest(absltest.TestCase):
  """Tests for StencilNet model."""

  def setUp(self):
    """Setup testing components."""
    length = 2 * np.pi
    size = 200
    step = length / size
    self.grid = grids.Grid(size, size, step)

    num_layers = 2
    kernel_size = 3
    num_filters = 16
    stencil_size = 4
    input_shift = 0.5
    input_variance = 4.98
    self.model = models.StencilNet(num_layers, kernel_size, num_filters,
                                   stencil_size, input_shift, input_variance)

  def test_network_construction(self):
    """Tests that model generates networks for spatial derivatives."""
    tf.reset_default_graph()
    input_state = {advection_equations.C: np.random.random((1,) + self.grid.get_shape())}
    request = (advection_equations.C_EDGE_X, advection_equations.C_EDGE_Y, advection_equations.C_XX, advection_equations.C_YY)
    derivs = self.model.state_derivatives(input_state, 0., self.grid, request)
    with tf.train.MonitoredSession() as sess:
      derivs_values = sess.run(derivs)

    expected_shape = (1,) + self.grid.get_shape()
    for key in request:
      self.assertEqual(np.shape(derivs_values[key]), expected_shape)


class StencilVNetTest(absltest.TestCase):
  """Tests StencilVNet model."""

  def setUp(self):
    """Setup testing components."""
    length = 2 * np.pi
    size = 200
    step = length / size
    self.grid = grids.Grid(size, size, step)

    num_terms = 4
    max_periods = 2
    seed = 3
    self.velocity_field = velocity_fields.ConstantVelocityField.from_seed(
        num_terms, max_periods, seed)

    num_layers = 2
    kernel_size = 3
    num_filters = 16
    stencil_size = 4
    input_shift = 0.5
    input_variance = 4.98
    self.model = models.StencilVNet(
        self.velocity_field, self.grid, num_layers, kernel_size, num_filters,
        stencil_size, input_shift, input_variance)

  def test_network_construction(self):
    """Tests that model generates networks for spatial derivatives."""
    tf.reset_default_graph()
    input_state = {advection_equations.C: np.random.random((1,) + self.grid.get_shape())}
    request = (advection_equations.C_EDGE_X, advection_equations.C_EDGE_Y, advection_equations.C_XX, advection_equations.C_YY)
    derivs = self.model.state_derivatives(input_state, 0., self.grid, request)
    with tf.train.MonitoredSession() as sess:
      derivs_values = sess.run(derivs)

    expected_shape = (1,) + self.grid.get_shape()
    for key in request:
      self.assertEqual(np.shape(derivs_values[key]), expected_shape)


if __name__ == '__main__':
  absltest.main()
