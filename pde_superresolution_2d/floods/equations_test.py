"""Tests for google3.third_party.py.pde_superresolution_2d.floods.equations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pde_superresolution_2d.core import equations as core_equations
from pde_superresolution_2d.core import grids
from pde_superresolution_2d.floods import equations

from absl.testing import absltest


class EquationsTest(absltest.TestCase):

  def test_equation_map(self):
    self.assertIn('saint_venant', core_equations.CONTINUOUS_EQUATIONS)


class BaseSaintVenantEquationTestCases(object):

  def setUp(self):
    self.grid = grids.Grid(200, 200, 2 * np.pi / 200)
    self.equation = self.EQUATION()

  def test_random_state(self):
    # we use subTest() here because we can't use parmeterized tests with our
    # mixin/subclassing approach
    for batch_size in [(), (1,), (2, 3)]:
      with self.subTest(batch_size):
        init_state = self.equation.random_state(
            self.grid, batch_size=batch_size, seed=1)

        # keys and shapes should match
        self.assertEqual(set(init_state), set(self.equation.STATE_KEYS))
        for array in init_state.values():
          self.assertEqual(array.shape, batch_size + self.grid.shape)

        # seed should be deterministic
        init_state2 = self.equation.random_state(
            self.grid, batch_size=batch_size, seed=1)
        for key in init_state:
          np.testing.assert_allclose(init_state[key], init_state2[key])


class FiniteDifferenceSaintVenantTest(BaseSaintVenantEquationTestCases,
                                      absltest.TestCase):
  EQUATION = equations.FiniteDifferenceSaintVenant


class FiniteVolumeSaintVenantTest(BaseSaintVenantEquationTestCases,
                                  absltest.TestCase):
  EQUATION = equations.FiniteVolumeSaintVenant


if __name__ == '__main__':
  absltest.main()
