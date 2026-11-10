"""Gradient-finiteness tests for the parametric PDD climate helpers.

``compute_integral_positive_temperature`` and
``compute_negative_temperature_ratio`` both evaluate ``acos(y)`` and
``sqrt(1 - y**2)`` at ``y = smooth_piecewise(T_m / T_s)``. Both have an
infinite derivative at ``|y| = 1``, so the whole point of ``smooth_piecewise``
is to keep ``|y|`` strictly inside the unit interval. It used to do that on the
left only: the right branch expanded to ``1 - w*(1 - p)**2`` and reached
exactly ``1.0`` in float32, which put an ``inf`` into the adjoint for any cell
whose ``T_m / T_s`` landed just short of ``1 + w``.

These run on CPU in under a second and need no emulator artifact.
"""

import os
import sys
import unittest

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from core.climate import (
    compute_integral_positive_temperature,
    compute_negative_temperature_ratio,
    smooth_piecewise,
)


class TestSmoothPiecewise(unittest.TestCase):
    """The saturating map must never actually reach +-1."""

    def test_never_reaches_unity(self):
        x = torch.linspace(-2.0, 2.0, 2_000_001, dtype=torch.float32)
        y = smooth_piecewise(x)
        self.assertLess(float(y.abs().max()), 1.0)
        self.assertTrue(bool((1.0 - y**2 > 0).all()),
                        "1 - y**2 must stay strictly positive for sqrt/acos")

    def test_stays_monotone_and_bounded(self):
        x = torch.linspace(-2.0, 2.0, 200_001, dtype=torch.float32)
        y = smooth_piecewise(x)
        self.assertTrue(bool((y[1:] - y[:-1] >= 0).all()), "must be non-decreasing")

    def test_identity_in_the_linear_band(self):
        x = torch.linspace(-0.85, 0.85, 1001, dtype=torch.float32)
        torch.testing.assert_close(smooth_piecewise(x), x)


class TestPDDGradients(unittest.TestCase):
    """The PDD terms must be differentiable across the whole input range."""

    T_s = 10.0

    def _grad(self, lo, hi, n=2_000_001):
        x = torch.linspace(lo, hi, n, dtype=torch.float32, requires_grad=True)
        T_s = torch.tensor(self.T_s)
        f = (compute_integral_positive_temperature(x * T_s, T_s)
             + compute_negative_temperature_ratio(x * T_s, T_s))
        grad, = torch.autograd.grad(f.sum(), x)
        return grad

    def test_finite_across_both_transitions(self):
        # -1.55 .. 0.8 is what Aletsch spans at T_lowest = 8 C; the sweep is
        # widened past both seams so a regression cannot hide outside it.
        for lo, hi, label in [(-2.0, 2.0, "full range"),
                              (-1.2, -0.8, "left transition"),
                              (0.8, 1.2, "right transition")]:
            with self.subTest(band=label):
                grad = self._grad(lo, hi)
                bad = int((~torch.isfinite(grad)).sum())
                self.assertEqual(bad, 0, f"{bad} non-finite gradients in {label}")

    def test_gradient_magnitude_is_bounded(self):
        # eps = 1e-6 in smooth_piecewise caps the acos/sqrt derivative; if the
        # cap were lost this would run to ~1e4 or to inf.
        grad = self._grad(-2.0, 2.0)
        self.assertLess(float(grad.abs().max()), 1e3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
