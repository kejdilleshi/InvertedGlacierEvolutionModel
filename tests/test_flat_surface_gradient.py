"""A flat ice surface must not blow up the emulator's adjoint.

``SIADecompNet`` feeds the network a downhill unit vector,
``dir = -grad(s) / |grad(s)|``, and a ``log|grad(s)|`` deformation proxy. Both
are bounded in value however small the slope gets -- a unit vector just points
somewhere arbitrary -- but both differentiate like ``1 / |grad(s)|``. IGM's
``FIXED_EPS = 1e-8`` floors ``|grad(s)|`` at 1e-4, so a cell whose ice surface
happens to be flat hands the adjoint a derivative of order 1e4. Inside a time
loop that feeds velocity back into geometry every step, that compounds: the
LIA inversion detonated at a single flat cell after 7 iterations, with a
perfectly finite forward pass throughout.

``FIXED_SLOPE_EPS = 1e-6`` floors the slope at 1e-3 instead, capping the
derivative at 1e3. These tests pin both halves: that the default is bounded,
and that IGM's value really is the order of magnitude worse that motivated
the change.
"""

import os
import sys
import unittest

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from core.emulator import SIADecompNet


def _net(slope_eps=None):
    torch.manual_seed(0)
    net = SIADecompNet(
        input_names=["thk", "usurf", "tau_ref", "arrhenius", "dX"],
        Nz=2,
        network_params={"nb_layers": 4, "nb_out_filter": 8,
                        "context_dilation_schedule": [1, 2]},
        slope_eps=slope_eps,
    )
    return net.eval()


def _flat_patch_inputs(n=24):
    """Ice with a dead-flat surface over the middle of the domain."""
    y, x = torch.meshgrid(torch.arange(n, dtype=torch.float32),
                          torch.arange(n, dtype=torch.float32), indexing="ij")
    bed = 2000.0 - 4.0 * y - 2.0 * x
    thk = torch.full((n, n), 150.0)
    surf = bed + thk
    surf[8:16, 8:16] = float(surf[8:16, 8:16].mean())   # exactly flat patch
    thk = (surf - bed).clamp(min=1.0)
    fields = torch.stack([thk, surf,
                          torch.full((n, n), 0.2154),
                          torch.full((n, n), 78.0),
                          torch.full((n, n), 100.0)])
    return fields.unsqueeze(0)


def _max_grad(slope_eps):
    net = _net(slope_eps)
    x = _flat_patch_inputs().clone().requires_grad_(True)
    out = net(x)
    out.abs().sum().backward()
    grad = x.grad[:, 1]          # d(velocity) / d(usurf)
    assert torch.isfinite(grad).all(), "gradient went non-finite"
    return float(grad.abs().max())


class TestFlatSurfaceGradient(unittest.TestCase):

    def test_default_slope_floor_is_the_safe_one(self):
        self.assertEqual(SIADecompNet.FIXED_SLOPE_EPS, 1e-6)
        self.assertEqual(_net().slope_eps, 1e-6)

    def test_gradient_is_finite_and_bounded_over_a_flat_patch(self):
        self.assertLess(_max_grad(None), 1e3)

    def test_igm_slope_floor_is_an_order_of_magnitude_worse(self):
        """The regression this guards against: 1e-8 really is the hazard."""
        safe = _max_grad(SIADecompNet.FIXED_SLOPE_EPS)
        igm = _max_grad(SIADecompNet.FIXED_EPS)
        self.assertGreater(igm / safe, 5.0,
                           f"expected IGM's floor to be much worse; got {igm:.3e} vs {safe:.3e}")

    def test_slope_eps_round_trips_through_resolved_params(self):
        net = _net(slope_eps=1e-5)
        self.assertEqual(net.resolved_params()["slope_eps"], 1e-5)

    def test_sloped_surface_is_unaffected_by_the_floor(self):
        """On real slopes the two floors must agree -- this is not a physics change."""
        torch.manual_seed(0)
        n = 24
        y, x = torch.meshgrid(torch.arange(n, dtype=torch.float32),
                              torch.arange(n, dtype=torch.float32), indexing="ij")
        bed = 2000.0 - 4.0 * y - 2.0 * x
        thk = torch.full((n, n), 150.0)
        fields = torch.stack([thk, bed + thk,
                              torch.full((n, n), 0.2154),
                              torch.full((n, n), 78.0),
                              torch.full((n, n), 100.0)]).unsqueeze(0)
        with torch.no_grad():
            a = _net(SIADecompNet.FIXED_SLOPE_EPS)(fields)
            b = _net(SIADecompNet.FIXED_EPS)(fields)
        rel = float((a - b).abs().max() / b.abs().max())
        self.assertLess(rel, 1e-4, f"slope floor changed a sloped solution by {rel:.2e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
