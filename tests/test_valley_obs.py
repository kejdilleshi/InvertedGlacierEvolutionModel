"""Unit tests for core/valley.py and the valley-referenced kinds of core/moraine_obs.py.

CPU, synthetic geometry, no data files.
"""

import math

import numpy as np
import torch

from core.moraine_obs import (data_term, huber, misfit_at, terminus_position)
from core.valley import build_flowline


def _line(n=40, dx=200.0):
    path = torch.arange(n)
    s = torch.arange(n, dtype=torch.float32) * dx
    return path, s


def test_terminus_at_the_farthest_ice():
    path, s = _line()
    H = torch.zeros(40)
    H[:20] = 150.0                                   # ice on cells 0..19
    L = float(terminus_position(H, path, s))
    assert abs(L - float(s[19])) < 20.0
    assert float(terminus_position(torch.zeros(40), path, s)) == 0.0   # no ice: the head


def test_terminus_moves_continuously_as_the_snout_cell_fills():
    path, s = _line()
    Ls = []
    for h in (0.0, 1.0, 5.0, 20.0, 60.0, 150.0):
        H = torch.zeros(40)
        H[:20] = 150.0
        H[20] = h
        Ls.append(float(terminus_position(H, path, s)))
    assert all(b > a for a, b in zip(Ls, Ls[1:]))
    assert abs(Ls[0] - float(s[19])) < 20.0 and abs(Ls[-1] - float(s[20])) < 20.0
    assert max(b - a for a, b in zip(Ls, Ls[1:])) < float(s[1] - s[0])     # no jump of a cell


def test_far_from_the_moraine_the_gradient_pulls_toward_it():
    """A glacier 3 km short of a frontal moraine still feels it -- the case where
    an overlap misfit is flat."""
    path, s = _line()
    H = torch.zeros(40)
    H[:20] = 150.0
    H[20] = 8.0                                       # thin snout cell
    H.requires_grad_(True)
    o = dict(kind="terminus", path=path, s=s, s_obs=float(s[35]), sigma_L=300.0)
    m = misfit_at(o, H, torch.zeros(40))
    m.backward()
    assert float(m) > 50.0
    assert H.grad[20] < 0                             # thicker snout -> smaller misfit
    assert torch.all(H.grad[25:] == 0)                # nothing from cells without ice


def test_no_nan_gradient_without_ice():
    path, s = _line()
    H = torch.zeros(40, requires_grad=True)
    o = dict(kind="terminus", path=path, s=s, s_obs=1000.0, sigma_L=300.0)
    misfit_at(o, H, torch.zeros(40)).backward()
    assert torch.isfinite(H.grad).all()


def test_lateral_v_reach_then_height():
    path, s = _line()
    topo = torch.linspace(3000.0, 1000.0, 40)
    axis = torch.tensor([30])
    o = dict(kind="lateral_v", path=path, s=s, axis=axis, s_req=float(s[30]),
             z=torch.tensor([float(topo[30]) + 100.0]), delta=0.0, sigma_L=300.0, sigma_z=50.0)
    short = torch.zeros(40)
    short[:20] = 150.0
    long_thin = torch.zeros(40)
    long_thin[:32] = 60.0
    long_right = torch.zeros(40)
    long_right[:32] = 100.0
    m_short, m_thin, m_right = (float(misfit_at(o, H, topo)) for H in (short, long_thin, long_right))
    assert m_short > m_thin > m_right
    assert m_right < 0.05


def test_huber_is_quadratic_then_linear():
    v = torch.tensor([0.0, 4.0, 9.0, 100.0])
    h = huber(v, c=3.0)
    assert torch.allclose(h[:3], v[:3])
    assert math.isclose(float(h[3]), 2 * 3 * 10 - 9, rel_tol=1e-5)
    assert float(huber(torch.tensor(-1e-7))) == 0.0


def test_group_weighting_is_not_diluted_by_satisfied_constraints():
    terms = {"a": torch.tensor(4.0), "b": torch.tensor(0.0), "c": torch.tensor(0.0)}
    obs = [dict(id="a", weight=1.0)]
    more = obs + [dict(id="b", weight=1.0), dict(id="c", weight=1.0)]
    assert float(data_term(obs, terms, "groups")) == float(data_term(more, terms, "groups"))
    assert float(data_term(more, terms, "mean")) < float(data_term(obs, terms, "mean"))
    # members of one group are averaged, so five samples of one event count once
    same = [dict(id="a", weight=1.0, group_id="g"), dict(id="b", weight=1.0, group_id="g")]
    assert float(data_term(same, terms, "groups")) == 2.0


def _valley(ny=60, nx=41, dx=100.0):
    """A V-shaped valley draining toward +y, with its axis at x = nx // 2."""
    y, x = np.mgrid[0:ny, 0:nx]
    return 3000.0 - 15.0 * y + 25.0 * np.abs(x - nx // 2), dx


def test_flowline_follows_the_axis_and_projects_side_landforms():
    bed, dx = _valley()
    ny, nx = bed.shape
    moraine = np.zeros_like(bed, dtype=bool)
    moraine[40, nx // 2 + 6] = True                  # on the valley side, 600 m off the axis
    fl = build_flowline("v", {"m": moraine}, bed, dx, tail_m=500.0)
    iy, ix = np.unravel_index(fl.active, bed.shape)
    assert np.all(ix[1:] == nx // 2)                 # straight down the axis below the head
    k, d = fl.project(moraine)
    assert iy[k[0]] == 40 and abs(d[0] - 600.0) < 1e-6
    assert fl.s[fl.stop - 1] >= fl.s[k[0]] + 400.0   # runs on past the landform


def test_hillslope_ice_does_not_fit_a_lateral_moraine():
    """Ice draped on the valley side at the crest, with the valley glacier far
    below it, fits the old cell operator but not the valley operator."""
    bed, dx = _valley()
    ny, nx = bed.shape
    crest = np.zeros_like(bed, dtype=bool)
    crest[40, nx // 2 + 6] = True
    fl = build_flowline("v", {"m": crest}, bed, dx, tail_m=500.0)
    k, _ = fl.project(crest)
    topo = torch.tensor(bed, dtype=torch.float32)
    z_crest = float(bed[crest][0])
    H = torch.zeros_like(topo)
    H[:45, nx // 2] = 40.0                            # thin valley glacier past the moraine
    H[40, nx // 2 + 5] = 30.0                         # hillslope ice right under the crest
    old = dict(kind="lateral", cells=torch.tensor(np.roll(crest, -1, axis=1)),
               z=torch.full_like(topo, z_crest), sigma=50.0)
    new = dict(kind="lateral_v", path=torch.tensor(fl.active), s=torch.tensor(fl.s[:fl.stop]),
               axis=torch.tensor(fl.active[k]), s_req=float(fl.s[k].max()),
               z=torch.tensor([z_crest]), delta=0.0, sigma_L=300.0, sigma_z=50.0)
    assert float(misfit_at(old, H, topo)) < 1.0      # "fitted" by the hillslope ice
    assert float(misfit_at(new, H, topo)) > 4.0      # axis surface 110 m (2.2 sigma) below the crest


def test_another_glaciers_tongue_further_down_does_not_count():
    path, s = _line()
    H = torch.zeros(40)
    H[:20] = 150.0                                   # this glacier: cells 0..19
    L_alone = float(terminus_position(H, path, s))
    H[30:36] = 200.0                                 # another tongue on the same valley floor
    assert abs(float(terminus_position(H, path, s)) - L_alone) < 1e-3
    H[20:30] = 50.0                                  # once they join, the snout is the far one
    assert float(terminus_position(H, path, s)) > float(s[34])


def test_volume_surrogate_keeps_the_value_and_spreads_the_gradient():
    path, s = _line()
    H = torch.zeros(40)
    H[:20] = 150.0
    H[20] = 40.0                                     # thick snout cell: tiny native gradient
    H.requires_grad_(True)
    L0 = terminus_position(H, path, s)
    L1 = terminus_position(H, path, s, volume_scale=500.0)
    assert abs(float(L0) - float(L1)) < 1e-4
    g0, = torch.autograd.grad(L0, H)
    g1, = torch.autograd.grad(L1, H)
    assert float(g0[:20].abs().max()) < 1e-2          # upstream thickening is invisible natively
    assert torch.all(g1[:21] > 0.3)                   # ... and seen through the volume: ds / 500
    assert torch.all(g1[21:] == 0)


def test_adjoint_transport_keeps_the_forward_and_takes_the_upwind_gradient():
    import core.forward_schemes.igm_emulator_step as step
    from utils.emulator_tools import compute_divflux_limited
    g = torch.Generator().manual_seed(0)
    h = (torch.rand(12, 10, generator=g) * 100).requires_grad_(True)
    u, v = torch.rand(12, 10, generator=g) * 50, torch.rand(12, 10, generator=g) * 50
    w = torch.rand(12, 10, generator=g)
    try:
        step.use_adjoint_transport("godunov")
        div = step._divflux(u, v, h, 100.0, 100.0, 0.5)
        g_st, = torch.autograd.grad((div * w).sum(), h)
    finally:
        step.use_adjoint_transport(None)
    ref = compute_divflux_limited(u, v, h, 100.0, 100.0, 0.5)
    g_up, = torch.autograd.grad((compute_divflux_limited(u, v, h, 100.0, 100.0, 0.5, slope_type="godunov") * w).sum(), h)
    assert torch.allclose(div, ref)
    assert torch.allclose(g_st, g_up)
