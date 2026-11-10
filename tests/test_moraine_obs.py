"""Unit tests for core/moraine_obs.py (CPU, no data files needed)."""

import math

import torch

from core.moraine_obs import misfit_at, obs_times, select, soft_min


def test_soft_min_single_value_is_identity():
    v = torch.tensor([3.7])
    assert torch.isclose(soft_min(v), v[0])


def test_soft_min_bounded_by_min_and_mean():
    v = torch.tensor([0.0, 5.0, 9.0])
    s = soft_min(v)
    assert v.min() <= s <= v.mean()
    assert soft_min(v, tau=1e-3) < 1e-2


def test_obs_times_clip_to_window():
    o = dict(kind="lateral", t0=-100.0, t1=250.0)
    t = obs_times(o, 0.0, 200.0, 50.0)
    assert t[0] == 50.0 and t[-1] <= 200.0
    # narrow window: scored once, at its centre
    o = dict(kind="lateral", t0=110.0, t1=120.0)
    assert obs_times(o, 0.0, 200.0, 50.0) == [115.0]
    # entirely outside
    assert select([dict(kind="lateral", t0=300.0, t1=400.0)], 0.0, 200.0, 50.0) == []


def test_icefree_scores_only_ice_above_sample():
    topo = torch.tensor([[1000.0, 1000.0]])
    cells = torch.tensor([[True, False]])
    o = dict(kind="icefree", cells=cells, z_pt=1050.0, sigma=10.0)
    H_ok = torch.tensor([[40.0, 500.0]])        # surface 1040 < sample: fine
    H_bad = torch.tensor([[70.0, 0.0]])         # 20 m of ice above the sample
    assert misfit_at(o, H_ok, topo) == 0
    assert math.isclose(float(misfit_at(o, H_bad, topo)), 4.0)


def test_lateral_is_surface_residual():
    topo = torch.zeros(1, 3)
    cells = torch.tensor([[False, True, True]])
    o = dict(kind="lateral", cells=cells, z=torch.full((1, 3), 100.0), sigma=10.0)
    H = torch.tensor([[0.0, 100.0, 120.0]])
    assert math.isclose(float(misfit_at(o, H, topo)), 2.0)
