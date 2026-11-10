"""Along-valley geometry: which ice can have built a landform, and how far it is.

A moraine low in a valley was built by ice that came down that valley, not by
whatever ice happens to be nearest in a straight line. On the model grid that
becomes a *flowline*: the steepest-descent (D8) path along the valley floor of
the bed, from the head of the source glacier's valley down past its outermost
landform. Each landform cell is then referred to the flowline cell nearest to
it -- its cross-section on the valley axis. Distances are arc lengths along
the flowline, so a terminus 3 km up-valley of a frontal moraine is 3 km from
it, wherever the neighbouring glaciers are, and ice draped on a valley side
above a lateral moraine is not mistaken for the glacier that built it.

``core/moraine_obs.py`` scores a simulated glacier against a landform along
that line: the terminus position against a frontal moraine, the ice surface
on the valley axis against a lateral one.

Numpy only (no torch), so the data preparation can use it too.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

_NEIGHBOURS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]


def fill_depressions(z, eps=1e-3):
    """Priority-flood depression filling (Barnes et al. 2014) with an epsilon slope.

    The epsilon (mm per cell) keeps every filled flat draining, so a flow path
    never stalls on a lake surface or a flat valley floor.
    """
    ny, nx = z.shape
    filled = np.full(z.shape, np.inf)
    done = np.zeros(z.shape, dtype=bool)
    heap = []
    for iy in range(ny):
        for ix in range(nx):
            if iy in (0, ny - 1) or ix in (0, nx - 1):
                heapq.heappush(heap, (z[iy, ix], iy, ix))
                done[iy, ix] = True
                filled[iy, ix] = z[iy, ix]
    while heap:
        zc, iy, ix = heapq.heappop(heap)
        for dy, dx in _NEIGHBOURS:
            jy, jx = iy + dy, ix + dx
            if not (0 <= jy < ny and 0 <= jx < nx) or done[jy, jx]:
                continue
            done[jy, jx] = True
            filled[jy, jx] = max(z[jy, jx], zc + eps)
            heapq.heappush(heap, (filled[jy, jx], jy, jx))
    return filled


def d8_receivers(z, fill=True):
    """Steepest-descent receiver (flat index) of every cell, and the surface used.

    A cell with no lower neighbour is its own receiver (an outlet, or a pit
    when ``fill`` is False).
    """
    ny, nx = z.shape
    zz = fill_depressions(z) if fill else np.asarray(z, dtype=np.float64)
    flat_idx = np.arange(z.size).reshape(z.shape)
    recv = np.arange(z.size)
    best = np.zeros(z.size)
    for dy, dx in _NEIGHBOURS:
        shifted = np.full(z.shape, np.inf)
        idx = np.full(z.shape, -1)
        ys, xs = slice(max(0, dy), ny + min(0, dy)), slice(max(0, dx), nx + min(0, dx))
        ys_src = slice(max(0, -dy), ny + min(0, -dy))
        xs_src = slice(max(0, -dx), nx + min(0, -dx))
        shifted[ys_src, xs_src] = zz[ys, xs]
        idx[ys_src, xs_src] = flat_idx[ys, xs]
        slope = (zz - shifted).ravel() / np.hypot(dy, dx)
        better = slope > best
        best[better] = slope[better]
        recv[better] = idx.ravel()[better]
    return recv, zz.ravel()


def flow_accumulation(recv, zz, weights=None):
    """Number of cells (or summed ``weights``) draining through each cell, itself included."""
    acc = (np.ones(recv.size) if weights is None
           else np.asarray(weights, dtype=float).ravel().copy())
    for cell in np.argsort(-zz, kind="stable"):          # high to low: donors first
        r = recv[cell]
        if r != cell:
            acc[r] += acc[cell]
    return acc


def path_down(cell, recv, max_steps=100000):
    """Flat indices of the D8 path from ``cell`` down to its outlet."""
    out = [int(cell)]
    while len(out) < max_steps:
        nxt = int(recv[out[-1]])
        if nxt == out[-1]:
            break
        out.append(nxt)
    return np.array(out)


def head_of_valley(cell, recv, acc, prefer=None):
    """Walk up the main valley from ``cell``: always to the donor draining most.

    ``prefer`` (optional, per cell) is a second accumulation -- e.g. the area of
    one glacier's LIA polygon draining through each cell. Where a donor carries
    some of it, the walk follows the donor with the most, so it climbs into
    that glacier's valley rather than into a larger neighbour's.
    """
    donors: Dict[int, List[int]] = {}
    for c, r in enumerate(recv):
        if r != c:
            donors.setdefault(int(r), []).append(c)
    cell = int(cell)
    while cell in donors:
        cand = donors[cell]
        if prefer is not None and max(prefer[d] for d in cand) > 0:
            cell = max(cand, key=lambda d: (prefer[d], acc[d]))
        else:
            cell = max(cand, key=lambda d: acc[d])
    return cell


@dataclass
class Flowline:
    """One valley's flowline on the model grid.

    ``cells`` are flat indices from the head of the valley downward and ``s``
    the arc length (m) at each. Only the first ``stop`` cells count: the line
    is cut before the next glaciated tributary below the outermost landform,
    so another glacier's tongue is never taken for this one's. ``start`` is a
    cell the glacier always covers, from which its terminus is traced.
    """
    name: str
    cells: np.ndarray
    s: np.ndarray
    stop: int
    shape: tuple
    dx: float
    start: int = 0

    @property
    def active(self):
        return self.cells[: self.stop]

    def project(self, mask):
        """Nearest active flowline index of every cell of ``mask``, and the distance (m).

        The nearest valley-axis cell is the landform's cross-section: a lateral
        moraine is compared with the ice surface there, whatever the valley
        side drains toward.
        """
        fy, fx = np.unravel_index(self.active, self.shape)
        my, mx = np.nonzero(np.asarray(mask).reshape(self.shape))
        d = np.hypot(my[:, None] - fy[None, :], mx[:, None] - fx[None, :])
        k = d.argmin(axis=1)
        return k, d[np.arange(k.size), k] * self.dx

    def position_of(self, mask):
        """Arc length (m) of the farthest active flowline cell inside ``mask`` (a terminus)."""
        inside = np.asarray(mask).reshape(-1)[self.active]
        if not inside.any():
            return np.nan
        return float(self.s[np.flatnonzero(inside).max()])


def build_flowline(name, landforms, bed, dx, recv=None, zz=None, prefer_mask=None,
                   other_ice=None, tail_m=3000.0, valley_acc=25.0,
                   min_tributary_cells=10):
    """Flowline of the valley whose ice built ``landforms`` (dict id -> boolean grid).

    Each landform is followed downslope until it reaches a valley floor
    (``valley_acc`` cells draining through it); the lowest of those points
    belongs to the outermost landform. From there the line climbs the main
    valley to its head -- into the ``prefer_mask`` glacier (e.g. its LIA
    polygon) at junctions -- and then runs down the D8 path. It stops
    ``tail_m`` below the outermost landform's cross-section, or earlier where a
    tributary carrying at least ``min_tributary_cells`` of ``other_ice`` (other
    glaciers' LIA cells) joins.
    """
    shape = bed.shape
    if recv is None:
        recv, zz = d8_receivers(bed, fill=True)
    acc = flow_accumulation(recv, zz)
    prefer = (flow_accumulation(recv, zz, weights=np.asarray(prefer_mask, dtype=float))
              if prefer_mask is not None else None)

    lowest = None
    for m in landforms.values():
        for c in np.flatnonzero(np.asarray(m).ravel()):
            while acc[c] < valley_acc and recv[c] != c:
                c = recv[c]
            if lowest is None or zz[c] < zz[lowest]:
                lowest = int(c)
    cells = path_down(head_of_valley(lowest, recv, acc, prefer=prefer), recv)
    iy, ix = np.unravel_index(cells, shape)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(iy), np.diff(ix)) * dx)])
    fl = Flowline(name, cells, s, stop=len(cells), shape=shape, dx=dx)

    outer = max(int(fl.project(m)[0].max()) for m in landforms.values())
    stop = int(np.searchsorted(s, s[outer] + tail_m, side="right"))
    if other_ice is not None:
        other = flow_accumulation(recv, zz, weights=np.asarray(other_ice, dtype=float))
        on_path = set(int(c) for c in cells)
        for k in range(outer + 1, min(stop, len(cells))):
            c = int(cells[k])
            joins = [d for d in np.flatnonzero(recv == c) if d != c and d not in on_path]
            if any(other[d] >= min_tributary_cells for d in joins):
                stop = k
                break
    fl.stop = int(min(stop, len(cells)))
    return fl
