# YD-Aletsch — dated moraines of the Great Aletsch Glacier

## Source files (as delivered)

| file | grid | contents |
|---|---|---|
| `aletsch_glacier_YD_moraine1.nc` | 79 × 74 @ 30 m | binary landform mask + summary exposure ages |
| `aletsch_glacier_YD_moraine2.nc` | 26 × 40 @ 30 m | binary landform mask + summary exposure ages |
| `aletsch_glacier_LIA.nc` | 224 × 219 @ 30 m | binary landform mask, no ages |

All three are EPSG:32632. Each is a **moraine ridge**, not a glacier outline: the
LIA file holds the pair of lateral moraines that flank the historical tongue and
join at its terminus, and the two YD files hold single lateral segments.

## Prepared file

`tools/prepare_yd_data.py` joins them to the 100 m model grid of
`data/geology.nc` and writes:

* **`aletsch_yd_100m.nc`** — one file, 225 × 132 @ 100 m, holding `topg`,
  `usurf`, `thk`, `thk_1880`, `icemask`, `catchment`, and per-landform
  `moraine_<name>` / `z_cell_<name>` / `z_crest_<name>`, plus `moraine_id`.
* **`aletsch_yd_moraines.csv`** — the same 421 moraine cells as a table.
* **`aletsch_yd_overview.png`** — `tools/plot_yd_data.py`.

Load it with [`load_yd_aletsch`](../loader.py):

```python
from data.loader import load_yd_aletsch
topo, H_init, ice_mask, moraines = load_yd_aletsch()
target = moraines['YD1']          # 'mask', 'z_cell', 'z_crest', 'age'
```

## The four segments

| name | cells | elevation | age | position |
|---|---|---|---|---|
| `YD2`   |  35 | 1593–2028 m | 12 270–14 190 yr BP | true-right flank, **downvalley of the LIA terminus** |
| `YD1`   | 100 | 1828–2222 m | 11 780–12 710 yr BP | true-left flank, outboard of and ~300 m above `LIA_E` |
| `LIA_W` | 118 | 1424–2239 m | ~1850 CE | true-right lateral |
| `LIA_E` | 168 | 1435–2375 m | ~1850 CE | true-left lateral |

The LIA landform arrives as a single connected loop; the prep script cuts the
southernmost 500 m (the terminal arc) to separate the two limbs and then gives
the cut cells back to the nearer one.

## The observation operator

A lateral moraine records the ice surface at the time it was built, so the data
term is *simulated surface elevation*, not ice extent:

```python
resid = (topo + H)[m['inboard']] - m['z_target'][m['inboard']]
```

**Score on `inboard`, not on the moraine cells.** Sampling at the moraine cell
itself is degenerate: with no ice there the surface is the bed, which *is* the
moraine elevation, so an empty valley scores a perfect fit and the objective has
a spurious minimum at "no glacier". `inboard` is the same cells stepped 3 cells
(300 m) toward the ice, where the model has to grow ice to that height to score
well.

The LIA pair calibrates the operator, because `geology.nc` carries the 1880
surface — a known ice surface beside a moraine of known age:

| operator | residual vs the 1880 surface |
|---|---|
| `z_target` on `inboard` (3 cells in) | **+2 ± 30 m** ← use this |
| `z_cell` at the moraine cell | −2 ± 40 m (unbiased but degenerate) |
| `z_crest`, 3 × 3 DEM max | +80 m — biased, reference only |

σ ≈ 30 m is the weight for the data term. `z_crest` was an attempt to recover
the ridge top that the 100 m DEM smooths away, but on these valley walls a
3 × 3 maximum mostly climbs the slope.

## Masks

`icemask` (161 km²) comes from `geology.nc` and covers the modern glacier plus
the LIA corridor. `catchment` (226 km²) is the Massa basin: D8 drainage on the
modern surface, seeded with the modern ice and the moraines, unioned with the
dilated outflow corridor.

**The mask choice is the dominant systematic — about 1 °C.** Held at
T_m_lowest = 7 °C, P = 1.8 m/yr, melt factor 2.7/12 for 400 yr:

| mask | area | LIA_W | LIA_E | YD1 |
|---|---|---|---|---|
| `icemask`   |  98 km² | +33 m | +36 m | −192 m (ice on 6% of cells) |
| `catchment` | 159 km² | +119 m | +109 m | −160 m (46%) |

`icemask` reproduces the LIA laterals to within one σ, but it was drawn around
the observed LIA extent, so using it for a paleo inversion assumes part of the
answer — and it is too small to let a YD glacier grow. `catchment` lets the
glacier grow freely but adds a thin ice veneer on high steep rock that the model
has no way to shed (no avalanching), which overfeeds the tongue by ~100 m of
surface elevation. Invert both stages under the *same* mask and report
ΔT between them, where the bias largely cancels.

## Two things to know before using this

* **The YD terminus is not in the domain.** `geology.nc` stops at N = 5 135 350,
  and 8 of the 35 `YD2` cells fall inside the 5-cell boundary ramp where
  `apply_boundary_condition` forces the thickness to zero. `YD2` cannot be
  fitted on this grid; `YD1` (nearest cell 7 rows in) can. Extending the domain
  south down the Massa would be needed to use `YD2` or any terminus constraint.
* **`T_m_lowest` is referenced to a moving datum.** `apply_lapse_rate` subtracts
  `min(Z_surf)`, the minimum of the *ice surface* passed in, currently 799 m.
  Pin it explicitly before comparing temperatures across experiments.
