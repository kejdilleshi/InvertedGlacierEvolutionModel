# Argentière / upper Arve — dated constraints from 13.6 ka to today

Everything the Argentière experiment needs, on one grid that covers the whole
upper Arve: Argentière, Tour, the Mer de Glace system (Talèfre, Leschaux, Géant)
and the valley floor down past La Joux. The domain is that large because the
Lateglacial Argentière tongue reached La Joux on the Arve valley floor, and the
Talèfre and Mer de Glace records belong to the glacier that joins the Arve 2 km
further down.

## Pipeline

```
python tools/fetch_argentiere_sources.py     # igm env: geopandas + rasterio
python tools/prepare_argentiere_data.py --dx 200   (and --dx 100)
python tools/plot_argentiere_data.py --dx 200
```

`fetch` copies or downloads and crops the sources into `sources/` (≈6 MB).
`prepare` writes `argentiere_<dx>m.nc` and `argentiere_observations_<dx>m.csv`.
Load both with [`load_argentiere`](../loader.py).

## Sources

| what | source | licence |
|---|---|---|
| 2012/2021 DEM, thickness, bed, velocity; stake SMB 2012–2021 | `../igm-smb-inference/data/argentiere` | in-house |
| dated ice margins, bedrock/erratic exposure ages, ¹⁴C ages | **AlpIce-1**, Kamleitner et al. 2026, *Boreas*, doi:10.1111/bor.70084; data doi:10.5281/zenodo.21841122 | CC-BY 4.0 |
| LIA outlines + 30 m LIA ice surface | Reinthaler & Paul 2025, *TC*, doi:10.5194/tc-19-753-2025; doi:10.5281/zenodo.14336827 (French outlines from Gardent 2014) | CC-BY 4.0 |
| ice-free bed of the Alps, 50 m (AW3D30 − Cook et al. 2023) | Henz et al. 2025, *TC*, doi:10.5194/tc-19-5913-2025; doi:10.5281/zenodo.17608701 | CC-BY 4.0 |

## Domain

The sources are cropped to a 58 × 58 km box (Henz bed, LIA surface). The
model grid is cut from it automatically: the bounding box of everywhere ice
may grow, plus a 3 km margin, snapped to 1 km. At 200 m that is
x 325–353 km, y 5074–5103 km (145 × 140 cells), and the nearest ice-allowed
cell sits 2.6 km inside IGM's 5-cell boundary ramp. Ice may grow anywhere in
the upper Arve basin above the Arveyron (Mer de Glace) confluence, and down the
Arve valley for 8 km below it (3 km in the Holocene runs). Beyond that the
mask ends the tongue, so it never runs into the domain boundary.

## Things that were fixed on the way in

* **Vertical datum.** The Argentière radar/DEM products sit 52.3 m above AW3D30
  on terrain that has been ice-free since the LIA (residual σ 6.6 m): an
  ellipsoidal vs orthometric datum. AlpIce-1 elevations, the LIA surface and the
  stake altitudes are orthometric, so a fitted plane (≈ +52 m) is removed from
  the radar products. Without it every moraine would sit 52 m too low against
  the model surface.
* **Bed.** The radar product carries the *ice surface* under the neighbouring
  glaciers inside its box, so it is used only under the 2012 Argentière glacier
  and blended over 300 m into the Henz bed. On that ring the two agree to
  −3 m (median).
* **Lateral-moraine operator.** Unlike the YD-Aletsch prep, "toward the ice" is
  the downslope direction on the bed, not the nearest modern ice — the
  Lateglacial Arve ice margins are kilometres from any modern ice. The offset
  `k`, the bias and σ are calibrated on the one dated LIA lateral moraine
  (Talèfre T4, ¹⁰Be ≈ 1756 CE) against the LIA surface: at 100 m `k = 1`,
  bias −22 m, σ 30 m; at 200 m `k = 1`, bias +18 m, σ 60 m. That is one
  moraine, so treat σ as optimistic.
* **Drainage.** Depression filling now has an epsilon slope; without it flow
  paths stalled on flats and the Arve basin came out empty.

## Observations (200 m grid)

| kind | n | meaning | sources |
|---|---|---|---|
| `lateral` | 10 | ice surface reaches the crest at some time in the age window | Protin 2019 (Lognan L1–L3, J1, YD/EH lines), Protin 2021 (Talèfre T1, T2, T4) |
| `frontal` | 1 | ice just upstream, none just downstream (La Joux J2, 10.9 ka) | Protin 2019 |
| `icefree` | 18 | no ice above the sample after its exposure age | Protin 2019/2021, Lehmann 2019/2020 (Mer de Glace transects), Cara 2017, Courtial-Manent 2023 |
| `icefree_any` | 21 | Mer de Glace below the subfossil wood during its calibrated range | Le Roy 2015 |
| `covered_any` | 17 | … and back over it within 400 yr (the advance that buried it) | Le Roy 2015 |
| fields | 4 | LIA (1850) extent and thickness; Argentière 2012 and 2021 thickness | Reinthaler & Paul; radar products |

TCN ages are relative to the sampling year (taken from sample names such as
`ARG-16-9`), ¹⁴C ages are cal BP; everything is converted to calendar years CE.
Each moraine is scored at the best time inside `[age − err, age + err]`
(a soft minimum over snapshots), not at the nominal age.

**Morphostratigraphic ice margins** (group `morphostrat`, weight 0.5): J1 and
the undated lines that Protin et al. assign to the YD/early-Holocene
transition get the window 13.2–10.9 ka — younger than the erratic in front of
J1 (13.24 ka), older than J2 (10.91 ka). Drop them with
`load_argentiere(..., morphostrat=False)`.

**Valley-referenced scoring** (`load_argentiere(..., valley=True)`, runner
flag `--valley`). The operators above score the ice at each landform's own
cells; a glacier that stops short of a moraine then has a constant, gradient-free
misfit, and ice draped on a valley side or a nunatak next to a crest is scored as
if it were the glacier. With `valley=True` every landform is referred to the
flowline of the valley its ice came down (`core/valley.py`: Argentière, Talèfre,
Mer de Glace, plus Tour for its LIA front): frontal moraines and the mapped LIA,
2012 and 2021 fronts by the along-valley distance of the simulated terminus
(σ 300 m / 200 m); lateral moraines, bedrock and wood by the ice surface on the
valley axis at their cross-section (σ 60 m, offset +10 m: median and scatter of
axis-minus-margin surface on the LIA Argentière); dated moraines also by their
survival (no later advance beyond them, weight 0.5). Kinds and formulas are in
`core/moraine_obs.py`.

**Not used:** the Cremier moraines (ages flagged *unaccepted* in AlpIce-1), the
undated Jardin de Talèfre line, and everything outside the upper Arve basin
(Triolet/Val Ferret, Trient). The Arve Lateglacial chronology of Portal et al.
2026 (JQS, doi:10.1002/jqs.70093) is newer than AlpIce-1 and not included; it is
the natural extension back past 13.6 ka.

## Caveats

* The Mer de Glace tongue is debris-covered; the PDD model has no debris, so
  it will need a colder climate than reality to keep that tongue long.
* No avalanching: steep headwalls collect a thin ice veneer (same issue as
  Aletsch).
* Outside Argentière the bed is a model product (Cook et al. 2023 thickness).
