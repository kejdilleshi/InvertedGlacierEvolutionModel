# Inverted Glacier Evolution Model — README

Short overview
- A differentiable glacier-evolution forward model with gradient-based inversion. Ice dynamics come from a pretrained **IGM ice-flow emulator**, and gradients travel from simulated thickness back through every time step to whatever is being inverted for: an SMB field, an SMB elevation profile, an ELA, or climate parameters.

Ice dynamics — two forward schemes
- `igm` (default): the IGM `SIADecompNet` emulator. Sliding is the **Weertman** law, parameterized by the reference basal shear stress `tau_ref` (MPa).
- `SIA`: shallow ice, internal deformation only — **no sliding**.

Quick entry points
- [IGEM_2D_ELA_igm.ipynb](IGEM_2D_ELA_igm.ipynb) — invert for the ELA, end to end.
- [notebooks/LIA_temperature_inversion.ipynb](notebooks/LIA_temperature_inversion.ipynb) — invert for the 1500–2017 temperature history from extent alone, plus a parameter sensitivity study (melt factor, precipitation, `tau_ref`). Start here for the historical-climate use case.
- [notebooks/LIA_temperature_inversion_V2.ipynb](notebooks/LIA_temperature_inversion_V2.ipynb) — why V1 converges the way it does (conditioning, not budget), and a reparameterization (`--param level_dev`) that fixes the diagnosed mechanism but, in the one run tested so far, overshoots at its learning rate and needs a follow-up before it can be called a fix. Read after V1.
- [notebooks/Argentiere_temperature_inversion.ipynb](notebooks/Argentiere_temperature_inversion.ipynb) — second site, 13.6 ka to today: dated moraines, exposure ages and subfossil wood (AlpIce-1) for Argentière and the upper Arve. Data in [data/Argentiere/](data/Argentiere/README.md), runs in [tools/run_argentiere_inversion.py](tools/run_argentiere_inversion.py).
- [example_igm_emulator.py](example_igm_emulator.py) — invert for an SMB elevation profile.
- [example_profile_smb.py](example_profile_smb.py) — run the forward model with a profile SMB.
- [example_constrained_inversion.py](example_constrained_inversion.py), [invert_SMB.py](invert_SMB.py).
- [invert_T_m_lowest.py](invert_T_m_lowest.py) — **stale**: calls `inversion_extent`, which hard-codes 7 epochs and no longer matches `GlacierDynamicsCheckpointed`'s return signature. Use [`invert_extent`](core/inversion.py) instead (arbitrary epochs, see the LIA notebook above).

How to run (typical)
1. Install dependencies: see [requirements.txt](requirements.txt).
2. Configure via [`parse_arguments`](config/read_config.py) in [config/read_config.py](config/read_config.py), or override the returned `args` in a notebook.
3. Build the emulator and the forward model, then call an inversion helper from [core/inversion.py](core/inversion.py) — [`invert_ela`](core/inversion.py), [`invert_profile`](core/inversion.py), [`invert_field`](core/inversion.py), [`inversion_extent`](core/inversion.py), [`inversion_thicknes`](core/inversion.py) — and step an optimizer on the result.

## IGM ice-flow emulator (`forward_scheme='igm'`)

Ice velocities come from a **pretrained IGM emulator** — the same `SIADecompNet`
used in the `igm-smb-inference` work. `core/emulator/` is a PyTorch port of IGM's
TensorFlow architecture, so gradients flow through the emulator and the whole
forward model stays differentiable in one framework.

```python
from core.emulator import load_emulator
from core.glacier import GlacierDynamicsCheckpointed

emulator = load_emulator('emulators/SIADecompNet_18x80_dil1-32', device=device)
glacier = GlacierDynamicsCheckpointed(topo, H_init, icemask, device, args,
                                      emulator=emulator)
H = glacier.forward(..., forward_scheme='igm')
```

See [example_igm_emulator.py](example_igm_emulator.py) for a full SMB-profile
inversion.

### Inputs

| field | meaning | units |
|---|---|---|
| `thk` | ice thickness | m |
| `usurf` | surface elevation | m |
| `tau_ref` | Weertman reference basal shear stress | MPa |
| `arrhenius` | Glen's flow-law factor | MPa⁻³ yr⁻¹ |
| `dX` | grid spacing | m |

Output is velocity at `Nz=2` MOLHO vertical nodes.

**On τ_ref.** Sliding is the Weertman law, and τ_ref is its single parameter:
lower τ_ref means easier sliding, hence faster flow. Set it with `--tau_ref`
(default `0.2154` MPa, IGM's own default), or pass a 2D tensor as `tau_ref=` to
the glacier model to vary it in space or invert for it.

IGM's manifests still spell this channel `slidingco` — its own Weertman law reads
`tau_ref = fieldin["slidingco"]`. Pretrained artifacts are therefore read
unchanged, but nothing in this repository uses that name: `emulator.input_names`
reports `tau_ref`, and that is the keyword every call takes.

**Column averaging.** The emulator returns velocity at two vertical nodes on the
MOLHO basis. Mass transport uses the exact analytic column integral,
`ubar = 0.2·U_bed + 0.8·U_surf` for Glen exponent n=3 — the same weights IGM
computes by Gauss quadrature.

### Velocity spikes at the ice margin

On DEM-derived geometry the surface can drop hundreds of metres across a single
cell at the glacier margin. The emulator reads that as an enormous driving
stress and returns a correspondingly enormous velocity. On Aletsch's 1880
surface the median speed is a sane 93 m/yr and the 90th percentile 247 m/yr,
but the maximum is 2850 m/yr at one cliff pixel.

That is not merely cosmetic: the CFL condition reads the **global** maximum
speed, so a handful of bad pixels shrink the time step for the whole domain and
the run crawls. `--max_velbar` (IGM's `force_max_velbar`) caps the depth-averaged
speed, keeping the flow direction. It defaults to `0.0` — disabled, as in IGM —
so set it explicitly when your geometry has margin artifacts.

### Emulator artifacts

`emulators/<name>/` holds an IGM artifact unchanged: `manifest.yaml` plus
`export/weights.weights.h5`. The manifest pins the architecture, the frozen
input-normalization statistics and the output scale; `load_emulator` rebuilds
the network from it and converts the Keras kernels to PyTorch layout. Point
`--emulator_path` at any schema-v3 `SIADecompNet` artifact.

### Accuracy and limitations

* The port reproduces the TensorFlow original to **2.4e-6 relative error**
  (float32 convolution noise). `tests/test_igm_emulator.py` checks this against
  a committed reference dump regenerated by `tools/make_tf_reference.py`.
* With a fixed time step the adjoint matches finite differences **exactly**.
  With the CFL limiter on, `dt = min(cfl·dx/vel_max, dtmax)` reads `vel_max`
  through `.item()`, so autograd treats the step length as constant; the
  gradient stays right in sign and magnitude but is not exact.
* The emulator assumes a square grid (`dx == dy`); this is checked at run time.
* Emulator weights are frozen by default, so the adjoint does not accumulate
  into them. Pass `trainable=True` to `load_emulator` to retrain it the way
  IGM's data assimilation does.

### Tests

```
python -m unittest tests.test_igm_emulator     # 37 tests
```

Covers the TF-equivalence of the port, the weight conversion, MOLHO averaging,
the adjoint (against finite differences), the ELA inversion, and the SIA scheme.

Key modules and responsibilities
- [core/glacier.py](core/glacier.py)
  - [`GlacierDynamicsCheckpointed`](core/glacier.py): the forward model. Steps ice thickness through time, checkpointing each step so the whole run stays differentiable without holding every intermediate state in memory. `FORWARD_SCHEMES` is `('SIA', 'igm')`.
- [core/emulator/](core/emulator/)
  - PyTorch port of IGM's `SIADecompNet`, plus the manifest reader and the Keras→PyTorch weight conversion. [`load_emulator`](core/emulator/loader.py) turns an artifact directory into a ready [`IceFlowEmulator`](core/emulator/loader.py).
- [core/forward_schemes/](core/forward_schemes/)
  - One time step per scheme: [`checkpointed_igm_emulator_step`](core/forward_schemes/igm_emulator_step.py) and [`checkpointed_SIA_step`](core/forward_schemes/sia_step.py).
- [core/inversion.py](core/inversion.py)
  - Objective functions, each returning `(H_sim, loss, data_term, metrics)`: [`invert_ela`](core/inversion.py), [`invert_profile`](core/inversion.py), [`invert_field`](core/inversion.py), plus the multi-epoch [`inversion_extent`](core/inversion.py) and [`inversion_thicknes`](core/inversion.py).
- [core/smb.py](core/smb.py)
  - Surface mass balance: [`update_smb_ELA`](core/smb.py), [`update_smb_profile`](core/smb.py), [`update_smb_PDD`](core/smb.py), [`cosine_temperature_series`](core/smb.py).
- [utils/emulator_tools.py](utils/emulator_tools.py)
  - Mass transport: [`compute_divflux_limited`](utils/emulator_tools.py) (superbee-limited upwind flux), [`apply_boundary_condition`](utils/emulator_tools.py).
- [visualization/plots.py](visualization/plots.py)
  - [`plot_thickness_comparison`](visualization/plots.py), [`plot_ela_convergence`](visualization/plots.py), [`plot_thickness_divflux_velocities`](visualization/plots.py) and other inspection helpers.
- [data/loader.py](data/loader.py)
  - [`load_geology`](data/loader.py) reads `data/geology.nc` into `(topo, thk, mask)`.

Data
- `data/geology.nc` — Aletsch bed and dated surfaces (`surf_1880`, `surf_1926`, `surf_1957`, `surf_1980`, `surf_1999`, `surf_2009`, `surf_2017`), ice mask, 100 m grid.
- `emulators/<name>/` — IGM emulator artifacts, used unchanged.
- `results/` — figures and analysis written by runs.
- `notebooks/` — older exploratory notebooks.

Implementation notes
- The forward model is expensive, so every step goes through `torch.utils.checkpoint`: activations are recomputed during the backward pass rather than stored.
- The SIA step is diffusion-limited and picks its own `dt`; the `igm` step is CFL-limited on the emulator's depth-averaged velocity.
- Soft (sigmoid-thresholded) masks are used for extent-based data fidelity in [`inversion_extent`](core/inversion.py), so extent misfit stays differentiable.

Where to look when debugging
- Inversion not moving: check that the control has `requires_grad` and that its `.grad` is non-zero after `backward()`; then check the learning rate against the parameter's natural scale (an ELA in metres needs a much larger step than an SMB in m/yr).
- Velocities look wrong: run `emulator.velocities(...)` directly on the initial geometry and compare against what you expect for the glacier — for Aletsch, tens to low hundreds of m/yr. Check `tau_ref` and `arrhenius` before suspecting the emulator.
- Emulator will not load: `load_manifest` and the weight conversion both raise with the specific field or layer at fault; `python -m unittest tests.test_igm_emulator` isolates it further.
