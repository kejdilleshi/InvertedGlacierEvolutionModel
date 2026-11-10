"""Example: SMB profile inversion driven by the IGM ice-flow emulator.

This is the ``profile`` inversion of ``example_profile_smb.py``, but with the
``igm`` forward scheme: ice velocities come from a pretrained IGM emulator
(SIADecompNet) rather than the legacy 3-channel CNN or the SIA solver.

What changes compared with the legacy emulator
----------------------------------------------
* The emulator takes ``[thk, usurf, slidingco, arrhenius, dX]`` in physical
  units, instead of pre-scaled ``[thk, slopsurfx, slopsurfy]``.
* The channel IGM names ``slidingco`` carries the **reference basal shear
  stress tau_ref** (MPa), not a Weertman sliding coefficient. Lower tau_ref
  means easier sliding.
* It predicts velocities at ``Nz=2`` vertical nodes on the MOLHO basis; the
  depth-averaged velocity that drives mass transport is the exact analytic
  column integral, ``ubar = 0.2*U_bed + 0.8*U_surf`` for Glen exponent n=3.
* Gradients flow through the emulator by default, so the adjoint sees how
  velocity responds to geometry. Pass ``--no_emulator_grad`` to freeze it.

Run with::

    python example_igm_emulator.py
"""

import torch
import netCDF4

from config.read_config import parse_arguments
from core.emulator import load_emulator
from core.glacier import GlacierDynamicsCheckpointed
from core.inversion import invert_profile

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------
nc = netCDF4.Dataset('./data/geology.nc')
topo = torch.tensor(nc.variables['topg'][:], device=device, dtype=torch.float32)
H_init = torch.tensor(nc.variables['surf_1980'][:], device=device, dtype=torch.float32) - topo
icemask = torch.tensor(nc.variables['icemask'][:], device=device, dtype=torch.float32)
nc.close()

args = parse_arguments([])
args.t_start = torch.tensor(1980.0)
args.ttot = torch.tensor(2000.0)
args.vis_freq = 100.0
args.forward_scheme = 'igm'

# -----------------------------------------------------------------
# Ice-flow emulator
# -----------------------------------------------------------------
emulator = load_emulator(args.emulator_path, device=device)
print(f'emulator inputs : {emulator.input_names}')
print(f'vertical basis  : {emulator.manifest.basis_vertical}, Nz={emulator.Nz}, '
      f'column weights={emulator.V_bar.tolist()}')
print(f'tau_ref={args.tau_ref} MPa, arrhenius={args.arrhenius} MPa^-3 yr^-1')

glacier = GlacierDynamicsCheckpointed(
    Z_topo=topo,
    H_init=H_init,
    ice_mask=icemask,
    device=device,
    args=args,
    emulator=emulator,        # tau_ref / arrhenius default to the args values;
    visualize=False,          # pass tau_ref=<2D tensor> to vary them in space
)

# A quick look at the velocity field the emulator produces for this geometry.
with torch.no_grad():
    ubar, vbar = emulator.velocities(
        thk=H_init, usurf=topo + H_init,
        slidingco=glacier.tau_ref, arrhenius=glacier.arrhenius, dX=glacier.dX)
speed = torch.sqrt(ubar ** 2 + vbar ** 2)[H_init > 1.0]
print(f'surface speed   : mean {speed.mean():.1f}, p95 {speed.quantile(0.95):.1f} m/yr')

# -----------------------------------------------------------------
# Synthetic observation from a known SMB profile
# -----------------------------------------------------------------
z_min, dz = topo.min(), 200.0
n_bins = max(int(torch.ceil((topo.max() - z_min) / dz).item()), 2)
smb_true = torch.clamp(torch.linspace(-12.0, 5.0, n_bins, device=device), max=3.0)

with torch.no_grad():
    observation = glacier.forward(
        precip_tensor=None, T_m_lowest=None, T_s=None, melt_factor=None,
        smb_method='profile', smb_vec=smb_true, z_min=z_min, dz=dz)
print(f'observation     : max thickness {observation.max():.1f} m')

# -----------------------------------------------------------------
# Invert for the profile from a deliberately wrong first guess
# -----------------------------------------------------------------
smb_vec = torch.full((n_bins,), -10.0, device=device, requires_grad=True)
# lr=1.0 is large, but the misfit starts around 120 m and the profile has to
# move by ~10 m/yr. Decaying it (StepLR) converged more slowly here, so the
# plain schedule is kept.
optimizer = torch.optim.Adam([smb_vec], lr=1.0)

best = float('inf')
for iteration in range(60):
    optimizer.zero_grad()
    _, loss, data_term, metrics = invert_profile(
        smb_vec=smb_vec, z_min=z_min, dz=dz,
        observation=observation, glacier_model=glacier, reg_lambda=0.01)
    loss.backward()
    optimizer.step()
    best = min(best, loss.item())
    print(f'iter {iteration + 1:3d}  loss {loss.item():8.4f}  '
          f'MAE {data_term.item():7.3f} m  RMSE {metrics["rmse"].item():7.3f} m')

error = (smb_vec.detach() - smb_true).abs().max()
print(f'\nbest loss reached           : {best:.4f} m')
print(f'largest SMB profile error   : {error:.3f} m/yr')
