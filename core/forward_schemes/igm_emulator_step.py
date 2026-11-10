"""Forward step driven by an IGM ice-flow emulator (SIADecompNet).

The network is fed named fields ``[thk, usurf, tau_ref, arrhenius, dX]`` in
physical units and returns per-node velocities, which are averaged over the
column with the MOLHO basis weights exactly as IGM does.

Sliding follows the Weertman law, parameterized by the reference basal shear
stress tau_ref (MPa): lower tau_ref means easier sliding, hence faster flow.
"""

import torch

from utils.emulator_tools import apply_boundary_condition, compute_divflux_limited

# Optional torch.compile'd transport (see use_compiled_transport). The flux
# limiter is ~50 tiny kernels; fused, a step on a 145 x 140 grid runs ~2.8x faster
# forward + backward together with a compiled emulator network.
_divflux_compiled = None


def use_compiled_transport(enabled: bool = True) -> None:
    """Route the transport through a torch.compile'd ``compute_divflux_limited``.

    ``dt`` changes every step, so it is passed as a 0-d tensor: a Python float
    would be baked into the graph and trigger a recompile each step.
    """
    global _divflux_compiled
    _divflux_compiled = torch.compile(compute_divflux_limited) if enabled else None


# Optional slope limiter the transport is *differentiated* with (see use_adjoint_transport).
_adjoint_slope = None


def use_adjoint_transport(slope_type=None) -> None:
    """Differentiate the transport as if it used ``slope_type``; the forward is unchanged.

    The superbee-limited flux is what the model should integrate, but its
    derivative is not a stable adjoint: over a Lateglacial window the
    sensitivity it reports grows about twofold per 200 years going back in
    time, while finite differences show the model forgets its state within a
    few centuries. On the Argentière (2400 yr, T = -1 C) it gives the ice
    surface at Lognan a sensitivity of +452 m/C against -42 m/C by finite
    differences. Differentiated as ``"godunov"`` (first-order upwind,
    monotone) instead, it gives -39 m/C, and ~0 for a temperature node 2400
    years earlier, as finite differences do. ``None`` differentiates the
    superbee flux itself.
    """
    global _adjoint_slope
    _adjoint_slope = slope_type


def _divflux(u, v, h, dx, dy, dt):
    if _divflux_compiled is None:
        flux, dt_ = compute_divflux_limited, dt
    else:
        flux, dt_ = _divflux_compiled, torch.as_tensor(dt, device=h.device, dtype=h.dtype)
    div = flux(u, v, h, dx, dy, dt_)
    if _adjoint_slope is None:
        return div
    # straight-through: superbee's value, the monotone scheme's derivative
    div_adj = flux(u, v, h, dx, dy, dt_, slope_type=_adjoint_slope)
    return div_adj + (div - div_adj).detach()


def checkpointed_igm_emulator_step(
    H_ice,
    Z_surf,
    smb,
    time,
    Z_topo,
    dx,
    dy,
    dtmax,
    emulator,
    cfl,
    tau_ref,
    arrhenius,
    dX,
    differentiable_velocity=True,
    max_velbar=0.0,
):
    """Advance ice thickness by one adaptive (CFL-limited) time step.

    Parameters
    ----------
    H_ice, Z_surf, Z_topo : torch.Tensor
        Ice thickness, surface and bed elevation, ``[ny, nx]``, metres.
    smb : torch.Tensor
        Surface mass balance, ``[ny, nx]``, m ice eq. per year.
    tau_ref, arrhenius, dX : torch.Tensor
        Emulator forcing fields, ``[ny, nx]``: Weertman reference basal shear
        stress (MPa), Glen's flow-law factor (MPa^-3 yr^-1), grid spacing (m).
    max_velbar : float
        Cap on depth-averaged speed (m/yr), IGM's ``force_max_velbar``. 0
        disables it. Useful on DEM-derived geometry, where a one-cell cliff at
        the ice margin can produce a velocity spike that throttles the CFL time
        step for the whole domain.
    differentiable_velocity : bool
        When True the adjoint carries the velocity's dependence on geometry,
        as IGM's data assimilation does. When False the emulator runs under
        ``no_grad`` and the velocity field is treated as frozen within the
        step -- cheaper, and what the legacy ``emulator`` scheme does.

    Returns
    -------
    H_ice, Z_surf, time, ubar, vbar
    """
    if differentiable_velocity:
        ubar, vbar = emulator.velocities(
            max_velbar=max_velbar,
            thk=H_ice, usurf=Z_surf, tau_ref=tau_ref, arrhenius=arrhenius, dX=dX
        )
    else:
        with torch.no_grad():
            ubar, vbar = emulator.velocities(
                max_velbar=max_velbar,
                thk=H_ice, usurf=Z_surf, tau_ref=tau_ref, arrhenius=arrhenius, dX=dX
            )

    # Time step from the CFL condition on the depth-averaged velocity.
    vel_max = max(ubar.abs().max().item(), vbar.abs().max().item())
    dt = min(cfl * dx / vel_max, dtmax) if vel_max > 0.0 else dtmax

    # Transport: dH/dt = -div(u H), with a superbee-limited upwind flux.
    dHdt = -_divflux(ubar, vbar, H_ice, dx, dy, dt)

    H_ice = H_ice + dt * dHdt
    H_ice = torch.clamp(H_ice, min=0)

    # Mass balance
    H_ice = H_ice + dt * smb
    H_ice = apply_boundary_condition(H_ice.clone())

    Z_surf = Z_topo + H_ice
    return H_ice, Z_surf, time + dt, ubar, vbar
