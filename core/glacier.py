from core.smb import update_smb_PDD, update_smb_ELA, update_smb_profile, cosine_temperature_series
from visualization.plots import visualize, visualize_velocities,plot_thickness_divflux_velocities
import math
import torch
from torch.utils.checkpoint import checkpoint
from core.forward_schemes.igm_emulator_step import checkpointed_igm_emulator_step
from core.forward_schemes.sia_step import checkpointed_SIA_step


# 'SIA': shallow-ice, internal deformation only -- no sliding.
# 'igm': IGM ice-flow emulator, Weertman sliding parameterized by tau_ref.
FORWARD_SCHEMES = ('SIA', 'igm')


def _year_index_from_time(time: torch.Tensor,
                          n_years: int,
                          year_len: float = 1.0,
                          t0: float = 1864.0,
                          cycle: bool = False) -> torch.Tensor:
    """
    Map continuous model time -> integer year index [0..n_years-1].
    Assumes `time` is in *years since t0*, with each year having length `year_len`.
    """
    # floor((time - t0)/year_len)
    yi = torch.floor((time - t0) / year_len)
    if cycle:
        yi = yi % n_years
    else:
        yi = yi.clamp(0, n_years - 1)
    return yi.to(torch.long)
 
class GlacierDynamicsCheckpointed(torch.nn.Module):
    def __init__(self, Z_topo, H_init, ice_mask, device, args, visualize=False,
                 emulator=None, tau_ref=None, arrhenius=None):
        """
        emulator : core.emulator.IceFlowEmulator, optional
            IGM ice-flow emulator, required by the 'igm' forward scheme.
        tau_ref, arrhenius : float or torch.Tensor, optional
            Emulator forcing. ``tau_ref`` is the reference basal shear stress
            (MPa) of the Weertman sliding law; ``arrhenius`` is Glen's flow-law
            factor (MPa^-3 yr^-1). Scalars are broadcast over the grid; a 2D
            tensor of the topography's shape is used as-is, so either can serve
            as an inversion control. Default to ``args.tau_ref`` /
            ``args.arrhenius``.
        """
        super().__init__()
        self.Z_topo = Z_topo
        self.ice_mask = ice_mask
        self.H_init = H_init
        self.device = device
        self.emulator = emulator
        self.visualize = visualize

        # Assign from args
        self.ttot = args.ttot
        self.t_start = args.t_start
        self.rho = args.rho
        self.g = args.g
        self.fd = args.fd
        self.Lx = Z_topo.shape[1] * args.dx
        self.Ly = Z_topo.shape[0] * args.dy
        self.dx = args.dx
        self.dy = args.dy
        self.dtmax = args.dtmax
        self.cfl = args.cfl
        self.vis_freq = args.vis_freq
        self.forward_scheme = getattr(args, 'forward_scheme', 'igm')

        # --- IGM emulator forcing ---------------------------------------
        # Sliding follows the Weertman law, parameterized by the reference
        # basal shear stress tau_ref (MPa). arrhenius is in MPa^-3 yr^-1.
        self.emulator_grad = bool(getattr(args, 'emulator_grad', True))
        self.max_velbar = float(getattr(args, 'max_velbar', 0.0))
        self.tau_ref = self._as_field(
            getattr(args, 'tau_ref', 0.2154) if tau_ref is None else tau_ref, 'tau_ref')
        self.arrhenius = self._as_field(
            getattr(args, 'arrhenius', 78.0) if arrhenius is None else arrhenius, 'arrhenius')
        self.dX = self._as_field(float(args.dx), 'dX')

        # --- options for long (millennial) runs ------------------------------
        # z_ref: elevation (m) the PDD temperature control refers to. None keeps
        #   the historical datum, min(ice surface), which moves as ice comes and
        #   goes at the lowest cell.
        # smb_freq: years between SMB updates. None keeps the historical coupling
        #   (SMB recomputed only when the control advances, every vis_freq
        #   years). When set, the SMB follows the surface every smb_freq years
        #   and the control is interpolated linearly between nodes placed every
        #   vis_freq years from t_start.
        # chunk_steps: >0 checkpoints the run in blocks of this many time steps
        #   (two-level checkpointing): memory then grows with
        #   n_steps / chunk_steps + chunk_steps instead of n_steps, for one
        #   extra forward pass. Needed for runs of several thousand years.
        self.z_ref = getattr(args, 'z_ref', None)
        self.smb_freq = getattr(args, 'smb_freq', None)
        self.chunk_steps = int(getattr(args, 'chunk_steps', 0) or 0)
        # precip_cc: Clausius-Clapeyron scaling of precipitation with the
        #   control temperature, fraction per deg C (e.g. 0.07). Precipitation
        #   is then P * (1 + precip_cc) ** (T(t) - precip_T_ref), so P is the
        #   value at the reference temperature. Long-run path only.
        self.precip_cc = getattr(args, 'precip_cc', None)
        self.precip_T_ref = getattr(args, 'precip_T_ref', None)

    def _as_field(self, value, name):
        """Broadcast a scalar to the model grid, or validate a supplied 2D field."""
        if isinstance(value, torch.Tensor) and value.ndim == 2:
            if tuple(value.shape) != tuple(self.Z_topo.shape):
                raise ValueError(
                    f"{name} field has shape {tuple(value.shape)}, "
                    f"expected {tuple(self.Z_topo.shape)}"
                )
            return value.to(device=self.Z_topo.device, dtype=self.Z_topo.dtype)
        return torch.full_like(self.Z_topo, float(value))

    

    def forward(self, precip_tensor, T_m_lowest, T_s, melt_factor,
                smb_method='PDD', ELA=None, grad_b=None, b_max=None, smb_field=None,
                smb_vec=None, z_min=None, dz=None, save_times=None, forward_scheme=None):
        return self.solve_glacier_dynamics(
            self.Z_topo, self.ttot, self.t_start, precip_tensor, T_m_lowest, T_s,
            melt_factor, smb_method, ELA, grad_b, b_max, smb_field, smb_vec, z_min, dz, save_times, forward_scheme
        )    


    def solve_glacier_dynamics(self, Z_topo, ttot, time, precip_tensor, T_m_lowest, T_s,
                              melt_factor, smb_method='PDD',
                              ELA=None, grad_b=None, b_max=None, smb_field=None,
                              smb_vec=None, z_min=None, dz=None, save_times=None, forward_scheme=None):

        H_ice=self.H_init

        Z_surf = Z_topo + H_ice

        # dt = torch.tensor(self.dtmax, device=self.device)
        it = torch.tensor(0., device=self.device)
        t_freq = torch.tensor(self.vis_freq, device=self.device)
        t_last_update = torch.tensor(self.t_start, device=self.device)
        # initial smb
        idx=0

        # Determine forward scheme (use parameter if provided, else use instance default)
        scheme = forward_scheme if forward_scheme is not None else self.forward_scheme

        # Validate forward scheme choice
        if scheme not in FORWARD_SCHEMES:
            raise ValueError(
                f"forward_scheme must be one of {FORWARD_SCHEMES}, got '{scheme}'"
            )
        if scheme == 'igm':
            if self.emulator is None:
                raise ValueError(
                    "forward_scheme='igm' requires an emulator; pass "
                    "emulator=core.emulator.load_emulator(<artifact dir>)"
                )
            if self.dx != self.dy:
                raise ValueError(
                    f"The IGM emulator assumes a square grid, got dx={self.dx}, dy={self.dy}"
                )

        # Validate SMB method choice
        if smb_method not in ['PDD', 'ELA', 'field', 'profile']:
            raise ValueError(f"smb_method must be 'PDD', 'ELA', 'field', or 'profile', got '{smb_method}'")

        # Handle both scalar and array for time-varying parameters
        if smb_method == 'PDD':
            # Convert to tensor if not already
            if not isinstance(T_m_lowest, torch.Tensor):
                T_m_lowest = torch.tensor(T_m_lowest, device=self.device)
            # Check if T_m_lowest is a scalar (0-dim tensor or single element)
            is_scalar_temp = T_m_lowest.ndim == 0 or (T_m_lowest.ndim == 1 and T_m_lowest.numel() == 1)
        elif smb_method == 'ELA':
            # Validate ELA parameters
            if ELA is None or grad_b is None or b_max is None:
                raise ValueError("ELA method requires 'ELA', 'grad_b', and 'b_max' parameters")
            # Convert to tensor if not already
            if not isinstance(ELA, torch.Tensor):
                ELA = torch.tensor(ELA, device=self.device)
            # Check if ELA is a scalar (0-dim tensor or single element)
            is_scalar_ELA = ELA.ndim == 0 or (ELA.ndim == 1 and ELA.numel() == 1)
        elif smb_method == 'field':
            # Validate SMB field parameter
            if smb_field is None:
                raise ValueError("'field' method requires 'smb_field' parameter")
            # Convert to tensor if not already
            if not isinstance(smb_field, torch.Tensor):
                smb_field = torch.tensor(smb_field, device=self.device)
            # Check if smb_field is time-varying (3D: [time, ny, nx]) or constant (2D: [ny, nx])
            if smb_field.ndim == 2:
                # Constant SMB field
                is_constant_smb = True
            elif smb_field.ndim == 3:
                # Time-varying SMB field
                is_constant_smb = False
            else:
                raise ValueError(f"smb_field must be 2D [ny, nx] or 3D [time, ny, nx], got shape {smb_field.shape}")
        elif smb_method == 'profile':
            # Validate SMB profile parameters
            if smb_vec is None or z_min is None or dz is None:
                raise ValueError("'profile' method requires 'smb_vec', 'z_min', and 'dz' parameters")
            # Convert to tensors if not already
            if not isinstance(smb_vec, torch.Tensor):
                smb_vec = torch.tensor(smb_vec, device=self.device)
            if not isinstance(z_min, torch.Tensor):
                z_min = torch.tensor(z_min, device=self.device)
            if not isinstance(dz, torch.Tensor):
                dz = torch.tensor(dz, device=self.device)
            # Check if smb_vec is time-varying (2D: [time, N]) or constant (1D: [N])
            if smb_vec.ndim == 1:
                # Constant SMB profile
                is_constant_profile = True
            elif smb_vec.ndim == 2:
                # Time-varying SMB profile
                is_constant_profile = False
            else:
                raise ValueError(f"smb_vec must be 1D [N] or 2D [time, N], got shape {smb_vec.shape}")


        if self.smb_freq is not None or self.chunk_steps > 0:
            return self._solve_interpolated(
                H_ice, time, ttot, scheme, smb_method, precip_tensor, T_m_lowest,
                T_s, melt_factor, ELA, grad_b, b_max, save_times)

        # Compute initial SMB based on chosen method
        if smb_method == 'PDD':
            T_current = T_m_lowest if is_scalar_temp else T_m_lowest[idx]
            smb = update_smb_PDD(Z_surf, precip_tensor, T_current, T_s, melt_factor=melt_factor, z_ref=self.z_ref) * self.ice_mask
        elif smb_method == 'ELA':
            ELA_current = ELA if is_scalar_ELA else ELA[idx]
            smb = update_smb_ELA(Z_surf, ELA_current, grad_b, b_max) * self.ice_mask
        elif smb_method == 'field':
            if is_constant_smb:
                smb = smb_field * self.ice_mask
            else:
                smb = smb_field[idx] * self.ice_mask
        else:  # profile method
            if is_constant_profile:
                smb = update_smb_profile(Z_surf, smb_vec, z_min, dz) * self.ice_mask
            else:
                smb = update_smb_profile(Z_surf, smb_vec[idx], z_min, dz) * self.ice_mask
        smb = torch.where((smb < 0) | (self.ice_mask > 0.5), smb, torch.tensor(-10.0, device=self.device))

        # Initialize dictionary to store H at specific times
        saved_H = {}
        if save_times is not None:
            for t in save_times:
                saved_H[t] = None

        while time < ttot:

            if scheme == 'SIA':
                H_ice, Z_surf, time = checkpoint(checkpointed_SIA_step,
                                            H_ice, Z_surf, smb, time,
                                            self.Z_topo, self.dx, self.dy, self.dtmax,
                                            self.rho, self.g, self.fd, self.device
                                        )
                # Initialize dummy velocities for visualization compatibility
                ubar = torch.zeros_like(H_ice)
                vbar = torch.zeros_like(H_ice)
            elif scheme == 'igm':
                H_ice, Z_surf, time, ubar, vbar = checkpoint(
                                            checkpointed_igm_emulator_step,
                                            H_ice, Z_surf, smb, time,
                                            self.Z_topo, self.dx, self.dy, self.dtmax,
                                            self.emulator,
                                            self.cfl,
                                            self.tau_ref, self.arrhenius, self.dX,
                                            self.emulator_grad, self.max_velbar
                                        )

            # Store H_ice at specified times
            if save_times is not None:
                for t in save_times:
                    if saved_H[t] is None and time >= t:
                        saved_H[t] = H_ice.clone()

            it += 1
            # print(f"Time: {time.item():.2f} years, Time step: {it.item()}")
            
            if (time - t_last_update) >= t_freq :
                idx+=1
                # Get current parameter value (handle both scalar and array) based on SMB method
                if smb_method == 'PDD':
                    T_current = T_m_lowest if is_scalar_temp else T_m_lowest[idx]
                    smb = update_smb_PDD(Z_surf, precip_tensor, T_current, T_s, melt_factor=melt_factor, z_ref=self.z_ref) * self.ice_mask
                elif smb_method == 'ELA':
                    ELA_current = ELA if is_scalar_ELA else ELA[idx]
                    smb = update_smb_ELA(Z_surf, ELA_current, grad_b, b_max) * self.ice_mask
                elif smb_method == 'field':
                    if is_constant_smb:
                        smb = smb_field * self.ice_mask
                    else:
                        smb = smb_field[idx] * self.ice_mask
                else:  # profile method
                    if is_constant_profile:
                        smb = update_smb_profile(Z_surf, smb_vec, z_min, dz) * self.ice_mask
                    else:
                        smb = update_smb_profile(Z_surf, smb_vec[idx], z_min, dz) * self.ice_mask
                smb = torch.where((smb < 0) | (self.ice_mask > 0.5), smb,
                                  torch.tensor(-10.0, device=self.device))
                t_last_update = time.clone()

                if self.visualize:
                    # visualize_velocities(ubar.detach(), vbar.detach(),H_ice.detach(), smb.detach(), time)
                    plot_thickness_divflux_velocities(H_ice, ubar, vbar, dx=100, dy=100, time=time)
              
        # Return results based on whether we're saving intermediate times
        if save_times is not None:
            return saved_H, H_ice
        else:
            return H_ice


    # ------------------------------------------------------------------
    # Long runs: interpolated control, frequent SMB updates, chunked checkpoints
    # ------------------------------------------------------------------
    def _control_at(self, series, time):
        """Control value at ``time``: linear between nodes every ``vis_freq`` yr."""
        if series.ndim == 0 or series.numel() == 1:
            return series.reshape(())
        flat = series.reshape(-1)
        s = ((time - float(self.t_start)) / float(self.vis_freq)).clamp(0, flat.numel() - 1)
        i0 = torch.floor(s).long().clamp(max=flat.numel() - 2)
        w = (s - i0).to(flat.dtype)
        return flat[i0] * (1 - w) + flat[i0 + 1] * w

    def _step(self, scheme, H_ice, Z_surf, smb, time):
        reentrant = dict(use_reentrant=False)
        if scheme == 'SIA':
            H_ice, Z_surf, time = checkpoint(
                checkpointed_SIA_step, H_ice, Z_surf, smb, time,
                self.Z_topo, self.dx, self.dy, self.dtmax,
                self.rho, self.g, self.fd, self.device, **reentrant)
        else:
            H_ice, Z_surf, time, _, _ = checkpoint(
                checkpointed_igm_emulator_step, H_ice, Z_surf, smb, time,
                self.Z_topo, self.dx, self.dy, self.dtmax, self.emulator,
                self.cfl, self.tau_ref, self.arrhenius, self.dX,
                self.emulator_grad, self.max_velbar, **reentrant)
        return H_ice, Z_surf, time

    def _advance(self, H_ice, Z_surf, smb, time, t_last, n_steps, ttot, scheme,
                 smb_fn, pending):
        """Up to ``n_steps`` time steps; snapshots for the times in ``pending``."""
        # Time is compared as a Python float: one host sync per step instead of
        # one per pending snapshot.
        snaps = {}
        t_end, t_prev = float(ttot), float(t_last)
        pending = sorted(pending)
        j = 0
        for _ in range(n_steps):
            if not float(time) < t_end:
                break
            H_ice, Z_surf, time = self._step(scheme, H_ice, Z_surf, smb, time)
            tf = float(time)
            while j < len(pending) and tf >= pending[j]:
                snaps[pending[j]] = H_ice.clone()
                j += 1
            if tf - t_prev >= self.smb_freq_eff:
                smb = smb_fn(Z_surf, time)
                t_last, t_prev = time.clone(), tf
        return H_ice, Z_surf, smb, time, t_last, snaps

    def _solve_interpolated(self, H_ice, time, ttot, scheme, smb_method,
                            precip_tensor, T_m_lowest, T_s, melt_factor,
                            ELA, grad_b, b_max, save_times):
        if smb_method not in ('PDD', 'ELA'):
            raise ValueError("smb_freq / chunk_steps support smb_method 'PDD' or 'ELA'")
        if self.precip_cc and self.precip_T_ref is None:
            raise ValueError("precip_cc needs precip_T_ref, the temperature at which "
                             "precip_tensor applies")
        self.smb_freq_eff = float(self.smb_freq if self.smb_freq is not None
                                  else self.vis_freq)
        control = T_m_lowest if smb_method == 'PDD' else ELA
        if not isinstance(control, torch.Tensor):
            control = torch.tensor(control, device=self.device)

        def smb_fn(Z_surf, t):
            c = self._control_at(control, t)
            if smb_method == 'PDD':
                P = precip_tensor
                if self.precip_cc:
                    P = precip_tensor * torch.exp(
                        math.log1p(self.precip_cc) * (c - self.precip_T_ref))
                smb = update_smb_PDD(Z_surf, P, c, T_s,
                                     melt_factor=melt_factor, z_ref=self.z_ref)
            else:
                smb = update_smb_ELA(Z_surf, c, grad_b, b_max)
            smb = smb * self.ice_mask
            return torch.where((smb < 0) | (self.ice_mask > 0.5), smb,
                               torch.tensor(-10.0, device=self.device))

        time = torch.as_tensor(time, device=self.device, dtype=torch.float32)
        ttot = torch.as_tensor(ttot, device=self.device, dtype=torch.float32)
        Z_surf = self.Z_topo + H_ice
        smb = smb_fn(Z_surf, time)
        t_last = time.clone()
        saved_H = {t: None for t in (save_times or [])}

        while float(time) < float(ttot):
            pending = [t for t, v in saved_H.items() if v is None]
            if self.chunk_steps > 0:
                H_ice, Z_surf, smb, time, t_last, snaps = checkpoint(
                    self._advance, H_ice, Z_surf, smb, time, t_last,
                    self.chunk_steps, ttot, scheme, smb_fn, pending,
                    use_reentrant=False)
            else:
                H_ice, Z_surf, smb, time, t_last, snaps = self._advance(
                    H_ice, Z_surf, smb, time, t_last, 10 ** 9, ttot, scheme,
                    smb_fn, pending)
            saved_H.update(snaps)

        if save_times is not None:
            return saved_H, H_ice
        return H_ice

