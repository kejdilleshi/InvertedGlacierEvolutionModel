"""PyTorch port of IGM's ``SIADecompNet`` ice-flow emulator.

This is a line-by-line translation of
``igm/processes/iceflow/emulate/utils/architectures/SR.py`` (TensorFlow/Keras)
so that pretrained IGM emulator weights can be used inside this repository's
differentiable, PyTorch-based forward model.

The network decomposes the horizontal velocity additively::

    total velocity = sliding + deformation + residual

Inputs are ``[thk, usurf, tau_ref, arrhenius, dX]``, where ``tau_ref`` is the
**reference basal shear stress** (MPa) of the Weertman sliding law. IGM stores
that channel under the legacy name ``slidingco`` in its manifests -- its own
Weertman law reads ``tau_ref = fieldin["slidingco"]`` -- so both spellings are
accepted when locating the channel, and ``tau_ref`` is the only name used here.

Layout differs from the TF original: tensors here are channel-first
``[B, C, H, W]`` throughout, as is idiomatic in PyTorch. The arithmetic,
padding, activation and feature ordering are identical.

Output is ``[B, 2*Nz, H, W]`` with ``out[:, :Nz]`` the x-component and
``out[:, Nz:]`` the y-component of the velocity at each vertical node.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# IGM manifests still spell the reference-shear-stress channel 'slidingco'.
# Accept either name so pretrained artifacts load unchanged.
TAU_REF_INPUT_NAMES = ("tau_ref", "slidingco")


def _pad_symmetric_1(x: torch.Tensor, dim: str) -> torch.Tensor:
    """One cell of TF ``SYMMETRIC`` padding along a single spatial dimension.

    For a pad width of 1, TF's SYMMETRIC mirroring (which repeats the edge
    element) coincides exactly with PyTorch's ``replicate``.
    """
    pad = (1, 1, 0, 0) if dim == "x" else (0, 0, 1, 1)
    return F.pad(x, pad, mode="replicate")


class FixedChannelStandardization(nn.Module):
    """Channel-wise ``(x - mean) / sqrt(var + eps)`` with frozen statistics.

    The statistics come from the emulator manifest -- they are the ones used
    during training and must never be recomputed on new data.
    """

    def __init__(self, mean_1d, var_1d, epsilon: float = 1e-7):
        super().__init__()
        mean = torch.as_tensor(mean_1d, dtype=torch.float32).reshape(1, -1, 1, 1)
        var = torch.as_tensor(var_1d, dtype=torch.float32).reshape(1, -1, 1, 1)
        self.epsilon = float(epsilon)
        self.register_buffer("mean", mean)
        self.register_buffer("inv_std", torch.rsqrt(var + self.epsilon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) * self.inv_std


class SIADecompNet(nn.Module):
    """Physics-guided ice-flow emulator with additive decomposition."""

    # Fixed architecture constants (part of the architecture, not the manifest)
    FIXED_N_GLEN = 3.0
    FIXED_RHO = 917.0
    FIXED_G = 9.81

    FIXED_M_SLIDE = 3.0
    FIXED_U_REF = 100.0

    FIXED_EPS = 1e-8

    # Floor inside the surface/bed slope magnitudes. FIXED_EPS bounds the
    # *value* of |grad s| at 1e-4, which is all the forward pass needs, but the
    # downhill unit vector and log(|grad s|) below both differentiate like
    # 1/|grad s| -- so a cell whose ice surface happens to be flat hands the
    # adjoint a derivative of 1e4 and, fed back through velocity -> flux -> H
    # each step, detonates it while the forward stays perfectly finite.
    # SLOPE_EPS = 1e-6 floors the slope at 1e-3 (a 0.1% surface slope, an order
    # of magnitude below anything real; this domain runs 0.05-0.7), which caps
    # that derivative at 1e3. Constructing with slope_eps=FIXED_EPS restores
    # bit-exact agreement with IGM's TensorFlow original.
    FIXED_SLOPE_EPS = 1e-6

    FIXED_H_REF = 200.0
    FIXED_SLOPE_REF = 0.1
    FIXED_A_REF = 7.6e-24

    FIXED_H_PROXY_FLOOR = 10.0  # stabilizes thin-ice log proxies and inverse gradients

    def __init__(
        self,
        *,
        input_names: List[str],
        Nz: int,
        network_params: Dict[str, Any],
        dx_const: Optional[float] = None,
        slope_eps: Optional[float] = None,
    ):
        super().__init__()

        self.input_names = list(input_names)
        self.Nz = int(Nz)
        self.nb_inputs = len(self.input_names)
        self.nb_outputs = 2 * self.Nz

        # --- fixed physical constants -----------------------------------
        self.n_glen = float(self.FIXED_N_GLEN)
        self.rho = float(self.FIXED_RHO)
        self.g = float(self.FIXED_G)
        self.m_slide = float(self.FIXED_M_SLIDE)
        self.u_ref = float(self.FIXED_U_REF)

        self.eps = float(self.FIXED_EPS)
        self.slope_eps = float(self.FIXED_SLOPE_EPS if slope_eps is None else slope_eps)
        self.H_ref = float(self.FIXED_H_REF)
        self.slope_ref = float(self.FIXED_SLOPE_REF)
        self.A_ref = float(self.FIXED_A_REF)
        self.tau_ref_scale = self.rho * self.g * self.H_ref * self.slope_ref
        self.H_proxy_floor = float(self.FIXED_H_PROXY_FLOOR)

        # --- input channel bookkeeping ----------------------------------
        self.idx_thk = self.input_names.index("thk")
        self.idx_usurf = self.input_names.index("usurf")
        self.idx_tau_ref = next(
            (self.input_names.index(name)
             for name in TAU_REF_INPUT_NAMES if name in self.input_names),
            None,
        )
        self.idx_arrhenius = (
            self.input_names.index("arrhenius") if "arrhenius" in self.input_names else None
        )
        self.idx_dX = self.input_names.index("dX") if "dX" in self.input_names else None

        if self.idx_dX is None:
            self.dx_const_value = 90.0 if dx_const is None else float(dx_const)
        else:
            self.dx_const_value = None

        self.input_normalizer: Optional[nn.Module] = None

        # --- architecture parameters ------------------------------------
        params = dict(network_params)
        allowed = {"nb_layers", "nb_out_filter", "context_dilation_schedule"}
        unexpected = sorted(set(params) - allowed)
        if unexpected:
            raise ValueError(
                f"Unexpected keys in network_params: {unexpected}. Allowed: {sorted(allowed)}"
            )
        for key in allowed:
            if key not in params:
                raise ValueError(f"network_params must contain {key!r}")

        self.nb_layers = int(params["nb_layers"])
        self.nb_out_filter = int(params["nb_out_filter"])
        self.context_dilation_schedule = [int(v) for v in params["context_dilation_schedule"]]
        self.n_context_blocks = max(2, self.nb_layers // 2)

        if len(self.context_dilation_schedule) != self.n_context_blocks:
            raise ValueError(
                f"context_dilation_schedule must have length {self.n_context_blocks} "
                f"(nb_layers={self.nb_layers}), got {len(self.context_dilation_schedule)}"
            )

        # --- fixed scaling for the explicit physics features ------------
        self.log_H_ref = math.log(self.H_ref + 1.0)
        self.log_tau_ref_scale = math.log(self.tau_ref_scale + self.eps)
        self.log_A_ref = math.log(self.A_ref + self.eps)
        self.B_ref = 2.0 * self.A_ref ** (-1.0 / self.n_glen)
        self.log_B_ref = math.log(self.B_ref + self.eps)
        self.log_u_slide_ref = math.log(self.u_ref + self.eps)
        self.log_u_def_ref = (
            self.log_A_ref
            + (self.n_glen + 1.0) * math.log(self.H_ref + self.H_proxy_floor)
            + self.n_glen * math.log(self.slope_ref + self.eps)
        )

        # number of explicit physics channels feeding each head
        n_slide_phys = 13 + (2 if self.idx_tau_ref is not None else 0)
        n_def_phys = 10 + (3 if self.idx_arrhenius is not None else 0)
        n_all_phys = n_slide_phys + n_def_phys + 2  # + the two curvature channels
        nf = self.nb_out_filter

        # --- shared context encoder -------------------------------------
        self.context_in = nn.Conv2d(self.nb_inputs, nf, 3, padding=1)
        self.context_blocks = nn.ModuleList()
        for dilation in self.context_dilation_schedule:
            self.context_blocks.append(
                nn.ModuleDict(
                    {
                        "conv1": nn.Conv2d(nf, nf, 3, padding=dilation, dilation=dilation),
                        "conv2": nn.Conv2d(nf, nf, 3, padding=dilation, dilation=dilation),
                    }
                )
            )

        # --- sliding head: depth-uniform plug flow, [B, 2, H, W] ---------
        self.slide_head_conv1 = nn.Conv2d(n_slide_phys + nf, nf, 3, padding=1)
        self.slide_head_conv2 = nn.Conv2d(nf, nf, 3, padding=1)
        self.slide_head_out = nn.Conv2d(nf, 2, 1)

        # --- deformation head: depth-dependent, [B, 2*Nz, H, W] ---------
        self.def_head_conv1 = nn.Conv2d(n_def_phys + nf, nf, 3, padding=1)
        self.def_head_conv2 = nn.Conv2d(nf, nf, 3, padding=1)
        self.def_head_out = nn.Conv2d(nf, 2 * self.Nz, 1)

        # --- residual head: zero-initialized correction ------------------
        self.res_head_filters = max(nf // 2, 8)
        self.res_head_conv1 = nn.Conv2d(n_all_phys + nf, self.res_head_filters, 3, padding=1)
        self.res_head_conv2 = nn.Conv2d(self.res_head_filters, self.res_head_filters, 3, padding=1)
        self.res_head_out = nn.Conv2d(self.res_head_filters, 2 * self.Nz, 1)
        nn.init.zeros_(self.res_head_out.weight)
        nn.init.zeros_(self.res_head_out.bias)

        # 3x3 binomial blur used to de-spike thickness for bed-gradient features
        kernel = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=torch.float32
        ).reshape(1, 1, 3, 3) / 16.0
        self.register_buffer("binomial_kernel", kernel)

    # ------------------------------------------------------------------
    def resolved_params(self) -> Dict[str, Any]:
        """Constructor payload, in the same shape IGM's manifest stores it."""
        return {
            "input_names": list(self.input_names),
            "Nz": int(self.Nz),
            "network_params": {
                "nb_layers": int(self.nb_layers),
                "nb_out_filter": int(self.nb_out_filter),
                "context_dilation_schedule": list(self.context_dilation_schedule),
            },
            "dx_const": None if self.dx_const_value is None else float(self.dx_const_value),
            "slope_eps": float(self.slope_eps),
        }

    @property
    def dtype(self) -> torch.dtype:
        """Working precision, taken from the weights.

        The TensorFlow original hardcodes float32; following the parameters
        instead keeps that behaviour for a float32 model while letting
        ``.double()`` give a float64 model for gradient checking.
        """
        return self.context_in.weight.dtype

    def set_input_normalizer(self, layer: nn.Module) -> None:
        self.input_normalizer = layer
        self.add_module("input_norm", layer)

    # ------------------------------------------------------------------
    # finite differences with symmetric padding
    # ------------------------------------------------------------------
    def _get_dx_field(self, x: torch.Tensor) -> torch.Tensor:
        if self.idx_dX is not None:
            return x[:, self.idx_dX : self.idx_dX + 1]
        thk = x[:, self.idx_thk : self.idx_thk + 1]
        return torch.ones_like(thk) * self.dx_const_value

    def _central_diff_x(self, field: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
        f = _pad_symmetric_1(field, "x")
        return (f[:, :, :, 2:] - f[:, :, :, :-2]) / (2.0 * dx + self.eps)

    def _central_diff_y(self, field: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
        f = _pad_symmetric_1(field, "y")
        return (f[:, :, 2:, :] - f[:, :, :-2, :]) / (2.0 * dx + self.eps)

    def _second_diff_x(self, field: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
        f = _pad_symmetric_1(field, "x")
        return (f[:, :, :, 2:] - 2.0 * field + f[:, :, :, :-2]) / (dx * dx + self.eps)

    def _second_diff_y(self, field: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
        f = _pad_symmetric_1(field, "y")
        return (f[:, :, 2:, :] - 2.0 * field + f[:, :, :-2, :]) / (dx * dx + self.eps)

    # ------------------------------------------------------------------
    # explicit physics features from raw inputs
    # ------------------------------------------------------------------
    def _physics_features(
        self, raw_inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        x = raw_inputs.to(self.dtype)

        H = torch.clamp(x[:, self.idx_thk : self.idx_thk + 1], min=0.0)
        s = x[:, self.idx_usurf : self.idx_usurf + 1]
        dx = self._get_dx_field(x)

        # Smooth thickness only for bed-gradient features, to suppress
        # pixel-scale spikes: 3x3 binomial blur with reflect padding.
        H_for_bed_grad = F.conv2d(F.pad(H, (1, 1, 1, 1), mode="reflect"), self.binomial_kernel)

        b = s - H
        b_for_bed_grad = s - H_for_bed_grad

        tau_ref = None
        if self.idx_tau_ref is not None:
            tau_ref = torch.clamp(x[:, self.idx_tau_ref : self.idx_tau_ref + 1], min=self.eps)

        A = None
        if self.idx_arrhenius is not None:
            A = torch.clamp(x[:, self.idx_arrhenius : self.idx_arrhenius + 1], min=self.eps)

        # --- geometry ---------------------------------------------------
        dsdx = self._central_diff_x(s, dx)
        dsdy = self._central_diff_y(s, dx)
        dbdx = self._central_diff_x(b_for_bed_grad, dx)
        dbdy = self._central_diff_y(b_for_bed_grad, dx)

        d2sdx2 = self._second_diff_x(s, dx)
        d2sdy2 = self._second_diff_y(s, dx)

        grad_s = torch.sqrt(dsdx**2 + dsdy**2 + self.slope_eps)
        grad_b = torch.sqrt(dbdx**2 + dbdy**2 + self.slope_eps)

        # --- driving-stress proxy ---------------------------------------
        tau_dx = -self.rho * self.g * H * dsdx
        tau_dy = -self.rho * self.g * H * dsdy
        tau_d = torch.sqrt(tau_dx**2 + tau_dy**2 + self.eps)

        # Floored thickness is used only inside log-based proxies; the linear
        # stress features stay on raw H.
        H_for_log_proxy = H + self.H_proxy_floor
        tau_dx_for_log = -self.rho * self.g * H_for_log_proxy * dsdx
        tau_dy_for_log = -self.rho * self.g * H_for_log_proxy * dsdy
        tau_d_for_log = torch.sqrt(tau_dx_for_log**2 + tau_dy_for_log**2 + self.eps)

        # --- fixed scaling to O(1) --------------------------------------
        log_H = (torch.log(H + 1.0) - self.log_H_ref) / 3.0
        H_lin = H / (self.H_ref + self.eps)
        H_grad_interaction = (H * grad_s) / (self.H_ref * self.slope_ref + self.eps)

        dsdx_n = dsdx / self.slope_ref
        dsdy_n = dsdy / self.slope_ref
        dbdx_n = dbdx / self.slope_ref
        dbdy_n = dbdy / self.slope_ref
        grad_s_n = grad_s / self.slope_ref
        grad_b_n = grad_b / self.slope_ref

        tau_dx_n = tau_dx / (self.tau_ref_scale + self.eps)
        tau_dy_n = tau_dy / (self.tau_ref_scale + self.eps)
        tau_d_n = tau_d / (self.tau_ref_scale + self.eps)

        log_tau_d_raw = torch.log(tau_d_for_log + self.eps)
        log_tau_d = (log_tau_d_raw - self.log_tau_ref_scale) / 5.0

        # downhill unit vector, masked off ice
        ice_mask = (H > 1.0).to(x.dtype)
        dir_x = -dsdx / (grad_s + self.slope_eps) * ice_mask
        dir_y = -dsdy / (grad_s + self.slope_eps) * ice_mask

        curv_x_n = d2sdx2 * 1000.0
        curv_y_n = d2sdy2 * 1000.0

        # --- sliding-head features --------------------------------------
        slide_feats = [
            log_H, dsdx_n, dsdy_n, grad_s_n, dbdx_n, dbdy_n, grad_b_n,
            tau_dx_n, tau_dy_n, tau_d_n, log_tau_d, dir_x, dir_y,
        ]

        log_tau_ref_raw = None
        log_u_slide_proxy_raw = None
        if tau_ref is not None:
            log_tau_ref_raw = torch.log(tau_ref + self.eps)
            log_tau_ref = (log_tau_ref_raw - self.log_tau_ref_scale) / 5.0
            log_u_slide_proxy_raw = self.log_u_slide_ref + self.m_slide * (
                log_tau_d_raw - log_tau_ref_raw
            )
            log_u_slide_proxy = (log_u_slide_proxy_raw - self.log_u_slide_ref) / 5.0
            slide_feats.extend([log_tau_ref, log_u_slide_proxy])

        slide_feats = torch.cat(slide_feats, dim=1)

        # --- deformation-head features ----------------------------------
        def_feats = [
            log_H, dsdx_n, dsdy_n, grad_s_n,
            tau_dx_n, tau_dy_n, tau_d_n, log_tau_d,
            H_lin, H_grad_interaction,
        ]

        log_A_raw = None
        log_B_raw = None
        log_u_def_proxy_raw = None
        if A is not None:
            log_A_raw = torch.log(A + self.eps)
            B = 2.0 * torch.pow(A, -1.0 / self.n_glen)
            log_B_raw = torch.log(B + self.eps)
            log_u_def_proxy_raw = (
                log_A_raw
                + (self.n_glen + 1.0) * torch.log(H_for_log_proxy)
                + self.n_glen * torch.log(grad_s + self.slope_eps)
            )
            log_A_n = (log_A_raw - self.log_A_ref) / 5.0
            log_B_n = (log_B_raw - self.log_B_ref) / 5.0
            log_u_def_proxy = (log_u_def_proxy_raw - self.log_u_def_ref) / 5.0
            def_feats.extend([log_A_n, log_B_n, log_u_def_proxy])

        def_feats = torch.cat(def_feats, dim=1)

        # --- residual-head features -------------------------------------
        all_feats = torch.cat([slide_feats, def_feats, curv_x_n, curv_y_n], dim=1)

        aux = {
            "H": H, "s": s, "b": b, "dx": dx,
            "dsdx": dsdx, "dsdy": dsdy, "dbdx": dbdx, "dbdy": dbdy,
            "d2sdx2": d2sdx2, "d2sdy2": d2sdy2,
            "grad_s": grad_s, "grad_b": grad_b,
            "tau_dx": tau_dx, "tau_dy": tau_dy, "tau_d": tau_d,
            "tau_d_for_log": tau_d_for_log,
            "dir_x": dir_x, "dir_y": dir_y,
            "H_for_bed_grad": H_for_bed_grad,
            "b_for_bed_grad": b_for_bed_grad,
            "H_for_log_proxy": H_for_log_proxy,
        }
        if tau_ref is not None:
            aux.update(tau_ref=tau_ref, log_tau_ref=log_tau_ref_raw,
                       log_u_slide_proxy=log_u_slide_proxy_raw)
        if A is not None:
            aux.update(A=A, log_A=log_A_raw, log_B=log_B_raw,
                       log_u_def_proxy=log_u_def_proxy_raw)

        return slide_feats, def_feats, all_feats, aux

    # ------------------------------------------------------------------
    def _context_features(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.to(self.dtype)
        if self.input_normalizer is not None:
            x = self.input_normalizer(x)

        h = F.gelu(self.context_in(x))
        for block in self.context_blocks:
            residual = h
            h = F.gelu(block["conv1"](h))
            h = F.gelu(block["conv2"](h))
            h = h + residual
        return h

    # ------------------------------------------------------------------
    def forward(self, inputs: torch.Tensor, return_components: bool = False):
        """``inputs``: [B, nb_inputs, H, W] -> ``[B, 2*Nz, H, W]``."""
        raw = inputs.to(self.dtype)

        slide_phys, def_phys, all_phys, aux = self._physics_features(raw)
        context = self._context_features(raw)

        slide_h = F.gelu(self.slide_head_conv1(torch.cat([slide_phys, context], dim=1)))
        slide_h = F.gelu(self.slide_head_conv2(slide_h))
        slide_xy = self.slide_head_out(slide_h)  # [B, 2, H, W]

        def_h = F.gelu(self.def_head_conv1(torch.cat([def_phys, context], dim=1)))
        def_h = F.gelu(self.def_head_conv2(def_h))
        def_flat = self.def_head_out(def_h)  # [B, 2*Nz, H, W]

        res_h = F.gelu(self.res_head_conv1(torch.cat([all_phys, context], dim=1)))
        res_h = F.gelu(self.res_head_conv2(res_h))
        res_flat = self.res_head_out(res_h)  # [B, 2*Nz, H, W]

        # broadcast the depth-uniform sliding over the Nz vertical nodes
        slide_flat = torch.cat(
            [
                slide_xy[:, 0:1].expand(-1, self.Nz, -1, -1),
                slide_xy[:, 1:2].expand(-1, self.Nz, -1, -1),
            ],
            dim=1,
        )

        total_flat = slide_flat + def_flat + res_flat

        if not return_components:
            return total_flat

        return {
            "total": total_flat,
            "slide_xy": slide_xy,
            "slide_flat": slide_flat,
            "deformation_flat": def_flat,
            "residual_flat": res_flat,
            "physics_aux": aux,
        }
