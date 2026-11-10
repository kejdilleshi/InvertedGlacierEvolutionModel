"""Build a ready-to-use ice-flow emulator from an IGM artifact directory."""

import math
from typing import Optional

import torch
import torch.nn as nn

from core.emulator.keras_weights import load_keras_weights_into
from core.emulator.manifest import EmulatorManifest, load_manifest
from core.emulator.siadecompnet import FixedChannelStandardization, SIADecompNet


ARCHITECTURES = {"SIADecompNet": SIADecompNet}

# IGM manifests spell the reference-shear-stress channel 'slidingco' for
# historical reasons. Everything in this repository calls it what it is.
CANONICAL_INPUT_NAMES = {"slidingco": "tau_ref"}


def canonical_input_name(name: str) -> str:
    return CANONICAL_INPUT_NAMES.get(name, name)


def vertical_average_weights(basis_vertical: str, Nz: int, exp_glen: float = 3.0):
    """Weights that turn the emulator's per-node velocities into a column average.

    ``ubar = sum_j V_bar[j] * U[j]``. These are the exact analytic integrals of
    IGM's vertical basis functions over the normalized column, i.e. the same
    numbers IGM computes by Gauss quadrature in
    ``igm/processes/iceflow/vertical/``.
    """
    basis = basis_vertical.lower()

    if basis == "molho":
        if Nz != 2:
            raise ValueError(f"MOLHO vertical basis only supports Nz=2, got Nz={Nz}")
        # phi_bed integrates to 1/(n+2), phi_surf to (n+1)/(n+2)
        n = float(exp_glen)
        return torch.tensor([1.0 / (n + 2.0), (n + 1.0) / (n + 2.0)], dtype=torch.float32)

    if basis == "ssa":
        if Nz != 1:
            raise ValueError(f"SSA vertical basis only supports Nz=1, got Nz={Nz}")
        return torch.tensor([1.0], dtype=torch.float32)

    raise ValueError(
        f"Unsupported basis_vertical {basis_vertical!r}; expected 'molho' or 'ssa'"
    )


class IceFlowEmulator(nn.Module):
    """An IGM emulator plus the bookkeeping needed to drive a forward model.

    Knows its own input names, so callers hand it named fields rather than a
    pre-stacked tensor in a magic order.
    """

    def __init__(self, network: nn.Module, manifest: EmulatorManifest, exp_glen: float = 3.0):
        super().__init__()
        self.network = network
        self.manifest = manifest
        self.input_names = [canonical_input_name(name) for name in manifest.inputs]
        self.Nz = int(manifest.Nz)
        self.output_scale = float(manifest.output_scale)
        self.exp_glen = float(exp_glen)
        self.register_buffer(
            "V_bar",
            vertical_average_weights(manifest.basis_vertical, self.Nz, exp_glen),
        )

    def forward(self, **fields: torch.Tensor) -> torch.Tensor:
        """Stack the named 2D fields in manifest order and run the network.

        Fields are named as this repository names them (``tau_ref``, not IGM's
        legacy ``slidingco``). Each is ``[ny, nx]``; the result is
        ``[2*Nz, ny, nx]``.
        """
        missing = [name for name in self.input_names if name not in fields]
        if missing:
            raise ValueError(
                f"Emulator expects inputs {self.input_names}; missing {missing}"
            )
        X = torch.stack([fields[name] for name in self.input_names], dim=0).unsqueeze(0)
        Y = self.network(X) * self.output_scale
        return Y[0]

    def velocities(self, max_velbar: float = 0.0, **fields: torch.Tensor):
        """Return the vertically averaged velocity ``(ubar, vbar)``, each ``[ny, nx]``.

        Velocities are zeroed where there is no ice, matching IGM's emulated
        iceflow update.

        ``max_velbar`` (m/yr) caps the depth-averaged speed, as IGM's
        ``force_max_velbar`` does; 0 disables it, which is IGM's default. It is
        worth setting on DEM-derived geometry, where a one-cell cliff at the ice
        margin produces a huge driving stress and hence a spurious velocity
        spike. Such a spike is not merely cosmetic: the CFL condition reads the
        *global* maximum speed, so a handful of bad pixels can shrink the time
        step for the whole domain.
        """
        Y = self.forward(**fields)
        U = Y[: self.Nz]
        V = Y[self.Nz :]

        thk = fields["thk"]
        ice = (thk > 0.0).to(Y.dtype)
        U = U * ice
        V = V * ice

        weights = self.V_bar.to(Y.dtype).view(-1, 1, 1)
        ubar = (U * weights).sum(dim=0)
        vbar = (V * weights).sum(dim=0)

        if max_velbar > 0.0:
            # Scale both components by a common factor, so direction is kept.
            magnitude = torch.sqrt(ubar ** 2 + vbar ** 2 + 1e-12)
            scale = torch.clamp(max_velbar / magnitude, max=1.0)
            ubar, vbar = ubar * scale, vbar * scale

        return ubar, vbar


def load_emulator(
    path,
    device: Optional[torch.device] = None,
    exp_glen: float = 3.0,
    eval_mode: bool = True,
    trainable: bool = False,
    slope_eps: Optional[float] = None,
) -> IceFlowEmulator:
    """Load an IGM emulator artifact directory into a PyTorch ``IceFlowEmulator``.

    The emulator's own weights are frozen by default. Gradients still flow
    *through* the network to whatever is being inverted for -- freezing only
    stops the adjoint from accumulating into the emulator's parameters, which
    would otherwise happen silently because the emulator is registered as a
    submodule of the glacier model. Pass ``trainable=True`` to retrain it in
    the manner of IGM's data assimilation.

    ``slope_eps`` floors the surface- and bed-slope magnitudes inside the
    network's physics features. It defaults to ``SIADecompNet.FIXED_SLOPE_EPS``
    (1e-6, a 1e-3 slope floor) because the alternative is an adjoint that
    detonates on any cell whose ice surface is flat -- the downhill unit vector
    differentiates like 1/|grad s|, so IGM's own 1e-8 lets that reach 1e4 and
    compound through the time loop while the forward pass stays finite. Pass
    ``slope_eps=SIADecompNet.FIXED_EPS`` for bit-exact agreement with IGM's
    TensorFlow original; the forward difference is ~5e-5 relative at ordinary
    slopes and unmeasurable on a real domain.
    """
    manifest = load_manifest(path)

    if manifest.architecture_name not in ARCHITECTURES:
        raise ValueError(
            f"Unsupported architecture {manifest.architecture_name!r}; "
            f"available: {sorted(ARCHITECTURES)}"
        )

    network = ARCHITECTURES[manifest.architecture_name](
        **manifest.architecture_params, slope_eps=slope_eps
    )
    network.set_input_normalizer(
        FixedChannelStandardization(manifest.norm_mean, manifest.norm_var, manifest.norm_epsilon)
    )
    load_keras_weights_into(network, path)

    emulator = IceFlowEmulator(network, manifest, exp_glen=exp_glen)
    if not trainable:
        for parameter in emulator.parameters():
            parameter.requires_grad_(False)
    if eval_mode:
        emulator.eval()
    if device is not None:
        emulator = emulator.to(device)
    return emulator
