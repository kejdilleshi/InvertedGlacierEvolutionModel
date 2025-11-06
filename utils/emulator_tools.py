import torch
import torch.nn.functional as F

def compute_divflux(u, v, h, dx, dy):
    """
    Upwind computation of the divergence of the flux: d(u h)/dx + d(v h)/dy.
    """
    # Compute u and v on the staggered grid
    u = torch.cat([u[:, :1], 0.5 * (u[:, :-1] + u[:, 1:]), u[:, -1:]], dim=1)  # shape (ny, nx+1)
    v = torch.cat([v[:1, :], 0.5 * (v[:-1, :] + v[1:, :]), v[-1:, :]], dim=0)  # shape (ny+1, nx)

    # Extend h with constant value at the domain boundaries
    Hx = torch.nn.functional.pad(h, (1, 1, 0, 0), mode="constant")  # shape (ny, nx+2)
    Hy = torch.nn.functional.pad(h, (0, 0, 1, 1), mode="constant")  # shape (ny+2, nx)

    # Compute fluxes by selecting the upwind quantities
    Qx = u * torch.where(u > 0, Hx[:, :-1], Hx[:, 1:])  # shape (ny, nx+1)
    Qy = v * torch.where(v > 0, Hy[:-1, :], Hy[1:, :])  # shape (ny+1, nx)

    # Compute the divergence, final shape is (ny, nx)
    divflux = (Qx[:, 1:] - Qx[:, :-1]) / dx + (Qy[1:, :] - Qy[:-1, :]) / dy
    return divflux


# Define PyTorch version of compute_gradient
def compute_gradient(s, dx, dy):
    """
    Compute spatial 2D gradient of a given field.
    """
    EX = torch.cat([1.5 * s[:, :1] - 0.5 * s[:, 1:2], 0.5 * (s[:, :-1] + s[:, 1:]), 1.5 * s[:, -1:] - 0.5 * s[:, -2:-1]], dim=1)
    diffx = (EX[:, 1:] - EX[:, :-1]) / dx

    EY = torch.cat([1.5 * s[:1, :] - 0.5 * s[1:2, :], 0.5 * (s[:-1, :] + s[1:, :]), 1.5 * s[-1:, :] - 0.5 * s[-2:-1, :]], dim=0)
    diffy = (EY[1:, :] - EY[:-1, :]) / dy

    return diffx, diffy


# Apply boundary condition in PyTorch
def apply_boundary_condition(H_ice, boundary_width=5):
    """
    Apply boundary condition to the ice thickness field `H_ice`.
    """
    ny, nx = H_ice.shape

    # Create linear ramps
    ramp = torch.linspace(1, 0, boundary_width, device=H_ice.device)

    # Apply boundary condition to the left and right boundaries
    H_ice[:, :boundary_width] *= ramp.flip(0)  # Left
    H_ice[:, -boundary_width:] *= ramp  # Right

    # Apply boundary condition to the top and bottom boundaries
    H_ice[:boundary_width, :] *= ramp.flip(0).unsqueeze(1)  # Top
    H_ice[-boundary_width:, :] *= ramp.unsqueeze(1)  # Bottom

    return H_ice
def minmod(a, b):
    """Minmod slope limiter."""
    cond = (a * b > 0.0)  # same sign
    return torch.where(
        cond,
        torch.where(torch.abs(a) < torch.abs(b), a, b),
        torch.zeros_like(a)
    )

def maxmod(a, b):
    """Maxmod slope limiter."""
    cond = (a * b > 0.0)  # same sign
    return torch.where(
        cond,
        torch.where(torch.abs(a) > torch.abs(b), a, b),
        torch.zeros_like(a)
    )


def compute_divflux_limited(u, v, h, dx, dy, dt, slope_type="superbee"):
    """
    Upwind computation of div(uh, vh) with slope limiter (Godunov/Minmod/Superbee).
    u: (ny, nx), v: (ny, nx), h: (ny, nx)
    dx, dy, dt: scalars (float)
    slope_type: "godunov" | "minmod" | "superbee"
    Returns: divergence tensor (ny, nx)
    """
    # Staggered velocities (face-centered)
    u = torch.cat([u[:, :1], 0.5 * (u[:, :-1] + u[:, 1:]), u[:, -1:]], dim=1)  # (ny, nx+1)
    v = torch.cat([v[:1, :], 0.5 * (v[:-1, :] + v[1:, :]), v[-1:, :]], dim=0)  # (ny+1, nx)

    # Pad h for slope computation (2 ghost cells like in your TF code)
    Hx = F.pad(h, (2, 2, 0, 0), mode="constant", value=0.0)  # (ny, nx+4)
    Hy = F.pad(h, (0, 0, 2, 2), mode="constant", value=0.0)  # (ny+4, nx)

    # One-sided slopes in x
    sigpx = (Hx[:, 2:]   - Hx[:, 1:-1]) / dx   # (ny, nx+2)
    sigmx = (Hx[:, 1:-1] - Hx[:, :-2])  / dx   # (ny, nx+2)

    # One-sided slopes in y
    sigpy = (Hy[2:, :]   - Hy[1:-1, :]) / dy   # (ny+2, nx)
    sigmy = (Hy[1:-1, :] - Hy[:-2,  :]) / dy   # (ny+2, nx)

    if slope_type.lower() == "godunov":
        slopex = torch.zeros_like(sigpx)
        slopey = torch.zeros_like(sigpy)

    elif slope_type.lower() == "minmod":
        slopex = minmod(sigmx, sigpx)
        slopey = minmod(sigmy, sigpy)

    elif slope_type.lower() == "superbee":
        sig1x = minmod(sigpx,  2.0 * sigmx)
        sig2x = minmod(sigmx,  2.0 * sigpx)
        slopex = maxmod(sig1x, sig2x)

        sig1y = minmod(sigpy,  2.0 * sigmy)
        sig2y = minmod(sigmy,  2.0 * sigpy)
        slopey = maxmod(sig1y, sig2y)
    else:
        raise ValueError("slope_type must be 'godunov', 'minmod', or 'superbee'.")

    # MUSCL reconstruction of left/right (x) and bottom/top (y) interface states
    # Shapes below all -> (ny, nx+1) for x-interfaces and (ny+1, nx) for y-interfaces
    w = Hx[:, 1:-2] + 0.5 * dx * (1.0 - (u * dt / dx)) * slopex[:, :-1]   # left state at i+1/2
    e = Hx[:, 2:-1] - 0.5 * dx * (1.0 + (u * dt / dx)) * slopex[:, 1:]    # right state at i+1/2

    s = Hy[1:-2, :] + 0.5 * dy * (1.0 - (v * dt / dy)) * slopey[:-1, :]   # bottom state at j+1/2
    n = Hy[2:-1, :] - 0.5 * dy * (1.0 + (v * dt / dy)) * slopey[1:,  :]   # top state at j+1/2

    # Upwind numerical fluxes at interfaces
    Qx = u * torch.where(u > 0, w, e)  # (ny, nx+1)
    Qy = v * torch.where(v > 0, s, n)  # (ny+1, nx)

    # Divergence back to cell centers
    divflux = (Qx[:, 1:] - Qx[:, :-1]) / dx + (Qy[1:, :] - Qy[:-1, :]) / dy
    return divflux
