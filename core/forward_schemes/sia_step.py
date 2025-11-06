import torch


def checkpointed_SIA_step(H_ice, Z_surf, smb, time, Z_topo, dx, dy, dtmax, rho, g, fd, device):
    epsilon = torch.tensor(1.e-10, device=device)
    # Compute H_avg
    H_avg = 0.25 * (H_ice[:-1, :-1] + H_ice[1:, 1:] + H_ice[:-1, 1:] + H_ice[1:, :-1])
    # Compute Snorm
    Sx = (Z_surf[:, 1:] - Z_surf[:, :-1]) / dx
    Sy = (Z_surf[1:, :] - Z_surf[:-1, :]) / dy
    Sx = 0.5 * (Sx[:-1, :] + Sx[1:, :])
    Sy = 0.5 * (Sy[:, :-1] + Sy[:, 1:])
    Snorm = torch.sqrt(Sx**2 + Sy**2 + epsilon)
    D = (rho * g)**3.0 *(fd * H_avg**5 )* Snorm**2 + epsilon
    # Compute adaptive time step.
    dt_value = min(min(dx, dy)**2 / (4.1 * torch.max(D).item()), dtmax)
    dt = torch.tensor(dt_value, dtype=torch.float32, device=device, requires_grad=True)
    # Compute fluxes
    qx = -(0.5 * (D[:-1, :] + D[1:, :])) * (Z_surf[1:-1, 1:] - Z_surf[1:-1, :-1]) / dx
    qy = -(0.5 * (D[:, :-1] + D[:, 1:])) * (Z_surf[1:, 1:-1] - Z_surf[:-1, 1:-1]) / dy
    # Compute thickness change rate
    dHdt = -(torch.diff(qx, dim=1) / dx + torch.diff(qy, dim=0) / dy)
    # Update ice thickness
    H_ice = H_ice.clone()
    H_ice[1:-1, 1:-1] = H_ice[1:-1, 1:-1] + dt * dHdt
    H_ice = H_ice.clone()
    H_ice[1:-1, 1:-1] = H_ice[1:-1, 1:-1] + dt * smb[1:-1, 1:-1]
    # Ensure ice thickness remains positive
    H_ice = torch.relu(H_ice)
    # Update surface topography
    Z_surf = Z_topo + H_ice
    return H_ice, Z_surf, time + dt