import torch
from utils import device
from torch.utils.checkpoint import checkpoint



class GlacierDynamicsCheckpointed(torch.nn.Module):
    def __init__(self, Z_topo, ttot, grad_b, b_max, rho, g, fd, Lx, Ly, dx, dy, dtmax, device,H_initial):
        super().__init__()
        self.Z_topo = Z_topo
        self.ttot = ttot
        self.grad_b = grad_b
        self.b_max = b_max
        self.rho = rho
        self.g = g
        self.fd = fd
        self.Lx = Lx
        self.Ly = Ly
        self.dx = dx
        self.dy = dy
        self.dtmax = dtmax
        self.device = device
        self.H_initial=H_initial

    def forward(self, Z_ELA):
        return self.solve_glacier_dynamics(self.Z_topo, self.ttot, self.grad_b, self.b_max, self.H_initial, Z_ELA)

    def solve_glacier_dynamics(self, Z_topo, ttot, grad_b, b_max, H_initial, Z_ELA):
        nx = int(self.Lx / self.dx)
        ny = int(self.Ly / self.dy)

        epsilon = torch.tensor(1.e-20, dtype=torch.bfloat16, device=self.device)
        # H_ice = torch.zeros((ny, nx), device=self.device, dtype=torch.float32)
        H_ice = H_initial.to(device=device)
        
        Z_surf = Z_topo + H_ice

        time = torch.tensor(1880., dtype=torch.float32, device=self.device)
        # dt = torch.tensor(self.dtmax, dtype=torch.bfloat16, device=self.device)
        it = 0

        def checkpointed_step(H_ice, Z_surf, Z_ELA, time):
            # Compute H_avg
            H_avg = 0.25 * (H_ice[:-1, :-1] + H_ice[1:, 1:] + H_ice[:-1, 1:] + H_ice[1:, :-1])

            # Compute Snorm
            Sx = (Z_surf[:, 1:] - Z_surf[:, :-1]) / self.dx
            Sy = (Z_surf[1:, :] - Z_surf[:-1, :]) / self.dy
            Sx = 0.5 * (Sx[:-1, :] + Sx[1:, :])
            Sy = 0.5 * (Sy[:, :-1] + Sy[:, 1:])
            Snorm = torch.sqrt(Sx**2 + Sy**2 + epsilon)

            # Compute diffusivity
            D = self.fd * (self.rho * self.g)**3.0 * H_avg**5 * Snorm**2 + epsilon

            # Compute adaptive time step
            dt = min(min(self.dx, self.dy)**2 / (2.7 * torch.max(D).item()), self.dtmax)

            # Compute fluxes
            qx = -(0.5 * (D[:-1, :] + D[1:, :])) * (Z_surf[1:-1, 1:] - Z_surf[1:-1, :-1]) / self.dx
            qy = -(0.5 * (D[:, :-1] + D[:, 1:])) * (Z_surf[1:, 1:-1] - Z_surf[:-1, 1:-1]) / self.dy

            # Compute thickness change rate
            dHdt = -(torch.diff(qx, dim=1) / self.dx + torch.diff(qy, dim=0) / self.dy)

            # Update ice thickness
            H_ice = H_ice.clone()
            H_ice[1:-1, 1:-1] = H_ice[1:-1, 1:-1] + dt * dHdt


            # Compute surface mass balance (SMB)
            b = torch.minimum(grad_b * (Z_surf - Z_ELA), b_max)
            H_ice = H_ice.clone()
            H_ice[1:-1, 1:-1] = H_ice[1:-1, 1:-1] + dt * b[1:-1, 1:-1]

            # Ensure ice thickness remains positive
            H_ice = torch.maximum(H_ice, torch.tensor(0.0, dtype=torch.float32, device=self.device))

            # Update surface topography
            Z_surf = Z_topo + H_ice
            return H_ice, Z_surf, time + dt

        while time < ttot:
            H_ice, Z_surf, time = checkpoint(checkpointed_step, H_ice, Z_surf, Z_ELA, time)
            it += 1

        return H_ice