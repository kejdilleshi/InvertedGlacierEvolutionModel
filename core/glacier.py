from core.smb import update_smb
from visualization.plots import visualize
import torch
from torch.utils.checkpoint import checkpoint

def reduce_hook(g):
    scale= 0.51  # should be >0; The closer to 0 the smaller the change to `g`,  is simple arctangent
    g2 = torch.atan(scale*g)/scale
    # print(f"[reduce_hook]  incoming {torch.norm(g)} → outgoing {torch.norm(g2)}")
    return g2
def reduce_hook2(g):
    scale= 0.051
    g2 = torch.atan(scale*g)/scale
    # print(f"[H_Ice]  incoming {torch.norm(g)} → outgoing {torch.norm(g2)}")
    return g2

class GlacierDynamicsCheckpointed(torch.nn.Module):
    def __init__(self, Z_topo, ttot, rho, g, fd, Lx, Ly, dx, dy, dtmax, device):
        super().__init__()
        self.Z_topo = Z_topo
        self.ttot = ttot
        self.rho = rho
        self.g = g
        self.fd = fd
        self.Lx = Lx
        self.Ly = Ly
        self.dx = dx
        self.dy = dy
        self.dtmax = dtmax
        self.device = device

    def forward(self, precip_tensor, T_m_lowest, T_s):
        return self.solve_glacier_dynamics(self.Z_topo, self.ttot,precip_tensor, T_m_lowest, T_s)

    def solve_glacier_dynamics(self, Z_topo, ttot,precip_tensor,T_m_lowest, T_s):
        nx = int(self.Lx / self.dx)
        ny = int(self.Ly / self.dy)

        epsilon = torch.tensor(1.e-10, device=self.device)
        H_ice = torch.zeros((ny, nx), device=self.device,requires_grad=True)

        # H_ice = H_initial.to(device=device)
        
        Z_surf = Z_topo + H_ice

        time = torch.tensor(0., device=self.device)
        # dt = torch.tensor(self.dtmax, device=self.device)
        it = torch.tensor(0., device=self.device)
        t_freq=torch.tensor(5., device=self.device)
        t_last_update=torch.tensor(0., device=self.device)
        #initial smb 
        idx=0
        smb = update_smb(Z_surf,precip_tensor[idx],T_m_lowest, T_s) 
        # Initialize storage variables
        H1 = H2 = H3 = None

        def checkpointed_step(H_ice, Z_surf,smb, time):
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

            # Compute adaptive time step.
            dt_value = min(min(self.dx, self.dy)**2 / (2.7 * torch.max(D).item()), self.dtmax)
            dt = torch.tensor(dt_value, dtype=torch.float32, device=self.device, requires_grad=True)

            # Compute fluxes
            qx = -(0.5 * (D[:-1, :] + D[1:, :])) * (Z_surf[1:-1, 1:] - Z_surf[1:-1, :-1]) / self.dx
            qy = -(0.5 * (D[:, :-1] + D[:, 1:])) * (Z_surf[1:, 1:-1] - Z_surf[:-1, 1:-1]) / self.dy

            # Compute thickness change rate
            dHdt = -(torch.diff(qx, dim=1) / self.dx + torch.diff(qy, dim=0) / self.dy)

            # Update ice thickness
            H_ice = H_ice.clone()
            H_ice[1:-1, 1:-1] = H_ice[1:-1, 1:-1] + dt * dHdt

            H_ice = H_ice.clone()

            smb.register_hook(reduce_hook)

            H_ice[1:-1, 1:-1] = H_ice[1:-1, 1:-1] + dt * smb[1:-1, 1:-1]

            # Ensure ice thickness remains positive
            H_ice = torch.relu(H_ice)

            # Update surface topography
            Z_surf = Z_topo + H_ice
            if Z_surf.requires_grad:
                Z_surf.retain_grad()                # Required to keep .grad field
                Z_surf.register_hook(reduce_hook2)   # Now hook will be called during backward
            return H_ice, Z_surf, time + dt

        while time < ttot:           
            
            H_ice, Z_surf, time = checkpoint(checkpointed_step, H_ice, Z_surf, smb, time)
            it += 1
            # Save H_ice at specific times
            t_years = time.item()
            if H1 is None and t_years >= 120:
                H1 = H_ice.clone().detach()
            if H2 is None and t_years >= 160:
                H2 = H_ice.clone().detach()
            
            # Compute surface mass balance (SMB)
            if (time-t_last_update)>=t_freq:
                idx+=1
                smb = update_smb(Z_surf,precip_tensor[idx],T_m_lowest, T_s)
                t_last_update=time.clone()
                # Visualization call
                # visualize(Z_surf.detach(), time.item(), H_ice.detach(),self.Lx,self.Ly)    
                      
        return H1, H2, H_ice

