from core.smb import update_smb
import torch
from torch.utils.checkpoint import checkpoint


class GlacierDynamicsCheckpointed(torch.nn.Module):
    def __init__(self, Z_topo, ttot, rho, g, fd, Lx, Ly, dx, dy, dtmax, device, ice_mask):
        super().__init__()
        self.Z_topo = Z_topo
        self.ice_mask=ice_mask
        self.ttot = ttot
        # self.grad_b = grad_b
        # self.b_max = b_max
        self.rho = rho
        self.g = g
        self.fd = fd
        self.Lx = Lx
        self.Ly = Ly
        self.dx = dx
        self.dy = dy
        self.dtmax = dtmax
        self.device = device
        

    def forward(self, precip_tensor, T_ma_lowest,T_mj_lowest):
        return self.solve_glacier_dynamics(self.Z_topo, self.ttot,precip_tensor, T_ma_lowest,T_mj_lowest)

    def solve_glacier_dynamics(self, Z_topo, ttot,precip_tensor, T_ma_lowest,T_mj_lowest):
        nx = int(self.Lx / self.dx)
        ny = int(self.Ly / self.dy)

        epsilon = torch.tensor(1.e-10, device=self.device)
        H_ice = torch.zeros((ny, nx), device=self.device)
        # H_ice = H_initial.to(device=device)
        
        Z_surf = Z_topo + H_ice

        time = torch.tensor(0., device=self.device)
        # dt = torch.tensor(self.dtmax, device=self.device)
        it = torch.tensor(0., device=self.device)
        t_freq=torch.tensor(5., device=self.device)
        t_last_update=torch.tensor(0., device=self.device)
        #initial smb 
        smb = update_smb(Z_surf,precip_tensor,T_ma_lowest,T_mj_lowest,self.ice_mask)

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
            H_ice[1:-1, 1:-1] = H_ice[1:-1, 1:-1] + dt * smb[1:-1, 1:-1]

            # Ensure ice thickness remains positive
            H_ice = torch.maximum(H_ice, torch.tensor(0.0, dtype=torch.float32, device=self.device))

            # Update surface topography
            Z_surf = Z_topo + H_ice
            # print(f"Max H avarage : {torch.max(H_avg).item()} andMax of D : {torch.max(D).item()}, dt : {dt.item()}, max smb : {torch.max(smb).item()}")
            return H_ice, Z_surf, time + dt

        while time < ttot:           
            
            # unroll = 1000          # window length

            H_ice, Z_surf, time = checkpoint(checkpointed_step, H_ice, Z_surf, smb, time)
            it += 1
            # Compute surface mass balance (SMB)
            if (time-t_last_update)>=t_freq:
                smb = update_smb(Z_surf,precip_tensor,T_ma_lowest,T_mj_lowest,self.ice_mask)
                t_last_update=time.clone()
                
            # if it % unroll == 0:
            # #     # cut the graph
            #     H_ice = H_ice.detach().requires_grad_()
            #     Z_surf = Z_surf.detach().requires_grad_()
            #     # precip_tensor.detach().requires_grad_()
            #     # T_ma_lowest.detach().requires_grad_()
            #     # T_mj_lowest.detach().requires_grad_()
        return H_ice