from core.smb import update_smb, cosine_temperature_series
from visualization.plots import visualize, visualize_velocities
import torch
from torch.utils.checkpoint import checkpoint
from core.forward_schemes.emulator_step import checkpointed_emulator_step
from core.forward_schemes.sia_step import checkpointed_SIA_step


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
    def __init__(self, Z_topo, H_init, ice_mask, device, args, model=None, visualize=False):
        super().__init__()
        self.Z_topo = Z_topo
        self.ice_mask = ice_mask
        self.H_init = H_init
        self.device = device
        self.model = model
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

    

    def forward(self, precip_tensor, T_m_lowest, T_s,P_daily,T_daily, melt_factor):
        return self.solve_glacier_dynamics(self.Z_topo, self.ttot,self.t_start, precip_tensor, T_m_lowest, T_s, P_daily,T_daily,melt_factor)    


    def solve_glacier_dynamics(self, Z_topo, ttot,time, precip_tensor, T_m_lowest, T_s,P_daily,T_daily,melt_factor):
        ny, nx = Z_topo.shape
        # H_ice = torch.zeros((ny, nx), device=self.device)
        H_ice=self.H_init
      
        Z_surf = Z_topo + H_ice

        # dt = torch.tensor(self.dtmax, device=self.device)
        it = torch.tensor(0., device=self.device)
        t_freq = torch.tensor(10., device=self.device)
        t_last_update = torch.tensor(self.t_start, device=self.device)
        # initial smb 
        idx=0
        # Infer number of climate years (supports single-year inputs too)
        # if P_daily.ndim >= 2:
        #     n_years = P_daily.shape[0]
        # else:
        #     n_years = 1

        # # Initial SMB using year index from t_start=0
        # prev_yi = _year_index_from_time(t_start, n_years,t0=1847.0)
        # smb = update_smb(Z_surf, P_daily=P_daily[prev_yi], T_daily=T_daily[prev_yi],melt_factor=melt_factor) * self.ice_mask

        smb = update_smb(Z_surf, precip_tensor, T_m_lowest[idx], T_s,melt_factor=melt_factor) * self.ice_mask 
        smb = torch.where((smb < 0) | (self.ice_mask > 0.5), smb, torch.tensor(-10.0, device=self.device))

        H_ice_1880 = None  # store H_ice at 3/4 * ttot
        H_ice_1926 = None  # store H_ice at 3/4 * ttot
        H_ice_1957 = None  # store H_ice at 3/4 * ttot
        H_ice_1980 = None  # store H_ice at 3/4 * ttot
        H_ice_1999 = None  # store H_ice at 3/4 * ttot
        H_ice_2009 = None  # store H_ice at 3/4 * ttot

        while time < ttot:           
            
            # H_ice, Z_surf, time = checkpointed_SIA_step(
            #                             H_ice, Z_surf, smb, time, 
            #                             self.Z_topo, self.dx, self.dy, self.dtmax, 
            #                             self.rho, self.g, self.fd, self.device
            #                         )
            H_ice, Z_surf, time, ubar, vbar = checkpoint(
                                        checkpointed_emulator_step, 
                                        H_ice, Z_surf, smb, time, 
                                        self.Z_topo, self.dx, self.dy, self.dtmax, 
                                        self.model
                                    )
            # store H_ice when time 
            if (H_ice_1880 is None) and (time >= 1880):
                H_ice_1880 = H_ice.clone()
            if (H_ice_1926 is None) and (time >= 1926):
                H_ice_1926 = H_ice.clone()
            if (H_ice_1957 is None) and (time >= 1957):
                H_ice_1957 = H_ice.clone()
            if (H_ice_1980 is None) and (time >= 1980):   
                H_ice_1980 = H_ice.clone()
            if (H_ice_1999 is None) and (time >= 1999):
                H_ice_1999 = H_ice.clone()
            if (H_ice_2009 is None) and (time >= 2009):
                H_ice_2009 = H_ice.clone()

            it += 1
            # Recompute SMB 
            # yi = _year_index_from_time(time, n_years,t0=1847.0)
            # crossed_year = (yi.item() != prev_yi.item())
            if (time - t_last_update) >= t_freq :
                idx+=1
                # smb = update_smb(Z_surf, P_daily=P_daily[yi], T_daily=T_daily[yi],melt_factor=melt_factor) * self.ice_mask
                smb = update_smb(Z_surf, precip_tensor, T_m_lowest[idx], T_s,melt_factor=melt_factor) * self.ice_mask 
                smb = torch.where((smb < 0) | (self.ice_mask > 0.5), smb,
                                  torch.tensor(-10.0, device=self.device))
                t_last_update = time.clone()

                # prev_yi = yi
                if self.visualize:

                    visualize_velocities(ubar.detach(), vbar.detach(),H_ice.detach(), smb.detach(), time)

        # print(f"number of iterations is {it}")
        return H_ice_1880, H_ice_1926,H_ice_1957,H_ice_1980,H_ice_1999, H_ice_2009, H_ice  
