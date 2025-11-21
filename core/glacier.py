from core.smb import update_smb_PDD, update_smb_ELA, cosine_temperature_series
from visualization.plots import visualize, visualize_velocities,plot_thickness_divflux_velocities
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
        self.vis_freq = args.vis_freq

    

    def forward(self, precip_tensor, T_m_lowest, T_s, P_daily, T_daily, melt_factor,
                smb_method='PDD', ELA=None, grad_b=None, b_max=None, smb_field=None):
        return self.solve_glacier_dynamics(
            self.Z_topo, self.ttot, self.t_start, precip_tensor, T_m_lowest, T_s,
            P_daily, T_daily, melt_factor, smb_method, ELA, grad_b, b_max, smb_field
        )    


    def solve_glacier_dynamics(self, Z_topo, ttot, time, precip_tensor, T_m_lowest, T_s,
                              P_daily, T_daily, melt_factor, smb_method='PDD',
                              ELA=None, grad_b=None, b_max=None, smb_field=None):
        ny, nx = Z_topo.shape
        # H_ice = torch.zeros((ny, nx), device=self.device)
        H_ice=self.H_init

        Z_surf = Z_topo + H_ice

        # dt = torch.tensor(self.dtmax, device=self.device)
        it = torch.tensor(0., device=self.device)
        t_freq = torch.tensor(self.vis_freq, device=self.device)
        t_last_update = torch.tensor(self.t_start, device=self.device)
        # initial smb
        idx=0

        # Validate SMB method choice
        if smb_method not in ['PDD', 'ELA', 'field']:
            raise ValueError(f"smb_method must be 'PDD', 'ELA', or 'field', got '{smb_method}'")

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

    
        # Compute initial SMB based on chosen method
        if smb_method == 'PDD':
            T_current = T_m_lowest if is_scalar_temp else T_m_lowest[idx]
            smb = update_smb_PDD(Z_surf, precip_tensor, T_current, T_s, melt_factor=melt_factor) * self.ice_mask
        elif smb_method == 'ELA':
            ELA_current = ELA if is_scalar_ELA else ELA[idx]
            smb = update_smb_ELA(Z_surf, ELA_current, grad_b, b_max) * self.ice_mask
        else:  # field method
            if is_constant_smb:
                smb = smb_field * self.ice_mask
            else:
                smb = smb_field[idx] * self.ice_mask
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
                                        self.model,
                                        0.05
                                    )
            # # store H_ice when time 
            # if (H_ice_1880 is None) and (time >= 1880):
            #     H_ice_1880 = H_ice.clone()
            # if (H_ice_1926 is None) and (time >= 1926):
            #     H_ice_1926 = H_ice.clone()
            # if (H_ice_1957 is None) and (time >= 1957):
            #     H_ice_1957 = H_ice.clone()
            # if (H_ice_1980 is None) and (time >= 1980):   
            #     H_ice_1980 = H_ice.clone()
            # if (H_ice_1999 is None) and (time >= 1999):
            #     H_ice_1999 = H_ice.clone()
            # if (H_ice_2009 is None) and (time >= 2009):
            #     H_ice_2009 = H_ice.clone()

            it += 1
            # print(f"Time: {time.item():.2f} years, Time step: {it.item()}")
            
            if (time - t_last_update) >= t_freq :
                idx+=1
                # Get current parameter value (handle both scalar and array) based on SMB method
                if smb_method == 'PDD':
                    T_current = T_m_lowest if is_scalar_temp else T_m_lowest[idx]
                    smb = update_smb_PDD(Z_surf, precip_tensor, T_current, T_s, melt_factor=melt_factor) * self.ice_mask
                elif smb_method == 'ELA':
                    ELA_current = ELA if is_scalar_ELA else ELA[idx]
                    smb = update_smb_ELA(Z_surf, ELA_current, grad_b, b_max) * self.ice_mask
                else:  # field method
                    if is_constant_smb:
                        smb = smb_field * self.ice_mask
                    else:
                        smb = smb_field[idx] * self.ice_mask
                smb = torch.where((smb < 0) | (self.ice_mask > 0.5), smb,
                                  torch.tensor(-10.0, device=self.device))
                t_last_update = time.clone()

                if self.visualize:
                    # visualize_velocities(ubar.detach(), vbar.detach(),H_ice.detach(), smb.detach(), time)
                    plot_thickness_divflux_velocities(H_ice, ubar, vbar, dx=100, dy=100, time=time)
        # return H_ice_1880, H_ice_1926,H_ice_1957,H_ice_1980,H_ice_1999, H_ice_2009, H_ice  
        # torch.save(smb, 'data/smb_initial.pt')
        # print(f"Number of time steps: {it.item()}")
        return H_ice  
    
