# Import the necessary libraries
import torch
import netCDF4
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.checkpoint import checkpoint
from utils import device
from visualization import plot_gradient_evolution,plot_loss_components

def compute_integral_positive_temperature(T_ma, T_mj):
    """
    Computes the integral of T_abl(t) over the period where T_abl > 0 (PyTorch version).
    """
    A = 12.0  # months
    ratio = T_ma / (T_mj - T_ma)
    integral = torch.zeros_like(T_ma)

    valid = ratio < 1
    ratio_valid = torch.clamp(ratio[valid], -1.0, 1.0)

    integral[valid] = (
        T_ma[valid] * (A - (A / torch.pi) * torch.acos(ratio_valid)) +
        ((T_mj[valid] - T_ma[valid]) * A / torch.pi) * torch.sqrt(1 - ratio_valid**2)
    )

    return integral

def apply_lapse_rate(topography, T_ma_lowest, T_mj_lowest):
    lapse_rate = 7.0 / 1000.0  # 6°C/km
    min_altitude = torch.min(topography)
    delta_alt = topography - min_altitude

    T_ma = T_ma_lowest - lapse_rate * delta_alt
    T_mj = T_mj_lowest - lapse_rate * delta_alt

    return T_ma, T_mj

def compute_negative_temperature_ratio(T_ma, T_mj):
    """
    Computes the ratio of the year when the temperature is negative (PyTorch version).
    Parameters:
        T_ma (Tensor): 2D tensor of mean annual temperatures (on device)
        T_mj (Tensor): 2D tensor of hottest month temperatures (on device)
    Returns:
        Tensor: 2D tensor of negative temperature ratios (values between 0 and 1)
    """
    ratio = T_ma / (T_mj - T_ma)
    neg_temp_ratio = torch.zeros_like(T_ma)
    # Case 1: Always positive temp
    mask_always_positive = ratio >= 1
    neg_temp_ratio[mask_always_positive] = 0.0
    # Case 2: Always negative temperatures
    mask_always_negative = ratio <= -1
    neg_temp_ratio[mask_always_negative] = 1.0
    # Case 3: Valid
    mask_valid = (~mask_always_positive) & (~mask_always_negative)
    ratio_valid = torch.clamp(ratio[mask_valid], -1.0, 1.0)
    neg_temp_ratio[mask_valid] = (1.0 / torch.pi) * torch.acos(ratio_valid)

    return neg_temp_ratio

# Define SMB parameters directly
melt_f =2/12 # m water / (C year)
smb_oggm_wat_density = 1000.0
smb_oggm_ice_density = 910.0  #kg/m^3
def update_smb(Z_topo,precipitation, T_ma_lowest,T_mj_lowest):
    """Compute the surface mass balance (SMB)
         Input:  precipitation [Unit: m * y^(-1)]
                 air_temp      [Unit: °C           ]
         Output  smb           [Unit: m ice eq. / y]
    This mass balance routine implements the surface mass balance model of OGGM
    """

    T_ma, T_mj = apply_lapse_rate(Z_topo, T_ma_lowest, T_mj_lowest)
    
    # Compute accumulation

    accumulation= precipitation* compute_negative_temperature_ratio(T_ma, T_mj) # unit: [ m * y^(-1) water ]
    

    # Compute ablation
    ablation = melt_f  *  compute_integral_positive_temperature(T_ma, T_mj)# unit: [m water / (C year)] * [C year]  => m water 
    # Compute SMB and convert to ice equivalent
    smb = (accumulation - ablation).sum(dim=0) * (smb_oggm_wat_density / smb_oggm_ice_density)
    
    return smb


nc_file = netCDF4.Dataset('geology_200m.nc')
Z_topo = torch.tensor(nc_file.variables['topg'][:], device=device)
topo_1880 = torch.tensor(nc_file.variables['surf_2017'][:], device=device)
thk_1880=topo_1880-Z_topo
thk_1880[thk_1880<=1]=0
thk_1880.to(device=device)
ice_mask = torch.tensor(nc_file.variables['icemask'][:], device=device)

ttot = 130  # Time limit (yr)
t_start=1880.

rho, g, fd = torch.tensor([910.0, 9.81,0.25e-16], device=device) # units [kg/m^3, m/s^-2, Pa^-3year^-1]
dx=200
dy=200
Lx=Z_topo.shape[1]*dx
Ly=Z_topo.shape[0]*dy
nc_file.close()
torch.save(thk_1880,'Obs_2D.pt')


# Convert to NumPy
Z_topo_npy = Z_topo.to(torch.float32).cpu().numpy()
thk_1880_npy = thk_1880.to(torch.float32).cpu().numpy()

class GlacierDynamicsCheckpointed(torch.nn.Module):
    def __init__(self, Z_topo, ttot, rho, g, fd, Lx, Ly, dx, dy, dtmax, device, ice_mask):
        super().__init__()
        self.Z_topo = Z_topo
        self.ice_mask=ice_mask
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
        

    def forward(self, precip_tensor, T_ma_lowest,T_mj_lowest,unroll):
        return self.solve_glacier_dynamics(self.Z_topo, self.ttot,precip_tensor, T_ma_lowest,T_mj_lowest,unroll)

    def solve_glacier_dynamics(self, Z_topo, ttot,precip_tensor, T_ma_lowest,T_mj_lowest,unroll):
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
        smb = update_smb(Z_surf,precip_tensor,T_ma_lowest,T_mj_lowest)*self.ice_mask

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
            
            H_ice, Z_surf, time = checkpoint(checkpointed_step, H_ice, Z_surf, smb, time)
            it += 1
            # Compute surface mass balance (SMB)
            if (time-t_last_update)>=t_freq:
                smb = update_smb(Z_surf,precip_tensor,T_ma_lowest,T_mj_lowest) * self.ice_mask
                t_last_update=time.clone()
            if it % unroll == 0:
                # cut the graph
                H_ice = H_ice.detach().requires_grad_()
                Z_surf = Z_surf.detach().requires_grad_()
        return H_ice

# Wrap the solve_glacier_dynamics function in the checkpointed module
glacier_model = GlacierDynamicsCheckpointed(Z_topo, ttot, rho, g, fd, Lx, Ly, dx, dy, 1, device,ice_mask)

# Define a function for the forward computation to use with checkpointing
def inversion_thicknes(precip_tensor, T_ma_lowest,T_mj_lowest,observed_thk, reg_lambda,unroll):
    # Perform forward simulation
    H_simulated = glacier_model(precip_tensor, T_ma_lowest,T_mj_lowest,unroll)

    # Compute data fidelity term
    data_fidelity = torch.mean(torch.abs(H_simulated - observed_thk) ** 2)

    # Compute smoothness regularization
    smoothness_x = torch.sum((precip_tensor[:, 1:] - precip_tensor[:, :-1]) ** 2)
    smoothness_y = torch.sum((precip_tensor[1:, :] - precip_tensor[:-1, :]) ** 2)
    smoothness_reg = smoothness_x + smoothness_y

    # Total loss
    loss = data_fidelity + reg_lambda * smoothness_reg
    return loss,H_simulated





# Observed glacier thickness (assumed already loaded as observed_thk tensor)
# observed_thk = torch.load('Obs_2D.pt').to(device)
observed_thk = torch.load('Obs_2D.pt').to(device)


# Hyperparameters
initial_lr = 0.1
reg_lambda = 1
n_iterations = 10

# Optimizer setup
# optimizer = torch.optim.Adam([precip_tensor], lr=initial_lr)


### Try to smooth out the Precipitation field after each forward run.

unrolls=[300]
# Main loop
for ur in unrolls:
    # Reset the starting point in tracking maximum GPU memory occupied by tensors in bytes for a given device.
    torch.cuda.reset_peak_memory_stats()
    # Tracking variables
    total_loss_history = []
    data_fidelity_history = []
    regularization_history = []
    total_gradients_history = []
    # Initial guesses for inversion problem
    precip_tensor = torch.full((1, *Z_topo.shape), 2.4, requires_grad=True, device=device)
    # precip_tensor = torch.load('percip.pt').to(device).requires_grad_()

    T_mj_lowest = torch.tensor(19., requires_grad=False, device=device) 
    T_ma_lowest = torch.tensor(9., requires_grad=False, device=device) 
    optimizer = torch.optim.Adam([precip_tensor], lr=initial_lr)


    for i in range(n_iterations):
        optimizer.zero_grad()
        print(f'Nb iterations: {i + 1} {50*"-"}')

        # Forward pass with gradient checkpointing
        loss, H_simulated = inversion_thicknes(precip_tensor, T_ma_lowest, T_mj_lowest, observed_thk, reg_lambda, ur)

        # Backward pass
        loss.backward()
        # Collect gradient norms
        grad_norms = []
        for param in optimizer.param_groups[0]['params']:
            norm = torch.norm(param.grad)
            print(f'Gradient norm of parameter: {norm:.4f}')

        # Optimizer step
        optimizer.step()

        # Log timing
        # Store history
        total_loss_history.append(loss.item())
        total_gradients_history.append(torch.norm(precip_tensor.grad).item())

        with torch.no_grad():
            # visualize(precip_tensor[0],i,H_simulated,Lx,Ly)
            data_fidelity = torch.mean((H_simulated - observed_thk) ** 2).item()
            data_fidelity_history.append(data_fidelity)
            print(data_fidelity)
        # Convert tensors to NumPy arrays for plotting
        Z_ELA_np = precip_tensor.to(torch.float32).detach().cpu().numpy()
        H_ice_np = H_simulated.to(torch.float32).detach().cpu().numpy()
        mask_observed = observed_thk#torch.sigmoid((observed_thk - 5) * 10.0)
        observed_thk_np = mask_observed.to(torch.float32).detach().cpu().numpy()

    # Create a figure with three subplots side by side
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))  # Adjust figsize for better layout
    # Plot the ELA field
    im1 = ax[0].imshow(Z_ELA_np[0], cmap='terrain', origin='lower')
    fig.colorbar(im1, ax=ax[0], orientation='vertical', label='Elevation (m)')
    ax[0].set_title('Reconstructed ELA Field')
    ax[0].set_xlabel('Distance, km')
    ax[0].set_ylabel('Distance, km')
    # Second subplot: Ice thickness (simulated)
    im2 = ax[1].imshow(np.where(H_ice_np > 0, H_ice_np, np.nan), cmap='jet', origin='lower')
    fig.colorbar(im2, ax=ax[1], orientation='vertical', label='Ice Thickness (m)')
    ax[1].set_title('Simulated Ice Thickness')
    ax[1].set_xlabel('Distance, km')
    # Third subplot: Observed ice thickness
    im3 = ax[2].imshow(np.where(observed_thk_np > 0, observed_thk_np, np.nan), cmap='jet', origin='lower')
    fig.colorbar(im3, ax=ax[2], orientation='vertical', label='Ice Thickness (m)')
    ax[2].set_title('Observed Ice Thickness')
    ax[2].set_xlabel('Distance, km')
    # Add a main title
    fig.suptitle('Glacier Evolution Analysis', fontsize=16)
    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # Display the plots
    plt.show()

    plot_gradient_evolution(total_gradients_history,name=f'Gradients{ur}.png')

    plot_loss_components(total_loss_history, data_fidelity_history, regularization_history,name=f'Loss{ur}.png')
torch.save(precip_tensor,"percip.pt")



