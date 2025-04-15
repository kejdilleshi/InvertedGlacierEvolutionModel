import torch
import netCDF4
from torch.utils.checkpoint import checkpoint
from utils import print_gpu_utilization, print_peak_gpu_memory, device
from visualization import plot_gradient_evolution,plot_loss_components,visualize,plot_resulting_ELA
from inversion import checkpointed_inversion_thicknes
from climate import surface_mass_balance
from read_config import parse_arguments
from glacier_dynamics import GlacierDynamicsCheckpointed
args = parse_arguments()


# Prepare hooks for tensors 
def print_hook_b(grad):
    print("\n db/dELA:", torch.mean(grad))
    return grad

def reduce_hook(grad):
    return grad * 0.8

nc_file = netCDF4.Dataset('geology.nc')
Z_topo = torch.tensor(nc_file.variables['topg'][:], device=device, dtype=torch.bfloat16)
topo_1880 = torch.tensor(nc_file.variables['surf_1880'][:], device=device, dtype=torch.bfloat16)
thk_1880=topo_1880-Z_topo
thk_1880.to(device=device)
# Convert the parsed arguments into torch.float32 tensors
grad_b = torch.tensor(args.grad_b, dtype=torch.float32, device=device)
b_max = torch.tensor(args.b_max, dtype=torch.float32, device=device)
Z_ELA = torch.tensor(args.Z_ELA, dtype=torch.float32, device=device)
rho = torch.tensor(args.rho, dtype=torch.float32, device=device)
g = torch.tensor(args.g, dtype=torch.float32, device=device)
fd = torch.tensor(args.fd, dtype=torch.float32, device=device)

Lx=Z_topo.shape[1]*args.dx
Ly=Z_topo.shape[0]*args.dy

def solve_glacier_dynamics(Z_topo, ttot, grad_b, b_max, Z_ELA,H_initial,t_start=args.t_start, rho=args.rho, g=args.g, fd=args.fd, Lx=Lx, Ly=Ly, dx=args.dx, dy=args.dy, dtmax=1, device=device):
    """
    Solve the glacier dynamics using a diffusion-based solver with PyTorch.
    """
    # nx = int(Lx / dx)
    # ny = int(Ly / dy)
    epsilon = torch.tensor(1.e-20, dtype=torch.bfloat16, device=device)

    # Initialize ice thickness and surface
    H_ice = H_initial.to(device=device)
    # H_ice = torch.zeros((ny, nx), device=device, dtype=torch.float32)
    Z_surf = Z_topo + H_ice
    b = surface_mass_balance(Z_surf)

    time = torch.tensor(t_start, dtype=torch.float32, device=device) 
    dt = torch.tensor(dtmax, dtype=torch.bfloat16, device=device)
    it=0

    while time < ttot:

        time += dt
        it += 1
        # Compute H_avg
        H_avg = 0.25 * (H_ice[:-1, :-1] + H_ice[1:, 1:] + H_ice[:-1, 1:] + H_ice[1:, :-1])
  

        # Compute Snorm
        Sx = (Z_surf[:, 1:] - Z_surf[:, :-1]) / dx
        Sy = (Z_surf[1:, :] - Z_surf[:-1, :]) / dy
        Sx = 0.5 * (Sx[:-1, :] + Sx[1:, :])
        Sy = 0.5 * (Sy[:, :-1] + Sy[:, 1:])
        Snorm = torch.sqrt(Sx**2 + Sy**2)

        # Compute diffusivity
        # Perform high-precision computation for stability

        D = fd * (rho * g)**3.0 * H_avg**5 * Snorm**2+epsilon
        # Compute adaptive time step
        dt = min(min(dx, dy)**2 / (2.7 * torch.max(D).item()), dtmax)

        # Compute fluxes
        qx = -(0.5 * (D[:-1, :] + D[1:, :])) * (Z_surf[1:-1, 1:] - Z_surf[1:-1, :-1]) / dx
        qy = -(0.5 * (D[:, :-1] + D[:, 1:])) * (Z_surf[1:, 1:-1] - Z_surf[:-1, 1:-1]) / dy

        # Compute thickness change rate
        dHdt = -(torch.diff(qx, dim=1) / dx + torch.diff(qy, dim=0) / dy)
        print(torch.mean(H_avg[1:,:]),torch.mean(H_avg[:,1:])) 
        print(torch.mean(qx/H_avg[1:,:]),torch.mean(qy/H_avg[:,1:]))

        # Update ice thickness
        H_ice[1:-1, 1:-1] += dt * dHdt

        # Compute surface mass balance (SMB)           
        b = surface_mass_balance(Z_surf,Z_ELA=Z_ELA)

        H_ice[1:-1, 1:-1] += dt * b[1:-1, 1:-1]

        # Ensure ice thickness remains positive
        H_ice = torch.maximum(H_ice, torch.tensor(0.0, dtype=torch.float32, device=device))

        # Update surface topography
        Z_surf = Z_topo + H_ice
     
        # Visualization at specified intervals
        if it % 1 == 0:
            print(it,time)
            visualize(Z_surf,time,H_ice,Lx,Ly)
   

    return H_ice , it

# h,it=solve_glacier_dynamics(Z_topo, args.ttot, args.grad_b, args.b_max, args.Z_ELA,H_initial=thk_1880)

# Wrap the solve_glacier_dynamics function in the checkpointed module
glacier_model = GlacierDynamicsCheckpointed(Z_topo, args.ttot, grad_b, b_max, rho, g, fd, Lx, Ly, args.dx, args.dy, 1, device,thk_1880)

#----------------------------------------
# Reset the starting point in tracking maximum GPU memory occupied by tensors in bytes for a given device.
torch.cuda.reset_peak_memory_stats()

# Initial guesses for inversion problem
Z_ELA = torch.full(Z_topo.shape, 3200.0, requires_grad=True, device=device)

# Observed glacier thickness (assumed already loaded as observed_thk tensor)
# observed_thk = torch.load('Obs_2D.pt', weights_only=True).to(device,dtype=torch.float16)

topo_2017 = torch.tensor(nc_file.variables['surf_2017'][:], device=device, dtype=torch.bfloat16)
observed_thk=topo_2017-Z_topo
observed_thk.to(device=device)

# Define initial and final learning rates
initial_lr = 7
final_lr = 5
reg_lambda=0.00001
n_iterations = 30
# Optimizer setup remains unchanged
optimizer = torch.optim.Adam([Z_ELA], lr=initial_lr)

# Initialize lists to track loss components
total_loss_history = []
data_fidelity_history = []
regularization_history = []
total_gradients_history=[]


for i in range(n_iterations):
    # Update the learning rate
    lr = initial_lr - (i / (n_iterations - 1)) * (initial_lr - final_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    optimizer.zero_grad()  # Zero gradients
    print('Nb iterations: ',i+1)

    # Perform forward pass with gradient checkpointing
    loss,H_simulated = checkpoint(
        checkpointed_inversion_thicknes, Z_ELA, observed_thk,reg_lambda,glacier_model
    )
    Z_ELA.register_hook(reduce_hook)
    # Backpropagate loss and update parameters
    loss.backward()
    optimizer.step()

    # Log the components of the loss for tracking
    total_loss_history.append(loss.item())
    total_gradients_history.append(torch.max(Z_ELA.grad).item())

print_gpu_utilization
print_peak_gpu_memory

plot_gradient_evolution(total_gradients_history)

plot_loss_components(total_loss_history, data_fidelity_history, regularization_history)

plot_resulting_ELA(Z_ELA,H_simulated,observed_thk)
