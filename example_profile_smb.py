"""
Example: Using the 'profile' SMB forcing method

This script demonstrates how to use the new 'profile' SMB forcing type
which maps a 1D elevation-based SMB profile to a 2D topography.
"""

import torch
import netCDF4
from core.glacier import GlacierDynamicsCheckpointed
from config.read_config import parse_arguments

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load topography
nc = netCDF4.Dataset('./data/geology.nc')
topo = torch.tensor(nc.variables['topg'][:], device=device, dtype=torch.float32)
nc.close()

# -----------------------------------------
# Build simple SMB profile: -15 -> 5 linearly
# length = (zmax - zmin)/100 m  (using ceil)
# -----------------------------------------
zmin = topo.min()
zmax = topo.max()
dz = 100.0  # 100m elevation bins

N = int(torch.ceil((zmax - zmin) / dz).item())  # number of bins
N = max(N, 2)  # safety minimum

# Create SMB profile: linearly from -15 m/yr at lowest elevation to 5 m/yr at highest
smb_vec = torch.linspace(-15.0, 5.0, steps=N, device=device, dtype=torch.float32)

# Optional: clamp and manually adjust specific values
smb_vec = torch.clamp(smb_vec, min=-15.0, max=1.0)
# Example: set ELA by manually adjusting a specific elevation bin
# smb_vec[18] = 0.0  # Set SMB to 0 at bin 18 (approximate ELA)

# -----------------------------------------
# Time-varying SMB profile (optional)
# -----------------------------------------
# To create a time-varying profile, stack multiple profiles into a 2D tensor
# with shape [n_timesteps, N], where each row is an SMB profile at a different time

# Example: Create 3 time steps with linearly changing profiles
n_timesteps = 3
smb_vec_time_varying = torch.stack([
    torch.linspace(-15.0, 5.0, steps=N, device=device),   # Initial
    torch.linspace(-12.0, 6.0, steps=N, device=device),   # Mid
    torch.linspace(-10.0, 7.0, steps=N, device=device),   # Final
])

# -----------------------------------------
# Initialize glacier model
# -----------------------------------------
args = parse_arguments()
H_init = torch.zeros_like(topo)  # Initial ice thickness
ice_mask = torch.ones_like(topo)  # Ice mask (1 where ice can exist)

glacier = GlacierDynamicsCheckpointed(
    Z_topo=topo,
    H_init=H_init,
    ice_mask=ice_mask,
    device=device,
    args=args,
    model=None,  # Or provide emulator model if using emulator scheme
    visualize=False
)

# -----------------------------------------
# Run glacier dynamics with profile SMB
# -----------------------------------------

# Example 1: Constant SMB profile (1D smb_vec)
H_final_constant = glacier.forward(
    precip_tensor=None,    # Not used for profile method
    T_m_lowest=None,       # Not used for profile method
    T_s=None,              # Not used for profile method
    melt_factor=None,      # Not used for profile method
    smb_method='profile',
    smb_vec=smb_vec,       # 1D profile [N]
    z_min=zmin,            # Minimum elevation
    dz=dz,                 # Elevation spacing
    forward_scheme='emulator'  # or 'SIA'
)

# Example 2: Time-varying SMB profile (2D smb_vec)
H_final_varying = glacier.forward(
    precip_tensor=None,
    T_m_lowest=None,
    T_s=None,
    melt_factor=None,
    smb_method='profile',
    smb_vec=smb_vec_time_varying,  # 2D profile [time, N]
    z_min=zmin,
    dz=dz,
    forward_scheme='emulator'
)

print(f"Final ice thickness range (constant profile): {H_final_constant.min():.2f} to {H_final_constant.max():.2f} m")
print(f"Final ice thickness range (varying profile): {H_final_varying.min():.2f} to {H_final_varying.max():.2f} m")

# -----------------------------------------
# Helper function to create SMB profile from topography
# -----------------------------------------
def create_smb_profile(topo, smb_low=-15.0, smb_high=5.0, dz=100.0, device='cpu'):
    """
    Create an SMB elevation profile for a given topography.

    Parameters:
    -----------
    topo : torch.Tensor
        Topography/DEM (elevation in meters)
    smb_low : float
        SMB value at lowest elevation (m ice eq / year)
    smb_high : float
        SMB value at highest elevation (m ice eq / year)
    dz : float
        Elevation bin spacing (m)
    device : str
        Torch device ('cpu' or 'cuda')

    Returns:
    --------
    smb_vec : torch.Tensor
        1D SMB profile [N]
    z_min : float
        Minimum elevation
    dz : float
        Elevation spacing
    """
    zmin = topo.min().item()
    zmax = topo.max().item()

    N = int(torch.ceil(torch.tensor((zmax - zmin) / dz)).item())
    N = max(N, 2)

    smb_vec = torch.linspace(smb_low, smb_high, steps=N,
                            device=device, dtype=torch.float32)

    return smb_vec, zmin, dz

# Example usage of helper function
smb_profile, z_minimum, elevation_spacing = create_smb_profile(topo, device=device)
