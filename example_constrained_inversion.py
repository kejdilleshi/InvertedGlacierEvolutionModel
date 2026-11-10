"""
Example: Constrained SMB Profile Inversion with Physical Bounds

This script demonstrates how to invert for an SMB profile with physical constraints:
- smb_min: minimum SMB value (e.g., -15.0 m/yr for maximum ablation)
- smb_max: maximum SMB value (e.g., 3.0 m/yr for maximum accumulation)

The constraints are enforced using torch.clamp() which is differentiable.
"""

import torch
from core.glacier import GlacierDynamicsCheckpointed
from core.inversion import invert_profile
from core.emulator import load_emulator
from data.loader import load_geology
from config.read_config import parse_arguments
import matplotlib.pyplot as plt
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model and data
args = parse_arguments([])
args.ttot = torch.tensor(2010.0)
args.t_start = torch.tensor(2000.0)

# Load the IGM ice-flow emulator
emulator = load_emulator(args.emulator_path, device=device)

# Load geology
Z_topo, H_init, icemask = load_geology("data/geology.nc")

# Create glacier model
glacier_model = GlacierDynamicsCheckpointed(
    Z_topo, H_init, ice_mask=icemask, device=device,
    args=args, emulator=emulator, visualize=False
)

# -----------------------------------------
# Build SMB profile parameters
# -----------------------------------------
zmin = Z_topo.min().item()
zmax = Z_topo.max().item()
dz = 100.0  # 100m elevation bins

N = int(torch.ceil((torch.tensor(zmax) - torch.tensor(zmin)) / dz).item())
N = max(N, 2)

print(f"\nSMB Profile Setup:")
print(f"  Number of elevation bins: {N}")
print(f"  Elevation range: {zmin:.1f} to {zmax:.1f} m")
print(f"  Bin spacing: {dz:.1f} m")

# -----------------------------------------
# Create TRUE SMB profile (for generating observations)
# -----------------------------------------
smb_vec_true = torch.linspace(-10.0, 5.0, steps=N, device=device, dtype=torch.float32)
smb_vec_true = torch.clamp(smb_vec_true, min=-15.0, max=3.0)

print(f"\nTrue SMB profile:")
print(f"  Range: {smb_vec_true.min().item():.2f} to {smb_vec_true.max().item():.2f} m/yr")

# -----------------------------------------
# Generate observations
# -----------------------------------------
print("\nGenerating observations from true SMB profile...")
with torch.no_grad():
    H_observation = glacier_model(
        precip_tensor=None, T_m_lowest=None, T_s=None, melt_factor=None,
        smb_method='profile', smb_vec=smb_vec_true, z_min=zmin, dz=dz
    )

print(f"Observation thickness range: {H_observation.min().item():.2f} to {H_observation.max().item():.2f} m")

# -----------------------------------------
# Initialize SMB profile for inversion
# -----------------------------------------
smb_vec_init = torch.linspace(-15.0, 5.0, steps=N, device=device, dtype=torch.float32)
smb_vec_init.requires_grad = True

print(f"\nInitial SMB profile:")
print(f"  Range: {smb_vec_init.min().item():.2f} to {smb_vec_init.max().item():.2f} m/yr")

# -----------------------------------------
# Define physical constraints
# -----------------------------------------
SMB_MIN = -15.0  # Maximum ablation (m/yr)
SMB_MAX = 3.0    # Maximum accumulation (m/yr)

print(f"\nPhysical constraints:")
print(f"  SMB minimum: {SMB_MIN} m/yr")
print(f"  SMB maximum: {SMB_MAX} m/yr")

# -----------------------------------------
# Optimization with constraints
# -----------------------------------------
learning_rate = 0.1
reg_lambda = 0.01
max_iters = 500

optimizer = torch.optim.Adam([smb_vec_init], lr=learning_rate)

print(f"\nOptimization settings:")
print(f"  Learning rate: {learning_rate}")
print(f"  Regularization: {reg_lambda}")
print(f"  Max iterations: {max_iters}")
print()

loss_hist = []
smb_min_hist = []
smb_max_hist = []

for i in range(max_iters):
    optimizer.zero_grad()

    # Call invert_profile with physical constraints
    H_sim, loss, data_term, metrics = invert_profile(
        smb_vec=smb_vec_init,
        z_min=zmin,
        dz=dz,
        observation=H_observation,
        glacier_model=glacier_model,
        reg_lambda=reg_lambda,
        smb_min=SMB_MIN,  # Enforce minimum SMB
        smb_max=SMB_MAX   # Enforce maximum SMB
    )

    loss.backward()
    optimizer.step()

    loss_hist.append(loss.item())

    # Track SMB range (before clamping in next iteration)
    with torch.no_grad():
        smb_min_hist.append(smb_vec_init.min().item())
        smb_max_hist.append(smb_vec_init.max().item())

    if (i+1) % 10 == 0:
        print(f"Iter {i+1:4d}: Loss={loss.item():.6f}, "
              f"RMSE={metrics['rmse'].item():.4f}, "
              f"SMB range=[{smb_vec_init.min().item():.2f}, {smb_vec_init.max().item():.2f}]")

    # Early stopping
    if i > 40 and loss.item() < 0.0001:
        print(f"\nConverged at iteration {i+1}")
        break

# -----------------------------------------
# Final clamping (in case optimizer pushed beyond bounds)
# -----------------------------------------
with torch.no_grad():
    smb_vec_final = torch.clamp(smb_vec_init, min=SMB_MIN, max=SMB_MAX)

print(f"\n{'='*60}")
print(f"Final Results:")
print(f"{'='*60}")
print(f"Final loss: {loss_hist[-1]:.6f}")
print(f"Final thickness RMSE: {metrics['rmse'].item():.4f} m")
print(f"Final SMB range: [{smb_vec_final.min().item():.2f}, {smb_vec_final.max().item():.2f}] m/yr")
print(f"Expected range: [{SMB_MIN}, {SMB_MAX}] m/yr")
print(f"Constraints satisfied: {smb_vec_final.min().item() >= SMB_MIN and smb_vec_final.max().item() <= SMB_MAX}")
print(f"{'='*60}\n")

# -----------------------------------------
# Visualization
# -----------------------------------------
elevations = zmin + np.arange(N) * dz
true_cpu = smb_vec_true.detach().cpu().numpy()
final_cpu = smb_vec_final.detach().cpu().numpy()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: SMB Profile comparison
axes[0, 0].plot(elevations, true_cpu, 'b-', linewidth=2.5, label='True SMB', marker='o')
axes[0, 0].plot(elevations, final_cpu, 'r--', linewidth=2.5, label='Inverted SMB', marker='s')
axes[0, 0].axhline(y=0, color='k', linestyle=':', linewidth=1, alpha=0.5)
axes[0, 0].axhline(y=SMB_MIN, color='gray', linestyle='--', linewidth=1, alpha=0.5, label=f'Min={SMB_MIN}')
axes[0, 0].axhline(y=SMB_MAX, color='gray', linestyle='--', linewidth=1, alpha=0.5, label=f'Max={SMB_MAX}')
axes[0, 0].set_xlabel('Elevation (m)', fontsize=12)
axes[0, 0].set_ylabel('SMB (m ice eq/yr)', fontsize=12)
axes[0, 0].set_title('SMB Profile: True vs Inverted (Constrained)', fontsize=13, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# Panel 2: Loss curve
axes[0, 1].plot(loss_hist, 'b-', linewidth=1.5)
axes[0, 1].set_xlabel('Iteration', fontsize=12)
axes[0, 1].set_ylabel('Loss', fontsize=12)
axes[0, 1].set_title('Loss Convergence', fontsize=13, fontweight='bold')
axes[0, 1].set_yscale('log')
axes[0, 1].grid(True, alpha=0.3)

# Panel 3: SMB range evolution
axes[1, 0].plot(smb_min_hist, 'b-', linewidth=1.5, label='Min SMB')
axes[1, 0].plot(smb_max_hist, 'r-', linewidth=1.5, label='Max SMB')
axes[1, 0].axhline(y=SMB_MIN, color='b', linestyle='--', linewidth=1, alpha=0.5)
axes[1, 0].axhline(y=SMB_MAX, color='r', linestyle='--', linewidth=1, alpha=0.5)
axes[1, 0].set_xlabel('Iteration', fontsize=12)
axes[1, 0].set_ylabel('SMB (m ice eq/yr)', fontsize=12)
axes[1, 0].set_title('SMB Range Evolution (shows clamping effect)', fontsize=13, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

# Panel 4: Residual
residual = final_cpu - true_cpu
axes[1, 1].plot(elevations, residual, 'k-', linewidth=2, marker='o')
axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=1.5)
axes[1, 1].fill_between(elevations, 0, residual, alpha=0.3)
axes[1, 1].set_xlabel('Elevation (m)', fontsize=12)
axes[1, 1].set_ylabel('Residual (m ice eq/yr)', fontsize=12)
axes[1, 1].set_title('SMB Residual (Inverted - True)', fontsize=13, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('constrained_inversion_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("Results saved to 'constrained_inversion_results.png'")
