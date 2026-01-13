"""
Temporary script to compare SMB and thickness evolution from different runs
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import numpy as np

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Load SMB fields (static)
print("Loading SMB fields...")
SMB_swift = torch.load('results/SMB/swift-sweep-1/smb_inverted.pt', map_location=device)
SMB_rose = torch.load('results/SMB/rose-sweep-1/smb_inverted.pt', map_location=device)

print(f"SMB swift shape: {SMB_swift.shape}")
print(f"SMB rose shape: {SMB_rose.shape}")

# Get list of thickness files from both runs
swift_run_path = Path('results/SMB/swift-run')
rose_run_path = Path('results/SMB/rose-run')

swift_files = sorted(swift_run_path.glob('H_ice_*.pt'))
rose_files = sorted(rose_run_path.glob('H_ice_*.pt'))

print(f"\nFound {len(swift_files)} swift thickness files")
print(f"Found {len(rose_files)} rose thickness files")

# Extract time values from filenames
def extract_time(filepath):
    """Extract time from filename like 'H_ice_1951.03.pt'"""
    stem = filepath.stem  # e.g., 'H_ice_1951.03'
    time_str = stem.split('_')[-1]  # e.g., '1951.03'
    return float(time_str)

swift_times = [extract_time(f) for f in swift_files]
rose_times = [extract_time(f) for f in rose_files]

print(f"\nSwift time range: {swift_times[0]:.2f} to {swift_times[-1]:.2f}")
print(f"Rose time range: {rose_times[0]:.2f} to {rose_times[-1]:.2f}")

# Create figure with subplots
# Layout: 2 rows x 3 columns
# Row 1: SMB_swift, Thickness_swift(t), Residual_thickness_swift
# Row 2: SMB_rose, Thickness_rose(t), Residual_thickness_rose

def create_comparison_plot(time_idx):
    """Create comparison plot for a given time index"""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Load thickness at this time
    H_swift = torch.load(swift_files[time_idx], map_location=device).detach().cpu().numpy()
    H_rose = torch.load(rose_files[time_idx], map_location=device).detach().cpu().numpy()

    time = swift_times[time_idx]

    # Convert to numpy
    SMB_swift_np = SMB_swift.detach().cpu().numpy()
    SMB_rose_np = SMB_rose.detach().cpu().numpy()

    # Row 1: Swift
    # SMB Swift
    vmin_smb = min(SMB_swift_np.min(), SMB_rose_np.min())
    vmax_smb = max(SMB_swift_np.max(), SMB_rose_np.max())

    im00 = axes[0, 0].imshow(SMB_swift_np, cmap='RdBu_r', vmin=vmin_smb, vmax=vmax_smb, aspect='auto')
    axes[0, 0].set_title('SMB (swift-sweep-1)', fontsize=12, fontweight='bold')
    axes[0, 0].invert_yaxis()
    plt.colorbar(im00, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # Thickness Swift
    vmin_h = 0
    vmax_h = 850

    im01 = axes[0, 1].imshow(H_swift, cmap='viridis', vmin=vmin_h, vmax=vmax_h, aspect='auto')
    axes[0, 1].set_title(f'Thickness (swift-run) @ t={time:.2f}', fontsize=12, fontweight='bold')
    axes[0, 1].invert_yaxis()
    plt.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Residual SMB Swift
    residual_smb_swift = SMB_swift_np - SMB_rose_np
    lim_smb = np.max(np.abs(residual_smb_swift))

    im02 = axes[0, 2].imshow(residual_smb_swift, cmap='RdBu', vmin=-lim_smb, vmax=lim_smb, aspect='auto')
    axes[0, 2].set_title('Residual SMB (swift - rose)', fontsize=12, fontweight='bold')
    axes[0, 2].invert_yaxis()
    plt.colorbar(im02, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Row 2: Rose
    # SMB Rose
    im10 = axes[1, 0].imshow(SMB_rose_np, cmap='RdBu_r', vmin=vmin_smb, vmax=vmax_smb, aspect='auto')
    axes[1, 0].set_title('SMB (rose-sweep-1)', fontsize=12, fontweight='bold')
    axes[1, 0].invert_yaxis()
    plt.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Thickness Rose
    im11 = axes[1, 1].imshow(H_rose, cmap='viridis', vmin=vmin_h, vmax=vmax_h, aspect='auto')
    axes[1, 1].set_title(f'Thickness (rose-run) @ t={time:.2f}', fontsize=12, fontweight='bold')
    axes[1, 1].invert_yaxis()
    plt.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # Residual Thickness
    residual_h = H_swift - H_rose
    lim_h = np.max(np.abs(residual_h))

    im12 = axes[1, 2].imshow(residual_h, cmap='RdBu', vmin=-lim_h, vmax=lim_h, aspect='auto')
    axes[1, 2].set_title('Residual Thickness (swift - rose)', fontsize=12, fontweight='bold')
    axes[1, 2].invert_yaxis()
    plt.colorbar(im12, ax=axes[1, 2], fraction=0.046, pad=0.04)

    plt.suptitle(f'SMB and Thickness Comparison @ t={time:.2f}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig

# Create plots for every time step
print("\nCreating comparison plots for all time steps...")

# Create kot directory if it doesn't exist
from pathlib import Path
Path('kot').mkdir(exist_ok=True)

# Plot for all time steps
for idx in range(len(swift_files)):
    fig = create_comparison_plot(idx)
    time = swift_times[idx]
    output_path = f'kot/comparison_t_{time:.2f}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    if (idx + 1) % 10 == 0:
        print(f"Progress: {idx + 1}/{len(swift_files)} plots created")
    plt.close()

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Created {len(swift_files)} comparison plots")
print(f"Saved in: kot/comparison_t_*.png")
print(f"Time range: {swift_times[0]:.2f} to {swift_times[-1]:.2f}")
print("="*60)



# print("\nCreating animation...")
# fig = plt.figure(figsize=(15, 10))
# def animate(i):
#     plt.clf()
#     # Load thickness at this time
#     H_swift = torch.load(swift_files[i], map_location=device).detach().cpu().numpy()
#     H_rose = torch.load(rose_files[i], map_location=device).detach().cpu().numpy()
#     time = swift_times[i]
#     # Convert to numpy
#     SMB_swift_np = SMB_swift.detach().cpu().numpy()
#     SMB_rose_np = SMB_rose.detach().cpu().numpy()
#     axes = fig.subplots(2, 3)
#     # Same plotting code as above
#     vmin_smb = min(SMB_swift_np.min(), SMB_rose_np.min())
#     vmax_smb = max(SMB_swift_np.max(), SMB_rose_np.max())
#     im00 = axes[0, 0].imshow(SMB_swift_np, cmap='RdBu_r', vmin=vmin_smb, vmax=vmax_smb, aspect='auto')
#     axes[0, 0].set_title('SMB (swift-sweep-1)', fontsize=12, fontweight='bold')
#     axes[0, 0].invert_yaxis()
#     plt.colorbar(im00, ax=axes[0, 0], fraction=0.046, pad=0.04)
#     vmin_h = 0
#     vmax_h = 850
#     im01 = axes[0, 1].imshow(H_swift, cmap='viridis', vmin=vmin_h, vmax=vmax_h, aspect='auto')
#     axes[0, 1].set_title(f'Thickness (swift-run) @ t={time:.2f}', fontsize=12, fontweight='bold')
#     axes[0, 1].invert_yaxis()
#     plt.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.04)
#     residual_smb_swift = SMB_swift_np - SMB_rose_np
#     lim_smb = np.max(np.abs(residual_smb_swift))
#     im02 = axes[0, 2].imshow(residual_smb_swift, cmap='RdBu', vmin=-lim_smb, vmax=lim_smb, aspect='auto')
#     axes[0, 2].set_title('Residual SMB (swift - rose)', fontsize=12, fontweight='bold')
#     axes[0, 2].invert_yaxis()
#     plt.colorbar(im02, ax=axes[0, 2], fraction=0.046, pad=0.04)
#     im10 = axes[1, 0].imshow(SMB_rose_np, cmap='RdBu_r', vmin=vmin_smb, vmax=vmax_smb, aspect='auto')
#     axes[1, 0].set_title('SMB (rose-sweep-1)', fontsize=12, fontweight='bold')
#     axes[1, 0].invert_yaxis()
#     plt.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.04)
#     im11 = axes[1, 1].imshow(H_rose, cmap='viridis', vmin=vmin_h, vmax=vmax_h, aspect='auto')
#     axes[1, 1].set_title(f'Thickness (rose-run) @ t={time:.2f}', fontsize=12, fontweight='bold')
#     axes[1, 1].invert_yaxis()
#     plt.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.04)
#     residual_h = H_swift - H_rose
#     lim_h = np.max(np.abs(residual_h))
#     im12 = axes[1, 2].imshow(residual_h, cmap='RdBu', vmin=-lim_h, vmax=lim_h, aspect='auto')
#     axes[1, 2].set_title('Residual Thickness (swift - rose)', fontsize=12, fontweight='bold')
#     axes[1, 2].invert_yaxis()
#     plt.colorbar(im12, ax=axes[1, 2], fraction=0.046, pad=0.04)
#     plt.suptitle(f'SMB and Thickness Comparison @ t={time:.2f}', fontsize=14, fontweight='bold')
#     plt.tight_layout()
# # Sample every 5th frame to reduce file size
# frames = range(0, len(swift_files), 5)
# anim = animation.FuncAnimation(fig, animate, frames=frames, interval=200, repeat=True)
# anim.save('kot/comparison_animation.mp4', writer='ffmpeg', fps=5, dpi=100)
# print("Animation saved: kot/comparison_animation.mp4")
# plt.close()
