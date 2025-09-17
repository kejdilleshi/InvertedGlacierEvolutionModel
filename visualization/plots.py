import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
import torch
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Visualize glacier surface and thickness
def visualize(Z_surf, time, H_ice, Lx, Ly):
    clear_output(wait=True)  # Clear the previous output in the notebook
    plt.figure(2, figsize=(11, 4), dpi=200)
    # Convert tensors to float32 for NumPy compatibility
    Z_surf_np = Z_surf.to(torch.float32).cpu().numpy()
    H_ice_np = H_ice.to(torch.float32).cpu().numpy()

    # First subplot: Ice surface
    plt.subplot(1, 2, 1)
    plt.imshow(Z_surf_np, extent=[0, Lx / 1000, 0, Ly / 1000], cmap='terrain', origin='lower')
    plt.colorbar(label='Elevation (m)')
    plt.title('Ice Surface at ' + str(int(time)) + ' y')
    plt.xlabel('Distance, km')
    plt.ylabel('Distance, km')

    # Second subplot: Ice thickness
    plt.subplot(1, 2, 2)
    plt.imshow(np.where(H_ice.cpu().numpy() > 0, H_ice.cpu().numpy(), np.nan), extent=[0, Lx/1000, 0, Ly/1000], cmap='jet', origin='lower')
    plt.colorbar(label='Ice Thickness (m)')
    plt.title('Ice Thickness at ' + str(int(time)) + ' y')
    plt.xlabel('Distance, km')
    plt.ylabel('Distance, km')
    # Display the plot briefly, then close
    # plt.tight_layout()
    # plt.pause(2)
    # plt.close()
    plt.show()

# Plot loss components
def plot_loss_components(total_loss_history, data_fidelity_history, name):
    plt.figure(figsize=(10, 6))
    plt.plot(total_loss_history, label='Total Loss', color='b', linewidth=2)
    plt.plot(data_fidelity_history, label='Data Fidelity', color='g', linestyle='--', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss Value')
    plt.title('Loss Function Components Over Iterations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(name)


def plot_loss_and_precipitation(data_fidelity_history, Precip_history, name, true_precip=1.5):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))  # Two subplots side by side

    # First subplot: Loss components
    axs[0].plot(data_fidelity_history, label='Data Fidelity', color='g', linestyle='--', linewidth=2)
    axs[0].set_title('Loss Components Over Iterations')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Loss Value')
    axs[0].legend()
    axs[0].grid(True)

    # Second subplot: Precipitation evolution
    axs[1].plot(Precip_history, label='Estimated Precipitation', color='b', linewidth=2)
    axs[1].axhline(y=true_precip, color='k', linestyle='--', label='True Precipitation')
    axs[1].set_title('Precipitation Evolution Over Iterations')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Precipitation (m/yr)')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(name)
    plt.show()

def plot_loss_and_temperature(data_fidelity_history, Temp_history, name, true_temp=7.0):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))  # Two subplots side by side

    # First subplot: Loss components
    axs[0].plot(data_fidelity_history, label='Data Fidelity', color='g', linestyle='--', linewidth=2)
    axs[0].set_title('Loss Components Over Iterations')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Loss Value')
    axs[0].legend()
    axs[0].grid(True)

    # Second subplot: Temperature evolution
    axs[1].plot(Temp_history, label='Estimated Temperature', color='r', linewidth=2)
    axs[1].axhline(y=true_temp, color='k', linestyle='--', label='True Temperature')
    axs[1].set_title('Temperature Evolution Over Iterations')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Temperature (°C)')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(name)
    plt.show()


def plot_gradient_evolution(total_gradients_history,name):
    plt.figure(figsize=(10,6))
# plt.plot(ELA_evolution,label="evolution of ELA",color='b')
    plt.plot(total_gradients_history, label='Evolution of gradients')
    plt.xlabel('Iteration')
    plt.legend()
    plt.savefig(name)

def plot_resulting_ELA(Z_ELA,H_simulated,observed_thk):
    # Convert tensors to NumPy arrays for plotting
    Z_ELA_np = Z_ELA.to(torch.float32).detach().cpu().numpy()
    H_ice_np = H_simulated.to(torch.float32).detach().cpu().numpy()
    observed_thk_np = observed_thk.to(torch.float32).detach().cpu().numpy()

    # Create a figure with three subplots side by side
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))  # Adjust figsize for better layout

    # Plot the ELA field
    im1 = ax[0].imshow(Z_ELA_np, cmap='terrain', origin='lower')
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
def plot_loss_topography_griddata(df, true_P=None, true_T=None, levels=5, savepath=None):
    """
    Plot a topographic map of the loss function using griddata interpolation,
    with contour lines for key minima.
    """
    # Validate input
    if not {'P', 'T', 'loss'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'P', 'T', and 'loss' columns")

    # Extract raw data
    points = df[['P', 'T']].values
    loss_vals = df['loss'].values

    # Grid definition
    P_vals = df['P']
    T_vals = df['T']
    P_grid = np.linspace(P_vals.min(), P_vals.max(), 200)
    T_grid = np.linspace(T_vals.min(), T_vals.max(), 200)
    P_mesh, T_mesh = np.meshgrid(P_grid, T_grid)

    # Interpolate loss values on the grid
    Z = griddata(points, loss_vals, (P_mesh, T_mesh), method='linear')

    # Plot filled contour
    plt.figure(figsize=(8, 6))
    contourf = plt.contourf(P_mesh, T_mesh, Z, levels=levels, cmap='viridis')
    cbar = plt.colorbar(contourf)
    cbar.set_label("Loss")

    # Overlay fine contour lines at key minima levels
    contour_lines = plt.contour(P_mesh, T_mesh, Z, levels=[0.001, 0.002], 
                                colors='white', linewidths=0.5, linestyles='--')
    plt.clabel(contour_lines, fmt='%.3f', inline=True, fontsize=8)

    # Scatter sampled points
    plt.scatter(P_vals, T_vals, s=5, c='black')

    # Optional: overlay true point
    if true_P is not None and true_T is not None:
        plt.plot(true_P, true_T, 'r*', markersize=15, label=f'True T={true_T}, P={true_P}')
        plt.legend()
    #plot stars for 2 examples
    plt.plot(0.842, 4.584, marker='*', color='blue', markersize=12,
         label='low temp low precip')
    plt.legend()

    plt.plot(2.567, 8.931, marker='*', color='green', markersize=12,
         label='high temp high precip')
    plt.legend()


    plt.xlabel("Precipitation (P)")
    plt.ylabel("Temperature (T)")
    plt.title("Topographic Map of Loss Function")
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=300)
    plt.show()


def visualize_velocities(ubar, vbar, H_ice, smb, time, dx=100, dy=100):
    clear_output(wait=True)
    plt.figure(2, figsize=(10, 6), dpi=200)

    # ---- tensors -> numpy ----
    H_ice_np = H_ice.detach().to(torch.float32).cpu().numpy()
    u_np     = ubar.detach().to(torch.float32).cpu().numpy()
    v_np     = vbar.detach().to(torch.float32).cpu().numpy()
    smb_np   = smb.detach().to(torch.float32).cpu().numpy()

    # Clamp negatives to -5; positives unchanged
    smb_plot = np.maximum(smb_np, -5.0)

    ny, nx = H_ice_np.shape
    extent = [0, nx*dx, 0, ny*dy]
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y)

    # --- Subplot 1: Ice Thickness ---
    ax1 = plt.subplot(1, 3, 1)
    im1 = ax1.imshow(np.where(H_ice_np > 0, H_ice_np, np.nan),
                     cmap='jet', origin='lower', extent=extent)
    cax1 = make_axes_locatable(ax1).append_axes("right", size="3%", pad=0.05)
    plt.colorbar(im1, cax=cax1, label="Ice Thickness (m)")
    ax1.set_title(f"Ice Thickness at {int(time)} y")
    ax1.set_xlabel("x (m)"); ax1.set_ylabel("y (m)")

    # --- Subplot 2: SMB (clamped) + zero contour ---
    ax2 = plt.subplot(1, 3, 2)
    im2 = ax2.imshow(smb_plot, cmap='plasma', origin='lower', extent=extent, vmin=-5)
    cax2 = make_axes_locatable(ax2).append_axes("right", size="3%", pad=0.05)
    plt.colorbar(im2, cax=cax2, label="SMB")

    # zero-SMB contour (equilibrium line)
    ax2.contour(X, Y, smb_np, levels=[0.0], colors='blue', linewidths=1.5)
    ax2.set_title(f"SMB (≤-5 clamped) at {int(time)} y")
    ax2.set_xlabel("x (m)"); ax2.set_ylabel("y (m)")

    # --- Subplot 3: Velocity Vectors ---
    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(np.where(H_ice_np > 0, H_ice_np, np.nan),
               cmap="Greys", origin="lower", alpha=0.6, extent=extent)
    step = max(1, min(ny, nx) // 30)
    x_coords = np.arange(0, nx, step) * dx
    y_coords = np.arange(0, ny, step) * dy
    ax3.quiver(x_coords, y_coords,
               u_np[::step, ::step], v_np[::step, ::step],
               color="red", scale_units="xy", scale=1, width=0.002)
    ax3.set_title(f"Velocity Field at {int(time)} y")
    ax3.set_xlabel("x (m)"); ax3.set_ylabel("y (m)")

    plt.tight_layout()
    plt.show()