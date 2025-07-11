import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
import torch

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
    plt.imshow( H_ice_np, cmap='jet', origin='lower')
    plt.colorbar(label='Ice Thickness (m)')
    plt.title('Ice Thickness at ' + str(int(time)) + ' y')
    plt.xlabel('Distance, km')
    plt.ylabel('Distance, km')
    # Display the plot briefly, then close
    plt.tight_layout()
    plt.pause(2)
    plt.close()

# Plot loss components
def plot_loss_components(total_loss_history, data_fidelity_history, regularization_history,name):
    plt.figure(figsize=(10, 6))
    plt.plot(total_loss_history, label='Total Loss', color='b', linewidth=2)
    plt.plot(data_fidelity_history, label='Data Fidelity', color='g', linestyle='--', linewidth=2)
    plt.plot(regularization_history, label='Regularization (Smoothness)', color='r', linestyle='-.', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss Value')
    plt.title('Loss Function Components Over Iterations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(name)

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

    # Display the plots
    plt.show()