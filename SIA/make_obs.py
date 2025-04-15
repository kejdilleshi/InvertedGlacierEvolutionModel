# Import the necessary libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import netCDF4

device = torch.device('cpu')
print(device)



nc_file = netCDF4.Dataset('bedrock.nc')
Z_topo = torch.tensor(nc_file.variables['topg'][:], device=device, dtype=torch.bfloat16)
Z_topo=Z_topo[10:250,20:415]

Lx=790000
Ly=480000
dx=2000
dy=2000


# load the map
H_ice=torch.load('Obs_2D.pt')



# Define global variables to store the clicked points
clicked_points = []

def on_click(event):
    """Handle click events to record selected points."""
    if event.xdata is not None and event.ydata is not None:
        clicked_points.append((round(event.ydata),round(event.xdata)))
        print(f"Point selected: x={event.xdata}, y={event.ydata}")

def create_polygon_mask(H_ice, clicked_points, dx, dy):
    """
    Create a mask where grid points inside a user-defined polygon are 1, and outside are 0.
    
    Parameters:
        H_ice (torch.Tensor): Glacier thickness grid (2D).
        clicked_points (list of tuples): List of (x, y) points selected by user clicks.
        dx, dy (float): Grid spacing in x and y directions.

    Returns:
        torch.Tensor: Mask with 1 inside the polygon and 0 outside.
    """
    ny, nx = H_ice.shape
    x = torch.arange(0, nx * dx, dx)
    y = torch.arange(0, ny * dy, dy)
    xv, yv = torch.meshgrid(x, y, indexing='ij')  # Create grid coordinates
    
    # Flatten the grid for efficient processing
    points = torch.stack([xv.flatten(), yv.flatten()], dim=-1).cpu().numpy()
    
    # Create a closed polygon from clicked points
    polygon_path = Path(clicked_points)  # Automatically closes the polygon

    # Check which grid points are inside the polygon
    mask_flat = polygon_path.contains_points(points)
    
    # Reshape the mask back to the grid shape
    mask = torch.tensor(mask_flat, dtype=torch.float32, device=H_ice.device).reshape(nx, ny).T


    return mask

# Example visualization
def visualize_mask(H_ice, mask):
    """
    Visualize the glacier thickness and the mask.
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Glacier Thickness")
    plt.imshow(H_ice.cpu().numpy(), origin='lower', cmap='Blues')
    plt.colorbar(label='Thickness (m)')
    
    plt.subplot(1, 2, 2)
    plt.title("Mask")
    plt.imshow(mask.cpu().numpy(), origin='lower', cmap='Reds')
    plt.colorbar(label='Mask Value')
    plt.show()



# Display the glacier map and set up the click event
fig, ax = plt.subplots()
ax.set_title("Click on the map to select points")
im = ax.imshow(H_ice.cpu().numpy(), origin='lower', cmap='Blues')
fig.colorbar(im, ax=ax)

# Connect the click event
cid = fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()

# After clicking points, create the mask
mask = create_polygon_mask(H_ice, clicked_points, dx, dy)
print("points: ",clicked_points)

x_left=clicked_points[0][1] 
x_right=clicked_points[1][1] 
y_down= clicked_points[1][0]
y_up= clicked_points[0][0]

mask[y_down:y_up,x_left:x_right]=1
print(x_left,x_right,y_down,y_up)

# Visualize the result
visualize_mask(H_ice, H_ice*mask)





