
import torch
from config.scaling_factors import scaling_factors
from utils.emulator_tools import compute_divflux, compute_gradient, apply_boundary_condition,compute_divflux_limited

def checkpointed_emulator_step(H_ice, Z_surf, smb, time, Z_topo, dx, dy, dtmax,model, cfl=0.20):
    # Compute initial gradients of surface elevation (slopes)
    slopsurfx, slopsurfy = compute_gradient(Z_surf, dx, dy)
    # Scale the inputs with stored scaling factors
    H_ice_scaled = H_ice / scaling_factors["thk"]
    slopsurfx_scaled = slopsurfx / scaling_factors["slopsurfx"]
    slopsurfy_scaled = slopsurfy / scaling_factors["slopsurfy"]
    # Combine scaled inputs and add batch dimension
    input_data_scaled = torch.stack([H_ice_scaled, slopsurfx_scaled, slopsurfy_scaled], dim=-1).unsqueeze(0)
    # Use the trained model to predict ubar (x-velocity) and vbar (y-velocity)
    with torch.no_grad():
        ubar_vbar_pred = model(input_data_scaled.permute(0, 3, 1, 2))  # Change to (batch, channels, height, width)
        ubar = ubar_vbar_pred[0, 0, :, :] * scaling_factors["ubar"]  # x-component of velocity
        vbar = ubar_vbar_pred[0, 1, :, :] * scaling_factors["vbar"]  # y-component of velocity
    # Compute maximum velocity for CFL condition
    vel_max = max(ubar.abs().max().item(), vbar.abs().max().item())
    # Compute time step (CFL condition)
    dt = min(cfl * dx / vel_max, dtmax)
    # Update rule (diffusion): Compute the change in thickness (dH/dt)
    dHdt = -compute_divflux(ubar, vbar, H_ice, dx, dy)
    # Update ice thickness and ensure no negative values
    H_ice = H_ice.clone()
    H_ice += dt * dHdt
    H_ice = torch.clamp(H_ice, min=0)

    # Update rule (mass balance)
    H_ice = H_ice.clone()
    H_ice += dt * smb
    # Apply the boundary condition before the next iteration
    H_ice = apply_boundary_condition(H_ice)
    # Ensure ice thickness remains positive
    # H_ice = torch.relu(H_ice)


    # Update surface topography
    Z_surf = Z_topo + H_ice
    return H_ice, Z_surf, time + dt, ubar,vbar

