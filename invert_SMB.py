#!/usr/bin/env python3
"""
SMB Inversion with Weights & Biases Sweep
==========================================
Minimal script for sweeping regularization and learning rate values.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from pathlib import Path

from core.glacier import GlacierDynamicsCheckpointed
from core.inversion import invert_field
from core.cnn_model import CNN
from data.loader import load_geology 
from config.read_config import parse_arguments
from visualization.plots import  visualize_velocities,plot_loss_components


def plot_3panel_combined(smb_inverted, smb_true, H_sim, H_obs, icemask, save_path):
    """
    Create 3-panel plot: SMB inverted, delta SMB, delta H.
    With fixed colorbars and white-centered colormap for residuals.
    """
    # Convert to numpy
    smb_inv_cpu = smb_inverted.detach().cpu().numpy()
    smb_true_cpu = smb_true.detach().cpu().numpy()
    H_sim_cpu = H_sim.detach().cpu().numpy() * icemask.cpu().numpy()
    H_obs_cpu = H_obs.detach().cpu().numpy()

    # Calculate differences
    delta_smb = smb_inv_cpu - smb_true_cpu
    delta_H = H_sim_cpu - H_obs_cpu

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: SMB inverted with fixed colorbar [-21, 2]
    im0 = axes[0].imshow(smb_inv_cpu, cmap='viridis', vmin=-21, vmax=2, aspect='auto')
    axes[0].invert_yaxis()
    axes[0].set_title('SMB Inverted')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label='m w.e./yr')

    # Panel 2: delta SMB with fixed colorbar [-0.5, 0.5] and white at 0
    im1 = axes[1].imshow(delta_smb, cmap='RdBu_r', vmin=-0.4, vmax=0.4, aspect='auto')
    axes[1].invert_yaxis()
    axes[1].set_title('Delta SMB')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label='m w.e./yr')

    # Panel 3: delta H with fixed colorbar [-80, 80] and white at 0
    im2 = axes[2].imshow(delta_H, cmap='RdBu_r', vmin=-30, vmax=30, aspect='auto')
    axes[2].invert_yaxis()
    axes[2].set_title('Delta H')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label='m')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig


def plot_3panel_smb(smb_inverted, smb_true, icemask, save_path):
    """Create 3-panel SMB comparison plot as in notebook."""
    smb_cpu = smb_inverted.detach().cpu().numpy()
    smb_precomputed_cpu = smb_true.detach().cpu().numpy()
    residual_cpu = smb_cpu - smb_precomputed_cpu

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    # Shared color limits for SMB panels
    vmin = min(smb_cpu.min(), smb_precomputed_cpu.min())
    vmax = max(smb_cpu.max(), smb_precomputed_cpu.max())

    im0 = axes[0].imshow(smb_cpu, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
    axes[0].invert_yaxis()
    axes[0].set_title('SMB_inverted')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(smb_precomputed_cpu, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
    axes[1].invert_yaxis()
    axes[1].set_title('True SMB')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Residual: use diverging colormap centered at 0
    lim = np.max(np.abs(residual_cpu))*0.5
    im2 = axes[2].imshow(residual_cpu, cmap='RdBu', vmin=-lim, vmax=lim, aspect='auto')
    axes[2].invert_yaxis()
    axes[2].set_title('Residual')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig


def plot_3panel_thickness(H_sim, observation, icemask, save_path):
    """Create 3-panel thickness comparison plot as in notebook."""
    smb_cpu = H_sim.detach().cpu().numpy() * icemask.cpu().numpy()
    smb_precomputed_cpu = observation.detach().cpu().numpy()
    residual_cpu = smb_cpu - smb_precomputed_cpu

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    # Shared color limits
    vmin = min(smb_cpu.min(), smb_precomputed_cpu.min())
    vmax = max(smb_cpu.max(), smb_precomputed_cpu.max())

    im0 = axes[0].imshow(smb_cpu, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
    axes[0].invert_yaxis()
    axes[0].set_title('Simulated Glacier Thickness')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(smb_precomputed_cpu, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
    axes[1].invert_yaxis()
    axes[1].set_title('Observed Glacier Thickness')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Residual: use diverging colormap centered at 0
    lim = np.max(np.abs(residual_cpu))*0.9
    im2 = axes[2].imshow(residual_cpu, cmap='RdBu', vmin=-lim, vmax=lim, aspect='auto')
    axes[2].invert_yaxis()
    axes[2].set_title('Residual')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig

def train(run_name=None, run_config=None):
    """Main training function for wandb sweep."""

    # Initialize wandb
    run = wandb.init()

    # Use provided config or fall back to wandb.config
    if run_config is not None:
        config = run_config
    else:
        config = wandb.config

    # Use provided run_name or fall back to wandb run name
    if run_name is not None:
        actual_run_name = run_name
    else:
        actual_run_name = run.name

    # Device
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # Create output directory
    output_dir = Path('results/SMB') / actual_run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and data
    args = parse_arguments([])
    args.ttot = torch.tensor(config.get('ttot', 2010.0))
    args.t_start = torch.tensor(2000.0)
    args.outdir = output_dir

    # Load CNN emulator
    cnn_config = {
        "nb_layers": 8,
        "nb_out_filter": 64,
        "conv_ker_size": 5,
        "activation": "lrelu",
        "dropout_rate": 0.1,
    }
    state = torch.load('data/emulator_model.pth', map_location=device, weights_only=False)
    model = CNN(3, 2, cnn_config).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Load geology
    Z_topo, H_init, icemask = load_geology("data/geology.nc")

    # Create glacier model
    glacier_model = GlacierDynamicsCheckpointed(
        Z_topo, H_init, ice_mask=icemask, device=device,
        args=args, model=model, visualize=False
    )

    # Load data
    # smb_initial = torch.load('data/smb_initial.pt', map_location=device)
    smb_true = torch.load('data/smb_final.pt', map_location=device)

    H = glacier_model(
                precip_tensor=None,
                T_m_lowest=None,
                T_s=None,
                P_daily=None,
                T_daily=None,
                melt_factor=None,
                smb_method='field',
                smb_field=smb_true
            )
    torch.save(H,'data/H_observation.pt')
    observation = torch.load('data/H_observation.pt', map_location=device)

    # Initialize SMB for optimization
    SMB_inverted = torch.full_like(smb_true, 0.0, device=device)
    SMB_inverted.requires_grad = True


    # Optimizer
    optimizer = torch.optim.Adam([SMB_inverted], lr=config.get('learning_rate', 0.01))

    # Training
    loss_hist, data_hist = [], []
    best_loss = float('inf')
    no_improve_count = 0

    for i in range(config.get('max_iters', 10000)):
        optimizer.zero_grad()

        H_sim, loss, data_term, metrics = invert_field(
            smb_field=SMB_inverted,
            observation=observation,
            glacier_model=glacier_model,
            reg_lambda=config.get('reg_lambda', 0.0)
        )

        loss.backward()
        optimizer.step()

        loss_hist.append(loss.item())
        data_hist.append(data_term.item())

        # Log to wandb
        wandb.log({
            'loss': loss.item(),
            'data_term': data_term.item(),
            'rmse': metrics['rmse'].item(),
            'mae': metrics['mae'].item(),
            'bias': metrics['bias'].item()})

        # Early stopping
        if i > 40:
            if best_loss - loss.item() > config.get('early_stop_threshold', 0.00001):
                best_loss = loss.item()
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= config.get('early_stop_patience', 50):
                print(f"Early stopping at iteration {i+1}")
                break
            if loss.item() <= 0.00001:
                print(f"Loss threshold reached at iteration {i+1}")
                break
        if (i+1) % 50 == 0:
            print(f"Iteration {i+1}, Loss: {loss.item():.6f}, RMSE: {metrics['rmse'].item():.4f}, MAE: {metrics['mae'].item():.4f}")
            plot_3panel_combined(
                SMB_inverted, smb_true, H_sim, observation, icemask,
                output_dir / f'combined_iter_{i+1}.png'
            )
            torch.save(SMB_inverted, output_dir / f'smb_inverted_iter_{i+1}.pt')
    # Save results
    SMB_inverted = torch.where(icemask > 0.5,
    SMB_inverted,    torch.tensor(-10.0, dtype=SMB_inverted.dtype, device=SMB_inverted.device))

    torch.save(SMB_inverted, output_dir / 'smb_inverted.pt')

    # Create and save plots
    loss_fig = plot_loss_components(loss_hist, data_hist, args)
    smb_fig = plot_3panel_smb(SMB_inverted, smb_true, icemask, output_dir / 'smb_comparison.png')
    thickness_fig = plot_3panel_thickness(H_sim, observation, icemask, output_dir / 'thickness_comparison.png')

    # Log figures to wandb
    wandb.log({
        'loss_curve': wandb.Image(str(output_dir / 'loss_curve.png')),
        'smb_comparison': wandb.Image(str(output_dir / 'smb_comparison.png')),
        'thickness_comparison': wandb.Image(str(output_dir / 'thickness_comparison.png')),
        'final_loss': loss_hist[-1],
        'final_rmse': metrics['rmse'].item(),
        'final_mae': metrics['mae'].item(),
        'converged_iter': i+1
    })

    print(f"Results saved to {output_dir}")
    wandb.finish()


if __name__ == "__main__":
    # For running sweep, use: wandb agent <sweep_id>
    # For single run without wandb:
    os.environ["WANDB_MODE"] = "disabled"  # Disable wandb completely

    # Configuration for single run
    run_name = "rose-sweep-1"
    run_config = {
        'ttot': 2010.0,
        'learning_rate': 0.01,
        'reg_lambda': 0.0,
        'max_iters': 451,
        'early_stop_patience': 250,
        'early_stop_threshold': 0.00001
    }

    wandb.init(name=run_name, project="smb-inversion", config=run_config)
    train(run_name=run_name, run_config=run_config)
