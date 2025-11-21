# SMB Inversion Parameter Sweep

Minimal parameter sweep for SMB inversion using Weights & Biases.

## Setup

1. Install wandb if not already installed:
```bash
pip install wandb
```

2. Login to wandb:
```bash
wandb login
```

## Running the Sweep

### Method 1: Initialize and run sweep

```bash
# Initialize the sweep (returns a sweep ID)
wandb sweep sweep_config.yaml

# Run the sweep agent (replace <sweep_id> with the ID from above)
wandb agent <your-entity>/<project-name>/<sweep_id>
```

### Method 2: Run directly with specific parameters

Edit the bottom of `invert_SMB.py` and uncomment the wandb.init() section:

```python
if __name__ == "__main__":
    wandb.init(project="smb-inversion", config={
        'learning_rate': 0.02,
        'reg_lambda': 1e-5,
        'max_iters': 100,
        'early_stop_patience': 10,
        'early_stop_threshold': 1e-5
    })
    train()
```

Then run:
```bash
python invert_SMB.py
```

## Output

Results are saved to `results/SMB/<run_name>/`:
- `loss_curve.png` - Loss evolution plot
- `smb_comparison.png` - 3-panel SMB comparison (inverted, true, residual)
- `thickness_comparison.png` - 3-panel thickness comparison (simulated, observed, residual)
- `smb_inverted.pt` - Final optimized SMB field

All metrics and plots are also logged to Weights & Biases dashboard.

## Sweep Parameters

The sweep tests all combinations of:
- **Learning rates**: [0.001, 0.005, 0.01, 0.02, 0.05]
- **Regularization**: [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

Total: 25 runs

Edit `sweep_config.yaml` to modify parameter ranges.
