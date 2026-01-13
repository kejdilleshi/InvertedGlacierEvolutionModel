# SMB Profile Inversion - Documentation

## Overview

This implementation adds a new SMB forcing method called **'profile'** that allows you to:
1. Define a 1D elevation-based SMB profile `smb(z)`
2. Automatically map it to 2D topography `smb(x,y)` using elevation
3. Invert for the SMB profile from glacier thickness observations

The implementation is **fully differentiable** and suitable for gradient-based optimization with automatic differentiation (AD).

---

## Key Features

✅ **AD-Friendly**: All operations use PyTorch tensors and differentiable operations
✅ **Piecewise-Linear Interpolation**: Uses triangular hat basis functions for smooth mapping
✅ **Time-Varying Support**: Supports both constant (1D) and time-varying (2D) SMB profiles
✅ **Regularization**: Includes smoothness regularization for physically plausible profiles
✅ **Easy Integration**: Works seamlessly with existing glacier dynamics model

---

## Files Modified/Created

### Core Files Modified

1. **[core/smb.py](core/smb.py)** (lines 148-213)
   - Added `update_smb_profile()` function
   - Maps 1D SMB profile to 2D topography using piecewise-linear interpolation

2. **[core/glacier.py](core/glacier.py)**
   - Added 'profile' to valid SMB methods (line 86)
   - Added parameters: `smb_vec`, `z_min`, `dz` to forward methods
   - Added validation and handling for profile method (lines 119-140, 153-159, 184-188)

3. **[core/inversion.py](core/inversion.py)** (lines 300-400)
   - Added `invert_profile()` function
   - Includes smoothness regularization (second derivative penalty)
   - Returns loss, data term, and evaluation metrics

### New Files Created

4. **[invert_SMB_profile.py](invert_SMB_profile.py)**
   - Complete inversion script with WandB logging
   - Generates synthetic observations
   - Runs optimization loop
   - Saves plots and results

5. **[IGEM_2D_profile_inversion.ipynb](IGEM_2D_profile_inversion.ipynb)**
   - Interactive Jupyter notebook version
   - Step-by-step workflow
   - Visualization of results

6. **[example_profile_smb.py](example_profile_smb.py)**
   - Example usage of the profile SMB method
   - Helper functions for creating profiles

---

## Mathematical Details

### SMB Profile Mapping

The SMB profile is defined at discrete elevations:
```
z[k] = z_min + k * dz,  k = 0, 1, ..., N-1
smb_vec[k] = SMB value at elevation z[k]
```

The 2D SMB field is computed using **piecewise-linear (triangular hat basis) interpolation**:

```python
def update_smb_profile(Z_topo, smb_vec, z_min, dz):
    # Continuous index: which elevation bin?
    t = (Z_topo - z_min) / dz

    # Triangular weights (hat functions)
    k = torch.arange(N)
    w = torch.relu(1.0 - torch.abs(t[..., None] - k))

    # Weighted sum
    smb = torch.sum(w * smb_vec, dim=-1)
    return smb
```

**Key properties:**
- Each point in `Z_topo` gets contributions from at most 2 adjacent bins
- Weights sum to 1 (partition of unity)
- Continuous and piecewise-linear
- Fully differentiable w.r.t. `smb_vec`

### Inversion Loss Function

The inversion optimizes:
```
L = MSE(H_sim, H_obs) + λ * R(smb_vec)
```

Where:
- **Data term**: `MSE(H_sim, H_obs)` = Mean squared error between simulated and observed thickness
- **Regularization**: `R(smb_vec)` = Sum of squared second derivatives (curvature penalty)

```python
second_deriv = smb_vec[:-2] - 2*smb_vec[1:-1] + smb_vec[2:]
R = torch.sum(second_deriv**2)
```

This encourages smooth, physically plausible SMB profiles.

---

## Usage

### Basic Usage (Forward Model)

```python
import torch
from core.glacier import GlacierDynamicsCheckpointed

# Create SMB profile
zmin = topo.min()
zmax = topo.max()
dz = 100.0  # 100m bins

N = int(torch.ceil((zmax - zmin) / dz).item())
smb_vec = torch.linspace(-15.0, 5.0, steps=N, device=device)

# Optional: manually adjust specific elevations
smb_vec = torch.clamp(smb_vec, min=-15.0, max=3.0)
smb_vec[18] = 0.0  # Set ELA

# Run glacier model
H_final = glacier_model.forward(
    precip_tensor=None,
    T_m_lowest=None,
    T_s=None,
    melt_factor=None,
    smb_method='profile',
    smb_vec=smb_vec,  # 1D profile [N]
    z_min=zmin,
    dz=dz
)
```

### Inversion Usage

```python
from core.inversion import invert_profile

# Initialize SMB profile for optimization
smb_vec_init = torch.linspace(-15.0, 5.0, steps=N, device=device)
smb_vec_init.requires_grad = True

# Optimizer
optimizer = torch.optim.Adam([smb_vec_init], lr=0.1)

# Optimization loop
for i in range(max_iters):
    optimizer.zero_grad()

    H_sim, loss, data_term, metrics = invert_profile(
        smb_vec=smb_vec_init,
        z_min=zmin,
        dz=dz,
        observation=H_obs,
        glacier_model=glacier_model,
        reg_lambda=0.01  # Regularization weight
    )

    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item():.6f}, RMSE: {metrics['rmse'].item():.4f}")
```

---

## Example Workflow

### 1. Generate Synthetic Observations

```python
# Define "true" SMB profile
smb_vec_true = torch.linspace(-10.0, 5.0, steps=N, device=device)
smb_vec_true = torch.clamp(smb_vec_true, min=-15.0, max=3.0)

# Generate observations
with torch.no_grad():
    H_obs = glacier_model(
        precip_tensor=None, T_m_lowest=None, T_s=None, melt_factor=None,
        smb_method='profile',
        smb_vec=smb_vec_true,
        z_min=zmin, dz=dz
    )
```

### 2. Initialize from Different Guess

```python
# Start from linear profile
smb_vec_init = torch.linspace(-15.0, 5.0, steps=N, device=device)
smb_vec_init.requires_grad = True
```

### 3. Run Inversion

```python
optimizer = torch.optim.Adam([smb_vec_init], lr=0.1)

for i in range(500):
    optimizer.zero_grad()
    H_sim, loss, data_term, metrics = invert_profile(
        smb_vec=smb_vec_init, z_min=zmin, dz=dz,
        observation=H_obs, glacier_model=glacier_model,
        reg_lambda=0.01
    )
    loss.backward()
    optimizer.step()
```

### 4. Analyze Results

```python
# Compute profile errors
residual = smb_vec_init - smb_vec_true
smb_rmse = torch.sqrt(torch.mean(residual**2))
smb_mae = torch.mean(torch.abs(residual))

print(f"SMB RMSE: {smb_rmse.item():.4f} m/yr")
print(f"Thickness RMSE: {metrics['rmse'].item():.4f} m")
```

---

## Running the Inversion

### Option 1: Python Script

```bash
python invert_SMB_profile.py
```

This will:
- Generate synthetic observations from a true profile
- Invert starting from a different initial guess
- Save plots and results to `results/SMB_profile/`
- Log metrics to WandB (optional)

### Option 2: Jupyter Notebook

Open and run `IGEM_2D_profile_inversion.ipynb` for an interactive workflow.

---

## Parameters

### SMB Profile Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `smb_vec` | `torch.Tensor [N]` or `[T, N]` | SMB values at discrete elevations. 1D for constant, 2D for time-varying |
| `z_min` | `float` | Minimum elevation (m) corresponding to `smb_vec[0]` |
| `dz` | `float` | Elevation spacing (m) between consecutive values |

### Inversion Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reg_lambda` | `float` | `0.0` | Regularization weight for profile smoothness |
| `learning_rate` | `float` | `0.1` | Adam optimizer learning rate |
| `max_iters` | `int` | `500` | Maximum optimization iterations |
| `thickness_thresh` | `float` | `1.0` | Threshold (m) for defining glacier presence |

---

## Validation: AD-Friendly Implementation

All operations in `update_smb_profile()` are differentiable:

```python
# ✅ All these operations support autograd:
t = (Z_topo - z_min) / dz                    # Division, subtraction
k = torch.arange(N, device=device, dtype=dtype)  # No gradients needed
w = torch.relu(1.0 - torch.abs(t[..., None] - k))  # ReLU, abs (subgradient)
smb = torch.sum(w * smb_vec, dim=-1)         # Multiplication, sum
```

**Test AD compatibility:**
```python
smb_vec = torch.randn(N, requires_grad=True)
smb_2d = update_smb_profile(topo, smb_vec, z_min, dz)
loss = smb_2d.sum()
loss.backward()
assert smb_vec.grad is not None  # ✅ Gradients flow correctly
```

---

## Tips and Best Practices

### 1. Choosing Elevation Bin Spacing (`dz`)

- **Fine resolution** (dz = 50-100m): Better captures local SMB variations
- **Coarse resolution** (dz = 200-500m): Smoother profiles, fewer parameters

### 2. Regularization Weight (`reg_lambda`)

- Start with `reg_lambda = 0.01` for smooth profiles
- Increase if inverted profile is too noisy
- Decrease if you expect sharp SMB gradients (e.g., near ELA)

### 3. Learning Rate

- `lr = 0.1` works well for most cases
- Reduce if optimization is unstable
- Use learning rate scheduling for better convergence:
  ```python
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20)
  ```

### 4. Initialization

- Linear profiles (`torch.linspace`) are good starting points
- Can also initialize from physical models (e.g., degree-day balance)
- Multiple random initializations can help avoid local minima

### 5. Time-Varying Profiles

For transient SMB:
```python
# 2D profile: [time_steps, N_elevations]
n_timesteps = 10
smb_vec_varying = torch.zeros(n_timesteps, N, requires_grad=True)

# Initialize each timestep
for t in range(n_timesteps):
    smb_vec_varying[t] = torch.linspace(-15.0, 5.0, steps=N)
```

---

## Comparison with Other SMB Methods

| Method | Parameters | Use Case | Differentiable |
|--------|------------|----------|----------------|
| **PDD** | T, precip, melt factor | Physics-based, climate forcing | ✅ Yes |
| **ELA** | ELA, gradient, b_max | Simple linear model | ✅ Yes |
| **field** | Full 2D SMB field | Maximum flexibility, many params | ✅ Yes |
| **profile** ⭐ | 1D elevation profile | Physical interpretation, fewer params | ✅ Yes |

**Advantages of 'profile' method:**
- **Fewer parameters**: N bins vs (nx × ny) for field method
- **Physical interpretation**: SMB varies with elevation
- **Smoothness**: Natural along-elevation smoothness
- **Easy to constrain**: Can set specific elevations (e.g., ELA = 0)

---

## Troubleshooting

### Issue: Inversion not converging

**Solutions:**
1. Reduce learning rate (try 0.05 or 0.01)
2. Increase regularization (`reg_lambda = 0.1`)
3. Check observation data quality
4. Try different initialization

### Issue: Inverted profile is noisy

**Solutions:**
1. Increase regularization weight
2. Use coarser elevation bins (larger `dz`)
3. Add additional constraints (e.g., clamp values)

### Issue: Gradients are NaN

**Solutions:**
1. Check for division by zero in `dz`
2. Ensure `smb_vec` has `requires_grad=True`
3. Use gradient clipping:
   ```python
   torch.nn.utils.clip_grad_norm_([smb_vec], max_norm=1.0)
   ```

---

## Performance Considerations

### Memory Usage

- **Profile method**: ~N parameters (typically 10-50)
- **Field method**: ~(nx × ny) parameters (typically 10,000-100,000)

**Profile method uses 100-1000× fewer parameters!**

### Computational Cost

Forward pass with profile method:
1. Interpolation: O(N × nx × ny) per timestep
2. Glacier dynamics: O(nx × ny) per timestep

The interpolation overhead is negligible compared to glacier dynamics.

---

## References and Citation

This implementation uses:
- **Piecewise-linear interpolation** with triangular hat basis functions
- **Automatic differentiation** via PyTorch autograd
- **Tikhonov regularization** (second derivative penalty)

If you use this code, please cite the relevant glacier model and this implementation.

---

## Future Extensions

Possible improvements:
1. **Multi-resolution profiles**: Coarse-to-fine optimization
2. **Temporal regularization**: Smooth evolution for time-varying profiles
3. **Physical constraints**: Enforce mass conservation, realistic gradients
4. **Uncertainty quantification**: Bayesian inversion, ensemble methods
5. **Multi-objective optimization**: Balance thickness, extent, velocities

---

## Contact and Support

For questions or issues:
1. Check this README
2. Review example notebooks
3. Inspect error messages and gradients
4. Open a GitHub issue (if applicable)

Good luck with your SMB inversions! 🏔️❄️
