from torch.utils.checkpoint import checkpoint
import torch
from typing import Dict, Tuple

def _mask(x: torch.Tensor, thresh: float) -> torch.Tensor:
    """
    mask observation/simulations using a sigmoid function
    """
    scale =0.1 # Steeper sigmoid using a scaling factor
    return torch.sigmoid(scale * (x - 1)).to(x.dtype)

def _eval_pair(H: torch.Tensor, obs: torch.Tensor, thickness_thresh: float = 0.5):
    """
    Compute masked residual diagnostics between sim and obs.
    thickness_thresh: meters; defines glacier presence for union-of-extents masking
    """
    # union of glacier extents (obs or sim), then intersect with base mask
    m = ((obs > thickness_thresh) | (H > thickness_thresh))

    resid = (H - obs) # meters
    mse = torch.mean(resid**2) if resid.numel() > 0 else torch.tensor(0.0, device=H.device)
    rmse = torch.sqrt(mse)
    mae = torch.sum(torch.abs(resid)) if resid.numel() > 0 else torch.tensor(11110.0, device=H.device)
    bias = torch.mean(resid) if resid.numel() > 0 else torch.tensor(0.0, device=H.device)
    # sample std (unbiased) of residuals; fall back to 0 if <2 pixels
    std = torch.std(resid, correction=1) if resid.numel() > 1 else torch.tensor(0.0, device=H.device)
    count = m.sum()

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "std": std,
        "area": count*0.01,
    }

def inversion_thicknes(
    precip_tensor, T_m_lowest, T_s, P_daily, T_daily, melt_factor,
    obs1880, obs26, obs57, obs80, obs99, obs09, obs17,
    glacier_model,
    reg_lambda: float = 0.001,
    thickness_thresh: float = 1.0,   # meters to define glacier presence
    w1880: float = 1.0,
    w26: float = 1.0, w57: float = 1.0, w80: float = 1.0, w99: float = 1.0, w09: float = 1.0, w17: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]:
    """
    Returns:
      H99: simulated thickness for 1999 (or last epoch you care about)
      loss: scalar tensor
      metrics: dict of per-epoch diagnostics computed **inside the glacier union mask**
               keys: '26','57','80','99' with mse, rmse, mae, bias, std, count
    Notes:
      - Loss uses RMSE (meters) for physical interpretability.
      - Data fidelity is computed only where (obs>thresh OR sim>thresh) & mask.
    """

    # ---- Forward simulation (you likely already have this inside your model) ----
    # Expect glacier_model to return thickness maps (meters) for target epochs
    H1880, H26, H57, H80, H99,H09, H17 = glacier_model(
        precip_tensor=precip_tensor,
        T_m_lowest=T_m_lowest,
        T_s=T_s,
        P_daily=P_daily,
        T_daily=T_daily,
        melt_factor=melt_factor
    )

    # ---- Per-epoch diagnostics (masked) ----
    m1880 = _eval_pair(H1880, obs1880, thickness_thresh)
    m26 = _eval_pair(H26, obs26, thickness_thresh)
    m57 = _eval_pair(H57, obs57, thickness_thresh)
    m80 = _eval_pair(H80, obs80, thickness_thresh)
    m99 = _eval_pair(H99, obs99, thickness_thresh)
    m09 = _eval_pair(H09, obs09, thickness_thresh)
    m17 = _eval_pair(H17, obs17, thickness_thresh)

    metrics = {
        "1880": m1880,  # each has rmse, mae, bias, std in meters
        "26": m26, 
        "57": m57,
        "80": m80,
        "99": m99,
        "09": m09,
        "17": m17
    }

    # ---- Smoothness regularization (simple 1D TV-like quadratic) ----
    smoothness_x = torch.sum((T_m_lowest[1:] - T_m_lowest[:-1])**2)

    # ---- Physically meaningful loss: weighted sum of RMSEs (meters) + regularization ----
    data_term = (
        w1880 * m1880["mse"] +
        w26 * m26["mse"] +
        w57 * m57["mse"] +
        w80 * m80["mse"] +
        w99 * m99["mse"] +
        w09 * m09["mse"] +
        w17 * m17["mse"]
    )

    data= data_term/(w1880+w26+w57+w80+w99+w09+w17)

    loss =  data+ reg_lambda * smoothness_x

    return [H1880,H26,H57,H80,H99,H09,H17], loss,data, metrics

def inversion_extent(
    precip_tensor, T_m_lowest, T_s, P_daily, T_daily, melt_factor,
    obs1880, obs26, obs57, obs80, obs99, obs09, obs17,
    glacier_model,
    reg_lambda: float = 0.001,
    thickness_thresh: float = 1.0,   # meters to define glacier presence
    w1880: float = 0.0,
    w26: float = 0.0, w57: float = 0.0, w80: float = 0.0, w99: float = 0.0, w09: float = 0.0, w17: float = 0.0
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]:
    """
    Extent-only inversion.
    Returns:
      extents: dict of simulated **binary** extents per epoch {'1880','26','57','80','99','17'} (0/1 tensors)
      loss: scalar tensor
      metrics: dict per epoch with IoU, Dice, Precision, Recall, Accuracy, BCE, RMSE(0/1), counts
    Notes:
      - Simulated maps are binarized with STE so gradients flow.
      - Loss is a weighted sum of (1 - Dice) with optional BCE mix, plus smoothness regularization.
    """
    # ---- Forward simulation (thickness maps; same signature as before) ----
    H1880, H26, H57, H80, H99, H09, H17 = glacier_model(
        precip_tensor=precip_tensor,
        T_m_lowest=T_m_lowest,
        T_s=T_s,
        P_daily=P_daily,
        T_daily=T_daily,
        melt_factor=melt_factor
    )

    # ---- Binarize observations (hard, no gradients) ----
    O1880 = _mask(obs1880, thickness_thresh)
    O26   = _mask(obs26,   thickness_thresh)
    O57   = _mask(obs57,   thickness_thresh)
    O80   = _mask(obs80,   thickness_thresh)
    O99   = _mask(obs99,   thickness_thresh)
    O09   = _mask(obs09,   thickness_thresh)
    O17   = _mask(obs17,   thickness_thresh)

    # ---- Binarize simulations with STE (keeps gradients alive) ----
    S1880 = _mask(H1880, thickness_thresh)
    S26   = _mask(H26,   thickness_thresh)
    S57   = _mask(H57,   thickness_thresh)
    S80   = _mask(H80,   thickness_thresh)
    S99   = _mask(H99,   thickness_thresh)
    S09   = _mask(H09,   thickness_thresh)
    S17   = _mask(H17,   thickness_thresh)

    # ---- Per-epoch diagnostics (masked) ----
    m1880 = _eval_pair(S1880, O1880)
    m26 = _eval_pair(S26, O26)
    m57 = _eval_pair(S57, O57 )
    m80 = _eval_pair(S80, O80)
    m99 = _eval_pair(S99, O99)
    m09 = _eval_pair(S09, O09)
    m17 = _eval_pair(S17, O17)

    metrics = {
        "1880": m1880,  
        "26": m26,
        "57": m57,
        "80": m80,
        "99": m99,
        "09": m09,
        "17": m17
    }
    # ---- Smoothness regularization (simple 1D TV-like quadratic) ----
    eps = 1e-8
    smoothness_x = torch.sum((T_m_lowest[1:] - T_m_lowest[:-1])**2)/ (T_m_lowest.pow(2).sum() + eps)

    # ---- Physically meaningful loss: weighted sum of RMSEs (meters) + regularization ----
    data_term = (
        w1880 * m1880["mse"] +
        w26 * m26["mse"] +
        w57 * m57["mse"] +
        w80 * m80["mse"] +
        w99 * m99["mse"] +
        w09 * m09["mse"] +
        w17 * m17["mse"]
    )
    data= data_term/(w1880+w26+w57+w80+w99+w09+w17)

    loss =  data+ reg_lambda * smoothness_x

    return [H1880,H26,H57,H80,H99,H09,H17], loss,data, metrics


def invert_field(
    smb_field: torch.Tensor,
    observation: torch.Tensor,
    glacier_model,
    reg_lambda: float = 0.01,
    thickness_thresh: float = 1.0,
    smooth_type: str = 'gradient',  # 'gradient' or 'laplacian'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Invert for SMB field given a single observation (2D ice thickness field).

    This function optimizes an SMB field so that when used in the glacier model,
    it produces ice thickness that matches the observation.

    Parameters:
    -----------
    smb_field : torch.Tensor
        SMB field to optimize (2D tensor [ny, nx])
        Should have requires_grad=True for optimization
    observation : torch.Tensor
        Observed ice thickness field (2D tensor [ny, nx])
    glacier_model : GlacierDynamicsCheckpointed
        Glacier model instance
    reg_lambda : float
        Regularization weight for spatial smoothness (default: 0.01)
    thickness_thresh : float
        Thickness threshold for glacier presence (default: 1.0 m)
    smooth_type : str
        Type of smoothness regularization:
        - 'gradient': penalize spatial gradients (total variation)
        - 'laplacian': penalize curvature (second derivatives)

    Returns:
    --------
    H_sim : torch.Tensor
        Simulated ice thickness field (2D tensor [ny, nx])
    loss : torch.Tensor
        Total loss (data term + regularization)
    data_term : torch.Tensor
        Data fidelity term only
    metrics : dict
        Dictionary with diagnostic metrics (mse, rmse, mae, bias, std, area)

    Notes:
    ------
    - The SMB field should be initialized before calling this function
    - Spatial smoothness regularization encourages neighboring SMB values to be similar
    - Use this in an optimization loop, updating smb_field with gradient descent

    Example:
    --------
    >>> # Initialize SMB field
    >>> smb_field = torch.zeros(ny, nx, device=device, requires_grad=True)
    >>> optimizer = torch.optim.Adam([smb_field], lr=0.01)
    >>>
    >>> # Optimization loop
    >>> for i in range(100):
    >>>     optimizer.zero_grad()
    >>>     H_sim, loss, data, metrics = invert_field(
    >>>         smb_field, observation, glacier_model, reg_lambda=0.01
    >>>     )
    >>>     loss.backward()
    >>>     optimizer.step()
    """

    # ---- Forward simulation using the SMB field ----
    H_sim = glacier_model(
        precip_tensor=None,
        T_m_lowest=None,
        T_s=None,
        P_daily=None,
        T_daily=None,
        melt_factor=None,
        smb_method='field',
        smb_field=smb_field
    )

    # ---- Compute data fidelity (masked misfit) ----
    metrics = _eval_pair(H_sim, observation, thickness_thresh)
    data_term = metrics["mse"]

    # ---- Spatial smoothness regularization ----
    if smooth_type == 'gradient':
        # Total variation-like: penalize gradients in x and y directions
        # This encourages spatially smooth SMB fields
        grad_x = smb_field[:, 1:] - smb_field[:, :-1]  # differences in x
        grad_y = smb_field[1:, :] - smb_field[:-1, :]  # differences in y
        smoothness = torch.sum(grad_x**2) + torch.sum(grad_y**2)

    elif smooth_type == 'laplacian':
        # Laplacian regularization: penalize curvature (second derivatives)
        # This encourages even smoother fields
        # Compute discrete Laplacian: ∇²u ≈ u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4*u[i,j]
        laplacian = (
            smb_field[:-2, 1:-1] +  # above
            smb_field[2:, 1:-1] +   # below
            smb_field[1:-1, :-2] +  # left
            smb_field[1:-1, 2:] -   # right
            4 * smb_field[1:-1, 1:-1]  # center
        )
        smoothness = torch.sum(laplacian**2)

    else:
        raise ValueError(f"smooth_type must be 'gradient' or 'laplacian', got '{smooth_type}'")

    # ---- Total loss ----
    loss = data_term + reg_lambda * smoothness

    return H_sim, loss, data_term, metrics

