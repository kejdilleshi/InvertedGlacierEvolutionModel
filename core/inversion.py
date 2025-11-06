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

# # ----------------------------------------------------------------------------------- delete below -----------------------------------------------------------------------------------
# import torch
# import torch.nn.functional as F
# from typing import Tuple

# # ---------- Safety helpers ----------
# def _ensure_tensor(x, like: torch.Tensor, name: str) -> torch.Tensor:
#     if x is None:
#         raise ValueError(f"{name} is None (glacier_model likely returned None).")
#     if not isinstance(x, torch.Tensor):
#         x = torch.as_tensor(x, device=like.device, dtype=like.dtype)
#     else:
#         x = x.to(device=like.device, dtype=like.dtype)
#     if not torch.isfinite(x).all():
#         raise ValueError(f"{name} contains inf/nan.")
#     return x

# # Central-diff grads (same as before), but we’ll compute grads on `x` directly.
# def _spatial_grads(x: torch.Tensor, spacing=(1.0,1.0)):
#     if x.ndim == 2:  # (H,W)
#         dy, dx = spacing if len(spacing)==2 else spacing[-2:]
#         x4 = x[None,None]
#         kx = torch.tensor([[[[-0.5,0.0,0.5]]]], dtype=x.dtype, device=x.device)/dx
#         ky = torch.tensor([[[[-0.5],[0.0],[0.5]]]], dtype=x.dtype, device=x.device)/dy
#         gx = F.conv2d(F.pad(x4,(1,1,0,0),mode='replicate'), kx)[0,0]
#         gy = F.conv2d(F.pad(x4,(0,0,1,1),mode='replicate'), ky)[0,0]
#         return gy, gx  # (d/dy, d/dx)
#     elif x.ndim == 3:  # (D,H,W)
#         dz, dy, dx = spacing if len(spacing)==3 else (spacing[0], spacing[0], spacing[1])
#         x5 = x[None,None]
#         kx = torch.zeros((1,1,1,1,3), dtype=x.dtype, device=x.device); kx[0,0,0,0,:] = torch.tensor([-0.5,0,0.5], dtype=x.dtype, device=x.device)/dx
#         ky = torch.zeros((1,1,1,3,1), dtype=x.dtype, device=x.device); ky[0,0,0,:,0] = torch.tensor([-0.5,0,0.5], dtype=x.dtype, device=x.device)/dy
#         kz = torch.zeros((1,1,3,1,1), dtype=x.dtype, device=x.device); kz[0,0,:,0,0] = torch.tensor([-0.5,0,0.5], dtype=x.dtype, device=x.device)/dz
#         gx = F.conv3d(F.pad(x5,(1,1,0,0,0,0),mode='replicate'), kx)[0,0]
#         gy = F.conv3d(F.pad(x5,(0,0,1,1,0,0),mode='replicate'), ky)[0,0]
#         gz = F.conv3d(F.pad(x5,(0,0,0,0,1,1),mode='replicate'), kz)[0,0]
#         return gz, gy, gx
#     else:
#         raise ValueError("x must be 2-D or 3-D.")

# def _mask(x: torch.Tensor, thresh: float, scale: float = 20.0) -> torch.Tensor:
#     # Slightly gentler slope than 100 to avoid super-steep logits early on.
#     return torch.sigmoid(scale * (x - thresh)).to(x.dtype)

# # ---------- Stable pseudo-SDF ----------
# def _phi_from_field_stable(x: torch.Tensor, thresh: float,
#                            spacing=(1.0,1.0), g_floor: float = 1e-2,
#                            band_m: float = 100.0) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     φ ≈ (x - thresh) / ||∇x||  with gradient floor, then soft-clip φ to ±band_m via tanh.
#     Everything is differentiable.
#     g_floor: minimum gradient norm in units of (thickness per meter).
#     band_m : meters; soft clipping range for φ to prevent huge values away from the front.
#     """
#     p = _mask(x, thresh)  # soft occupancy
#     grads = _spatial_grads(x, spacing=spacing)
#     g2 = sum(g*g for g in grads)
#     gnorm = torch.sqrt(g2 + torch.finfo(x.dtype).eps)
#     gnorm = torch.clamp(gnorm, min=g_floor)
#     phi = (x - thresh) / gnorm  # meters
#     # soft clip to keep diffs bounded but preserve gradients
#     phi = band_m * torch.tanh(phi / band_m)
#     return p, phi

# def sdf_band_huber(phi_s: torch.Tensor, phi_o: torch.Tensor,
#                    band_m: float = 50.0, tau: float = 5.0, beta: float = 1.0):
#     # band weights around both level-sets; uses soft φ already
#     w = torch.sigmoid((band_m - phi_s.abs())/(tau+1e-12)) + torch.sigmoid((band_m - phi_o.abs())/(tau+1e-12))
#     w = torch.clamp(w, 0.0, 1.0)
#     diff = phi_s - phi_o
#     l = F.smooth_l1_loss(diff, torch.zeros_like(diff), beta=beta, reduction='none')
#     return (w * l).sum() / (w.sum() + 1e-12)
# def inversion_extent(
#     precip_tensor, T_m_lowest, T_s, P_daily, T_daily, melt_factor,
#     obs1880, obs26, obs57, obs80, obs99, obs09, obs17,
#     glacier_model,
#     reg_lambda: float = 1e-3,
#     thickness_thresh: float = 1.0,
#     w1880: float = 0.0,
#     w26: float = 0.0, w57: float = 0.0, w80: float = 0.0, w99: float = 0.0, w09: float = 0.0, w17: float = 0.0,
#     sdf_band_m: float = 50.0,
#     sdf_tau: float = 5.0,
#     sdf_beta: float = 1.0,
#     spacing=(1.0,1.0),
#     g_floor: float = 1e-2,          # NEW: gradient floor
#     phi_soft_band: float = 100.0    # NEW: soft φ clip range
# ):
#     # Forward
#     H1880, H26, H57, H80, H99, H09, H17 = glacier_model( precip_tensor=precip_tensor,
#         T_m_lowest=T_m_lowest,
#         T_s=T_s,
#         P_daily=P_daily,
#         T_daily=T_daily,
#         melt_factor=melt_factor)

#     # Fail fast if model returned None or NaNs
#     ref = precip_tensor if isinstance(precip_tensor, torch.Tensor) else T_m_lowest
#     H1880 = _ensure_tensor(H1880, ref, "H1880")
#     H26   = _ensure_tensor(H26,   ref, "H26")
#     H57   = _ensure_tensor(H57,   ref, "H57")
#     H80   = _ensure_tensor(H80,   ref, "H80")
#     H99   = _ensure_tensor(H99,   ref, "H99")
#     H09   = _ensure_tensor(H09,   ref, "H09")
#     H17   = _ensure_tensor(H17,   ref, "H17")

#     # Pseudo-SDFs (stable)
#     S1880, PhiS1880 = _phi_from_field_stable(H1880, thickness_thresh, spacing, g_floor, phi_soft_band)
#     S26,   PhiS26   = _phi_from_field_stable(H26,   thickness_thresh, spacing, g_floor, phi_soft_band)
#     S57,   PhiS57   = _phi_from_field_stable(H57,   thickness_thresh, spacing, g_floor, phi_soft_band)
#     S80,   PhiS80   = _phi_from_field_stable(H80,   thickness_thresh, spacing, g_floor, phi_soft_band)
#     S99,   PhiS99   = _phi_from_field_stable(H99,   thickness_thresh, spacing, g_floor, phi_soft_band)
#     S09,   PhiS09   = _phi_from_field_stable(H09,   thickness_thresh, spacing, g_floor, phi_soft_band)
#     S17,   PhiS17   = _phi_from_field_stable(H17,   thickness_thresh, spacing, g_floor, phi_soft_band)

#     O1880, PhiO1880 = _phi_from_field_stable(obs1880, thickness_thresh, spacing, g_floor, phi_soft_band)
#     O26,   PhiO26   = _phi_from_field_stable(obs26,   thickness_thresh, spacing, g_floor, phi_soft_band)
#     O57,   PhiO57   = _phi_from_field_stable(obs57,   thickness_thresh, spacing, g_floor, phi_soft_band)
#     O80,   PhiO80   = _phi_from_field_stable(obs80,   thickness_thresh, spacing, g_floor, phi_soft_band)
#     O99,   PhiO99   = _phi_from_field_stable(obs99,   thickness_thresh, spacing, g_floor, phi_soft_band)
#     O09,   PhiO09   = _phi_from_field_stable(obs09,   thickness_thresh, spacing, g_floor, phi_soft_band)
#     O17,   PhiO17   = _phi_from_field_stable(obs17,   thickness_thresh, spacing, g_floor, phi_soft_band)

#     # SDF-band loss (robust)
#     L1880 = sdf_band_huber(PhiS1880, PhiO1880, band_m=sdf_band_m, tau=sdf_tau, beta=sdf_beta)
#     L26   = sdf_band_huber(PhiS26,   PhiO26,   band_m=sdf_band_m, tau=sdf_tau, beta=sdf_beta)
#     L57   = sdf_band_huber(PhiS57,   PhiO57,   band_m=sdf_band_m, tau=sdf_tau, beta=sdf_beta)
#     L80   = sdf_band_huber(PhiS80,   PhiO80,   band_m=sdf_band_m, tau=sdf_tau, beta=sdf_beta)
#     L99   = sdf_band_huber(PhiS99,   PhiO99,   band_m=sdf_band_m, tau=sdf_tau, beta=sdf_beta)
#     L09   = sdf_band_huber(PhiS09,   PhiO09,   band_m=sdf_band_m, tau=sdf_tau, beta=sdf_beta)
#     L17   = sdf_band_huber(PhiS17,   PhiO17,   band_m=sdf_band_m, tau=sdf_tau, beta=sdf_beta)

#     weights = torch.tensor([w1880,w26,w57,w80,w99,w09,w17], device=H80.device, dtype=H80.dtype)
#     losses  = torch.stack([L1880,L26,L57,L80,L99,L09,L17])
#     data_term = (weights * losses).sum() / torch.clamp(weights.sum(), min=1e-12)

#     smoothness_x = torch.sum((T_m_lowest[1:] - T_m_lowest[:-1])**2)
#     loss = data_term + reg_lambda * smoothness_x

#     return [H1880,H26,H57,H80,H99,H09,H17], loss,data_term, {
#         "1880":{"sdf_band":L1880}, "26":{"sdf_band":L26}, "57":{"sdf_band":L57},
#         "80":{"sdf_band":L80}, "99":{"sdf_band":L99}, "09":{"sdf_band":L09}, "17":{"sdf_band":L17}
#     }
