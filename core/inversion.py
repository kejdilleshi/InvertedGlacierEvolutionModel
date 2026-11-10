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

    resid = (H - obs)[m]# meters
    mse = torch.mean(resid**2) if resid.numel() > 0 else torch.tensor(0.0, device=H.device)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(resid)) if resid.numel() > 0 else torch.tensor(11110.0, device=H.device)
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
    save_times: list = None,
    reg_lambda: float = 0.0,
    thickness_thresh: float = 1.0,
    smooth_type: str = 'gradient',  # 'gradient' or 'laplacian'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Invert for SMB field given ice thickness observations.

    Parameters
    ----------
    smb_field : torch.Tensor
        SMB field to optimize (2D or 3D)
    observation : torch.Tensor
        Observed ice thickness at final time
    glacier_model : GlacierDynamicsCheckpointed
        Glacier dynamics model
    save_times : list, optional
        Times at which to save H (if None, only returns final H)
    reg_lambda : float
        Regularization weight for spatial smoothness
    thickness_thresh : float
        Threshold for glacier presence in metrics
    smooth_type : str
        Type of smoothness regularization ('gradient' or 'laplacian')

    Returns
    -------
    H_sim : torch.Tensor
        Simulated ice thickness at final time
    loss : torch.Tensor
        Total loss (data + regularization)
    data_term : torch.Tensor
        Data fidelity term only
    metrics : dict
        Evaluation metrics (mse, rmse, mae, bias, std, area)
    """

    # ---- Forward simulation ----
    if save_times is not None:
        saved_H_dict, H_sim = glacier_model(
            precip_tensor=None,
            T_m_lowest=None,
            T_s=None,
            melt_factor=None,
            smb_method='field',
            smb_field=smb_field,
            save_times=save_times,
        )
    else:
        H_sim = glacier_model(
            precip_tensor=None,
            T_m_lowest=None,
            T_s=None,
            melt_factor=None,
            smb_method='field',
            smb_field=smb_field,
        )

    # ---- Compute data fidelity (masked misfit) ----
    metrics = _eval_pair(H_sim, observation, thickness_thresh)
    data_term = metrics["mse"]

    # ---- Spatial smoothness regularization ----
    if reg_lambda > 0:
        # For 3D SMB fields [time, ny, nx], apply smoothness only to the last slice (the one being inverted)
        # For 2D SMB fields [ny, nx], apply smoothness to the entire field
        if smb_field.ndim == 3:
            # 3D case: use last time slice
            smb_slice = smb_field[-1, :, :]
        elif smb_field.ndim == 2:
            # 2D case: use entire field
            smb_slice = smb_field
        else:
            raise ValueError(f"smb_field must be 2D or 3D, got shape {smb_field.shape}")

        if smooth_type == 'gradient':
            # Total variation-like: penalize gradients in x and y directions
            # This encourages spatially smooth SMB fields
            grad_x = smb_slice[:, 1:] - smb_slice[:, :-1]  # differences in x
            grad_y = smb_slice[1:, :] - smb_slice[:-1, :]  # differences in y
            smoothness = torch.sum(grad_x**2) + torch.sum(grad_y**2)

        elif smooth_type == 'laplacian':
            # Laplacian regularization: penalize curvature (second derivatives)
            # This encourages even smoother fields
            # Compute discrete Laplacian: ∇²u ≈ u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4*u[i,j]
            laplacian = (
                smb_slice[:-2, 1:-1] +  # above
                smb_slice[2:, 1:-1] +   # below
                smb_slice[1:-1, :-2] +  # left
                smb_slice[1:-1, 2:] -   # right
                4 * smb_slice[1:-1, 1:-1]  # center
            )
            smoothness = torch.sum(laplacian**2)

        else:
            raise ValueError(f"smooth_type must be 'gradient' or 'laplacian', got '{smooth_type}'")

        # ---- Total loss ----
        loss = data_term + reg_lambda * smoothness
    else:
        loss = data_term

    return H_sim, loss, data_term, metrics


def invert_profile(
    smb_vec: torch.Tensor,
    z_min: torch.Tensor | float,
    dz: torch.Tensor | float,
    observation: torch.Tensor,
    glacier_model,
    save_times: list = None,
    reg_lambda: float = 0.0,
    thickness_thresh: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Invert for SMB elevation profile given ice thickness observations.

    This function optimizes a 1D SMB profile smb(z) that is mapped to 2D topography
    using piecewise-linear interpolation. This is fully differentiable and suitable
    for gradient-based optimization.
    """

    # ---- Forward simulation ----
    if save_times is not None:
        saved_H_dict, H_sim = glacier_model(
            precip_tensor=None,
            T_m_lowest=None,
            T_s=None,
            melt_factor=None,
            smb_method='profile',
            smb_vec=smb_vec,
            z_min=z_min,
            dz=dz,
            save_times=save_times,
        )
    else:
        H_sim = glacier_model(
            precip_tensor=None,
            T_m_lowest=None,
            T_s=None,
            melt_factor=None,
            smb_method='profile',
            smb_vec=smb_vec,
            z_min=z_min,
            dz=dz,
        )

    # ---- Compute data fidelity (masked misfit) ----
    metrics = _eval_pair(H_sim, observation, thickness_thresh)
    data_term = metrics["mae"]

    # ---- Smoothness regularization for 1D profile ----
    if reg_lambda > 0:
        # Penalize second derivative (curvature) of SMB profile
        # This encourages smooth, physically plausible profiles
        # Second derivative approximation: smb''[i] ≈ smb[i-1] - 2*smb[i] + smb[i+1]
        second_deriv = smb_vec[:-2] - 2*smb_vec[1:-1] + smb_vec[2:]
        smoothness = torch.sum(second_deriv**2)

        # ---- Total loss ----
        loss = data_term + reg_lambda * smoothness
    else:
        loss = data_term

    return H_sim, loss, data_term, metrics


def invert_ela(
    ELA: torch.Tensor,
    grad_b: torch.Tensor | float,
    b_max: torch.Tensor | float,
    observation: torch.Tensor,
    glacier_model,
    save_times: list = None,
    thickness_thresh: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Invert for the equilibrium line altitude given ice thickness observations.

    The SMB follows the linear ELA law ``smb = min(grad_b * (z - ELA), b_max)``.
    Any of the three parameters may carry ``requires_grad``; whichever do are
    the controls, and the rest stay fixed. ELA alone is the common case.

    Unlike ``invert_profile`` there is nothing to regularize: a scalar ELA is
    already the strongest possible prior on the shape of smb(z).

    Parameters
    ----------
    ELA : torch.Tensor
        Equilibrium line altitude (m), the quantity being inverted for.
    grad_b : torch.Tensor or float
        Mass-balance gradient (m ice eq. per m of elevation).
    b_max : torch.Tensor or float
        Cap on accumulation (m ice eq. per year).
    observation : torch.Tensor
        Observed ice thickness at the final time (m).
    glacier_model : GlacierDynamicsCheckpointed
        Forward model. Its ``forward_scheme`` selects the ice dynamics.
    save_times : list, optional
        Times at which to also return intermediate thickness fields.
    thickness_thresh : float
        Thickness (m) above which a pixel counts as glaciated, for the metrics.

    Returns
    -------
    H_sim : torch.Tensor
        Simulated ice thickness at the final time.
    loss : torch.Tensor
        Scalar objective; here the masked mean squared thickness error.
    data_term : torch.Tensor
        The data-fidelity part of the loss, identical to ``loss``.
    metrics : dict
        mse, rmse, mae, bias, std and glaciated area.
    """

    forward_kwargs = dict(
        precip_tensor=None,
        T_m_lowest=None,
        T_s=None,
        melt_factor=None,
        smb_method='ELA',
        ELA=ELA,
        grad_b=grad_b,
        b_max=b_max,
    )

    if save_times is not None:
        _, H_sim = glacier_model(save_times=save_times, **forward_kwargs)
    else:
        H_sim = glacier_model(**forward_kwargs)

    metrics = _eval_pair(H_sim, observation, thickness_thresh)
    data_term = metrics["mse"]

    return H_sim, data_term, data_term, metrics


def _extent(H: torch.Tensor, thresh: float, width: float) -> torch.Tensor:
    """Soft glacier-presence field: 0 where there is no ice, 1 where there is.

    ``thresh`` is the thickness that counts as the margin and ``width`` the
    transition over which the indicator goes from 0 to 1. The width is the one
    knob that matters: gradients only exist where a cell sits inside it, so a
    narrow width makes the objective a true extent misfit but lets only the few
    cells at the margin carry information, while a wide one lets thickness
    errors leak into what is supposed to be an extent-only loss.
    """
    return torch.sigmoid((H - thresh) / width)


def invert_extent(
    T_m_lowest,
    observations: Dict[int, torch.Tensor],
    glacier_model,
    *,
    precip_tensor,
    T_s,
    melt_factor,
    weights: Dict[int, float] | None = None,
    reg_lambda: float = 0.001,
    thickness_thresh: float = 1.0,
    width: float = 5.0,
    analysis_mask: torch.Tensor | None = None,
    smb_method: str = 'PDD',
    **forward_kwargs,
):
    """Extent-only inversion against an arbitrary set of dated epochs.

    Replaces :func:`inversion_extent`, which hard-codes seven epochs and unpacks
    seven return values from a forward model that now returns one tensor plus a
    dict of snapshots.

    Parameters
    ----------
    T_m_lowest : Tensor
        The control: mean annual temperature at the domain's lowest point, one
        value per ``args.vis_freq`` interval (or a scalar for a constant
        climate).
    observations : dict
        ``{year: observed thickness}``. The forward model is asked for a
        snapshot at each year, so the years must lie inside
        ``[args.t_start, args.ttot]``.
    weights : dict, optional
        ``{year: weight}``, defaulting to 1 for every epoch. Zero an epoch to
        bring it in later, as a warm-up schedule.
    analysis_mask : Tensor, optional
        Weight grid in [0, 1] (a boolean grid also works); the misfit is the
        weighted mean over cells, so a smooth sigmoid mask fades the loss out
        rather than cutting it. Without it
        the loss averages over the whole domain, where most of the disagreement
        sits high in the accumulation basin -- above the equilibrium line for
        every temperature the inversion could try, so no value of the control
        can remove it. Masking to the elevations the control can actually act
        on stops that irreducible error from dominating the gradient.

    Returns
    -------
    simulated : dict
        ``{year: simulated thickness}``.
    loss, data : Tensor
        Total objective and its data term.
    metrics : dict
        Per epoch: the differentiable ``mse`` actually optimized, plus hard
        ``iou``, ``dice`` and areas for reporting. Only ``mse`` carries a
        gradient.
    """
    years = sorted(observations)
    weights = {y: 1.0 for y in years} if weights is None else weights
    total_w = sum(weights[y] for y in years)
    if total_w <= 0:
        raise ValueError("at least one epoch needs a non-zero weight")

    saved, _ = glacier_model(
        precip_tensor=precip_tensor,
        T_m_lowest=T_m_lowest,
        T_s=T_s,
        melt_factor=melt_factor,
        smb_method=smb_method,
        save_times=[float(y) for y in years],
        **forward_kwargs,
    )
    missing = [y for y in years if saved.get(float(y)) is None]
    if missing:
        raise ValueError(
            f"the forward model saved no snapshot for {missing}; check that "
            f"these years lie inside [{float(glacier_model.t_start)}, "
            f"{float(glacier_model.ttot)}]"
        )
    simulated = {y: saved[float(y)] for y in years}

    if analysis_mask is not None:
        w = analysis_mask.to(next(iter(simulated.values())).dtype)
        hard = w > 0.5          # reporting metrics use the hard 0.5 contour
    terms, metrics = [], {}
    for y in years:
        H, obs = simulated[y], observations[y]
        sim_soft = _extent(H, thickness_thresh, width)
        obs_soft = _extent(obs, thickness_thresh, width)
        sq = (sim_soft - obs_soft) ** 2
        mse = sq.mean() if analysis_mask is None else (sq * w).sum() / w.sum()
        terms.append(weights[y] * mse)

        with torch.no_grad():
            s, o = H > thickness_thresh, obs > thickness_thresh
            if analysis_mask is not None:
                s, o = s & hard, o & hard
            inter, union = (s & o).sum(), (s | o).sum()
            metrics[y] = dict(
                mse=mse.detach(),
                iou=inter / union if union > 0 else torch.zeros_like(inter),
                dice=2 * inter / (s.sum() + o.sum()) if (s.sum() + o.sum()) > 0
                else torch.zeros_like(inter),
                area=s.sum() * 0.01,
                obs_area=o.sum() * 0.01,
                weight=weights[y],
            )
    data = torch.stack(terms).sum() / total_w

    if isinstance(T_m_lowest, torch.Tensor) and T_m_lowest.numel() > 1:
        flat = T_m_lowest.flatten()
        smoothness = torch.mean((flat[1:] - flat[:-1]) ** 2)
    else:
        smoothness = torch.zeros_like(data)

    return simulated, data + reg_lambda * smoothness, data, metrics
