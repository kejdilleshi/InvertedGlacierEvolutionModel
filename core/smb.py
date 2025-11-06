from core.climate import apply_lapse_rate, compute_negative_temperature_ratio, compute_integral_positive_temperature, accumulation_from_daily, pdd_sum_daily,apply_lapse_rate_daily
import torch
from torch import Tensor



# Define SMB parameters directly
smb_oggm_wat_density = 1000.0  # kg/m³
smb_oggm_ice_density = 910.0   # kg/m³


def cosine_temperature_series(ttot, t_freq, T_high=9.0, T_low=7.0, device='cpu'):
    """
    Returns a tensor of length N = ttot / t_freq with one full cosine cycle:
    starts at T_high, dips to T_low mid-cycle, returns to T_high at the end.
    """
    N = int(round(ttot / float(t_freq)))
    if N < 2:
        raise ValueError("ttot/t_freq must be >= 2 to form a cosine cycle.")
    mean = 0.5 * (T_high + T_low)   # 8.0
    amp  = 0.5 * (T_high - T_low)   # 1.0
    theta = torch.linspace(0, 2*torch.pi, steps=N, device=device)  # inclusive of endpoints
    return mean + amp * torch.cos(theta)


def update_smb_PDD(
    Z_topo: Tensor,
    precipitation: Tensor | float | None = None,  # annual precip (m w.e./yr) for parametric mode
    T_m_lowest: Tensor | float | None = None,     # mean annual temp at lowest point (°C) for parametric mode
    T_s: Tensor | float | None = None,            # seasonal amplitude parameter for parametric mode
    melt_factor: Tensor | float = 2.0 / 360,          # ablation factor in m8day) for daily mode,
                                                  # or in m w.e. per (°C·A) matching compute_integral_positive_temperature in parametric mode
    smb_wat_rho: float = 1000.0,
    smb_ice_rho: float = 910.0,
    # Daily mode inputs (mutually exclusive with parametric mode)
    P_daily: Tensor | None = None,                # daily precip (m w.e. per day), shape [T, ...] or broadcastable
    T_daily: Tensor | None = None,                # daily temp (°C), same shape/broadcast as P_daily
    time_dim: int = 0,                            # which axis is time for daily inputs
    positive_degree_threshold_c: float = 1.0,     # your requested >1°C threshold for PDD
) -> Tensor:
    """
    Compute SMB (m ice eq / year) using Positive Degree Day (PDD) method.

    Modes:
      • Daily mode: if P_daily and T_daily are provided -> direct PDD and accumulation from daily data.
      • Parametric mode: otherwise use precipitation, T_m_lowest, T_s with the OGGM-style formulas.

    Returns:
      smb: Tensor of SMB in m ice eq per year (same spatial shape as inputs).
    """

    # --- DAILY MODE ---
    if (P_daily is not None) and (T_daily is not None):


        # Temperature field via lapse-rate
        T_daily_topo = apply_lapse_rate_daily(Z_topo, T_daily)

        # Accumulation: sum precip where T <= 0°C
        accumulation_wat = accumulation_from_daily(P_daily, T_daily_topo, time_dim=time_dim)

        # PDD (with threshold 1°C by default)
        pdd = pdd_sum_daily(T_daily_topo, time_dim=time_dim, threshold_c=positive_degree_threshold_c)

        # Ablation in water equivalent
        # melt_factor must be calibrated in m w.e. per (°C·day) for this branch
        ablation_wat = torch.as_tensor(melt_factor, dtype=pdd.dtype, device=pdd.device) * pdd


    else:
        # --- PARAMETRIC MODE ---
        if any(x is None for x in (precipitation, T_m_lowest, T_s)):
            raise ValueError("Parametric mode requires 'precipitation', 'T_m_lowest', and 'T_s' when daily inputs are not provided.")

        # Build fields on the same device/dtype as Z_topo
        precipitation = torch.as_tensor(precipitation, dtype=Z_topo.dtype, device=Z_topo.device)
        T_s = torch.as_tensor(T_s, dtype=Z_topo.dtype, device=Z_topo.device)

        # Temperature field via lapse-rate
        T_m = apply_lapse_rate(Z_topo, T_m_lowest)

        # Accumulation fraction of year with negative temps, times annual precip (water eq)
        neg_ratio = compute_negative_temperature_ratio(T_m, T_s)  # [0,1]
        accumulation_wat = precipitation * neg_ratio
        # print(f' Accumulation is {accumulation_wat.max()}')
        # Positive degree "integral" (units degree*A, with A=12 in the helper)
        p_int = compute_integral_positive_temperature(T_m, T_s)

        # melt_factor here must be calibrated to the same time-units as p_int
        ablation_wat = torch.as_tensor(melt_factor, dtype=p_int.dtype, device=p_int.device) * p_int
        # print(f' Ablation is {ablation_wat.max()}')


    # Convert water-equivalent to ice-equivalent and combine
    rho_w = torch.as_tensor(smb_wat_rho, dtype=Z_topo.dtype, device=Z_topo.device)
    rho_i = torch.as_tensor(smb_ice_rho, dtype=Z_topo.dtype, device=Z_topo.device)

    smb = (accumulation_wat - ablation_wat) * (rho_w / rho_i)

    return smb


def update_smb_ELA(
    Z_topo: Tensor,
    ELA: Tensor | float,
    grad_b: Tensor | float,
    b_max: Tensor | float,
) -> Tensor:
    """
    Compute SMB (m ice eq / year) using simple ELA (Equilibrium Line Altitude) method.

    The SMB is linearly proportional to the elevation difference from the ELA,
    capped at a maximum accumulation rate.

    SMB = min(grad_b * (Z - ELA), b_max)

    Parameters:
    -----------
    Z_topo : Tensor
        Surface elevation (m)
    ELA : Tensor or float
        Equilibrium Line Altitude (m)
    grad_b : Tensor or float
        Mass balance gradient (m ice eq / m elevation)
    b_max : Tensor or float
        Maximum accumulation rate (m ice eq / year)

    Returns:
    --------
    smb : Tensor
        Surface mass balance in m ice eq per year (same shape as Z_topo)
    """
    device = Z_topo.device
    dtype = Z_topo.dtype

    # Convert inputs to tensors on the same device/dtype as Z_topo
    ELA = torch.as_tensor(ELA, dtype=dtype, device=device)
    grad_b = torch.as_tensor(grad_b, dtype=dtype, device=device)
    b_max = torch.as_tensor(b_max, dtype=dtype, device=device)

    # Compute SMB: grad_b * (Z - ELA), capped at b_max
    smb = torch.minimum(grad_b * (Z_topo - ELA), b_max)

    return smb


# Backward compatibility alias
update_smb = update_smb_PDD



