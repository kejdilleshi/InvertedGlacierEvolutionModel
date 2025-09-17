import torch
from torch import Tensor
from typing import Union


def apply_lapse_rate(topography, T_m_lowest):
    lapse_rate = 7.0 / 1000.0  # 7°C/km
    min_altitude = torch.min(topography)
    delta_alt = topography - min_altitude

    T_m = T_m_lowest - lapse_rate * delta_alt
    
    return T_m


def apply_lapse_rate_daily(topography: torch.Tensor, T_m_lowest: torch.Tensor) -> torch.Tensor:
    """
    topography: (H, W) altitude in meters
    T_m_lowest: (366, 1) or (366,) daily temperatures at the lowest altitude
    returns:    (366, H, W) daily temperature fields
    """
    # Keep dtype/device consistent with topography
    lapse_rate = torch.as_tensor(7.0/1000.0, dtype=topography.dtype, device=topography.device)  # 7 °C/km

    # Altitude differences relative to minimum
    min_altitude = topography.amin()                # scalar
    delta_alt = topography - min_altitude           # (H, W)

    # Shape to enable broadcasting
    # T_m_lowest: (366, 1, 1), delta_alt: (1, H, W)
    T_m_lowest = T_m_lowest.squeeze(-1).to(dtype=topography.dtype, device=topography.device).view(-1, 1, 1)
    delta_alt   = delta_alt.unsqueeze(0)

    # Broadcasted computation -> (366, H, W)
    T_m = T_m_lowest - lapse_rate * delta_alt
    return T_m


def smooth_piecewise(x: Tensor, w: float = 0.1, eps: float = 1e-6) -> Tensor:
    one  = torch.tensor(1.0 - eps,  dtype=x.dtype, device=x.device)
    mone = -one

    left_p  = (x + 1.0 + w) / (2.0 * w)         # (-1-w , -1+w)
    right_p = (x - (1.0 - w)) / (2.0 * w)       # ( 1-w ,  1+w)

    left_poly  = mone + w * left_p.square()
    right_poly = (-w) * right_p.square() + 2.0*w*right_p + (1.0 - w)

    y = torch.where(x <= -1.0 - w, mone,
         torch.where(x >=  1.0 + w,  one,
           torch.where(x <  -1.0 + w, left_poly,
             torch.where(x >  1.0 - w,  right_poly, x))))
    return y



def compute_integral_positive_temperature(T_m, T_s):
    """
    Computes the integral of T_abl(t) over the period where T_abl > 0 (PyTorch version).
    """
    A = 12.0  # months
    return T_m * (A - (A / torch.pi) * torch.acos(smooth_piecewise(T_m / T_s)))+(T_s * A / torch.pi) * torch.sqrt(1 - (smooth_piecewise(T_m / T_s))**2)

def compute_negative_temperature_ratio(T_m, T_s):
    """
    Computes the ratio of the year when the temperature is negative (PyTorch version).
    Parameters:
        T_m (Tensor): 2D tensor of mean annual temperatures (on device)
        T_s (Tensor): Scalar representig the difference between summer & winter (on device)
    Returns:
        Tensor: 2D tensor of negative temperature ratios (values between 0 and 1)
    """
    return (1.0 / torch.pi) * torch.acos(smooth_piecewise(T_m / T_s))


# daily calculations
def _align_time_vector(vec: Tensor, ref: Tensor, time_dim: int) -> Tensor:
    """If vec is a time vector of shape (T,), reshape to broadcast along ref's time_dim."""
    time_dim = time_dim % ref.ndim
    if vec.ndim == 0:
        # scalar: fine as-is
        return vec
    if vec.ndim == 1 and vec.shape[0] == ref.shape[time_dim]:
        view_shape = [1] * ref.ndim
        view_shape[time_dim] = vec.shape[0]    # put T on the time axis
        return vec.view(*view_shape)
    # Otherwise assume it's already broadcastable; caller may still hit a runtime error if not
    return vec

def pdd_sum_daily(
    T_daily: Tensor,
    time_dim: int = 0,
    threshold_c: Union[float, Tensor] = 0.2
) -> Tensor:
    """
    Positive Degree Days: sum_t max(T - threshold, 0).
    T_daily: (T, X, Y) or with extra batch dims. Reduces over time_dim.
    threshold_c: scalar, (T,), (X, Y), or any broadcastable shape.
    """
    time_dim = time_dim % T_daily.ndim
    thr = threshold_c if isinstance(threshold_c, Tensor) else torch.as_tensor(
        threshold_c, dtype=T_daily.dtype, device=T_daily.device
    )
    if isinstance(thr, Tensor):
        thr = thr.to(dtype=T_daily.dtype, device=T_daily.device)
        # If it's a (T,) vector, align to time_dim
        if thr.ndim == 1 and thr.shape[0] == T_daily.shape[time_dim]:
            thr = _align_time_vector(thr, T_daily, time_dim)

    pdd = torch.clamp(T_daily - thr, min=0)
    return pdd.sum(dim=time_dim)

def accumulation_from_daily(
    P_daily: Tensor,  # can be scalar, (T,), or grid broadcastable to T_daily
    T_daily: Tensor,
    time_dim: int = 0,
    snow_temp_c: Union[float, Tensor] = 0.0
) -> Tensor:
    """
    Solid-precip accumulation: sum_t P where T <= snow_temp_c.
    Handles P_daily as (T,) by reshaping to (T,1,1,...) on the correct time_dim.
    """
    time_dim = time_dim % T_daily.ndim
    thr = snow_temp_c if isinstance(snow_temp_c, Tensor) else torch.as_tensor(
        snow_temp_c, dtype=T_daily.dtype, device=T_daily.device
    )
    thr = thr.to(dtype=T_daily.dtype, device=T_daily.device)

    # Cold-day mask (same shape as T_daily)
    mask_cold = (T_daily <= thr)

    # Align P_daily to broadcast along T_daily
    P_daily = P_daily.to(dtype=T_daily.dtype, device=T_daily.device)
    if P_daily.ndim == 1 and P_daily.shape[0] == T_daily.shape[time_dim]:
        P_daily = _align_time_vector(P_daily, T_daily, time_dim)
    # (If it's scalar or already broadcastable, we keep it as-is.)

    # Multiply by mask; scalar zero broadcasts fine
    acc = P_daily * mask_cold.to(T_daily.dtype)
    return acc.sum(dim=time_dim)