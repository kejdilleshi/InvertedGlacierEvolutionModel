import torch
import math

def apply_lapse_rate(Z, T_ma0, T_mj0, lapse=6.0/1000.0):
    """Return cellwise T_ma, T_mj given base temps and altitude Z."""
    Z0 = torch.min(Z)
    Δz = Z - Z0
    return T_ma0 - lapse*Δz, T_mj0 - lapse*Δz

def compute_integral_positive_temperature(T_ma, T_mj):
    A = 12.0
    ratio = T_ma / (T_mj - T_ma)
    valid = ratio < 1
    r = torch.clamp(ratio[valid], -1, 1)
    term1 = T_ma[valid]*(A - A/math.pi*torch.acos(r))
    term2 = (T_mj[valid] - T_ma[valid]) * A/math.pi * torch.sqrt(1 - r**2)
    out = torch.zeros_like(T_ma)
    out[valid] = term1 + term2
    return out

def compute_negative_temperature_ratio(T_ma, T_mj):
    ratio = T_ma / (T_mj - T_ma)
    out = torch.zeros_like(T_ma)
    mask_pos = ratio >= 1
    mask_neg = ratio <= -1
    mask_mid = ~(mask_pos | mask_neg)
    out[mask_neg] = 1.0
    out[mask_mid] = (1/math.pi)*torch.acos(torch.clamp(ratio[mask_mid], -1, 1))
    return out
