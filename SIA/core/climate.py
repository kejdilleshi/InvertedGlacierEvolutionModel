import torch
import math

def compute_integral_positive_temperature(T_m, T_s):
    """
    Computes the integral of T_abl(t) over the period where T_abl > 0 (PyTorch version).
    """
    A = 12.0  # months
    scale = 1.1 # to make the function more linear for [-0.5,0.5]
    return T_m * (A - (A / torch.pi) * torch.acos(torch.tanh(scale *T_m / T_s)))+(T_s * A / torch.pi) * torch.sqrt(1 - (torch.tanh(scale *T_m / T_s))**2)

def apply_lapse_rate(topography, T_m_lowest):
    lapse_rate = 6.0 / 1000.0  # 6°C/km
    min_altitude = torch.min(topography)
    delta_alt = topography - min_altitude

    T_m = T_m_lowest - lapse_rate * delta_alt
    
    return T_m

def compute_negative_temperature_ratio(T_m, T_s):
    """
    Computes the ratio of the year when the temperature is negative (PyTorch version).
    Parameters:
        T_m (Tensor): 2D tensor of mean annual temperatures (on device)
        T_mj (Tensor): 2D tensor of hottest month temperatures (on device)
    Returns:
        Tensor: 2D tensor of negative temperature ratios (values between 0 and 1)
    """
    scale = 1.1 # to make the function more linear for [-0.5,0.5]
    return (1.0 / torch.pi) * torch.acos(torch.tanh(scale *T_m / T_s))