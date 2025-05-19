from core.climate import apply_lapse_rate, compute_integral_positive_temperature, compute_negative_temperature_ratio

# constants
MELT_F = 2/12
ρw, ρi = 1000.0, 910.0

def update_smb(Z_topo, precipitation, T_ma0, T_mj0, ice_mask):
    T_ma, T_mj = apply_lapse_rate(Z_topo, T_ma0, T_mj0)
    accr = precipitation * compute_negative_temperature_ratio(T_ma, T_mj)
    abl = MELT_F * compute_integral_positive_temperature(T_ma, T_mj)
    smb = (accr - abl) * (ρw/ρi)
    return smb * ice_mask
