from core.climate import apply_lapse_rate, compute_integral_positive_temperature, compute_negative_temperature_ratio

# Define SMB parameters directly
melt_f = 2 / 12  # m water / (°C·year)
smb_oggm_wat_density = 1000.0  # kg/m³
smb_oggm_ice_density = 910.0   # kg/m³

def update_smb(Z_topo, precipitation, T_m_lowest, T_s):
    """
    Compute the surface mass balance (SMB)

    Inputs:
        Z_topo         : Topography [Unit: m]
        precipitation  : Annual precip. [Unit: m y⁻¹ (water)]
        T_m_lowest     : Mean annual temp at lowest point [Unit: °C]
        T_mj_lowest    : Max month temp at lowest point [Unit: °C]

    Output:
        smb            : Surface mass balance [Unit: m ice eq. / y]
    This mass balance routine implements the OGGM surface mass balance model.
    """

    # Apply lapse rate to get temperature fields
    T_m = apply_lapse_rate(Z_topo, T_m_lowest)

    # Compute accumulation in water equivalent [m y⁻¹]
    accumulation = precipitation * compute_negative_temperature_ratio(T_m, T_s)

    # Compute ablation (melt) in water equivalent [m y⁻¹]
    ablation = melt_f * compute_integral_positive_temperature(T_m, T_s)

    # Convert to ice equivalent and compute final SMB
    smb = (accumulation - ablation) * (smb_oggm_wat_density / smb_oggm_ice_density)

    return smb
