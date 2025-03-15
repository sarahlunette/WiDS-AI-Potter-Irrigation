import numpy as np

def compute_evapotranspiration(tmean, tmax, tmin, humidity, wind_speed, lat, doy):
    """
    Compute ET₀ (Reference Evapotranspiration) using FAO-56 Penman-Monteith equation.
    """
    # Constants
    Gsc = 0.0820  # Solar constant (MJ/m²/min)
    phi = np.radians(lat)  # Convert latitude to radians
    delta = 0.409 * np.sin((2 * np.pi * doy / 365) - 1.39)  # Solar declination
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)  # Inverse relative distance Earth-Sun
    omega_s = np.arccos(-np.tan(phi) * np.tan(delta))  # Sunset hour angle

    # Extraterrestrial radiation (Ra) in MJ/m²/day
    Ra = (24 * 60 / np.pi) * Gsc * dr * (
        omega_s * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(omega_s)
    )

    # Psychrometric constant (γ) in kPa/°C
    gamma = 0.665 * 0.001 * 101.3  # Assuming 101.3 kPa standard pressure

    # Vapor pressure deficit (VPD)
    es = 0.6108 * np.exp((17.27 * tmean) / (tmean + 237.3))  # Saturation vapor pressure
    ea = es * (humidity / 100)  # Actual vapor pressure
    vpd = es - ea  # Vapor pressure deficit

    # Slope of saturation vapor pressure curve (Δ)
    delta_svp = (4098 * es) / ((tmean + 237.3) ** 2)

    # Net radiation approximation (Rn - G)
    Rn_minus_G = 0.75 * Ra  # Approximate net radiation

    # FAO-56 Penman-Monteith equation
    ET0 = ((0.408 * delta_svp * Rn_minus_G) + (gamma * (900 / (tmean + 273)) * wind_speed * vpd)) / \
          (delta_svp + gamma * (1 + 0.34 * wind_speed))

    return max(ET0, 0)  # Ensure ET0 is non-negative
