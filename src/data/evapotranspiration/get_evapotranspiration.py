import numpy as np
import pandas as pd

def extraterrestrial_radiation(lat, doy):
    """ Compute Ra (extraterrestrial radiation) using FAO-56 formula """
    Gsc = 0.0820  # Solar constant (MJ/m²/min)
    
    # Convert latitude to radians
    phi = np.radians(lat)
    
    # Compute solar declination (delta)
    delta = 0.409 * np.sin((2 * np.pi * doy / 365) - 1.39)
    
    # Compute inverse relative distance Earth-Sun (dr)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    
    # Compute sunset hour angle (omega_s)
    omega_s = np.arccos(-np.tan(phi) * np.tan(delta))
    
    # Compute extraterrestrial radiation (Ra) in MJ/m²/day
    Ra = (24 * 60 / np.pi) * Gsc * dr * (
        omega_s * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(omega_s)
    )
    
    return Ra

def hargreaves_et(tmin, tmax, Ra):
    """ Compute evapotranspiration using the Hargreaves equation """
    if np.isnan(Ra) or Ra <= 0:
        print("Warning: Invalid Ra value (NaN or negative)")
        return None

    tmean = (tmin + tmax) / 2

    # Ensure temperature difference is positive
    temp_diff = tmax - tmin
    if temp_diff <= 0:
        print("Warning: Invalid temperature difference")
        return None

    # Compute evapotranspiration
    ET0 = 0.0023 * Ra * np.sqrt(temp_diff) * (tmean + 17.8)

    return ET0 if ET0 >= 0 else None  # Ensure ET is not negative

def fetch_evapotranspiration_data(tmean, lat):
    """ Compute Evapotranspiration using the manual Hargreaves method """
    if tmean is None:
        print("Error: tmean is None")
        return None

    tmin = tmean - 2
    tmax = tmean + 2

    # Get the current day of the year
    doy = pd.Timestamp.today().day_of_year  
    Ra = extraterrestrial_radiation(lat, doy)  # Compute Ra

    # print(f"tmin: {tmin}, tmax: {tmax}, Ra: {Ra}")

    return hargreaves_et(tmin, tmax, Ra)
