import pyet

def fetch_evapotranspiration_data(solar_radiation=25, temperature, humidity): #TODO: add an api for getting solar radiation accordnig to date, lat, lon
    # Create a PET object
    pet = pyet.PET()

    # Calculate evapotranspiration using the Hargreaves method
    # Note: Hargreaves method typically uses min and max temperature
    # We'll estimate these from the given temperature
    tmin = temperature - 5  # Estimate minimum temperature
    tmax = temperature + 5  # Estimate maximum temperature

    # Calculate evapotranspiration (returns value in mm/day)
    evapo_rate = pet.hargreaves(tmin, tmax, solar_radiation, latitude=0)

    # Adjust for humidity (simple linear adjustment, not part of standard Hargreaves)
    humidity_factor = 1 - (humidity / 100)
    evapo_rate *= humidity_factor

    return evapo_rate
