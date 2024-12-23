---
title: "How can evaporation be accurately calculated in algorithmic terrain generation?"
date: "2024-12-23"
id: "how-can-evaporation-be-accurately-calculated-in-algorithmic-terrain-generation"
---

, let’s talk about evaporation in algorithmic terrain generation. It’s a tricky beast, more so than it initially seems. I’ve spent a good chunk of my career building terrain systems, and let me tell you, nailing the water cycle, especially evaporation, is crucial for believable results. Early on, I recall a project where we completely ignored evaporation, and our procedural rivers just kept filling and overflowing into flat, featureless swamps. Not exactly ideal. It’s not just about visual fidelity either; it significantly impacts erosion patterns, vegetation distribution, and even the feel of the climate within your simulated world.

The fundamental issue is that evaporation isn't a simple linear function; it's influenced by multiple interacting factors. Directly translating complex thermodynamic equations into a computationally feasible algorithm is usually overkill for terrain generation purposes. We need approximations that capture the essence of evaporation without tanking performance. Let’s break down how I've approached this in past projects.

First, consider the core principle: evaporation is driven by energy. The sun provides the energy to convert water from a liquid to a gas, so solar irradiance is a critical factor. This means your evaporation rate will naturally vary based on the time of day, the season, and the latitude, if you're simulating a planet. Then you have the local temperature, another key variable. Warmer water evaporates faster. Add to that wind speed, which accelerates the process by removing saturated air, and relative humidity, which limits how much water the air can hold. It quickly becomes a tangled web of interconnected variables.

My approach tends to rely on a simplified evaporation model that integrates these core factors. We begin with a base evaporation rate, determined by factors like the overall climate of the terrain. This base rate is modified by the influence of local parameters. I usually represent these parameters on a heightmap or a grid, just as I do the terrain itself. Think of them as texture maps but representing environmental data instead of surface colors.

Here's a snippet that demonstrates how I typically compute evaporation at each point on the terrain. This snippet uses python due to its expressiveness, but the principle remains the same for other languages. We assume that height is already present in a grid called `heightmap`:

```python
import numpy as np

def calculate_evaporation(heightmap, temperature_map, solar_irradiance_map, humidity_map, wind_speed_map, base_evaporation_rate=0.01):
    """
    Calculates evaporation based on temperature, solar irradiance, humidity, and wind speed.

    Args:
        heightmap (numpy.ndarray): 2D numpy array of terrain heights.
        temperature_map (numpy.ndarray): 2D numpy array of local temperatures.
        solar_irradiance_map (numpy.ndarray): 2D numpy array of solar irradiance values.
        humidity_map (numpy.ndarray): 2D numpy array of relative humidity values (0 to 1).
        wind_speed_map (numpy.ndarray): 2D numpy array of wind speed values.
        base_evaporation_rate (float): Base evaporation rate.

    Returns:
        numpy.ndarray: 2D numpy array of evaporation values.
    """

    evaporation_rate = np.zeros_like(heightmap, dtype=float)
    rows, cols = heightmap.shape
    for row in range(rows):
        for col in range(cols):
          # Normalize environmental factors to a range of 0 to 1
            norm_temp = np.clip(temperature_map[row, col] / 100.0, 0, 1) # temp divided by a typical maximum temperature
            norm_solar = np.clip(solar_irradiance_map[row, col] / 1000.0, 0, 1) # solar intensity divided by a typical maximum
            norm_humidity = 1 - humidity_map[row, col] # invert humidity, high humidity means low evaporation.
            norm_wind = np.clip(wind_speed_map[row,col] / 30.0 , 0 , 1) # wind speed divided by a typical maximum windspeed.

            # Apply scaling factors to the normalized factors
            evaporation_rate[row, col] = base_evaporation_rate * (
                1.0 +
                (norm_temp * 0.5) + # Temperature has a moderate effect
                (norm_solar * 1.0) + # Solar irradiance has a high effect
                (norm_humidity * 0.5) + # Humidity has a moderate effect
                (norm_wind * 0.2)  # Wind speed has a lower effect, may be lower in other systems
            )

    return evaporation_rate
```

This function takes pre-computed temperature, solar irradiance, humidity, and wind speed maps as inputs, along with a `base_evaporation_rate`. It then iterates through each point of the terrain and calculates evaporation rates based on the local environmental factors. It’s crucial to realize that I’m normalizing these factors between 0 and 1. This simplifies scaling and prevents any factor from dominating the result. I use linear scaling here to illustrate; in some more complex systems, you might want to use exponential scaling or a logarithmic curve for a more realistic effect.

This approach works reasonably well, but one often encounters situations where you need to incorporate subsurface water, especially in porous materials. So you can’t just assume all the water is on the surface. This adds a second layer of complexity, but it’s crucial if you want to accurately simulate wetlands, soil moisture, and the like. I’ve used variations of a simple hydrological model to simulate the movement of water through the soil, including vertical infiltration and lateral flow. When dealing with subsurface water, the evaporation calculation changes. The deeper the water table is, the less influence solar radiation and wind speed have and the more relevant becomes soil properties. The soil type influences the rate at which water is drawn from the subsurface to the surface and evaporates.

Here’s a simplified snippet that demonstrates how subsurface water impacts evaporation, building upon the prior model:

```python
def calculate_subsurface_evaporation(heightmap, temperature_map, solar_irradiance_map, humidity_map, wind_speed_map, subsurface_water_map, soil_type_map, base_evaporation_rate=0.01):
    """
    Calculates evaporation considering subsurface water and soil properties.

    Args:
        heightmap (numpy.ndarray): 2D numpy array of terrain heights.
        temperature_map (numpy.ndarray): 2D numpy array of local temperatures.
        solar_irradiance_map (numpy.ndarray): 2D numpy array of solar irradiance values.
        humidity_map (numpy.ndarray): 2D numpy array of relative humidity values (0 to 1).
        wind_speed_map (numpy.ndarray): 2D numpy array of wind speed values.
        subsurface_water_map (numpy.ndarray): 2D numpy array of water depth below the surface
        soil_type_map (numpy.ndarray) : 2D numpy array representing soil type of every position.
        base_evaporation_rate (float): Base evaporation rate.

    Returns:
        numpy.ndarray: 2D numpy array of evaporation values.
    """

    evaporation_rate = np.zeros_like(heightmap, dtype=float)
    rows, cols = heightmap.shape
    for row in range(rows):
        for col in range(cols):
            norm_temp = np.clip(temperature_map[row, col] / 100.0, 0, 1)
            norm_solar = np.clip(solar_irradiance_map[row, col] / 1000.0, 0, 1)
            norm_humidity = 1 - humidity_map[row, col]
            norm_wind = np.clip(wind_speed_map[row, col] / 30.0, 0, 1)
            soil_factor = calculate_soil_factor(soil_type_map[row,col])
            subsurface_influence = 1 - np.clip(subsurface_water_map[row, col] / 10.0, 0, 1)  # deeper water = less influence


            evaporation_rate[row, col] = base_evaporation_rate * (
                1.0 +
                (norm_temp * 0.5 * subsurface_influence * soil_factor ) +
                (norm_solar * 1.0 * subsurface_influence * soil_factor) +
                (norm_humidity * 0.5 * subsurface_influence * soil_factor) +
                (norm_wind * 0.2 * subsurface_influence * soil_factor)
            )

    return evaporation_rate


def calculate_soil_factor(soil_type):
  # Returns a scaling factor based on soil type.
  if soil_type == 0: #sand
    return 1.0
  elif soil_type == 1: #clay
    return 0.5
  elif soil_type == 2: #silt
    return 0.75
  else:
      return 0.7 # Default

```

The new function takes an additional map, `subsurface_water_map`, and `soil_type_map`. The influence of the environmental factors is reduced by `subsurface_influence`, which depends on how deep the subsurface water is. There is also a helper function `calculate_soil_factor`, which returns a soil specific scaling factor. This approach captures the reduction in evaporation as the water table drops, and the difference between soil types. However, it still uses normalized environmental parameters and a base evaporation rate for the calculation.

Finally, it’s important to consider the temporal aspect of evaporation. It's not instantaneous. Water doesn’t disappear in a single update; it gradually diminishes over time. To simulate this, you’ll typically need to implement a system where each point has a water level that is decreased over time based on the calculated evaporation rate. This process is iterative, and it's where delta time or time step values become particularly crucial to ensuring consistent results across different systems.

```python
def update_water_level(water_level_map, evaporation_map, delta_time):
  """
    Updates the water level map based on the evaporation map and time step.

    Args:
       water_level_map (numpy.ndarray): 2D numpy array of water levels at every position.
       evaporation_map (numpy.ndarray): 2D numpy array of evaporation values.
       delta_time (float): Time since the last update, usually in seconds.

    Returns:
       numpy.ndarray: Updated water level map
  """

  water_level_map = water_level_map - (evaporation_map * delta_time)
  water_level_map = np.clip(water_level_map, 0 , None) # Water cannot be negative.
  return water_level_map
```
Here, the function takes the `water_level_map` and modifies it based on `evaporation_map` and the time delta. The water level is clipped to zero to avoid negative values.

In terms of further reading, I'd highly recommend seeking out resources on hydrological modeling. Specifically, you should examine works from the field of environmental science and geographic information systems, as these often touch on how evaporation is handled in a simulation or modeling context. You won't need to know the intricacies of a full climate model, but a working understanding of the basics is invaluable. A good starting point would be introductory texts like *Applied Hydrology* by Ven Te Chow, David R. Maidment, and Larry W. Mays or *Remote Sensing and GIS for Ecologists: Using Open Source Software* by Martin Wegmann et al., which also cover environmental modelling more broadly. For the computational aspects, research papers about computational fluid dynamics often contain useful methods for approximating flow calculations, and *Numerical Recipes* by William H. Press et al. remains a great resource for working with numerical methods. You should also investigate modern computer graphics textbooks such as "Physically Based Rendering: From Theory To Implementation" by Matt Pharr, Wenzel Jakob and Greg Humphreys which contain chapters on simulating natural phenomena.

In conclusion, calculating evaporation accurately for terrain generation requires a multi-faceted approach. It's not about a single magic formula but rather a combination of simplifications, approximations, and careful consideration of how different environmental factors interact. I’ve always found that starting with a simple base evaporation rate and layering in the effects of temperature, solar irradiance, humidity, wind, soil type, and subsurface water tables provides a good balance between realism and computational feasibility. The key is to remember that the final result needs to be visually plausible and perform well, and that finding the correct balance between the complexity and performance is crucial to any simulation.
