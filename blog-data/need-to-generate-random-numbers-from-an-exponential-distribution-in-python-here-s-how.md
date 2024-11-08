---
title: "Need to Generate Random Numbers from an Exponential Distribution in Python? Here's How!"
date: '2024-11-08'
id: 'need-to-generate-random-numbers-from-an-exponential-distribution-in-python-here-s-how'
---

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Generate random numbers from an exponential distribution
data = np.random.exponential(5, size=1000)

# Create a histogram of the data
hist, edges = np.histogram(data, bins="auto", density=True)

# Get the middle of each bin for plotting
x = edges[:-1] + np.diff(edges) / 2.

# Plot the histogram
plt.scatter(x, hist)

# Define the exponential decay function
func = lambda x, beta: 1. / beta * np.exp(-x / beta)

# Fit the function to the data
popt, pcov = curve_fit(f=func, xdata=x, ydata=hist)

# Print the fitted parameters
print(popt)

# Plot the fitted curve
xx = np.linspace(0, x.max(), 101)
plt.plot(xx, func(xx, *popt), ls="--", color="k", label="fit, $beta = ${}".format(popt))

# Add a legend and show the plot
plt.legend()
plt.show()
```
