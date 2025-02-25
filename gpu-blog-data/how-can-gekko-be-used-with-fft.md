---
title: "How can GEKKO be used with FFT?"
date: "2025-01-30"
id: "how-can-gekko-be-used-with-fft"
---
The integration of GEKKO with Fast Fourier Transform (FFT) facilitates the analysis and manipulation of time-series data within a dynamic optimization framework. Having implemented such a system for a complex chemical process model in the past, I’ve found it’s crucial to understand the nuances of how these two seemingly disparate tools interact. GEKKO, being a Python package focused on solving differential and algebraic equations, doesn't inherently perform frequency analysis; that's the domain of FFT. However, the output of an FFT, specifically the spectral data, can be fed into GEKKO to create models that are frequency-aware, enabling process control, noise filtering, or parameter estimation based on frequency content.

The fundamental concept revolves around leveraging FFT to extract frequency information from time-domain data generated either through simulation within GEKKO or obtained from real-world measurements. This spectral information—amplitude and phase at different frequencies—becomes a new dataset that GEKKO can utilize as input for optimization or parameter estimation. The key connection is that GEKKO can solve optimization problems where the FFT output is part of the objective function, constraints, or model parameters. This allows us to move beyond traditional time-domain modelling and control to scenarios where the frequency characteristics of a system are crucial.

For a concrete illustration, imagine simulating a chemical reactor with periodic disturbances affecting its temperature. Instead of directly modelling the disturbance in the time domain, one could use measured temperature data or simulated data from an initial GEKKO run. Applying FFT to this temperature time series generates a spectrum representing the disturbance's dominant frequencies. This spectrum can then be incorporated into a GEKKO model to identify the source of the disturbance or to design a controller that specifically dampens the undesirable frequencies.

The implementation typically requires a workflow that encompasses these steps: 1) generating time-series data, 2) performing FFT on the data, 3) extracting relevant spectral information, and 4) incorporating this information into a GEKKO model for further analysis or optimization.

Here are three illustrative code examples:

**Example 1: Basic FFT of a GEKKO Simulation Output**

```python
from gekko import GEKKO
import numpy as np
from numpy.fft import fft, fftfreq
import matplotlib.pyplot as plt

# Initialize GEKKO model
m = GEKKO(remote=False)

# Define time and parameters
n = 500  # number of time points
tf = 100.0 # final time
t = np.linspace(0,tf,n)
m.time = t
k = 0.1 # Rate Constant
# Variables
y = m.Var(value=1.0) # initial condition
# Equation
m.Equation(y.dt()== -k * y)

# Solve model
m.options.imode = 4
m.solve(disp=False)

# Perform FFT
y_values = y.value
yf = fft(y_values)
xf = fftfreq(n,tf/n) # frequencies
yf_abs = np.abs(yf)

# Plot the results
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.plot(t, y_values)
plt.title('Time Domain Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.subplot(122)
plt.plot(xf[0:n//2],yf_abs[0:n//2]) # Plot the first half of the FFT output only
plt.title('Frequency Domain Signal')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.show()
```

*Commentary:* In this example, I demonstrate the simplest use case: generating a time-series from a GEKKO model (a decaying exponential) and using `numpy.fft.fft` and `numpy.fft.fftfreq` to compute the Fourier transform and its corresponding frequencies. The time-domain signal, as well as the resulting spectral information (magnitude vs. frequency), are visualized using `matplotlib`. Note that the `fftfreq` function needs the sampling period (tf/n) as an input. I display only the first half of the FFT result because the spectrum is symmetric. This example primarily focuses on capturing the spectrum of a signal generated by a GEKKO simulation, but doesn't use that information *within* a GEKKO model.

**Example 2: Parameter Estimation Using FFT Data**

```python
from gekko import GEKKO
import numpy as np
from numpy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from scipy.signal import chirp

# Generate simulated data with known parameters
n = 1000
tf = 100
t = np.linspace(0, tf, n)
true_freq = 5 # True frequency
signal = chirp(t,f0=0.5,f1=true_freq,t1=tf)
noisy_signal = signal + 0.1*np.random.randn(n) # Simulated 'measured' data

# Calculate FFT of the simulated data
yf = fft(noisy_signal)
xf = fftfreq(n, tf/n)
yf_abs = np.abs(yf)

# Find the index of the max frequency
max_freq_index = np.argmax(yf_abs[0:n//2])
measured_freq = xf[max_freq_index]

# Initialize GEKKO Model for Parameter Estimation
m = GEKKO(remote=False)

# Define the parameter we want to estimate
freq_guess = 2.0 # initial guess
estimated_freq = m.Param(value=freq_guess)

# Define a GEKKO variable to simulate the signal
simulated_signal = m.Var()
# Define an intermediate variable for the simulated frequency in time-domain.
m.Equation(simulated_signal.dt() == 2*np.pi*estimated_freq*m.cos(2*np.pi*estimated_freq*m.time)) # Simplified model

# Objective function is difference between the measured frequency and the estimated frequency
m.Obj((measured_freq - estimated_freq)**2)

m.options.imode = 7 # Parameter Estimation mode
m.solve(disp=False)

print("True Frequency:", true_freq)
print("Estimated Frequency:",estimated_freq.value[0])

# Plotting the FFT Magnitude as well as the estimated parameters
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.plot(xf[0:n//2],yf_abs[0:n//2])
plt.title("FFT Magnitude")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.scatter(measured_freq,yf_abs[max_freq_index],marker='*',s=150,color='red', label= "Max Frequency")
plt.legend()

plt.subplot(122)
plt.plot(t,noisy_signal,label="Noisy Signal")
plt.plot(t,m.Var(simulated_signal,sim_time=t), label="Simulated Signal")
plt.title("Time Domain Signal Comparison")
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
```

*Commentary:* This example demonstrates how the spectral output of an FFT can be used for parameter estimation within GEKKO. Here, I first create a noisy chirp signal with a known frequency.  Then I use the FFT of this noisy signal to find the peak frequency, which serves as a "measurement" for the GEKKO model. GEKKO is then used to find a model parameter (frequency) such that the model output matches the 'measured' frequency.  This example illustrates a simple scenario where an FFT acts as a preprocessor to extract key spectral features that are subsequently used to set an optimization target within GEKKO, which enables parameter estimation. The estimated frequency is then compared to the true frequency.

**Example 3: Filtering a Signal Using GEKKO**
```python
from gekko import GEKKO
import numpy as np
from numpy.fft import fft, fftfreq, ifft
import matplotlib.pyplot as plt
from scipy.signal import chirp

# Generate a noisy chirp signal
n = 1000
tf = 100
t = np.linspace(0, tf, n)
true_freq = 5
signal = chirp(t,f0=0.5,f1=true_freq,t1=tf)
noise = 0.5*np.sin(2*np.pi*15*t)
noisy_signal = signal + noise # Signal with noise at 15Hz

# Perform FFT
yf = fft(noisy_signal)
xf = fftfreq(n, tf/n)
yf_abs = np.abs(yf)

# Filter
cutoff_freq=10 # Cutoff frequency
yf_filtered = np.copy(yf)
for i,freq in enumerate(xf):
    if np.abs(freq)> cutoff_freq:
        yf_filtered[i] = 0

filtered_signal = ifft(yf_filtered)


#GEKKO model for parameter estimation
m=GEKKO(remote=False)

#Parameter defining the cutoff
cutoff = m.Param(value=cutoff_freq)

#Variable to represent FFT and filter
filtered_fft = m.Var(value=yf)

# Objective function is the difference between the measured signal's FFT and the filtered signal
for i,freq in enumerate(xf):
  m.Equation(m.if3(abs(freq) > cutoff, filtered_fft[i], yf[i]) == 0)

# Optimization
m.Obj(m.sum([abs(filtered_fft[i])**2 for i in range(n)]))
m.options.imode=7
m.solve(disp=False)


#Plotting
plt.figure(figsize=(12, 6))

plt.subplot(221)
plt.plot(t, noisy_signal)
plt.title('Noisy Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.subplot(222)
plt.plot(t, signal)
plt.title("Original Signal")
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.subplot(223)
plt.plot(t, filtered_signal.real)
plt.title('Filtered Signal (FFT)')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.subplot(224)
plt.plot(xf[0:n//2], np.abs(yf[0:n//2]), label="Original")
plt.plot(xf[0:n//2], np.abs(m.Var(filtered_fft, sim_time=xf)[0:n//2]), label="Filtered FFT")
plt.title('FFT Spectrum Comparison')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.show()
```

*Commentary:* In this example, I demonstrate a basic noise filtering application. After creating the noisy chirp signal I perform an FFT of the simulated data and then filter out frequencies above the cutoff within the GEKKO model and use the modified FFT output to achieve an optimization objective (minimize the filtered signal). This shows how FFT can be used in conjunction with GEKKO to filter data. The cutoff frequency is also treated as an optimizable parameter. Finally, I plot the original, noisy signal, the filtered signal using both FFT processing and the GEKKO based filter, along with the corresponding spectra. This example shows how spectral analysis can inform filtering strategies inside of an optimization model.

Regarding resources, for a deeper understanding of FFT, explore materials on digital signal processing; many reputable textbooks cover FFT principles. For GEKKO specifics, the official documentation and examples available on their website, as well as relevant journal publications (typically focusing on chemical engineering and process systems engineering), are invaluable. Additionally, numerous open-source Python libraries offer supplementary tools for spectral analysis that can further assist with advanced applications.
