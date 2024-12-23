---
title: "How can I predict a sine wave using Python?"
date: "2024-12-23"
id: "how-can-i-predict-a-sine-wave-using-python"
---

Alright, let's unpack this. Predicting a sine wave isn’t about divining the future; it's about understanding the underlying mathematical structure and then employing that knowledge within a predictive model. This task often crops up in signal processing, time series analysis, and even some areas of simulation. I’ve personally tackled this kind of thing multiple times in various embedded systems and data analysis projects, so I've got some hands-on experience to draw from.

The core concept here is that a sine wave, at its heart, is defined by a fairly simple mathematical function: `y(t) = A * sin(2 * pi * f * t + phi)`, where `A` is the amplitude, `f` is the frequency, `t` is time, and `phi` is the phase shift. Our goal, then, isn't about magic, but about estimating the parameters `A`, `f`, and `phi` (and maybe handling any noise present). This is where curve fitting and time series analysis techniques come into play.

One common approach, and arguably the most intuitive for this case, is curve fitting using non-linear least squares. Essentially, we use an algorithm, usually a variant of the Levenberg-Marquardt algorithm (which you might want to explore in detail in *Numerical Recipes* by Press et al.), to find the best parameters that fit the sine wave equation to our data. We start with initial guesses for A, f, and phi and then refine these guesses iteratively to minimize the error between the predicted sine wave and the observed data.

Let’s illustrate with a Python example. I'll use `scipy.optimize.curve_fit`, which provides a convenient implementation for this purpose:

```python
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def sine_wave(t, A, f, phi):
    return A * np.sin(2 * np.pi * f * t + phi)

# Generate some noisy data for demonstration
t = np.linspace(0, 5, 100)
A_true = 2.5
f_true = 1.2
phi_true = 0.5
y_true = sine_wave(t, A_true, f_true, phi_true)
noise = 0.5 * np.random.normal(size=len(t))
y_noisy = y_true + noise

# Initial parameter guesses
p0 = [1.0, 1.0, 0.0]

# Perform curve fitting
popt, pcov = curve_fit(sine_wave, t, y_noisy, p0=p0)

# Extract the fitted parameters
A_fit, f_fit, phi_fit = popt

# Generate the fitted sine wave
y_fit = sine_wave(t, A_fit, f_fit, phi_fit)

# Visualization
plt.plot(t, y_noisy, 'o', label='Noisy Data')
plt.plot(t, y_fit, label='Fitted Sine Wave')
plt.plot(t, y_true, label = 'True Sine Wave')
plt.legend()
plt.show()

print(f"Fitted Amplitude: {A_fit:.2f}")
print(f"Fitted Frequency: {f_fit:.2f}")
print(f"Fitted Phase Shift: {phi_fit:.2f}")
```

In this snippet, we generate a noisy sine wave, fit the `sine_wave` function, then plot both the noisy data and the fitted curve for comparison. The output will show how closely the estimated parameters match the original ones, showing that it accurately models the original signal despite the noise.

Curve fitting, while effective, requires an entire cycle or, ideally multiple periods, of the sine wave to get accurate parameters. If you are dealing with data that has much less than one cycle, you need a different approach. This is where time series analysis techniques like spectral analysis can be beneficial. By taking the discrete Fourier transform (DFT) of your data, you can identify the prominent frequencies present. In the case of a single sine wave, you should see a clear peak in the frequency spectrum. The location of that peak corresponds to your frequency *f*. From the amplitude at that location, we can get an estimate of A and also get a rough estimation of phi, though it's usually less accurate.

Let's look at how we can perform spectral analysis using the FFT with the `numpy` library:

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate some noisy data
t = np.linspace(0, 10, 200) # Reduced length to focus on frequency
A_true = 1.8
f_true = 2.5
phi_true = 0.8
y_true = A_true * np.sin(2 * np.pi * f_true * t + phi_true)
noise = 0.3 * np.random.normal(size=len(t))
y_noisy = y_true + noise

# Compute the FFT
yf = np.fft.fft(y_noisy)
T = t[1] - t[0]
N = len(t)
xf = np.fft.fftfreq(N, T)[:N//2]
y_abs = np.abs(yf[:N//2])

# Find the index of the maximum magnitude
peak_index = np.argmax(y_abs[1:]) + 1 # Ignore the DC component
peak_frequency = xf[peak_index]

# Amplitude is roughly 2 * magnitude/N (due to split across positive and negative frequencies)
peak_amplitude = 2 * y_abs[peak_index] / N

#Visualization
plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(t,y_noisy)
plt.title('Original Signal')
plt.subplot(2,1,2)
plt.plot(xf, y_abs)
plt.title("Frequency Spectrum")
plt.xlabel('Frequency')
plt.show()

print(f"Estimated Frequency: {peak_frequency:.2f}")
print(f"Estimated Amplitude: {peak_amplitude:.2f}")
```

The output will show the original time-domain signal and its frequency spectrum. The largest peak in the spectrum will correspond to the underlying frequency of the sine wave, and its amplitude can provide a corresponding estimate. It's crucial to remember that the FFT result is a complex array, and we are looking at the magnitude. This technique works well for stationary signals (those where the frequency doesn’t change significantly over time). For non-stationary signals, you'd want to look into more advanced techniques, such as wavelet transforms (a great reference for this is *A Wavelet Tour of Signal Processing* by Mallat).

Lastly, when dealing with time series data, you can also try to use recurrent neural networks (RNN), specifically LSTMs (Long Short-Term Memory networks). Although this method can be an overkill for a simple sine wave, it shines when you are dealing with highly noisy or more complex periodic data and need to predict not just parameters, but the signal at future time points. LSTMs are trained on past time series data, and can capture temporal dependencies that are not readily apparent with the above two methods. An implementation could look like the following:

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class SineWaveLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SineWaveLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
      lstm_out, hidden = self.lstm(input, hidden)
      output = self.fc(lstm_out.view(len(input), -1))
      return output, hidden

def create_dataset(time_series, seq_length):
    X = []
    y = []
    for i in range(len(time_series) - seq_length):
      X.append(time_series[i:i+seq_length])
      y.append(time_series[i+seq_length])
    return torch.tensor(X, dtype=torch.float).unsqueeze(2), torch.tensor(y, dtype=torch.float)

t = np.linspace(0, 10, 200)
A_true = 1.8
f_true = 2.5
phi_true = 0.8
y_true = A_true * np.sin(2 * np.pi * f_true * t + phi_true)
noise = 0.3 * np.random.normal(size=len(t))
y_noisy = y_true + noise

seq_length = 20

X, y = create_dataset(y_noisy, seq_length)

input_size = 1
hidden_size = 50
output_size = 1

model = SineWaveLSTM(input_size, hidden_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

num_epochs = 200

for epoch in range(num_epochs):
  hidden = (torch.zeros(1, 1, hidden_size), torch.zeros(1, 1, hidden_size))
  optimizer.zero_grad()
  output, hidden = model(X, hidden)
  loss = criterion(output.squeeze(), y)
  loss.backward()
  optimizer.step()
  if (epoch+1) % 50 == 0:
     print(f"Epoch: {epoch+1} loss: {loss.item():.4f}")

model.eval()
hidden = (torch.zeros(1, 1, hidden_size), torch.zeros(1, 1, hidden_size))
with torch.no_grad():
  predicted_values, hidden = model(X, hidden)
  predicted_values = predicted_values.squeeze().detach().numpy()

plt.plot(t[seq_length:], y[seq_length:].detach().numpy(), label = 'Actual Data')
plt.plot(t[seq_length:], predicted_values, label = 'Predicted Data')
plt.legend()
plt.show()
```

In this snippet, we first set up an LSTM neural network, and then perform the training. Finally, we then predict the values with our trained model. As mentioned, RNNs are powerful when your data is complex, or you need to predict the signal at further time points. An in-depth exploration of neural networks is found in *Deep Learning* by Goodfellow et al.

In summary, predicting a sine wave is about understanding its underlying parameters, and then using suitable methods to estimate them. Curve fitting using `scipy.optimize.curve_fit` is a good starting point if you have enough data. Spectral analysis through FFT is great for getting a feel for the frequency content. Finally, RNNs excel when you need to model more complex periodic data. Choose your approach based on the specific characteristics of your data and the goal of your analysis. Remember to always validate your model on a held-out dataset to ensure it generalizes well.
