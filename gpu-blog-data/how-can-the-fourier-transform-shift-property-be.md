---
title: "How can the Fourier Transform shift property be implemented in PyTorch?"
date: "2025-01-30"
id: "how-can-the-fourier-transform-shift-property-be"
---
The core property of the Fourier Transform, that a shift in the time domain corresponds to a linear phase ramp in the frequency domain, is a critical concept to understand for signal processing and image manipulation. Implementing this shift property efficiently in PyTorch, given its tensor-centric architecture, requires careful consideration of the framework's capabilities and limitations.

My experience working with seismic data analysis frequently involved manipulating signals where accurate time delays were crucial. Often, data would arrive with subtle variations in timing, requiring us to correct them before subsequent processing. Direct time domain shifts through interpolation were computationally expensive and introduced artifacts. Implementing this in PyTorch required understanding both the underlying mathematical concepts of the Fourier Transform, and how PyTorch handles complex numbers, which are inherent in the result of the DFT.

The Fourier Transform shift property states that if *x(t)* has a Fourier Transform *X(f)*, then *x(t - τ)* has a Fourier Transform *X(f)e^(-j2πfτ)*, where τ represents the time shift and *j* is the imaginary unit. Effectively, a shift in time, *τ*, introduces a linear phase shift in the frequency domain, scaled proportionally to the frequency *f*. Implementing this in PyTorch therefore requires two primary steps: first, performing the Fourier Transform, and second, applying the appropriate complex exponential phase ramp.

The Fast Fourier Transform (FFT) algorithm is what PyTorch uses for its FFT implementations, and for 1D signals this is exposed through `torch.fft.fft` and `torch.fft.ifft` for forward and inverse transforms respectively. When using these functions, we need to pay attention to the `dim` argument, indicating along which axis to perform the transform. PyTorch's `torch.complex` constructs complex numbers from real and imaginary components, and `torch.exp` provides the complex exponential. These tools, in combination with creating an appropriate frequency vector, are what we need to implement the shift.

**Code Example 1: 1D Signal Shift**

This example demonstrates a shift on a simple 1D signal, simulating a time delay on a waveform.

```python
import torch
import numpy as np

def shift_1d(signal, shift_samples, sample_rate):
  """
    Shifts a 1D signal in the time domain using the Fourier Transform.

    Args:
        signal (torch.Tensor): The input signal (1D tensor).
        shift_samples (int): The number of samples to shift (positive for right shift).
        sample_rate (float): The sampling rate of the signal.

    Returns:
        torch.Tensor: The shifted signal (1D tensor).
    """
  n = signal.shape[-1] # Length of the signal
  freqs = torch.fft.fftfreq(n, d=1/sample_rate).to(signal.device) # Frequency vector
  phase_ramp = torch.exp(-1j * 2 * torch.pi * freqs * shift_samples/sample_rate) # Phase ramp
  
  signal_fft = torch.fft.fft(signal) # Forward FFT
  shifted_fft = signal_fft * phase_ramp # Apply phase shift
  shifted_signal = torch.fft.ifft(shifted_fft) # Inverse FFT

  return shifted_signal.real

# Example Usage:
sample_rate = 1000.0
duration = 1.0
t = torch.linspace(0, duration, int(sample_rate * duration))
signal = torch.sin(2*torch.pi*10*t).float()

shift = 20 # 20 samples shift (20ms)

shifted_signal = shift_1d(signal, shift, sample_rate)

# For plotting purposes (outside the scope of the core implementation)
#  import matplotlib.pyplot as plt
#  plt.figure()
#  plt.plot(t, signal, label='Original Signal')
#  plt.plot(t, shifted_signal, label='Shifted Signal')
#  plt.legend()
#  plt.show()
```
In this first code block the key is calculating the phase ramp. We calculate frequency bins using `torch.fft.fftfreq`, which returns normalized frequencies and then scale those frequencies by the sampling rate.  This array then gets used with the desired sample shift to create the complex exponent that causes the phase shift. Finally, we take the inverse FFT, selecting the real component as we expect our input signal and the corresponding time shift to also be real valued.

**Code Example 2: 2D Image Shift**

This example extends the concept to 2D image shifting, relevant to situations such as aligning astronomical images where sub-pixel shifts are common.

```python
import torch
import numpy as np

def shift_2d(image, shift_pixels, sample_rate):
  """
  Shifts a 2D image in the spatial domain using the Fourier Transform.

    Args:
        image (torch.Tensor): The input image (2D tensor).
        shift_pixels (tuple): The pixel shift for each dimension (row, col)
        sample_rate (float): Sampling rate or spatial frequency of each pixel.
            Used here as a constant scaling on the fourier frequencies.

    Returns:
        torch.Tensor: The shifted image (2D tensor).
  """
  rows, cols = image.shape
  freqs_row = torch.fft.fftfreq(rows, d=1/sample_rate).to(image.device) # Frequency vector
  freqs_col = torch.fft.fftfreq(cols, d=1/sample_rate).to(image.device) # Frequency vector
    
  freqs_row_mesh, freqs_col_mesh = torch.meshgrid(freqs_row, freqs_col, indexing='ij') # Create coordinate grid for frequency
  phase_ramp = torch.exp(-1j * 2 * torch.pi * (freqs_row_mesh * shift_pixels[0]/sample_rate + freqs_col_mesh * shift_pixels[1]/sample_rate))

  image_fft = torch.fft.fft2(image) # Perform the 2D FFT
  shifted_fft = image_fft * phase_ramp # Apply the phase shift
  shifted_image = torch.fft.ifft2(shifted_fft) # Inverse 2D FFT

  return shifted_image.real

# Example Usage
image = torch.zeros((64, 64))
image[30:34, 30:34] = 1
shift = (5, 10) # 5 pixels down, 10 pixels right
sample_rate = 1.0

shifted_image = shift_2d(image.float(), shift, sample_rate)

# For plotting purposes:
# import matplotlib.pyplot as plt
# fig, axes = plt.subplots(1, 2)
# axes[0].imshow(image)
# axes[0].set_title('Original Image')
# axes[1].imshow(shifted_image)
# axes[1].set_title('Shifted Image')
# plt.show()
```
Here, we extend the 1D case to a 2D version. Two-dimensional fourier transforms are done with `torch.fft.fft2` and `torch.fft.ifft2`. The crucial change is the creation of a 2D mesh grid of frequencies with `torch.meshgrid` so that the correct phase is applied for each spatial frequency in both axes. Again we select the real portion of the result.

**Code Example 3: Batched Shifting**

Often, we have to deal with multiple signals at once (batched signals). Here, we demonstrate how to implement this efficiently with PyTorch.

```python
import torch

def shift_batched(signals, shift_samples, sample_rate):
    """
        Shifts a batch of 1D signals in the time domain using the Fourier Transform.

    Args:
        signals (torch.Tensor): A batch of input signals (N, L), where N is the batch size and L is signal length.
        shift_samples (int): The number of samples to shift (positive for right shift).
        sample_rate (float): The sampling rate of the signal.

    Returns:
        torch.Tensor: The batch of shifted signals.
    """
    n = signals.shape[-1] # Length of the signal
    freqs = torch.fft.fftfreq(n, d=1/sample_rate).to(signals.device) # Frequency vector
    phase_ramp = torch.exp(-1j * 2 * torch.pi * freqs * shift_samples/sample_rate) # Phase ramp

    signals_fft = torch.fft.fft(signals, dim=-1) # Forward FFT along the last dimension
    shifted_fft = signals_fft * phase_ramp # Apply phase shift
    shifted_signals = torch.fft.ifft(shifted_fft, dim=-1) # Inverse FFT along the last dimension

    return shifted_signals.real

# Example Usage:
batch_size = 3
sample_rate = 1000.0
duration = 1.0
t = torch.linspace(0, duration, int(sample_rate * duration))
signals = torch.stack([torch.sin(2*torch.pi*freq*t).float() for freq in [10, 20, 30]]) # Batch of signals

shift = 20 # 20 samples shift

shifted_signals = shift_batched(signals, shift, sample_rate)


# For plotting purposes:
# import matplotlib.pyplot as plt
# plt.figure()
# for i in range(batch_size):
#     plt.plot(t, signals[i], label=f'Original Signal {i}')
#     plt.plot(t, shifted_signals[i], label=f'Shifted Signal {i}')
# plt.legend()
# plt.show()

```
This example uses a stack of signals in the first dimension and the shifts are all applied in the last dimension with  `dim=-1` in the FFT functions. All the operations are performed in a vectorized fashion by PyTorch.

These examples highlight that while the core concept of the Fourier shift property is consistent, its implementation in PyTorch involves attention to detail, especially concerning dimensionality, complex number handling, and efficient vectorization. Understanding `torch.fft.fftfreq`, `torch.fft.fft`, `torch.fft.ifft`, and broadcasting is fundamental for performing the required operations correctly.

For further study, I recommend researching the properties of discrete Fourier transform, such as its linearity, time-shift, and scaling characteristics, and reading up on the fast Fourier transform (FFT) algorithm. Books on digital signal processing provide invaluable insights into the theoretical basis of these techniques. Also reviewing documentation from `torch.fft` will demonstrate what other operations are exposed. Familiarity with complex number operations in Python is also beneficial.
