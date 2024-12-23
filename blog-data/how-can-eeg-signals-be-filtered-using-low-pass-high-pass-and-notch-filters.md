---
title: "How can EEG signals be filtered using low-pass, high-pass, and notch filters?"
date: "2024-12-23"
id: "how-can-eeg-signals-be-filtered-using-low-pass-high-pass-and-notch-filters"
---

Alright, let's talk about cleaning up EEG data. I've spent more hours than I care to count staring at noisy EEG traces, and let me tell you, filtering is absolutely critical to getting anything meaningful out of those signals. It’s not just about applying a canned function; it’s about understanding the *why* behind each filter type and its limitations. Specifically, let’s unpack low-pass, high-pass, and notch filtering techniques as applied to electroencephalography.

First, before we even touch a filter, it's important to remember that raw EEG is inherently messy. It's a superposition of various neural oscillations, artifacts from muscle movement, eye blinks, and even power line interference. Our goal is to isolate the frequency bands relevant to our research or clinical goals while removing as much of the noise as possible.

Let’s kick things off with *low-pass filtering*. The core concept here is straightforward: we're attenuating high-frequency components in the signal while allowing lower frequencies to pass through relatively unattenuated. Think of it as smoothing out the high-frequency 'jitter' that doesn't contain much neurological information, or that is simply outside the range of what we're interested in.

In my experience, I’ve most commonly used low-pass filters when looking at slower brain rhythms, such as delta (typically below 4 hz) and theta (4-7 hz). Higher frequencies are generally less prominent at scalp level, but they can still introduce unwanted fluctuations if we’re interested in slower activity. For instance, in a project exploring sleep stages, high-frequency muscle artifacts would just add noise without providing much information. Typically, for such cases, a low-pass cutoff around 30-40hz is a good starting point.

Implementing a low-pass filter, we might use a Butterworth filter. This type of filter offers a good balance between a flat passband response (meaning the frequencies that we want don't get overly changed) and a decent roll-off in the stopband (the frequencies we want to remove). Here’s how that could look in python using scipy:

```python
import numpy as np
from scipy.signal import butter, filtfilt

def apply_lowpass(data, cutoff, fs, order=5):
  """Applies a low-pass Butterworth filter to the input data.

  Args:
    data: 1D numpy array representing EEG signal.
    cutoff: Cutoff frequency in Hz.
    fs: Sampling rate of the EEG data in Hz.
    order: Order of the Butterworth filter (higher order = sharper filter).

  Returns:
    Filtered data as a 1D numpy array.
  """
  nyquist = 0.5 * fs
  normal_cutoff = cutoff / nyquist
  b, a = butter(order, normal_cutoff, btype='low', analog=False)
  filtered_data = filtfilt(b, a, data) #zero-phase filtering
  return filtered_data

# Example Usage:
fs = 250  # Sampling rate
time = np.arange(0, 10, 1/fs)  # 10 seconds of data
signal = np.sin(2 * np.pi * 5 * time) + np.sin(2 * np.pi * 40 * time) + np.random.normal(0, 0.5, len(time)) #5 Hz and 40Hz plus noise
cutoff_freq = 30
filtered_signal = apply_lowpass(signal, cutoff_freq, fs)
```

In the snippet above, the `filtfilt` function is crucial because it applies the filter forward and then backward, eliminating phase distortion (which is important for time-sensitive analysis). This is also referred to as zero-phase filtering. The 'order' parameter dictates the sharpness of the filter; higher orders offer steeper roll-offs but can also introduce more ringing, so it’s generally something to adjust thoughtfully.

Now let’s shift to *high-pass filters*. These act in the opposite manner: they attenuate low-frequency components while passing higher ones. In my experience, a primary use case of high-pass filtering is to remove slow drifts and offsets that are common in EEG recordings – the kinds of slow fluctuations that may stem from electrode polarization or slow changes in skin potential that have nothing to do with neuronal activity. These baseline drifts can interfere with the analysis of faster brain signals.

For example, imagine you're trying to detect alpha rhythms (8-13 Hz), but you have very slow changes in your signal drifting below that range. A high-pass filter with a cutoff frequency of around 1 Hz would be effective in removing most of this drift. It’s important, however, to be mindful that very aggressive high-pass filtering can also remove legitimate low-frequency neural activity, like slow cortical potentials, so this step needs to be approached with care depending on the nature of the data and research question.

Here is the python implementation:

```python
import numpy as np
from scipy.signal import butter, filtfilt

def apply_highpass(data, cutoff, fs, order=5):
  """Applies a high-pass Butterworth filter to the input data.

  Args:
    data: 1D numpy array representing EEG signal.
    cutoff: Cutoff frequency in Hz.
    fs: Sampling rate of the EEG data in Hz.
    order: Order of the Butterworth filter (higher order = sharper filter).

  Returns:
    Filtered data as a 1D numpy array.
  """
  nyquist = 0.5 * fs
  normal_cutoff = cutoff / nyquist
  b, a = butter(order, normal_cutoff, btype='high', analog=False)
  filtered_data = filtfilt(b, a, data)
  return filtered_data

# Example Usage
fs = 250  # Sampling rate
time = np.arange(0, 10, 1/fs)  # 10 seconds of data
signal = np.sin(2 * np.pi * 0.5 * time) + np.sin(2 * np.pi * 10 * time) + np.random.normal(0, 0.5, len(time)) # 0.5 Hz and 10 Hz signal + noise
cutoff_freq = 1
filtered_signal = apply_highpass(signal, cutoff_freq, fs)
```

The code structure is largely similar to the low-pass implementation. The key difference is that we set `btype` to `high` when designing the butterworth filter. Once more, we are using `filtfilt` to maintain zero phase distortion.

Lastly, let's address *notch filtering*. This specific type of filter is used to attenuate a narrow band of frequencies. In EEG, one of the most common applications is removing the ubiquitous 50 or 60 Hz power-line noise. These artifacts, stemming from electrical wiring, can be very prominent and can significantly contaminate the desired brain signals. The notch filter is designed to target just those specific frequencies while leaving the other components intact.

In practice, I've found that a notch filter is almost always a necessity in an EEG pipeline, and it's crucial to apply it effectively without affecting the legitimate neural activity in the neighboring frequencies. The challenge here is to implement a narrow enough notch to target only the power line interference; otherwise, you might start affecting broader brain rhythms.

Here's how that looks in python:

```python
import numpy as np
from scipy.signal import butter, filtfilt

def apply_notch(data, notch_freq, fs, quality_factor=30):
  """Applies a notch filter to remove powerline interference.

    Args:
        data: 1D numpy array representing EEG signal.
        notch_freq: Center frequency of the notch (e.g., 50 or 60 Hz).
        fs: Sampling rate of the EEG data in Hz.
        quality_factor: Quality factor for the notch (higher = narrower notch).

    Returns:
        Filtered data as a 1D numpy array.
    """
  nyquist = 0.5 * fs
  bw = notch_freq / quality_factor
  low_cutoff = notch_freq - bw/2
  high_cutoff = notch_freq + bw/2

  b, a = butter(2, [low_cutoff/nyquist, high_cutoff/nyquist], btype='bandstop')
  filtered_data = filtfilt(b, a, data)

  return filtered_data


# Example Usage:
fs = 250  # Sampling rate
time = np.arange(0, 10, 1/fs)  # 10 seconds of data
signal = np.sin(2 * np.pi * 10 * time) + 0.5 * np.sin(2 * np.pi * 50 * time) + np.random.normal(0, 0.5, len(time)) #10 Hz and 50 Hz signal + noise
notch_freq = 50
filtered_signal = apply_notch(signal, notch_freq, fs)
```

Note that we are employing a bandstop filter in this function, rather than a notch filter explicitly, as the scipy library does not provide a notch function directly. We create a bandstop filter around the target frequency and define the 'quality factor' which controls the width of the 'notch' we intend to create to filter out noise.

It's important to realize these filters are not used in isolation, but combined strategically. In a typical EEG pre-processing pipeline, I’d first apply a high-pass filter to remove slow drifts, then a notch filter to address power-line noise, and finally, a low-pass filter to restrict the frequency range and reduce overall noise, for instance prior to applying ICA for further artifact removal. The specific cut-off frequencies are always dependent on the experimental question, data quality, and specific circumstances.

For further study, I would suggest reviewing "The Handbook of EEG Signal Processing" by Steven Luck for a broad understanding of EEG signal processing including various filtering techniques. "EEG Analysis: Methods and Applications" by Stefan Haufe and Matthias Treder can also be very helpful for a more detailed view of advanced methods. Specifically, look into the sections pertaining to digital filter design for more detail on the mathematics behind this process.

The proper application of filters isn't just a mechanical process; it requires a deep understanding of their effects on the signal. It’s an art form in itself: the art of clean signal extraction.
