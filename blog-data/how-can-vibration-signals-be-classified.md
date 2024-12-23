---
title: "How can vibration signals be classified?"
date: "2024-12-23"
id: "how-can-vibration-signals-be-classified"
---

Alright, let’s tackle the interesting world of vibration signal classification. I've spent quite a bit of time on this, notably back when I was working with predictive maintenance systems for heavy machinery. We were dealing with a ton of raw vibration data and needed to reliably identify the source of issues, like imbalances or bearing faults. It wasn't just about spotting deviations; it was about correctly *classifying* them. So, let's break down how it’s done.

Vibration signal classification primarily relies on transforming time-domain data into a more usable format and applying machine learning algorithms for pattern recognition. The initial, raw vibration data that you’d collect using accelerometers or other sensors is basically a record of amplitude fluctuations over time. It's rarely helpful in its raw form. It often appears noisy and lacks immediately interpretable features. Therefore, pre-processing and feature extraction are critical steps.

We typically start with signal conditioning, which might involve filtering to remove noise and unwanted frequencies. This could range from simple low-pass filters to more advanced methods such as wavelet denoising, depending on the specific application. The goal here is to isolate the relevant vibration signals from background noise. Then comes the crucial phase: feature extraction. There are two main routes we commonly take: time-domain and frequency-domain feature extraction.

In the time domain, we look at statistical parameters of the vibration signal. This includes:

*   **Root mean square (rms):** This gives you an idea of the overall energy level of the vibration. It’s calculated as the square root of the mean of the squared signal values.
*   **Peak values and peak-to-peak values:** These capture the extreme points of the signal and are useful for identifying sudden impacts or shocks.
*   **Crest factor:** This is the ratio of the peak value to the rms value. It’s useful for identifying impulsiveness. High crest factors often indicate impacts.
*   **Skewness and kurtosis:** These describe the shape of the signal's distribution. Skewness indicates asymmetry, while kurtosis measures the “tailedness” or peakedness of the distribution. They can help detect deviations from Gaussian noise.

In the frequency domain, we analyze the signal's frequency content. This involves transforming the time-domain signal into the frequency domain, primarily using the Fast Fourier Transform (FFT). After the FFT, we look at:

*   **Dominant frequencies:** These are the frequencies with the highest amplitude in the spectrum. They often correspond to the fundamental frequencies of rotating components or resonance frequencies of structures.
*   **Harmonic content:** The presence of harmonics (multiples of a fundamental frequency) can indicate issues such as misalignment or gear meshing problems.
*   **Band energy:** This is the energy within a specific frequency band. Monitoring band energy changes can be helpful for detecting specific component faults, e.g., bearing fault frequencies.

Once you've extracted these features (whether in the time or frequency domain, or both), they become the inputs for your machine learning classifier. Several types of algorithms work well for vibration classification. Some that I've had good results with are:

*   **Support vector machines (svm):** Effective for separating classes by finding optimal hyperplanes in the feature space. These tend to perform well with a moderate number of samples.
*   **K-nearest neighbors (knn):** A simple, non-parametric algorithm that classifies a data point based on the majority class among its 'k' nearest neighbors in the feature space. This is useful for exploratory analysis, although computational expense increases with sample size.
*   **Artificial neural networks (ann), particularly convolutional neural networks (cnn):** These are capable of learning complex patterns directly from the feature vectors, and are particularly well suited for time series data, although they typically require large datasets to perform effectively.

Now, let’s look at some code examples. These will be in Python, as it’s a common language in this space, leveraging libraries like `numpy`, `scipy`, and `scikit-learn`.

First, an example demonstrating time-domain feature extraction:

```python
import numpy as np
from scipy.stats import skew, kurtosis

def extract_time_domain_features(signal):
    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    peak_to_peak = np.max(signal) - np.min(signal)
    crest_factor = peak / rms if rms != 0 else 0  # prevent zero division
    sk = skew(signal)
    ku = kurtosis(signal)
    return rms, peak, peak_to_peak, crest_factor, sk, ku

# Example Usage
signal = np.random.randn(1000) # Simulate a signal
rms_val, peak_val, p2p_val, crest_val, skew_val, kurt_val = extract_time_domain_features(signal)
print(f"RMS: {rms_val:.3f}, Peak: {peak_val:.3f}, Peak-to-Peak: {p2p_val:.3f}, Crest: {crest_val:.3f}, Skew: {skew_val:.3f}, Kurtosis: {kurt_val:.3f}")
```

This shows you how to calculate the basic time-domain features directly from a signal array using `numpy` for mathematical operations and `scipy.stats` for the skew and kurtosis. Note how we’ve added a small check to prevent zero division for crest factor calculation.

Next, let's look at frequency domain feature extraction using the FFT:

```python
import numpy as np
from scipy.fft import fft, fftfreq

def extract_frequency_domain_features(signal, sampling_rate):
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1/sampling_rate)
    positive_freq = xf[:N//2]
    magnitudes = np.abs(yf[:N//2])
    dominant_frequency_idx = np.argmax(magnitudes)
    dominant_frequency = positive_freq[dominant_frequency_idx]
    return dominant_frequency, magnitudes, positive_freq

# Example Usage
sampling_rate = 1000
time = np.linspace(0, 1, sampling_rate, endpoint=False)
signal = np.sin(2 * np.pi * 50 * time) + 0.5 * np.sin(2 * np.pi * 150 * time)  # Simulate two freq components
dominant_freq, spectrum_values, frequencies = extract_frequency_domain_features(signal, sampling_rate)
print(f"Dominant Frequency: {dominant_freq:.2f} Hz")
# Optional: You can use matplotlib to plot the spectrum
# import matplotlib.pyplot as plt
# plt.plot(frequencies, spectrum_values)
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude')
# plt.show()
```

Here, we see how to perform the FFT and then identify the dominant frequency. We’re using `scipy.fft` here. Note how we isolate the positive frequency components as FFT results are symmetric around zero frequency.

Finally, let’s incorporate a simple SVM classifier using `scikit-learn`:

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Simulate some feature data and labels (replace with your extracted features)
features = np.random.rand(100, 6) # Simulate 100 samples with 6 features
labels = np.random.randint(0, 3, 100) # Simulate 3 classes

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Create an SVM classifier
classifier = SVC(kernel='linear', C=1.0)

# Train the classifier
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

This snippet shows a basic example of how to train an SVM classifier using synthetic data. Note that in a real-world application you’d use your feature arrays derived from the previous feature extraction steps.

For delving deeper, I would suggest starting with "Statistical Digital Signal Processing" by Monson H. Hayes for a good theoretical foundation. Then "Pattern Recognition and Machine Learning" by Christopher M. Bishop provides an excellent overview of machine learning algorithms suitable for classification. Specifically for signal processing in machinery diagnostics, "Vibration-Based Condition Monitoring: Industrial, Aerospace and Automotive Applications" by Robert B. Randall would be very beneficial.

In conclusion, classifying vibration signals is a multi-step process involving pre-processing, feature extraction, and machine learning. It's a powerful technique once you understand the process, and it has numerous applications, especially in the realm of predictive maintenance. The key is choosing the correct features to extract and finding the appropriate classification algorithm to suit the specific problem you are addressing. I hope this provides some useful insights from my experiences.
