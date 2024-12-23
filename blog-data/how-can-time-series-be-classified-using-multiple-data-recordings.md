---
title: "How can time series be classified using multiple data recordings?"
date: "2024-12-23"
id: "how-can-time-series-be-classified-using-multiple-data-recordings"
---

Alright, let’s dive into multi-variate time series classification. I’ve tackled this beast more than a few times, often in contexts where a single data stream just didn’t tell the whole story. Think about, for example, trying to classify the behavior of a complex machine based on readings from multiple sensors: temperature, vibration, power consumption, and perhaps even pressure. One sensor on its own could be misleading, but combined, they form a richer, more complete picture.

The fundamental challenge here is that each time series, by itself, represents a sequence of values ordered over time. Now, when you’re dealing with multiple of these, the problem space expands dramatically. You have to consider the temporal relationships within *each* individual time series, and the cross-correlations and interactions *between* the different time series at various points in time. Simple methods, like just flattening the data into a long vector, usually don’t cut it because they throw away crucial temporal and inter-series information.

So, let's look at a few techniques I've found particularly useful, along with some practical implementations in Python. I'm going to focus on methods that respect the temporal structure, since that's the key.

**1. Feature-Based Classification with Time Series Features:**

Instead of treating the raw data points as individual features, we can extract meaningful features from the time series. These could include things like mean, standard deviation, trend, auto-correlation, or frequency domain features (using Fourier transform). Once we’ve done this for each time series in each sample, we end up with a feature vector that *captures the core characteristics* of that set of time series, and we can feed this into a traditional classifier such as a support vector machine (SVM) or a random forest.

This approach has always been a go-to of mine because of its simplicity. It's relatively straightforward to implement and understand. The critical part is choosing the right features. Domain knowledge plays a huge role here. For example, if I’m working with vibration data, features like spectral entropy and peak frequency are crucial.

Here’s a basic Python example using `numpy` and `scikit-learn`. Let's assume we have data where each sample is a collection of two time series (series_1, series_2), each of length `n` and we have 100 samples total:

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.fft import fft

def extract_features(series_1, series_2):
    # Simple example: mean, std, and first frequency peak
    fft_series_1 = np.abs(fft(series_1))
    fft_series_2 = np.abs(fft(series_2))

    peak_freq_1 = np.argmax(fft_series_1[1:])
    peak_freq_2 = np.argmax(fft_series_2[1:])

    return np.array([
        np.mean(series_1),
        np.std(series_1),
        peak_freq_1,
        np.mean(series_2),
        np.std(series_2),
        peak_freq_2
        ])

# Sample Data Generation
np.random.seed(42)
n = 100
num_samples = 100
X = np.zeros((num_samples, 6))
y = np.random.randint(0, 2, num_samples) # Generate binary class labels
for i in range(num_samples):
    series_1 = np.random.randn(n) + (0.5 if y[i]==0 else 1)
    series_2 = np.random.randn(n) + (0.1 if y[i]==0 else 0.3)
    X[i, :] = extract_features(series_1, series_2)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(kernel='rbf')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
```

This example is basic, but it highlights the general approach. You can extend `extract_features` to include more sophisticated features or even use libraries like `tsfresh` for automatic feature extraction. It's a starting point and demonstrates the flexibility of this method.

**2. Time Series Classification with Dynamic Time Warping (DTW):**

DTW is especially good when you're dealing with time series that might be misaligned or have variable speeds. It calculates the optimal alignment between two time series, allowing for stretching or compressing one with respect to the other. In a classification context, DTW can be used as a distance measure within a *k*-nearest neighbors (KNN) algorithm. The distance to other series is used to classify a given set. It's conceptually different from feature extraction, as DTW operates directly on the raw time series, finding similar time series patterns even when they are phase-shifted.

I once used this for classifying human gait patterns recorded from multiple accelerometers, which varied in pace between different individuals. DTW’s capability to handle temporal distortions was critical to its success in accurately distinguishing different forms of gait abnormalities.

Here's a Python implementation for using DTW in a multi-variate time series scenario:

```python
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fastdtw import fastdtw

def dtw_distance(series_1, series_2):
    distance, _ = fastdtw(series_1, series_2, dist=euclidean)
    return distance


def multivariate_dtw_distance(series_collection_1, series_collection_2):
    # assuming each series_collection is a list of series (e.g., [series_1, series_2])
    total_distance = 0
    for s1, s2 in zip(series_collection_1, series_collection_2):
        total_distance += dtw_distance(s1, s2)
    return total_distance/len(series_collection_1) # Avg Distance

def knn_classifier(X_train, y_train, X_test, k=3):
    y_pred = []
    for test_series in X_test:
        distances = [multivariate_dtw_distance(test_series, train_series) for train_series in X_train]
        k_nearest_indices = np.argsort(distances)[:k]
        k_nearest_labels = y_train[k_nearest_indices]
        # Get majority class
        unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
        y_pred.append(unique_labels[np.argmax(counts)])
    return np.array(y_pred)



# Sample Data Generation
np.random.seed(42)
n = 100
num_samples = 100
X = []
y = np.random.randint(0, 2, num_samples)
for i in range(num_samples):
    series_1 = np.random.randn(n) + (0.5 if y[i]==0 else 1)
    series_2 = np.random.randn(n) + (0.1 if y[i]==0 else 0.3)
    X.append([series_1, series_2])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


y_pred = knn_classifier(X_train, y_train, X_test, k=3)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
```

This example shows how to extend DTW to multiple series by averaging the distance between individual series pairs. The `fastdtw` library provides an efficient way to calculate DTW.

**3. Deep Learning Architectures: Recurrent Neural Networks (RNNs):**

For more complex scenarios and especially with larger datasets, deep learning models like RNNs (particularly LSTMs and GRUs) can learn intricate temporal patterns. These networks process sequences sequentially, and they are great at handling dependencies between timesteps. By stacking multiple RNN layers, we can capture very high level features in the data. In multi-variate cases, you typically feed the different series in as separate channels, so the model will learn both within-channel and cross-channel dependencies. This method has proven quite powerful for a lot of time series problems including those involving complex multi-variate setups.

Here's an example using `TensorFlow/Keras`:

```python
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Sample Data Generation
np.random.seed(42)
n = 100
num_samples = 100
num_series = 2 # number of channels/series
X = np.zeros((num_samples, n, num_series))
y = np.random.randint(0, 2, num_samples)
for i in range(num_samples):
    X[i, :, 0] = np.random.randn(n) + (0.5 if y[i]==0 else 1)
    X[i, :, 1] = np.random.randn(n) + (0.1 if y[i]==0 else 0.3)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(units=50, input_shape=(n, num_series), return_sequences=False), # set return_sequences=True for another LSTM
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, verbose=0) # turn off verbosity during training
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
```

This code gives an example of a simple LSTM network working on multivariate time series data. Remember, optimizing model architecture and parameters can greatly improve results.

**Closing Thoughts**

These are three methods that have served me well across different scenarios. For a more theoretical understanding of time series analysis and classification, I would recommend starting with *Time Series Analysis* by James D. Hamilton, it provides a rigorous mathematical framework. For a more practical view with a focus on machine learning methods, *Hands-On Time Series Analysis with Python* by B. M. Fayssal is extremely helpful. And finally, if you wish to go deep into the world of sequence models, then *Deep Learning with Python* by Francois Chollet is invaluable.

The right approach, as always, comes down to the specifics of the problem: the size of your dataset, the nature of your time series, and the level of accuracy you require. Always keep experimenting and iterate to find the best approach for your data.
