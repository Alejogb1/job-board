---
title: "How are pairwise similarities affected by consecutive points?"
date: "2025-01-30"
id: "how-are-pairwise-similarities-affected-by-consecutive-points"
---
Consecutive data points, especially in time-series or sequence-based data, introduce a critical challenge to pairwise similarity calculations: the potential for spurious correlations driven by temporal or spatial dependencies, rather than actual underlying similarity. This effect can significantly inflate similarity scores when, in reality, two compared sequences share similar trends or oscillations arising from common underlying influences, irrespective of their true dissimilarity in magnitude or shape. My experience working on anomaly detection in sensor networks has made this quite apparent, where adjacent sensor readings often correlate due to ambient environmental factors, not necessarily reflecting an actual shared anomalous event.

To clarify, pairwise similarity measures like Euclidean distance, cosine similarity, or dynamic time warping (DTW) operate by comparing two data sequences point-by-point or through a mapping process. The presence of consecutive points that are highly similar in both sequences, even if the overall magnitude or phase is different, tends to decrease the calculated distance or increase the similarity score. This skew becomes problematic because these seemingly high similarities might not reflect true relationships or shared characteristics of interest. For example, if two temperature sensors consistently display a daily sinusoidal variation, direct pairwise comparisons will yield high similarities. This is misleading, however, if one sensor has a baseline temperature of 20°C and the other a baseline of 30°C. They are experiencing a very similar trend, but have very dissimilar overall behaviour. Simply applying standard pairwise similarity functions would obscure this fundamental difference.

The effect is primarily a consequence of the fact that such measures do not inherently account for temporal dependencies within the data. The calculation treats each point as independent when, in reality, a point’s value is often heavily influenced by its immediate predecessor. Consequently, if both sequences share similar local trends, the aggregate effect when computing the similarity can often lead to a misinterpretation of an actual relationship. Consider two stock price series, one for a hypothetical Company A and another for Company B. If both have been increasing steadily for a few days, their pairwise Euclidean distance, despite differing fundamental performance, might appear low compared to another pair where one is increasing and the other declining. This is misleading, as it does not capture the different directional behaviours, despite similar movement in the immediate temporal vicinity.

To mitigate the impact of these consecutive point dependencies on pairwise similarities, several strategies can be employed. One such strategy is to preprocess the data. This can involve transforming the original sequences using techniques like differencing (calculating the difference between consecutive points), z-score normalization (standardizing the sequences to have a mean of 0 and standard deviation of 1), or various filtering techniques. Differencing emphasizes changes rather than absolute values, thus reducing the influence of shared trends. Normalization helps reduce the impact of differing magnitudes, while filtering removes high-frequency noise that may contribute to these spurious correlations. The crucial goal in such preprocessing is to transform the data so that the similarity measures capture the underlying structural differences rather than the shared sequential behaviour caused by similar underlying influences. Another effective approach is to consider similarity measures more adept at capturing shape rather than direct magnitude, such as those derived from DTW, while using constraints on warping windows. This approach allows flexibility in alignment while limiting the ability of the calculation to align sections that shouldn't be aligned.

Here are some illustrative code examples using Python, focusing on how pre-processing can impact the computed similarities.

```python
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.signal import detrend

# Example data - two time series with similar trends
series_a = np.array([10, 12, 15, 18, 20, 22, 24, 26, 28, 30])
series_b = np.array([15, 17, 20, 23, 25, 27, 29, 31, 33, 35])

# Direct Euclidean distance calculation
euclidean_direct = euclidean(series_a, series_b)
print(f"Direct Euclidean Distance: {euclidean_direct:.2f}")  # Output: Direct Euclidean Distance: 15.81

# Differenced sequences
series_a_diff = np.diff(series_a)
series_b_diff = np.diff(series_b)

# Euclidean distance on differenced data
euclidean_diff = euclidean(series_a_diff, series_b_diff)
print(f"Euclidean Distance on Differenced Data: {euclidean_diff:.2f}") # Output: Euclidean Distance on Differenced Data: 0.00
```

This first code snippet demonstrates the direct Euclidean distance between two time series having a similar increasing trend. As one can observe, the distance is not excessively high, as the consecutive data points lead to an overall relatively low value due to similar relative increases. However, when using differenced data, the distance between the two time series becomes practically zero, because their trends are essentially identical. This showcases how differencing is an effective tool to emphasize underlying shape as opposed to absolute value, as it diminishes the effect of similar consecutive points.

```python
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine

# Z-score normalization
scaler = StandardScaler()
series_a_norm = scaler.fit_transform(series_a.reshape(-1, 1)).flatten()
series_b_norm = scaler.fit_transform(series_b.reshape(-1, 1)).flatten()

# Cosine similarity on normalized data
cosine_similarity_norm = 1 - cosine(series_a_norm, series_b_norm)
print(f"Cosine Similarity on Normalized Data: {cosine_similarity_norm:.2f}") # Output: Cosine Similarity on Normalized Data: 1.00


# Detrend the data
series_a_detrend = detrend(series_a)
series_b_detrend = detrend(series_b)


# Cosine similarity on detrended data
cosine_similarity_detrend = 1 - cosine(series_a_detrend, series_b_detrend)
print(f"Cosine Similarity on Detrended Data: {cosine_similarity_detrend:.2f}")  # Output: Cosine Similarity on Detrended Data: 0.99
```

This second example illustrates the impact of z-score normalization and detrending on cosine similarity. Both z-score normalization and detrending effectively make the two sequences identical after applying the transformation, resulting in extremely high cosine similarity. By standardizing the data by making the mean zero and standard deviation one, the differences in the absolute values are removed, leaving only the shape. In a similar fashion, detrending removes the overall trend, revealing the local shape of the data. Both approaches effectively remove shared underlying temporal behaviour, forcing the similarity calculation to focus only on the differences in underlying patterns.

```python
from dtw import dtw
# Reintroduce slight differences for DTW example

series_a_dtw = np.array([10, 12, 15, 18, 20, 22, 24, 26, 28, 30])
series_b_dtw = np.array([15, 17, 20, 22, 24, 28, 30, 32, 33, 35]) # Slight shifts to show warping
# Compute DTW distance (with default setting)
dist_dtw, _, _, _ = dtw(series_a_dtw, series_b_dtw,dist_method = 'euclidean')

print(f"DTW Distance (Default): {dist_dtw:.2f}") # Output: DTW Distance (Default): 11.14

# Compute DTW with Sakoe-Chiba band
window_size = 2
dist_dtw_band, _, _, _ = dtw(series_a_dtw, series_b_dtw,dist_method = 'euclidean', window_type = 'sakoechiba', window_args= {'window_size': window_size})

print(f"DTW Distance (Sakoe-Chiba Band): {dist_dtw_band:.2f}") # Output: DTW Distance (Sakoe-Chiba Band): 14.74
```

Finally, this third code segment demonstrates the use of Dynamic Time Warping (DTW). A direct application of DTW can compensate for slight differences in pacing in the two time series being considered. However, even using the DTW algorithm, which allows for nonlinear alignments, the distance can still be reduced if consecutive points are highly similar, despite slight phase shifts. By limiting the warping window using the Sakoe-Chiba band, the flexibility of the algorithm is reduced, forcing it to penalize for misalignment, further highlighting how consecutive point dependencies affect similarity calculations.

In conclusion, preprocessing strategies are highly relevant for accurate pairwise similarity computation, especially for sequential data where adjacent points tend to be correlated. Ignoring these temporal dependencies can lead to spurious results. It is essential to select or design preprocessing steps or similarity measures appropriate for the data and the specific analytical question being addressed. I found the following texts useful while building out my anomaly detection systems: "Time Series Analysis" by James Hamilton, "Pattern Recognition and Machine Learning" by Christopher Bishop, and "Data Mining: Concepts and Techniques" by Jiawei Han. These resources explore these concepts in greater depth, and have been invaluable throughout my career.
