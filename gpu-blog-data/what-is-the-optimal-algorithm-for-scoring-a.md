---
title: "What is the optimal algorithm for scoring a streaming classifier's time series probability?"
date: "2025-01-30"
id: "what-is-the-optimal-algorithm-for-scoring-a"
---
The core challenge in assessing a streaming classifier's performance on time-series probability lies in balancing computational efficiency with accuracy, particularly given that "optimal" often depends on specific application constraints. A naive approach of calculating accuracy on every single prediction is computationally wasteful and typically yields noisy metrics due to the fluctuating nature of streaming data and the inherent uncertainty in individual classifications. Instead, a composite scoring approach, focusing on trends and aggregate performance within sliding windows, offers a more robust and actionable evaluation.

My experience deploying real-time fraud detection systems has shown that directly evaluating the raw probabilities output by a streaming model is less useful than evaluating them within a temporal context. We're not as concerned with whether the model is *perfectly* calibrated at each individual prediction as we are concerned with the overall trend and effectiveness over a period. Therefore, I propose using a combination of metrics, tailored to the specific nature of the streaming task, rather than relying on a single, static score.

Let’s consider the problem of evaluating a classifier trained to detect anomalous network traffic. The model outputs, for each observation (say, a packet), the probability of it belonging to the 'anomalous' class. Simply averaging these probabilities doesn't provide meaningful information regarding the classifier’s effectiveness. Instead, it's essential to look at how these probabilities behave over time in relation to the known (ground truth) anomalies.

The suggested methodology involves these steps:

1.  **Sliding Window Aggregation:** Divide the continuous stream into overlapping or non-overlapping time windows (e.g., 1-minute intervals). Within each window, collect predicted probabilities and the corresponding ground truth labels (if available).

2.  **Probabilistic Binarization:** For each window, apply a threshold to the model’s predicted probabilities to derive binary predictions (0 or 1, representing non-anomalous or anomalous). This converts the probabilistic output into discrete classification. An adaptive threshold based on statistical characteristics of the recent prediction history could be used here. However, a fixed or slowly-changing threshold provides a good basis.

3.  **Window-Level Metrics Calculation:** Within each window, compute metrics such as:
    *   **Area Under the Curve (AUC):** This measures the model's ability to distinguish between anomalous and non-anomalous observations. Since the amount of true positives and false positives within a short window can be low, this should be considered along with other metrics.
    *   **Precision:** The proportion of identified positives which were truly positives. Useful when false positive costs are high.
    *   **Recall (Sensitivity):** The proportion of actual positives that were correctly identified. Useful when false negative costs are high.
    *   **F1-Score:** The harmonic mean of precision and recall, balancing between the two.
    *   **Mean Predicted Probability (for each class):** The average probability assigned to observations of a certain class within that window. Useful to detect drift.
    *  **Root Mean Squared Error (RMSE) of probabilities** (if the ground truth labels can be viewed as 'true probabilities', such as a 'probability of fraud'). This directly evaluates the accuracy of the probabilities.

4.  **Time Series Smoothing:** To reduce the noise of the metrics, smooth the time series of window-level metrics using techniques such as Exponential Moving Average (EMA).

5.  **Alarming & Monitoring:** Use these smoothed window-level metrics to set alarms based on defined thresholds. For example, trigger an alarm if the moving average of F1-score drops below a threshold.

Here are three code examples that demonstrate parts of this process:

**Example 1: Window Aggregation and Probabilistic Binarization**

```python
import numpy as np
from collections import deque
import time

def process_window(window_probs, window_labels, threshold=0.5):
    """Processes a window of predictions and labels."""
    binary_preds = np.array([1 if p > threshold else 0 for p in window_probs])
    return binary_preds

def stream_data(data_points, window_size, overlap):
    """Simulates streaming data with a sliding window."""
    window_queue = deque()
    labels_queue = deque()
    for i, (prob, label) in enumerate(data_points):
        window_queue.append(prob)
        labels_queue.append(label)
        if len(window_queue) == window_size:
            binary_predictions = process_window(list(window_queue), list(labels_queue))
            print(f"Time: {i}, Binary Predictions: {binary_predictions}")
            window_queue.popleft()
            labels_queue.popleft()
            if overlap == 0:
                window_queue.clear()
                labels_queue.clear()

# Example usage
data = [(0.1,0), (0.2,0),(0.7,1),(0.8,1),(0.3,0),(0.1,0),(0.9,1),(0.95,1),(0.2,0),(0.1,0)]
window_size = 4
overlap = 1
stream_data(data,window_size,overlap)
```

This example simulates a stream of prediction probabilities and corresponding labels, and creates a binary prediction based on a fixed threshold. The function `process_window` converts continuous probabilities into discrete class predictions by applying a threshold. The `stream_data` function iterates through the stream, and every time the window is full, it prints the calculated binary predictions. A more sophisticated approach would involve more advanced sliding window management (possibly with a non-constant or overlapping window) and more robust handling of partially completed windows.

**Example 2: Calculation of Window-Level Metrics**

```python
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, mean_squared_error

def calculate_metrics(binary_preds, true_labels, predicted_probs):
    """Calculates relevant metrics for a window."""
    auc = roc_auc_score(true_labels, predicted_probs)
    precision = precision_score(true_labels, binary_preds, zero_division=0)
    recall = recall_score(true_labels, binary_preds, zero_division=0)
    f1 = f1_score(true_labels, binary_preds, zero_division=0)
    rmse = np.sqrt(mean_squared_error(true_labels, predicted_probs))
    return auc, precision, recall, f1, rmse

# Example Usage (using data from previous example)
data = [(0.1,0), (0.2,0),(0.7,1),(0.8,1),(0.3,0),(0.1,0),(0.9,1),(0.95,1),(0.2,0),(0.1,0)]
window_size = 4
overlap = 1
window_queue = deque()
labels_queue = deque()

for i, (prob, label) in enumerate(data):
    window_queue.append(prob)
    labels_queue.append(label)
    if len(window_queue) == window_size:
        binary_preds = np.array([1 if p > 0.5 else 0 for p in window_queue])
        auc, precision, recall, f1, rmse = calculate_metrics(binary_preds, list(labels_queue), list(window_queue))
        print(f"AUC: {auc:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}, RMSE: {rmse:.2f}")
        window_queue.popleft()
        labels_queue.popleft()
        if overlap == 0:
            window_queue.clear()
            labels_queue.clear()
```

This snippet calculates AUC, precision, recall, F1-score, and RMSE within each time window based on the probabilities and corresponding labels. The `calculate_metrics` function makes use of Scikit-learn to compute these scores. A real system would need to be more cautious about situations where all predictions are in the same class, handling edge cases appropriately.

**Example 3: Exponential Moving Average Smoothing**

```python
def exponential_moving_average(data, alpha, previous_ema=None):
    """Applies Exponential Moving Average to a time series."""
    if previous_ema is None:
      previous_ema = data[0]
    ema_values = [previous_ema]
    for value in data[1:]:
        ema = alpha * value + (1 - alpha) * ema_values[-1]
        ema_values.append(ema)
    return ema_values

# Example Usage (using dummy AUC values)
auc_values = [0.6, 0.7, 0.5, 0.8, 0.9, 0.7, 0.6, 0.5]
alpha = 0.3
smoothed_auc = exponential_moving_average(auc_values, alpha)
print(f"Original AUC: {auc_values}")
print(f"Smoothed AUC: {smoothed_auc}")
```

This example demonstrates the application of the Exponential Moving Average (EMA) smoothing technique on a time series of AUC values. The `exponential_moving_average` function iterates through a series and calculates the smoothed values, given a smoothing factor (alpha). This helps in reducing noise in the measured metrics, providing a more stable view of the model performance. Choosing the correct value of alpha requires experimentation depending on the dataset.

In summary, there's no one-size-fits-all 'optimal' algorithm, but rather a collection of methodologies and considerations that should be taken into account. These include:

*   **Choosing the appropriate window size:** This should depend on the time scale of relevant events in the streaming data. Too short a window leads to noisy metrics, while a long window may obscure crucial information.
*   **Selecting the right metrics:** This varies based on the specific goal. For instance, focusing on recall might be essential in some scenarios (such as identifying rare but critical anomalies), while precision might be a greater concern in others (avoiding false alarms).
*  **Adaptive thresholding** The use of a fixed threshold is only a baseline method. Thresholds should be adaptive based on metrics of the recent history to account for changing environments.
*   **Proper implementation of smoothing:** It is important to correctly implement smoothing techniques to prevent information leakage (e.g. if smoothing is applied over a large period before being used for an alarm).
*   **Continuous refinement:** It’s necessary to continuously monitor and refine the chosen approach based on feedback from the live system.

For further study, I suggest research into time-series analysis techniques and adaptive thresholding methods. Books on statistical process control and online machine learning are also valuable resources. I have personally found that iterative experimentation with specific time-series properties yields better results than generic approaches.
