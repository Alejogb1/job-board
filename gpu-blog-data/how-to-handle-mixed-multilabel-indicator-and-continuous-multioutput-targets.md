---
title: "How to handle mixed multilabel-indicator and continuous-multioutput targets in classification metrics?"
date: "2025-01-30"
id: "how-to-handle-mixed-multilabel-indicator-and-continuous-multioutput-targets"
---
The core challenge in evaluating models with mixed multilabel-indicator and continuous-multioutput targets lies in the incompatibility of standard evaluation metrics designed for single target types.  Directly applying metrics like accuracy or F1-score, suited for multilabel classification, to a dataset containing both categorical and continuous outputs will yield meaningless results.  My experience working on anomaly detection in high-frequency trading datasets highlighted this issue acutely. We needed a system capable of identifying specific anomaly *types* (multilabel-indicator) alongside the magnitude of the anomalous event (continuous-multioutput). This necessitated a composite evaluation strategy.

**1.  A Composite Metric Approach:**

The optimal approach involves decomposing the evaluation into separate components, one for the multilabel classification task and another for the multioutput regression task.  This allows us to assess model performance on each aspect independently and then combine the results into a composite score, weighted according to their relative importance in the specific application.  For instance, in my trading application, correctly identifying the anomaly type (e.g., flash crash, order book imbalance) was prioritized over the precision of the magnitude prediction.  Therefore, the classification component received a higher weight in the composite score.

The choice of individual metrics within each component depends on the characteristics of the data and the desired properties of the evaluation.  Common choices include:

* **Multilabel Classification:**  Hamming loss, subset accuracy, F1-macro, precision@k, and recall@k are suitable options.  Hamming loss measures the average number of misclassified labels per instance. Subset accuracy checks if the predicted set of labels perfectly matches the true set.  F1-macro averages the F1-score across all labels, providing a balanced measure. Precision@k and Recall@k focus on the top k predicted labels.

* **Multioutput Regression:** Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) are widely used.  MSE and RMSE penalize larger errors more heavily, while MAE provides a more robust measure against outliers.  The selection often depends on the distribution of the continuous targets and the sensitivity to outliers.  A robust metric like MAE might be preferable if the continuous outputs exhibit heavy-tailed distributions.

The composite score can then be calculated as a weighted average:

`Composite Score = w_c * Classification_Metric + w_r * Regression_Metric`

where `w_c` and `w_r` are weights summing to 1, reflecting the relative importance of classification and regression accuracy. These weights are typically determined based on domain expertise and the specific application requirements.  Sensitivity analysis on different weight combinations can further refine the evaluation.


**2. Code Examples with Commentary:**

The following Python code snippets illustrate this composite evaluation approach using Scikit-learn.

**Example 1:  Using Hamming Loss and MAE**

```python
import numpy as np
from sklearn.metrics import hamming_loss, mean_absolute_error

# Sample data (replace with your actual data)
y_true_multilabel = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
y_pred_multilabel = np.array([[1, 0, 0], [0, 1, 1], [1, 0, 0]])
y_true_continuous = np.array([[2.5], [1.2], [3.8]])
y_pred_continuous = np.array([[2.2], [1.5], [4.1]])

# Calculate individual metrics
hamming_loss_score = hamming_loss(y_true_multilabel, y_pred_multilabel)
mae_score = mean_absolute_error(y_true_continuous, y_pred_continuous)

# Define weights (adjust as needed)
wc = 0.7
wr = 0.3

# Calculate composite score
composite_score = wc * (1 - hamming_loss_score) + wr * (1 - (mae_score / np.max(y_true_continuous)))  #Normalize MAE

print(f"Hamming Loss: {hamming_loss_score:.4f}")
print(f"MAE: {mae_score:.4f}")
print(f"Composite Score: {composite_score:.4f}")

```
This example utilizes Hamming loss for the multilabel portion and MAE for the continuous portion.  The MAE is normalized by dividing by the maximum value in the true continuous target to ensure both metrics contribute to the composite score on a comparable scale.


**Example 2:  Employing F1-macro and RMSE**

```python
from sklearn.metrics import f1_score, mean_squared_error

# ... (same sample data as Example 1) ...

# Calculate individual metrics
f1_macro_score = f1_score(y_true_multilabel, y_pred_multilabel, average='macro')
rmse_score = np.sqrt(mean_squared_error(y_true_continuous, y_pred_continuous))

# Define weights (adjust as needed)
wc = 0.6
wr = 0.4

# Calculate composite score (RMSE normalization not strictly necessary here, but good practice)
composite_score = wc * f1_macro_score + wr * (1 - (rmse_score / np.max(y_true_continuous)))

print(f"F1-macro: {f1_macro_score:.4f}")
print(f"RMSE: {rmse_score:.4f}")
print(f"Composite Score: {composite_score:.4f}")
```

Here, F1-macro and RMSE are used.  The choice between MAE and RMSE depends on your priorities; RMSE is more sensitive to larger errors.  Again, normalization is included to ensure comparable scales.


**Example 3:  Handling Imbalanced Datasets (Classification)**

```python
from sklearn.metrics import classification_report

# ... (sample data with class imbalance – more instances of one label) ...

# Classification report provides precision, recall, and F1 for each label
report = classification_report(y_true_multilabel, y_pred_multilabel)
print(report)

# Extract F1-weighted score (accounts for class imbalance)
weighted_f1 = float(report.split('weighted avg       f1-score      ')[1].split('\n')[0])

# Combine with regression metric (e.g., MAE)
# ... (rest of the composite score calculation as in previous examples) ...
```
In situations with class imbalance in the multilabel component, using F1-weighted instead of F1-macro ensures that the score appropriately reflects the performance across all classes. This snippet highlights the integration of the `classification_report` for a more comprehensive understanding of the multilabel classifier's performance before calculating the composite score.



**3. Resource Recommendations:**

For deeper understanding of multilabel classification metrics, consult the scikit-learn documentation and relevant chapters in machine learning textbooks by authors like Hastie, Tibshirani, and Friedman, or Bishop.  For detailed treatments of regression evaluation, similar resources focusing on statistical modeling and regression analysis are recommended.  Additionally, exploring research papers on multi-task learning will provide further insights into handling multiple intertwined prediction tasks.  Remember that the optimal composite score is problem-specific and its components need careful selection based on the specific requirements of your task.
