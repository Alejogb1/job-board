---
title: "How do plot confusion matrices differ across all samples?"
date: "2025-01-30"
id: "how-do-plot-confusion-matrices-differ-across-all"
---
The interpretation of confusion matrices can vary significantly depending on whether they are aggregated across all samples or considered individually for each sample in a multi-sample dataset, especially when dealing with datasets exhibiting class imbalance or inter-sample variability. Ignoring this distinction can lead to misleading performance assessments and compromised model interpretations. I've encountered this issue numerous times while working with multi-patient medical datasets, where disease prevalence and diagnostic criteria can differ across individuals, making this particular nuance of confusion matrix analysis quite critical.

When we compute a *single*, aggregated confusion matrix for all samples, we are essentially pooling all individual predictions and true labels together before evaluating performance. This gives us an overall view of model performance across the entire dataset, which is particularly useful for gauging broad trends. However, this approach effectively assumes a uniform data distribution and a uniform model behaviour across all samples. The total counts in an aggregate confusion matrix reflect all classifications, irrespective of their origins. Specifically, this involves treating all samples equally in an aggregation of the results of any classifier. This is the typical approach to model evaluation, especially in situations where we do not have reasons to assume intersample differences are relevant.

In contrast, focusing on *individual* sample confusion matrices provides a more granular understanding of the model's behaviour. This involves creating separate confusion matrices for each sample (e.g., each patient, each experimental condition) and then analyzing each matrix independently. This finer-grained view allows us to identify specific areas where the model struggles or excels within a specific sample. The result highlights individual sample behaviour. This can be particularly useful in assessing the effects of specific factors across a dataset. It is crucial in cases where data heterogeneity is present, where assuming identical performance between samples is inappropriate.

The aggregate matrix will be misleading when class distributions or model behaviour varies across samples. Specifically, samples that make up larger proportions of the dataset will have a disproportionate effect on any metrics calculated from the aggregate confusion matrix. For instance, a model may have excellent performance on a majority class and may have a poor performance on a minority class; in an aggregated confusion matrix, the performance on the minority class may be hidden from view or averaged away, whereas individual samples will show clearly which sample is problematic. By creating individual matrices, it becomes possible to account for these differences.

Let's illustrate this concept with code examples. I'll use Python and the `sklearn` library for demonstration since that’s a common environment for many users.

**Example 1: Generating Aggregate Confusion Matrix**

```python
import numpy as np
from sklearn.metrics import confusion_matrix

# Simulate predictions and true labels across three samples (imagine patients)
true_labels = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
predicted_labels = np.array([0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0])
sample_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])  # Each 'sample' has 4 data points

# Generate aggregate confusion matrix
aggregate_cm = confusion_matrix(true_labels, predicted_labels)
print("Aggregate Confusion Matrix:\n", aggregate_cm)

# Calculate overall accuracy
overall_accuracy = np.trace(aggregate_cm) / np.sum(aggregate_cm)
print(f"Overall Accuracy: {overall_accuracy:.2f}")
```

In this snippet, `true_labels` and `predicted_labels` represent the true and predicted class for a binary classification task, and `sample_ids` corresponds to which of three samples each prediction and label refers to. The code generates a single aggregate confusion matrix based on *all* the labels and predictions. Note that all predictions from all the samples are treated as one data set; we are *not* analysing individual samples. As such, any metric derived from it is an aggregate performance metric.

**Example 2: Generating and Evaluating Individual Confusion Matrices**

```python
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

# Same sample data as above
true_labels = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
predicted_labels = np.array([0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0])
sample_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])


unique_sample_ids = np.unique(sample_ids)

# Initialize lists to store results from individual sample confusion matrices
sample_cms = []
sample_accuracies = []


for sample_id in unique_sample_ids:
    sample_true_labels = true_labels[sample_ids == sample_id]
    sample_predicted_labels = predicted_labels[sample_ids == sample_id]
    sample_cm = confusion_matrix(sample_true_labels, sample_predicted_labels)
    sample_cms.append(sample_cm)
    sample_accuracy = accuracy_score(sample_true_labels, sample_predicted_labels)
    sample_accuracies.append(sample_accuracy)
    print(f"Confusion Matrix Sample {sample_id}:\n{sample_cm}")
    print(f"Accuracy of Sample {sample_id}: {sample_accuracy:.2f}")

# Sample cms stores individual matrices, and sample accuracies stores individual accuracues
# We can also use sample accuracies or confusion matrices to understand variance across samples
# For example:
average_accuracy = np.mean(sample_accuracies)
std_accuracy = np.std(sample_accuracies)
print(f"\nAverage Accuracy Across Samples: {average_accuracy:.2f}")
print(f"Standard Deviation of Accuracy Across Samples: {std_accuracy:.2f}")
```

Here, the code iterates through each unique `sample_id`, extracts the corresponding labels and predictions, and then computes and displays a separate confusion matrix and accuracy for each sample. This approach highlights the performance of the model on each individual sample. As can be seen in the example, both confusion matrices and accuracies differ between samples.

**Example 3: A Class Imbalance Scenario**

```python
import numpy as np
from sklearn.metrics import confusion_matrix

# Example with a single majority class and a single minority class per sample
true_labels = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1])
predicted_labels = np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1])
sample_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

aggregate_cm = confusion_matrix(true_labels, predicted_labels)
print("Aggregate Confusion Matrix:\n", aggregate_cm)


unique_sample_ids = np.unique(sample_ids)
for sample_id in unique_sample_ids:
    sample_true_labels = true_labels[sample_ids == sample_id]
    sample_predicted_labels = predicted_labels[sample_ids == sample_id]
    sample_cm = confusion_matrix(sample_true_labels, sample_predicted_labels)
    print(f"Confusion Matrix Sample {sample_id}:\n{sample_cm}")
```

This final example demonstrates a scenario where each sample has a substantial class imbalance, as is very common when classifying different individuals based on disease prevalence. We can observe the overall model performance in the aggregate confusion matrix, but individual confusion matrices highlight the different distribution of data, and differing error rates.

As can be seen, focusing on the aggregate confusion matrix provides an overall evaluation, but masks performance differences across samples. Looking at per-sample confusion matrices highlights where specific samples are causing problems. The most meaningful approach depends on the context of the experiment. If we are not concerned with inter-sample differences, we can just use an aggregate matrix. If, on the other hand, we are concerned with heterogeneity in the data set, the use of individual matrices is more appropriate.

For further study, I would recommend examining resources that discuss model evaluation in heterogeneous datasets. There are many well written resources on the handling of imbalanced datasets, especially in the context of healthcare. Publications focusing on personalized modeling and inter-subject variability in machine learning could also be useful. More generally, books detailing confusion matrix theory and model evaluation techniques should be considered mandatory reading for any practitioner. Finally, articles and books discussing the ethical and responsible use of model analysis, with specific focus on the dangers of masking heterogenous performance and the effect of this on subgroups, will also prove invaluable to any user considering these issues.
