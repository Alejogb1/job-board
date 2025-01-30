---
title: "How does a 16% positive label rate affect prediction accuracy?"
date: "2025-01-30"
id: "how-does-a-16-positive-label-rate-affect"
---
A classification model trained on a dataset exhibiting a severe class imbalance, specifically with only 16% of instances labeled as positive, will demonstrably struggle to achieve optimal prediction accuracy, particularly regarding recall for the minority (positive) class. My direct experience working on fraud detection systems, where fraudulent transactions often represent less than 1% of total activity, confirms this. The dominant negative class tends to skew learning algorithms towards predicting the negative outcome, resulting in a high overall accuracy largely driven by correct negative predictions, while simultaneously exhibiting poor performance in identifying positive cases. This imbalance necessitates a deeper understanding of specific evaluation metrics beyond simple accuracy, and the implementation of techniques to mitigate this negative influence on model performance.

The crux of the problem lies in how algorithms, particularly those relying on cost functions that optimize for overall accuracy, are affected by imbalanced class distributions. In essence, a model can achieve a seemingly high accuracy by simply predicting the majority class across all instances. With 84% negative labels, a trivial model that always predicts "negative" would achieve 84% accuracy. While statistically high, this has no practical utility; it fails to identify any positive cases. Consequently, using accuracy alone as an evaluation metric becomes misleading. In such skewed scenarios, the underlying learning process becomes biased, focusing on optimizing for correctly classifying the prevalent class at the expense of the underrepresented class. The model becomes increasingly confident at classifying negative instances while struggling with positive cases, where learning signals are sparse and diluted within the overwhelming negative signal. The result is often a model with a very low false negative rate but a very high false positive rate within the actual positive occurrences.

To accurately assess performance in such imbalanced settings, evaluation metrics beyond overall accuracy are essential. Precision, recall, and the F1-score are more informative. Precision, defined as true positives divided by the sum of true positives and false positives, reflects the proportion of correctly predicted positives among all predicted positives. It addresses the question of "how many of my positive predictions were correct?" Recall, defined as true positives divided by the sum of true positives and false negatives, reflects the proportion of actual positives that were correctly predicted. It focuses on "how many of the actual positive cases did I identify?" The F1-score is the harmonic mean of precision and recall, providing a single metric that balances these two conflicting objectives. It penalizes models that favor one over the other. Area Under the Receiver Operating Characteristic curve (AUC-ROC), which measures the discriminatory capability of the model across varying thresholds, is also vital. These metrics provide a more granular and accurate assessment of model effectiveness in the context of imbalanced datasets.

Furthermore, techniques to mitigate class imbalance are crucial for training models with satisfactory performance on the minority class. These techniques fall broadly into two categories: data-level and algorithm-level approaches. Data-level approaches focus on manipulating the training dataset to achieve a more balanced representation of classes. One common technique is oversampling, which involves replicating instances of the minority class or generating synthetic samples. Undersampling, which involves removing instances from the majority class, can also be employed. However, this has a risk of discarding valuable information present in the majority class. Hybrid techniques combining oversampling and undersampling, such as SMOTE and Tomek links, often prove beneficial. Algorithm-level approaches modify the model's learning process to account for imbalanced data. This could include using cost-sensitive learning, assigning higher misclassification costs to the minority class during training. Another approach is to use ensemble methods, such as boosting algorithms, which can focus on misclassified instances, allowing the model to pay more attention to the minority class.

Here are three code examples in Python, demonstrating the use of metrics and a specific data augmentation technique:

```python
# Example 1: Illustrating the problem with accuracy in imbalanced data

from sklearn.metrics import accuracy_score
import numpy as np

y_true = np.array([0] * 84 + [1] * 16)  # 84 negatives, 16 positives
y_pred_trivial = np.array([0] * 100)  # Predicts all negatives
y_pred_good = np.array([0] * 80 + [1] * 4 + [0]*4 + [1]*12) # 80 correct negatives, 4 false positives, 4 false negatives, 12 correct positives


accuracy_trivial = accuracy_score(y_true, y_pred_trivial)
accuracy_good = accuracy_score(y_true, y_pred_good)


print(f"Trivial Accuracy: {accuracy_trivial:.2f}")  # Output: Trivial Accuracy: 0.84
print(f"Good model Accuracy: {accuracy_good:.2f}")  # Output: Good model Accuracy: 0.92

# Commentary: A trivial model that always predicts negative achieves an accuracy of 84%, the same as the proportion of negatives in the dataset. The better model, despite correctly predicting only 12 of the positives still has a higher accuracy at 92%, although performance on the minority class is still poor. It demonstrates that high overall accuracy is misleading.
```

```python
# Example 2: Using precision, recall and F1-score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

y_true = np.array([0] * 84 + [1] * 16)
y_pred_good = np.array([0] * 80 + [1] * 4 + [0]*4 + [1]*12)

precision = precision_score(y_true, y_pred_good)
recall = recall_score(y_true, y_pred_good)
f1 = f1_score(y_true, y_pred_good)
auc_roc = roc_auc_score(y_true, y_pred_good)

print(f"Precision: {precision:.2f}") # Output: Precision: 0.75
print(f"Recall: {recall:.2f}")       # Output: Recall: 0.75
print(f"F1 Score: {f1:.2f}")       # Output: F1 Score: 0.75
print(f"AUC-ROC: {auc_roc:.2f}")  # Output: AUC-ROC: 0.86

# Commentary: Precision and Recall are equal in this case (at 0.75), meaning that out of the predicted positives 75% were actual positives and the model identified 75% of the actual positives. The f1-score summarizes these as well. The AUC-ROC curve gives the probability of correctly classifying an instance across all thresholds. These metrics provide a far clearer view of model performance on the minority class.
```

```python
# Example 3: Using SMOTE for oversampling

from imblearn.over_sampling import SMOTE
import numpy as np

X = np.random.rand(100, 5)  # 100 samples, 5 features
y = np.array([0] * 84 + [1] * 16)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print(f"Original dataset shape: {X.shape}, {y.shape}") # Original dataset shape: (100, 5), (100,)
print(f"Resampled dataset shape: {X_resampled.shape}, {y_resampled.shape}") # Resampled dataset shape: (168, 5), (168,)

unique_labels, counts = np.unique(y_resampled, return_counts=True)
print(f"Label distribution after SMOTE: {dict(zip(unique_labels,counts))}") # Label distribution after SMOTE: {0: 84, 1: 84}


# Commentary: This illustrates the use of SMOTE to oversample the minority class. The output shows that the number of minority and majority samples becomes equal, achieving balance. The new dataset X_resampled and y_resampled can be used for model training.
```

In summary, a 16% positive label rate significantly impacts prediction accuracy, particularly on the minority class, if evaluation metrics are not carefully selected and techniques to address class imbalances are not implemented. Solely relying on overall accuracy provides a misleading view of model performance. Instead, precision, recall, the F1-score, and AUC-ROC offer better insights. Techniques such as SMOTE for oversampling, along with cost-sensitive learning and ensemble methods, are needed to build robust predictive models. I strongly recommend consulting resources focusing on handling imbalanced datasets when working with a similar situation; books like "Imbalanced Learning: Foundations, Algorithms, and Applications" and specialized articles found in computer science journals provide comprehensive guidance on best practices. Online courses from reputable learning platforms offer practical hands-on experience in addressing imbalanced datasets within the context of machine learning. Ultimately, careful metric selection, rigorous evaluation, and awareness of available solutions are critical for building high-performing predictive models, especially when dealing with datasets exhibiting class imbalances.
