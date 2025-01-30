---
title: "Is MSE suitable as a loss function with label encoding for classification?"
date: "2025-01-30"
id: "is-mse-suitable-as-a-loss-function-with"
---
Mean Squared Error (MSE), despite its ubiquity in regression problems, presents a significant challenge when used directly as a loss function with label-encoded data in classification. This stems from the inherent nature of label encoding, which assigns arbitrary numerical values to categorical classes, and MSE’s sensitivity to distance, leading to unintended consequences.

Label encoding, while transforming categorical data into a numerical format suitable for many machine learning algorithms, does not imply any inherent ordering or magnitude relationship between classes. For instance, if we have classes "Red", "Green", and "Blue" encoded as 0, 1, and 2 respectively, these numerical values are purely labels. Class 'Blue' is not numerically “more” than 'Red' by the amount suggested by their numerical representation.

MSE, however, interprets the assigned numerical values as continuous outputs, implicitly assuming a meaningful distance between each class. MSE calculates the squared difference between the predicted output and the target. Applying this metric to label-encoded classes means we are effectively penalizing predictions based on the artificial numerical distance between labels, rather than the correctness of the classification. This leads to models that might incorrectly learn relationships between classes based on their encoding, resulting in poor generalization and sub-optimal classification performance. The core issue is MSE's focus on *magnitude* differences, which is not the correct metric to assess classification *category* differences.

Consider a multiclass classification problem with three classes encoded as 0, 1, and 2. A perfect prediction for an observation belonging to class 0 should ideally receive minimal loss. However, if a model makes a prediction closer to 1 (e.g., a predicted value of 0.9), MSE would penalize this prediction as heavily as a prediction near 2 would. In a classification setting, misclassifying an instance as belonging to *any* other class should be penalized equally, irrespective of the numerical representation of the classes. This is the key failing of MSE in this context.

Here are three code examples using Python, demonstrating these issues and highlighting better alternatives.

**Example 1: Illustrating MSE with Label-Encoded Data**

```python
import numpy as np
from sklearn.metrics import mean_squared_error

# True labels, represented by encoded class numbers.
y_true = np.array([0, 1, 2, 0, 1])
# Predicted output values of a hypothetical model.
y_pred = np.array([0.1, 0.8, 1.8, 0.9, 1.1])
# Notice the 'predicted' values are all floating-point and can be outside of valid class range
# The model is confused between classes 0 and 1.

mse_loss = mean_squared_error(y_true, y_pred)
print(f"MSE Loss: {mse_loss}")

# True labels, predicted perfectly
y_pred_perfect = np.array([0.0, 1.0, 2.0, 0.0, 1.0])
mse_loss_perfect = mean_squared_error(y_true, y_pred_perfect)
print(f"MSE Loss with perfect prediction: {mse_loss_perfect}")

#Now consider this scenario
y_pred_different_error = np.array([2.0, 2.0, 0.0, 1.0, 2.0])
mse_loss_different_error = mean_squared_error(y_true,y_pred_different_error)
print(f"MSE Loss with very different classification results: {mse_loss_different_error}")

# Note that the MSE loss in the last scenario is higher than when predictions are slightly off
# but these are more serious classification errors.
```

This example demonstrates how MSE penalizes intermediate predictions, even if they are closer to the true class from an ordinal perspective. The last scenario highlights the problematic nature: predictions that are completely incorrect, are not more heavily penalized than predictions that are partially correct.

**Example 2: Demonstrating Cross-Entropy as an alternative**

```python
import numpy as np
from sklearn.metrics import log_loss
# We need to transform labels to a one-hot encoded format, or we must use predict_proba from a classifier
y_true = np.array([0, 1, 2, 0, 1])
# Here, we would typically provide the predicted probabilities of each class for each input, but here we will use true values as a representation of perfect confidence
y_pred_perfect_proba = np.array([[1.0,0,0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 0, 0], [0, 1.0, 0]])
# Cross entropy is undefined for 0 probabilities, so make this small number, not zero.
y_pred_incorrect_proba = np.array([[0.9,0.05,0.05], [0.1, 0.9, 0], [0.1,0.1, 0.8], [0.05, 0.9, 0.05], [0, 0.1, 0.9]])
#Note that each row has sum 1.0 which is a requirement of the probability distribution of each input.

# We need one-hot encoded labels to compare with probabilities.
y_true_onehot = np.eye(3)[y_true]

cross_entropy_loss_perfect = log_loss(y_true_onehot, y_pred_perfect_proba)
print(f"Cross-Entropy Loss (Perfect prediction): {cross_entropy_loss_perfect}")
cross_entropy_loss_incorrect = log_loss(y_true_onehot, y_pred_incorrect_proba)
print(f"Cross-Entropy Loss (Incorrect Prediction): {cross_entropy_loss_incorrect}")

# Now, let's show how a prediction that is completely wrong but 'close' to another label is penalized.
y_pred_different_error_proba = np.array([[0,0,1], [0,0,1], [1,0,0], [0,1,0], [0,0,1]])
cross_entropy_loss_different_error = log_loss(y_true_onehot, y_pred_different_error_proba)
print(f"Cross Entropy loss (Serious errors): {cross_entropy_loss_different_error}")
```

This example uses the logarithmic loss, which penalizes a confident incorrect answer heavily. The key difference compared to MSE is that cross-entropy is based on probability distributions rather than arbitrary magnitudes, therefore it is more suited to classification tasks.

**Example 3: Demonstrating why label encoding with MSE is suboptimal**
```python
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Assume we have classes "Red", "Green", "Blue"
classes = ["Red", "Green", "Blue"]

# Using label encoding
label_encoder = LabelEncoder()
encoded_classes = label_encoder.fit_transform(classes)
print(f"Label encoded classes: {encoded_classes}")

# Assume we have true labels and a model's predictions
y_true = np.array([0,1,2,0,1]) # Using encoded labels, 0 = Red, 1 = Green, 2 = Blue
y_pred = np.array([0.2, 0.8, 1.7, 0.1, 1.2])

#Let us consider an alternative scenario, in which blue is assigned a label of 0
classes_alt = ["Red", "Green", "Blue"]
label_encoder_alt = LabelEncoder()
label_encoder_alt.fit(["Blue", "Red", "Green"])

encoded_classes_alt = label_encoder_alt.transform(classes_alt)
print(f"Label encoded classes (Alternative order): {encoded_classes_alt}")
y_true_alt = np.array([1,2,0,1,2]) # 1 = Red, 2 = Green, 0 = Blue, the new label
y_pred_alt = np.array([0.2, 0.8, 1.7, 0.1, 1.2])
mse_loss_original = mean_squared_error(y_true, y_pred)
mse_loss_alt = mean_squared_error(y_true_alt, y_pred_alt)
print(f"MSE Loss (original encoding): {mse_loss_original}")
print(f"MSE Loss (alternative encoding): {mse_loss_alt}")
#This code demostrates that the MSE loss changes by simply permuting the label encodings.
#This is a fatal flaw.
```
This example demonstrates that different label encodings can change the MSE loss even if predictions and true class remain identical. This reveals a critical issue: MSE penalizes based on an arbitrary numerical representation. With MSE, a perfect classification model can be penalized simply by changing the label encoding.

In summary, MSE is not suitable for classification with label encoding due to its inherent assumptions about the data and loss calculation. Using MSE introduces sensitivity to arbitrary numerical assignments, leading to misleading performance evaluations and unreliable model learning. For classification problems, alternatives like categorical cross-entropy, often coupled with one-hot encoding, are far more appropriate as they explicitly address the categorical nature of the target variables and penalize based on probability distributions.

For further study, I recommend investigating these topics: the mathematical basis of different loss functions, such as cross entropy, and the mechanics of one-hot encoding. Publications from the scikit-learn library developers and from universities that focus on machine learning offer detailed insights. Also, textbooks on statistical machine learning provide a theoretical background, allowing a more thorough understanding of the limitations of mean squared error in classification settings.
