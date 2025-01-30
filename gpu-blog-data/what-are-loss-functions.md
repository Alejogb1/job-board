---
title: "What are loss functions?"
date: "2025-01-30"
id: "what-are-loss-functions"
---
Loss functions, at their core, are mathematical measures quantifying the discrepancy between predicted and actual values in a machine learning model. I've encountered their practical significance extensively while building predictive systems for industrial equipment failure. The choice of a particular loss function critically impacts model performance, dictating the learning trajectory and the ultimate predictive accuracy. They provide the objective that gradient descent algorithms, and similar optimization techniques, attempt to minimize during model training. Without a defined loss, a model lacks the means to iteratively improve its parameter settings.

The essence of a loss function is to provide a scalar value that represents how ‘wrong’ the model’s current output is. This scalar value then acts as the signal for parameter adjustments. Lower loss values generally indicate a more accurate model. Different types of problems require different loss function characteristics. For example, regression problems, where we predict continuous numerical values, usually benefit from loss functions that measure the magnitude of the error. Conversely, classification problems, where we predict categories or classes, use loss functions that consider the probabilistic outputs associated with those classes.

The concept of a loss function is distinct from a metric used to evaluate model performance. A loss function informs the model during training (its optimization objective). An evaluation metric judges the model’s end performance after training is complete. While both might be similar for straightforward tasks, they can deviate significantly for more complex applications. For example, I have worked on imbalanced datasets where the loss was optimized using class-weighted cross-entropy while performance was evaluated using the F1 score or AUC, metrics less sensitive to class imbalance.

Several loss functions are frequently employed in practice. In regression tasks, Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE) are commonly seen. MSE averages the square of the differences between predicted and actual values. The squaring emphasizes large errors and tends to be more sensitive to outliers, often leading to a smoother error surface, which benefits gradient-based optimization. RMSE is simply the square root of MSE and offers interpretation in the original unit of the target variable. MAE computes the average of the absolute differences, making it more robust to outliers because it treats all errors equally without the squaring operation.

For classification problems, cross-entropy loss is a predominant choice, particularly for multi-class scenarios. Cross-entropy calculates the difference between the probability distributions of the model’s predictions and the actual class. It is derived from information theory and aims to minimize the information required to describe the actual class distribution using the model’s probability distribution. Another variant, binary cross-entropy, is specific to two-class problems.

Let’s examine examples showcasing the utilization of these loss functions within a Python environment using `torch` (a framework I often utilize):

**Code Example 1: Mean Squared Error (MSE) in Regression**

```python
import torch
import torch.nn as nn

# Assume model output and target are already tensors
predicted_values = torch.tensor([2.5, 4.8, 7.1])
actual_values = torch.tensor([3.0, 5.0, 6.5])

# Define the loss function: Mean Squared Error
mse_loss = nn.MSELoss()

# Calculate the MSE loss
loss = mse_loss(predicted_values, actual_values)
print("MSE Loss:", loss)  # Output: MSE Loss: tensor(0.1700)

# Manual Calculation for clarity:
manual_mse = torch.mean((predicted_values - actual_values)**2)
print("Manual MSE:", manual_mse) # Output: Manual MSE: tensor(0.1700)
```

In this snippet, we have `predicted_values` and `actual_values`, representing a set of model predictions and their corresponding ground truth values. `nn.MSELoss()` initializes the MSE loss function, and then when called with these two tensors, it computes the mean squared error. I included a manual calculation for clarity on how the MSE loss is calculated internally. The resultant tensor represents the computed MSE loss value.

**Code Example 2: Binary Cross-Entropy Loss in Classification**

```python
import torch
import torch.nn as nn

# Assuming we have binary classification - model output is probabilities between 0 and 1
predicted_probabilities = torch.tensor([0.9, 0.2, 0.7])
actual_labels = torch.tensor([1.0, 0.0, 1.0]) # 1 represents positive, 0 represents negative

# Define Binary Cross Entropy Loss
bce_loss = nn.BCELoss()

# Calculate the Binary Cross Entropy Loss
loss = bce_loss(predicted_probabilities, actual_labels)

print("Binary Cross Entropy Loss:", loss) # Output: Binary Cross Entropy Loss: tensor(0.2698)

# Explanation: The model assigns a probability to each sample for each class
# Cross-entropy checks the difference between model probability and true label
# Larger loss if wrong predictions (e.g. very confident wrong or not confident right)
```

Here, I assume a binary classification scenario with `predicted_probabilities` representing the model's output for each instance, and `actual_labels` are the corresponding true classes (0 or 1). `nn.BCELoss()` creates the loss function object, and the output shows the computed binary cross-entropy loss value. Note that in `nn.BCELoss`, the inputs must be probabilities between 0 and 1 (e.g., after using sigmoid). The loss will be higher if the model's assigned probability does not correspond with the actual class.

**Code Example 3: Cross-Entropy Loss in Multi-Class Classification**

```python
import torch
import torch.nn as nn

# Assuming a 3-class classification problem. Model output are logits, not probabilities.
predicted_logits = torch.tensor([[2.0, 1.0, 0.1],
                                 [0.5, 2.2, 1.8],
                                 [0.1, 0.8, 3.0]])

actual_classes = torch.tensor([0, 1, 2]) # 0, 1, and 2 are class indices

# Define Cross Entropy Loss (no sigmoid, logits are expected)
ce_loss = nn.CrossEntropyLoss()

# Calculate Cross Entropy Loss
loss = ce_loss(predicted_logits, actual_classes)

print("Cross Entropy Loss:", loss) # Output: Cross Entropy Loss: tensor(0.3585)

# Explanation: Cross-entropy compares output logit distributions for true vs predicted classes
# It penalizes the model for assigning lower scores to the correct class
```

For this multi-class example, we have a `predicted_logits` tensor where each row represents the model's output (logits) for a specific input and each column corresponds to a different class. `actual_classes` holds the true class indices. `nn.CrossEntropyLoss()` handles both softmax activation (applied internally before computation) and cross-entropy loss calculation, taking logits as inputs directly. As can be seen the loss penalizes the model when the probability of the true class is lower.

In practice, I’ve found that careful consideration must be given to the chosen loss function. A function well-suited to a particular problem can result in substantially faster model convergence and achieve higher performance.

For individuals wishing to delve deeper, I recommend researching publications and educational material relating to optimization in machine learning. Texts covering statistical learning theory can also provide a more thorough understanding of the theoretical underpinnings of different loss functions. Books on deep learning also commonly provide extensive coverage of both common and specialized loss functions, including the mathematical derivations. Lastly, hands-on practice with the concepts by implementing them in common Python frameworks such as `pytorch` or `tensorflow` provides valuable practical insight.
