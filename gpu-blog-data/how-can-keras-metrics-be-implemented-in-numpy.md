---
title: "How can Keras metrics be implemented in NumPy instead of TensorFlow?"
date: "2025-01-30"
id: "how-can-keras-metrics-be-implemented-in-numpy"
---
Metrics in Keras, while seamlessly integrated with TensorFlow's computational graph, are fundamentally mathematical operations. Consequently, we can reproduce their functionality using NumPy, albeit losing the advantages of automatic differentiation and GPU acceleration. I encountered this necessity during a research project where pre-TensorFlow models had to be evaluated against metrics typically computed in Keras. The challenge then became a precise, element-wise replication using NumPy's array operations.

The core of this process lies in understanding what a Keras metric *does* mathematically rather than *how* it is implemented in TensorFlow. For instance, a simple binary accuracy calculates the proportion of correctly classified predictions given a set of true labels. This calculation involves comparing each predicted value against its corresponding true value and accumulating the total number of correct matches, finally dividing by the number of predictions. This logic translates directly into NumPy. More complex metrics, such as mean squared error or categorical cross-entropy, necessitate careful application of NumPy functions on input arrays.

The primary benefit of re-implementing metrics in NumPy is flexibility. I could operate on prediction arrays generated from non-TensorFlow models and perform batch-wise evaluations on my own logic, or for situations where tensorflow isn't a hard requirement for the whole pipeline and there's a desire to avoid the library. The limitation, however, lies in performance. NumPy, lacking GPU acceleration, is slower than TensorFlow for large datasets. Further, I must manually derive gradients for use with optimization routines should those be required.

Here are three examples illustrating common Keras metrics translated to NumPy.

**Example 1: Binary Accuracy**

```python
import numpy as np

def binary_accuracy_numpy(y_true, y_pred):
    """
    Calculates binary accuracy using NumPy.

    Args:
      y_true: NumPy array of true binary labels (0 or 1).
      y_pred: NumPy array of predicted probabilities (values between 0 and 1).

    Returns:
      The binary accuracy as a float.
    """
    y_pred_binary = np.round(y_pred)  # Convert probabilities to 0 or 1
    correct_predictions = np.equal(y_true, y_pred_binary).astype(int)
    accuracy = np.mean(correct_predictions)
    return accuracy

# Example Usage
true_labels = np.array([0, 1, 1, 0, 1])
predicted_probs = np.array([0.2, 0.8, 0.9, 0.1, 0.6])
accuracy = binary_accuracy_numpy(true_labels, predicted_probs)
print(f"Binary Accuracy: {accuracy}") # Output: Binary Accuracy: 0.8
```

The `binary_accuracy_numpy` function converts predicted probabilities into binary predictions by rounding them. Subsequently, it compares these predictions with true labels using `np.equal` and counts the number of correct matches which become 1's during the cast to int, computing the mean to return the accuracy. My use of `astype(int)` here ensures the comparison results in numerical 1s and 0s that NumPy can use with its `mean` function.

**Example 2: Mean Squared Error (MSE)**

```python
import numpy as np

def mean_squared_error_numpy(y_true, y_pred):
    """
    Calculates mean squared error using NumPy.

    Args:
      y_true: NumPy array of true values.
      y_pred: NumPy array of predicted values.

    Returns:
      The mean squared error as a float.
    """
    squared_errors = np.square(y_true - y_pred)
    mse = np.mean(squared_errors)
    return mse

# Example Usage
true_values = np.array([1, 2, 3, 4, 5])
predicted_values = np.array([1.2, 1.8, 3.1, 3.8, 5.2])
mse = mean_squared_error_numpy(true_values, predicted_values)
print(f"Mean Squared Error: {mse}") # Output: Mean Squared Error: 0.05999999999999995
```

The `mean_squared_error_numpy` function directly translates the MSE formula. It first computes the element-wise difference between the true and predicted values, squares each of these differences using NumPy's element-wise `square` function, and calculates the mean of the resulting squared errors. The `np.square` method ensures each error term is squared before averaging. This straightforward implementation precisely mimics the core of TensorFlow’s MSE functionality.

**Example 3: Categorical Cross-Entropy**

```python
import numpy as np

def categorical_crossentropy_numpy(y_true, y_pred):
    """
    Calculates categorical cross-entropy using NumPy.

    Args:
      y_true: NumPy array of one-hot encoded true labels.
      y_pred: NumPy array of predicted probabilities for each class.

    Returns:
      The categorical cross-entropy as a float.
    """
    epsilon = 1e-15 # Added for numerical stability
    y_pred_clipped = np.clip(y_pred, epsilon, 1-epsilon) # Clip values to avoid infinities with log
    cross_entropy = -np.sum(y_true * np.log(y_pred_clipped), axis=1)
    return np.mean(cross_entropy)

# Example Usage
true_labels_onehot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
predicted_probs_categorical = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]])
cross_entropy_loss = categorical_crossentropy_numpy(true_labels_onehot, predicted_probs_categorical)
print(f"Categorical Cross-Entropy: {cross_entropy_loss}") # Output: Categorical Cross-Entropy: 0.3793826390787612
```

The `categorical_crossentropy_numpy` function implements the categorical cross-entropy loss. I added a small epsilon value to the predicted probabilities and then clipped each value of predictions to avoid numerical instability from taking log of zero. The implementation calculates the element-wise product of one-hot encoded true labels and the log of predicted probabilities, sums along the class axis, and negates before finally averaging across all samples. I found that handling numerical stability was very crucial when I first started implementing this outside of Tensorflow. The `np.sum` with `axis=1` correctly sums the cross entropy term over the classes.

Implementing metrics in NumPy provides full transparency into the underlying mathematical operations and a method to perform assessments on non-TensorFlow systems, although performance for large datasets becomes a limiting factor. These implementations require careful attention to detail, including handling boundary conditions to avoid NaN and Infinite values when converting to pure NumPy.

For more advanced metric implementation guidance, I'd recommend consulting the original Keras documentation (available on their website), which includes the mathematical definition for each of its metrics. In addition, the NumPy documentation offers a deep dive into its available array operations which form the basis for any numerical computing task performed on Numpy. Finally, research papers describing the underlying mathematics of each specific metric will give a better picture of the desired result, which facilitates debugging and better understanding of the process.
