---
title: "How can I create a Confusion Matrix for Binary Classification with a Neural Network?"
date: "2024-12-23"
id: "how-can-i-create-a-confusion-matrix-for-binary-classification-with-a-neural-network"
---

, let’s tackle this. From my experience working on a few projects involving image analysis and medical diagnosis, I've become pretty familiar with the ins and outs of confusion matrices, particularly when evaluating binary classification models. So, constructing one from neural network predictions is definitely something I can walk you through.

The core idea behind a confusion matrix is to provide a detailed breakdown of a model's prediction performance. In binary classification, we deal with two classes – say, ‘positive’ and ‘negative’ (or ‘1’ and ‘0,’ ‘true’ and ‘false,’ whatever fits your problem domain). The matrix then lays out how many instances were predicted correctly and incorrectly for each class. It’s a vital tool for moving beyond simple accuracy and understanding where your model is strong or weak.

A basic confusion matrix has four elements:

*   **True Positives (TP):** Instances correctly predicted as positive.
*   **True Negatives (TN):** Instances correctly predicted as negative.
*   **False Positives (FP):** Instances incorrectly predicted as positive (Type I error).
*   **False Negatives (FN):** Instances incorrectly predicted as negative (Type II error).

Let's frame this with a practical example from a past project. I was developing a neural network to detect a specific type of anomaly in sensor data. The data was inherently imbalanced, with anomalies being far less frequent than normal readings. Simply reporting accuracy was misleading, as the model could appear highly accurate by predominantly predicting the common ‘normal’ class. Here's where the confusion matrix became crucial.

To construct it from a neural network, you need two main components: the actual ground truth labels (the correct class for each input) and the predictions made by your model. Generally, the neural network outputs a probability for each class. In binary classification, this is usually a single value (let’s call it `prediction_prob`), where a probability greater than a certain threshold (often 0.5) is considered a positive prediction. The ground truth labels will just be binary values (0 or 1) representing negative or positive class respectively.

Here’s a code snippet using python with `numpy` and `sklearn`, which I’ve found to be highly practical and widely adopted:

```python
import numpy as np
from sklearn.metrics import confusion_matrix

def create_confusion_matrix(y_true, y_pred_prob, threshold=0.5):
  """
  Generates a confusion matrix for binary classification.

  Args:
    y_true: Array of true binary labels (0 or 1).
    y_pred_prob: Array of predicted probabilities (between 0 and 1).
    threshold: The probability threshold for classifying a prediction as positive.

  Returns:
    A 2x2 numpy array representing the confusion matrix.
  """
  y_pred = (y_pred_prob >= threshold).astype(int)
  cm = confusion_matrix(y_true, y_pred)
  return cm

# Example usage:
y_true_example = np.array([0, 1, 0, 1, 0, 1, 0, 0, 1, 0]) # Example ground truth
y_pred_prob_example = np.array([0.2, 0.7, 0.3, 0.8, 0.4, 0.6, 0.1, 0.2, 0.9, 0.5]) # Example predicted probabilities

cm_example = create_confusion_matrix(y_true_example, y_pred_prob_example)
print(cm_example)

```
This function, `create_confusion_matrix`, takes the true labels (`y_true`) and predicted probabilities (`y_pred_prob`), applies the threshold to classify to predicted binary labels and then leverages `sklearn.metrics.confusion_matrix` to calculate and output the matrix. Notice the use of `astype(int)` to convert boolean values `y_pred` to numerical values.

Now, what if you're working with probabilities coming directly from your neural network model, possibly using something like a sigmoid activation in the output layer? The output from a neural network needs to be processed to obtain binary values for comparison with the true labels. Here’s another snippet, showing how this might fit in an actual inference loop with, say, PyTorch, one of my go-to frameworks:
```python
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

# Example model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


def evaluate_model(model, data_loader, threshold=0.5):
    """
    Evaluates a PyTorch model and returns the confusion matrix.

    Args:
      model: The PyTorch model to evaluate.
      data_loader: A PyTorch DataLoader containing the evaluation data.
      threshold: The probability threshold for classification.

    Returns:
      A 2x2 numpy array representing the confusion matrix.
    """
    model.eval()  # Set model to evaluation mode
    all_true_labels = []
    all_predicted_probs = []
    with torch.no_grad(): # Disable gradient calculation
        for inputs, labels in data_loader:
            outputs = model(inputs)
            all_true_labels.extend(labels.numpy().flatten())
            all_predicted_probs.extend(outputs.numpy().flatten())
    
    return create_confusion_matrix(np.array(all_true_labels), np.array(all_predicted_probs), threshold)

# Setup dummy data
X_tensor = torch.randn(100, 10)  # 100 samples, 10 features
y_tensor = torch.randint(0, 2, (100,)).float()
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=10)

# Instantiate and test the function
model = SimpleNet()
cm = evaluate_model(model, dataloader)
print(cm)

```

Here, we set the model in evaluation mode using `model.eval()` and iterate through the DataLoader which provides both the features and the true labels. For every batch, we feed the features to our model, collect predicted probabilities, collect true labels, and return a confusion matrix calculated with the function we implemented earlier, while ensuring the gradient calculation is turned off within the loop for faster inference using `torch.no_grad()`.

One thing you’ll often find with binary classification problems is the need to adjust the threshold. In my sensor data project, the default threshold of 0.5 often led to too many false negatives, which were costly in terms of missed anomalies. Adjusting it to favor recall over precision led to a better overall system performance, even if this meant increased false positive rates. So, experimenting with various thresholds is often important. You may also find libraries such as `scikit-plot` to help you visualize the effects of a change in threshold.

Finally, you might encounter situations where your labels are not directly binary (e.g., they might be one-hot encoded or have a slightly different format). The code snippets above work with numerical values for true labels as well as predicted probabilities, but make sure your labels are integers (0 or 1) and your predicted output is between 0 and 1. Ensure the output from the network’s last layer is a sigmoid activation that confines the output between 0 and 1 to get the predicted probability. Here’s a final example demonstrating this, with a more complex data set, a one-hot encoded label format and a conversion to binary format:

```python
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

# Example model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 2) # output two classes
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


def evaluate_model_one_hot(model, data_loader, threshold=0.5):
    """
    Evaluates a PyTorch model and returns the confusion matrix.

    Args:
      model: The PyTorch model to evaluate.
      data_loader: A PyTorch DataLoader containing the evaluation data.
      threshold: The probability threshold for classification.

    Returns:
      A 2x2 numpy array representing the confusion matrix.
    """
    model.eval()  # Set model to evaluation mode
    all_true_labels = []
    all_predicted_probs = []
    with torch.no_grad(): # Disable gradient calculation
        for inputs, labels in data_loader:
            outputs = model(inputs)
            
            # Convert one-hot to binary labels using argmax
            _, predicted_classes = torch.max(outputs, 1)
            
            # Get the probability for the predicted class
            predicted_probs = torch.gather(outputs, 1, predicted_classes.unsqueeze(1)).squeeze(1)

            
            all_true_labels.extend(torch.argmax(labels, dim=1).numpy()) # convert one-hot true labels to numeric labels
            all_predicted_probs.extend(predicted_probs.numpy())

    return create_confusion_matrix(np.array(all_true_labels), np.array(all_predicted_probs), threshold)


# Setup dummy data, one-hot encoded
X_tensor = torch.randn(100, 10)  # 100 samples, 10 features
y_tensor = torch.randint(0, 2, (100,)).float() # initial target values
y_one_hot = torch.nn.functional.one_hot(y_tensor.long(), num_classes=2).float() # generate one-hot labels
dataset = TensorDataset(X_tensor, y_one_hot)
dataloader = DataLoader(dataset, batch_size=10)

# Instantiate and test the function
model = SimpleNet()
cm = evaluate_model_one_hot(model, dataloader)
print(cm)


```
Here, you can see the model’s output layer uses 2 outputs, one for each class, and the function `evaluate_model_one_hot` converts the output to binary labels and probability with the use of `torch.max` and `torch.gather`. The true labels are converted from one-hot encoding to numerical values with `torch.argmax`.

For further reading, I’d recommend “Pattern Recognition and Machine Learning” by Christopher Bishop. This book offers a rigorous treatment of statistical learning and is valuable for understanding the theory behind performance metrics. Another good option is “Deep Learning” by Goodfellow, Bengio, and Courville which provides comprehensive coverage of neural networks. Both are excellent resources for deepening your knowledge of model evaluation and deep learning practices.

So, in summary, you’ve got the tools to convert the probability outputs of your model into a useful confusion matrix, and a bit of understanding of the practical challenges you may face along the way. It's a fundamental tool for understanding the performance of your models beyond raw accuracy.
