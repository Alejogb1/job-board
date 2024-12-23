---
title: "How do I generate a confusion matrix for binary classification with a neural network?"
date: "2024-12-23"
id: "how-do-i-generate-a-confusion-matrix-for-binary-classification-with-a-neural-network"
---

,  Instead of jumping straight into code, I think it's worth first laying the groundwork and explaining *why* we need a confusion matrix, especially when working with binary classification models. I’ve spent a good chunk of my career building and debugging models, and trust me, a good confusion matrix can save you from making some serious missteps.

So, you’ve got a neural network spitting out predictions—probabilities, really—that a given input belongs to one of two classes. You've probably set a threshold, usually 0.5, to translate those probabilities into hard class assignments: 0 or 1, for instance. But just because your model’s overall accuracy is, say, 90%, doesn’t mean it's behaving well across both classes. That's where the confusion matrix comes in. It breaks down the model’s predictions into four critical categories: true positives (tp), true negatives (tn), false positives (fp), and false negatives (fn). These values tell a much more detailed story than a single accuracy score ever could, revealing imbalances or biases in your model's performance. Think of it like this: accuracy might tell you how often you're generally correct, but a confusion matrix tells you *where* you are going right and *where* you are stumbling.

Now, let's get into how you actually build one. The first key is to have your model's predictions and your ground truth labels at hand. I recall a project where I was working on identifying defective widgets on an assembly line, and initially relied solely on accuracy; the results were misleading. The majority of parts were good, so the high accuracy masked a real problem: the system was consistently misclassifying defective widgets as good—high false negatives—a far more costly error than incorrectly flagging a good widget as defective. This experience taught me the importance of specifically analyzing class-level performance.

So, here's a basic snippet, using python and numpy, to construct this matrix from scratch:

```python
import numpy as np

def create_confusion_matrix(y_true, y_pred, threshold=0.5):
    """
    Generates a confusion matrix for binary classification.

    Args:
        y_true (np.array): Ground truth labels (0s and 1s).
        y_pred (np.array): Predicted probabilities (values between 0 and 1).
        threshold (float): Classification threshold.

    Returns:
        tuple: (tp, tn, fp, fn)
    """
    y_pred_labels = (y_pred >= threshold).astype(int)

    tp = np.sum((y_true == 1) & (y_pred_labels == 1))
    tn = np.sum((y_true == 0) & (y_pred_labels == 0))
    fp = np.sum((y_true == 0) & (y_pred_labels == 1))
    fn = np.sum((y_true == 1) & (y_pred_labels == 0))

    return tp, tn, fp, fn

# Example usage:
y_true_example = np.array([1, 0, 1, 0, 1, 0, 1, 0])
y_pred_example = np.array([0.8, 0.2, 0.7, 0.4, 0.9, 0.6, 0.3, 0.1])

tp, tn, fp, fn = create_confusion_matrix(y_true_example, y_pred_example)
print(f"True Positives: {tp}, True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}")
```

In this function, we take your ground truth labels (`y_true`), which are binary (0s and 1s), and predicted probabilities (`y_pred`) from your network. We convert those probabilities into binary predictions by applying a threshold. Then, we use numpy's boolean indexing to count occurrences of tp, tn, fp, and fn based on comparisons between the true and predicted labels. The returned values can be used to calculate metrics like precision, recall, and the f1-score, giving us a much more granular look at model performance.

While the above is good for illustrating the fundamentals, most of us leverage libraries which provide more structured output. Libraries such as scikit-learn (sklearn) offer more sophisticated functionalities. Here's the same logic, implemented using `sklearn.metrics`:

```python
from sklearn.metrics import confusion_matrix
import numpy as np

def create_sklearn_confusion_matrix(y_true, y_pred, threshold=0.5):
    """
    Generates a confusion matrix using scikit-learn.

    Args:
        y_true (np.array): Ground truth labels (0s and 1s).
        y_pred (np.array): Predicted probabilities (values between 0 and 1).
        threshold (float): Classification threshold.

    Returns:
        np.array: The confusion matrix
    """
    y_pred_labels = (y_pred >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_labels)
    return cm

# Example usage
y_true_example = np.array([1, 0, 1, 0, 1, 0, 1, 0])
y_pred_example = np.array([0.8, 0.2, 0.7, 0.4, 0.9, 0.6, 0.3, 0.1])

cm = create_sklearn_confusion_matrix(y_true_example, y_pred_example)
print(f"Confusion Matrix:\n{cm}")
```

In this example, we utilize `sklearn.metrics.confusion_matrix`. It takes our ground truth labels and binarized predictions and returns a 2x2 array where the rows represent the true labels and the columns represent predicted labels. The top-left is the true negatives (tn), top-right false positives (fp), bottom-left false negatives (fn), and bottom-right true positives (tp). This structured output is often far more usable for automated report generation.

Now, in a neural network context, your `y_pred` will often be the output of your network's last layer, before you apply a classification threshold. Libraries like TensorFlow or PyTorch have their own mechanisms to handle batch predictions, so you'll need to accumulate the true labels and predictions across batches during your evaluation phase before constructing your confusion matrix. Here is an example, using PyTorch:

```python
import torch
import numpy as np
from sklearn.metrics import confusion_matrix

def evaluate_model_with_confusion_matrix(model, dataloader, device, threshold=0.5):
  """
    Evaluates a PyTorch model, accumulating labels and predictions, and generates a confusion matrix.
    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        dataloader (torch.utils.data.DataLoader): Dataloader for evaluation data.
        device (str): Device to run the model on ('cpu' or 'cuda').
        threshold (float): Classification threshold.
    Returns:
        np.array: The confusion matrix
  """

  model.eval()  # Set model to evaluation mode
  all_true_labels = []
  all_predictions = []

  with torch.no_grad(): # No gradients needed during evaluation
      for inputs, labels in dataloader:
          inputs = inputs.to(device)
          labels = labels.to(device)

          outputs = model(inputs)
          probabilities = torch.sigmoid(outputs) # Assumes binary classification with sigmoid activation

          all_true_labels.extend(labels.cpu().numpy().flatten())
          all_predictions.extend(probabilities.cpu().numpy().flatten())

  y_true_np = np.array(all_true_labels)
  y_pred_np = np.array(all_predictions)

  y_pred_labels = (y_pred_np >= threshold).astype(int)
  cm = confusion_matrix(y_true_np, y_pred_labels)
  return cm



# Example Usage (Dummy data and model)
class DummyModel(torch.nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc = torch.nn.Linear(10, 1)
    def forward(self, x):
        return self.fc(x)

dummy_model = DummyModel()

# Dummy data
dummy_data = torch.randn(100, 10)
dummy_labels = torch.randint(0, 2, (100, 1)).float()

from torch.utils.data import TensorDataset, DataLoader
dummy_dataset = TensorDataset(dummy_data, dummy_labels)
dummy_dataloader = DataLoader(dummy_dataset, batch_size=32)
device = 'cpu'

confusion_mat = evaluate_model_with_confusion_matrix(dummy_model, dummy_dataloader, device)
print(f"Confusion Matrix:\n{confusion_mat}")
```

In this PyTorch example, during evaluation we iterate through our dataloader and accumulate all labels and model output probabilities. We use the sigmoid function as it is standard for binary classification tasks. After collecting predictions and ground truths, the results are converted to numpy and we then generate a confusion matrix using `sklearn`. This demonstrates a common workflow where you extract outputs from a network and then apply a standard library tool to do the analysis.

For a comprehensive understanding of evaluation metrics in machine learning, I strongly recommend "Pattern Recognition and Machine Learning" by Christopher Bishop; it provides a very thorough foundation. For a hands-on approach with a focus on code, consult “Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow” by Aurélien Géron. In either case, going beyond simple accuracy and using the confusion matrix will undoubtedly make you a more effective model builder. It's not just about getting good numbers; it's about understanding what those numbers *mean*. It's this understanding that will help you iterate more effectively on your models.
