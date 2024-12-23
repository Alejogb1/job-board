---
title: "What's the right way to F1 score one-hot encoded Roberta outputs?"
date: "2024-12-16"
id: "whats-the-right-way-to-f1-score-one-hot-encoded-roberta-outputs"
---

,  I've seen my fair share of confusion around evaluating one-hot encoded outputs from models like Roberta, especially when the goal is a precise F1 score. It's a classic situation where the mechanics of evaluation can be as tricky as the model itself, and I've definitely spent some late nights debugging these things in the past.

The core issue, as you're likely experiencing, lies in the transformation from model outputs – typically probabilities or logits – to the discrete class predictions required for F1 calculation. Roberta, when used for multi-class classification tasks, often produces a tensor with dimensions like `[batch_size, num_classes]`, where each entry represents the likelihood of a particular class. But F1 score operates on actual, hard predictions, not probabilities. Let me walk you through how I usually handle this, including some code examples.

First, understand that the one-hot encoded ground truth labels, which you'd have as input, will likely look like arrays, each having the length of `num_classes` and one “hot” value at the right index; like `[0, 0, 1, 0]` for a sample belonging to the 3rd class.

Now, when dealing with your Roberta output, the first critical step is to convert those probabilities (or logits) into class predictions. For probabilities, we simply take the `argmax` along the `num_classes` dimension, assigning each sample to the class with the highest probability. If the model outputs logits, you might prefer to apply a softmax function first, turning logits to probabilities, before `argmax`. We tend to use softmax over logits as it allows better calibration for scoring. Here's how to do it using PyTorch, as that's what I use most often:

```python
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
import numpy as np

def calculate_f1_from_probabilities(probabilities, labels):
    """
    Calculates the F1 score from one-hot encoded labels.

    Args:
        probabilities: PyTorch tensor of shape [batch_size, num_classes]
        labels: Numpy array of shape [batch_size, num_classes] containing the true one-hot labels

    Returns:
        float: The micro-averaged F1 score.
    """
    # Ensure probabilities are on CPU and converted to NumPy for sklearn
    probabilities_cpu = probabilities.cpu().detach().numpy()
    
    # Convert probabilities to predictions by taking the argmax
    predicted_labels = np.argmax(probabilities_cpu, axis=1)
    
    # Convert one-hot labels to simple indices
    true_labels = np.argmax(labels, axis=1)
    
    # Calculate micro-averaged F1 score
    f1 = f1_score(true_labels, predicted_labels, average='micro')
    return f1

# Example usage
# Assume Roberta model output is already in probabilities
num_classes = 4
batch_size = 10

probabilities = torch.randn(batch_size, num_classes).softmax(dim=1) # Simulate probabilities
true_labels = np.eye(num_classes)[torch.randint(0, num_classes, (batch_size,)).numpy()] # Simulate one-hot labels

f1 = calculate_f1_from_probabilities(probabilities, true_labels)
print(f"F1 Score: {f1}")
```

In the code snippet above, I’ve used `torch.argmax()` to convert the probabilities into predicted class indices. This is the crucial step before using `sklearn.metrics.f1_score`, as that function requires class indices, not probabilities. Note the crucial conversion to CPU and NumPy for compatibility with `sklearn` and we also use `average='micro'` for multi-class tasks. Micro-averaging will average all the samples to get one global F1 score. It’s best practice for multilabel and multiclass cases.

Let's take a look at another example in case you have logits as outputs. In that scenario, we will apply softmax function first before taking the argmax.

```python
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
import numpy as np

def calculate_f1_from_logits(logits, labels):
    """
    Calculates the F1 score from one-hot encoded labels, applying softmax.

    Args:
        logits: PyTorch tensor of shape [batch_size, num_classes] containing the logits
        labels: Numpy array of shape [batch_size, num_classes] containing the true one-hot labels

    Returns:
        float: The micro-averaged F1 score.
    """
    # Ensure logits are on CPU and converted to NumPy for sklearn
    logits_cpu = logits.cpu().detach().numpy()

    # Convert logits to probabilities using softmax
    probabilities = F.softmax(torch.tensor(logits_cpu), dim=1)
    
    # Convert probabilities to predictions by taking the argmax
    predicted_labels = np.argmax(probabilities.cpu().detach().numpy(), axis=1)
    
    # Convert one-hot labels to simple indices
    true_labels = np.argmax(labels, axis=1)
    
    # Calculate micro-averaged F1 score
    f1 = f1_score(true_labels, predicted_labels, average='micro')
    return f1

# Example usage
# Assume Roberta model output is already in logits
num_classes = 4
batch_size = 10

logits = torch.randn(batch_size, num_classes)  # Simulate logits
true_labels = np.eye(num_classes)[torch.randint(0, num_classes, (batch_size,)).numpy()]  # Simulate one-hot labels

f1 = calculate_f1_from_logits(logits, true_labels)
print(f"F1 Score: {f1}")
```

The difference from the previous example is that here we first apply a softmax function on the logits to get probabilities before calling `argmax`. I have also made it explicit that the logits are converted into a pytorch tensor before the softmax as `F.softmax` can take only pytorch tensors as an input.

Now, there's another subtle, but often important detail: the `average` parameter in the `f1_score` function. For multi-class classifications, you almost always want `average='micro'`. This option calculates the F1 globally by considering total true positives, false negatives, and false positives. Alternatively, you could use macro averaging (`average='macro'`), which calculates the F1 score separately for each class and then averages those scores. However, in general, micro averaging is preferred for multi-class cases.

Finally, it's good practice to validate on a dedicated dataset. If you're training a model and you do not want to measure F1 on training data. You should use a held-out validation dataset that your model has never seen.

In terms of further reading, I'd recommend delving into the detailed explanation in *'Pattern Recognition and Machine Learning'* by Christopher Bishop. Specifically, look into the chapter on performance evaluation of classifiers. A good reference on evaluation metrics in general is also *'Data Mining: Practical Machine Learning Tools and Techniques'* by Ian H. Witten et al. These books will provide a solid statistical foundation to the practices described here.

And if you want a detailed read on different averaging methods, have a look at the paper titled 'A Note on Precision and Recall' by David D. Lewis, which details how metrics should be averaged for multi class problems.

There’s also plenty of documentation on the scikit-learn library that you can leverage directly, and specifically the `sklearn.metrics.f1_score` documentation will provide information for using the metrics correctly.

Let’s get to another useful code snippet that handles batch iteration for the predictions.

```python
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
import numpy as np

def evaluate_model(model, dataloader):
    """
    Evaluates the model on a dataset and calculates the F1 score.

    Args:
        model: A PyTorch model.
        dataloader: PyTorch DataLoader providing batches of inputs and labels.

    Returns:
        float: The micro-averaged F1 score across the entire dataset.
    """
    model.eval()  # Set model to evaluation mode
    all_predicted_labels = []
    all_true_labels = []

    with torch.no_grad():  # Disable gradient calculations during evaluation
        for inputs, labels in dataloader:

          outputs = model(inputs) # Get the model output which can be logits or probabilities. 

          if hasattr(outputs, 'logits'):
            logits = outputs.logits
            predicted_batch = torch.argmax(F.softmax(logits, dim=1), dim=1).cpu().numpy()
          else: # Assume we have raw probabilities as outputs
            predicted_batch = torch.argmax(outputs, dim=1).cpu().numpy() # Use softmax if it's raw logits

          true_batch = np.argmax(labels.cpu().numpy(), axis=1) # Assuming one-hot encoded labels
            
          all_predicted_labels.extend(predicted_batch)
          all_true_labels.extend(true_batch)

    f1 = f1_score(all_true_labels, all_predicted_labels, average='micro')
    return f1


# Example usage (Assuming you have a defined model and dataloader)
class DummyModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(DummyModel, self).__init__()
        self.linear = torch.nn.Linear(10, num_classes)

    def forward(self, x):
        return self.linear(x)

num_classes = 4
model = DummyModel(num_classes)
batch_size = 10
dummy_data = torch.randn(batch_size, 10)
dummy_labels = np.eye(num_classes)[torch.randint(0, num_classes, (batch_size,)).numpy()]
dummy_dataset = list(zip(dummy_data, dummy_labels))
dummy_dataloader = torch.utils.data.DataLoader(dummy_dataset, batch_size=batch_size)

f1 = evaluate_model(model, dummy_dataloader)
print(f"F1 Score (evaluated over all batches): {f1}")
```

This example highlights that a robust evaluation system should iterate over the dataset with data loaders, compute predictions, and evaluate against true labels. This example allows for a flexible evaluation setup where we handle both logits and probabilities to get the appropriate prediction array.

I hope this detailed explanation helps. It’s always about careful data handling and applying the right functions at the right time. Let me know if you have more specific situations or need clarifications.
