---
title: "Why do probability outputs from a BERT fine-tuned language model differ from predicted labels in a classification task?"
date: "2025-01-30"
id: "why-do-probability-outputs-from-a-bert-fine-tuned"
---
The discrepancies between a BERT model's probability outputs and its predicted labels, particularly in a classification setting, stem from a fundamental difference in their intended purpose and underlying mechanisms. Probabilities quantify the model's *confidence* in its prediction across all possible classes, while the predicted label is simply the class with the highest associated probability. This is not a flaw, but an inherent characteristic of how these models function and how softmax activation is employed. I've observed this frequently in my work building sentiment analysis models, where a model might assign a probability of 0.6 to 'positive,' 0.3 to 'neutral,' and 0.1 to 'negative,' yet definitively label the input as 'positive.' These probabilities are critical, as they communicate the relative likelihood across classes rather than guaranteeing the correctness of the single predicted label.

The heart of this discrepancy lies in how BERT, and other similar transformer models, operate on input data. After encoding the text input, the final hidden state from the [CLS] token is often used as an aggregated representation of the entire sequence. This representation is then passed through a linear layer and subsequently through a softmax activation function. This softmax function transforms raw output scores (logits) into a probability distribution. The softmax is crucial; it normalizes the raw scores into probabilities that sum up to 1 across all classes, allowing for a comparative analysis of which class the model considers the most probable.

Crucially, even when a predicted label is highly probable, it does not imply the model is “certain.” A probability of 0.9 for class A, for instance, does not guarantee A is correct; it only signifies that, relative to other classes, the model considers class A nine times more likely given the provided input. The probability outputs, therefore, are not binary indicators of correctness, but rather graded likelihoods which allow us to understand the model's internal state. High probabilities are desired but do not eradicate the potential for error. Similarly, low probabilities should not discourage investigation. A low-probability prediction could be the result of insufficient training data or the inherent ambiguity in the input text itself.

Furthermore, the model's training objective plays a critical role. Typically, BERT is fine-tuned using cross-entropy loss, which encourages the model to assign higher probabilities to the correct labels and lower probabilities to incorrect ones during training. This optimization is conducted not on individual labels, but the entire probability vector. Thus, during inference, the model's probability outputs still reflect this learnt distribution and are not directly geared to maximize the margin of the predicted label. It is optimized to minimize divergence between its output and actual labels across a batch of data. It’s optimizing for the entire probability distribution, not the highest single predicted label.

Here are some code examples that demonstrate this difference, using Python with the PyTorch framework (and a simplified example):

**Example 1:  Basic Classification and Probability Extraction**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Fictional Model
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        logits = self.linear(x)
        return logits

# Dummy Data
input_dim = 10
num_classes = 3
batch_size = 2
input_data = torch.randn(batch_size, input_dim)


# Model and Prediction
model = SimpleClassifier(input_dim, num_classes)
model.eval() # Set to evaluation mode
with torch.no_grad():
  logits = model(input_data)
  probabilities = F.softmax(logits, dim=1)
  predicted_labels = torch.argmax(probabilities, dim=1)

print("Logits:\n", logits)
print("Probabilities:\n", probabilities)
print("Predicted Labels:\n", predicted_labels)
```

*Commentary:* This code snippet illustrates the fundamental steps involved in obtaining predictions from a simple classifier model.  The `logits` are the raw scores output by the linear layer and are not normalized. Applying the `softmax` function converts these logits to probabilities. `torch.argmax` then returns the index of the maximum probability, effectively yielding the predicted class label. Observe that despite the argmax picking a single label, we retain access to the underlying probability distribution which reveals the model's confidence across all classes.

**Example 2: Visualizing the Discrepancy with a Specific Case**

```python
import torch
import torch.nn.functional as F

# Example logits
logits = torch.tensor([[1.5, 0.8, -0.2],
                       [0.1, 2.3, 1.8]])

probabilities = F.softmax(logits, dim=1)
predicted_labels = torch.argmax(probabilities, dim=1)

print("Logits:\n", logits)
print("Probabilities:\n", probabilities)
print("Predicted Labels:\n", predicted_labels)
```
*Commentary:* In this simplified example using manually defined logits, we can clearly observe how the `softmax` function transforms logits to probabilities. For instance, for the first data point, the logit with the highest value is 1.5, resulting in a predicted label of 0. However, the probabilities show that, whilst highest, it is not an overwhelming certainty. The probability for index 0 is approximately 0.58, compared to 0.31 for index 1. This highlights that even if a label is the predicted one (argmax), the probability distribution helps in further understanding of the model's confidence. A well-trained model should exhibit much higher probability assigned to the predicted label.

**Example 3: Incorporating Batch Processing with a Dummy Data**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Fictional Model
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        logits = self.linear(x)
        return logits

# Dummy Data
input_dim = 5
num_classes = 2
batch_size = 4
input_data = torch.randn(batch_size, input_dim)

# Model
model = SimpleClassifier(input_dim, num_classes)
model.eval()

with torch.no_grad():
  logits = model(input_data)
  probabilities = F.softmax(logits, dim=1)
  predicted_labels = torch.argmax(probabilities, dim=1)

print("Logits:\n", logits)
print("Probabilities:\n", probabilities)
print("Predicted Labels:\n", predicted_labels)
```

*Commentary:* This code demonstrates the same process as the previous examples but using batch processing. We now process multiple examples concurrently which is typical in real-world applications. Note how the code treats the batch and obtains probabilities, where each row corresponds to probability distribution across the classes for a given input sample and subsequently predicts the single label using argmax. The output demonstrates how even when operating on batches, the same principle applies with the model generating a probability distribution across classes, although it produces a single label as its prediction per sample. This emphasizes that even in batches, the probability distribution for each instance is crucial.

For further exploration, I would recommend studying the following topics, without requiring specific web resources:
*   **Softmax Activation Function:** A fundamental concept when discussing probabilities derived from any neural network. Understanding how it transforms logits into a probability distribution is critical.
*   **Cross-Entropy Loss:** This is the standard loss function used for classification tasks and will clarify the training goal of maximizing the probability of correct labels, as it minimizes divergence between predicted and true probability distribution.
*   **Model Evaluation Metrics:** Beyond the predicted label, it is beneficial to delve into evaluation metrics such as AUC and log-loss to better appreciate the entire output distribution of your model, not simply the single predicted labels.
*   **Bayesian Deep Learning:** For those seeking more sophisticated ways to interpret model uncertainty, this area provides techniques to generate more reliable probability estimations.

In summary, the difference between probability outputs and predicted labels is not an error or a shortcoming but is a deliberate outcome of how the softmax activation function and classification models are structured. The probabilities give you a view into the model’s uncertainty and confidence, while the predicted label provides a single definitive result. For proper understanding of performance, you need to evaluate the entire probability distribution and the predicted labels separately and simultaneously.
