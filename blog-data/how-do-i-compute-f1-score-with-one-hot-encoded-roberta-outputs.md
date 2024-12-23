---
title: "How do I compute F1 score with one-hot encoded Roberta outputs?"
date: "2024-12-16"
id: "how-do-i-compute-f1-score-with-one-hot-encoded-roberta-outputs"
---

 One-hot encoding the outputs of a Roberta model, particularly when evaluating performance, can introduce some interesting nuances when calculating the F1 score. I’ve certainly been down this road before, wrestling (, perhaps 'tackling') similar scenarios on past NLP projects involving multi-class classification where interpreting model output correctly was critical.

The core issue stems from the fact that the output of a Roberta model (or any transformer-based model, for that matter) is typically a probability distribution over classes. One-hot encoding transforms this probability distribution into a discrete prediction, essentially assigning a single class label to each instance. This is where the subtleties arise when calculating the F1 score. Recall that the F1 score is the harmonic mean of precision and recall, and both precision and recall are inherently tied to the concept of true positives, false positives, and false negatives, which in turn depends on proper categorical comparisons.

My team and I faced this during a particularly sticky project aimed at classifying customer support tickets into multiple pre-defined categories. Our initial approach involved simply taking the *argmax* of the output logits (the raw scores before softmax) and treating that as our predicted label, one-hot encoding both the predictions and true labels accordingly. While this worked to some extent, a crucial error we were making was in neglecting the probabilistic nature of Roberta’s output before making our class prediction. We needed to calculate the F1 correctly within the confines of a multi-class output.

The typical approach involves the following steps. First, you transform the raw model outputs into probabilities using a softmax activation function. Second, for each instance (say a sentence or a document being classified) you take the *argmax* of these probabilities to determine the predicted class for that instance, thereby turning them into categorical values. Lastly, you use these predicted classes, alongside your actual true classes (also typically one-hot encoded) to determine the various F1 scores, usually, this involves reporting micro, macro, and weighted F1 scores.

To illustrate this with code, here are a few python examples using `numpy` and `scikit-learn`, which I found invaluable when working on those projects.

**Example 1: Basic F1 Calculation with One-Hot Encoded Outputs**

This example showcases the fundamental transformation of softmax outputs to predictions and the subsequent calculation of precision, recall and then the F1 scores.

```python
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_f1_basic(y_true_onehot, y_pred_logits):
    """
    Calculates f1 scores from one-hot encoded true labels and logits.
    """
    y_pred_probs = np.exp(y_pred_logits) / np.sum(np.exp(y_pred_logits), axis=1, keepdims=True) # Softmax
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_true_onehot, axis=1)


    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    return precision, recall, f1


if __name__ == '__main__':
    y_true_onehot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]]) # One-hot encoded true labels
    y_pred_logits = np.array([[2.1, 1.1, 0.2], [0.1, 2.5, 0.2], [0.1, 0.2, 3.1], [2.8, 0.2, 0.1]]) # Simulated logits output

    precision, recall, f1 = calculate_f1_basic(y_true_onehot, y_pred_logits)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
```

Here, the `calculate_f1_basic` function takes one-hot encoded true labels and the raw logit outputs as arguments. Inside the function, we initially apply the softmax activation to get the probabilistic predictions. Then, `np.argmax` yields the predicted classes. The `sklearn.metrics` provides the precision, recall, and F1 scores.

**Example 2: Handling No Predicted Positives Cases**

A problem I encountered was scenarios when the model didn’t predict *any* instances of certain classes, leading to a zero division error when calculating recall or precision for these classes independently. The solution is not simply avoiding errors but to make sure these cases contribute as expected when calculating micro, macro, or weighted scores.

```python
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelBinarizer

def calculate_f1_no_predicted_positives(y_true_onehot, y_pred_logits):
    """
    Calculates weighted f1 score from one-hot encoded true labels and logits,
    handling cases where no positive predictions are made for a particular class.
    """
    y_pred_probs = np.exp(y_pred_logits) / np.sum(np.exp(y_pred_logits), axis=1, keepdims=True)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_true_onehot, axis=1)


    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0) # zero_division=0 to handle potential errors
    return f1


if __name__ == '__main__':
    y_true_onehot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    y_pred_logits = np.array([[2.1, 1.1, 0.2], [0.1, 2.5, 0.2], [0.1, 0.2, 3.1], [2.8, 0.2, 0.1]])
    # forcing example that does not have class 1 predictions
    y_pred_logits = np.array([[2.1, -10, 0.2], [0.1, -10, 0.2], [0.1, -10, 3.1], [2.8, -10, 0.1]])


    f1 = calculate_f1_no_predicted_positives(y_true_onehot, y_pred_logits)
    print(f"Weighted F1 Score: {f1:.4f}")
```

In the example, I’ve modified the `y_pred_logits` to ensure that class '1' is never predicted (by setting the corresponding logit to a very low number). The `zero_division=0` argument in `f1_score` gracefully handles these cases by treating them as having zero precision/recall when calculating the weighted average.

**Example 3: Micro, Macro, and Weighted F1 Calculation**

Different averaging strategies can offer different insights. The third example computes micro, macro, and weighted f1 scores. This can be useful for investigating classification biases, especially when you have classes that are imbalanced.

```python
import numpy as np
from sklearn.metrics import f1_score

def calculate_f1_multi_average(y_true_onehot, y_pred_logits):
    """
    Calculates micro, macro, and weighted f1 scores from one-hot encoded true labels and logits.
    """
    y_pred_probs = np.exp(y_pred_logits) / np.sum(np.exp(y_pred_logits), axis=1, keepdims=True)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_true_onehot, axis=1)

    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    return f1_micro, f1_macro, f1_weighted


if __name__ == '__main__':
    y_true_onehot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    y_pred_logits = np.array([[2.1, 1.1, 0.2], [0.1, 2.5, 0.2], [0.1, 0.2, 3.1], [2.8, 0.2, 0.1]])

    f1_micro, f1_macro, f1_weighted = calculate_f1_multi_average(y_true_onehot, y_pred_logits)
    print(f"Micro F1 Score: {f1_micro:.4f}")
    print(f"Macro F1 Score: {f1_macro:.4f}")
    print(f"Weighted F1 Score: {f1_weighted:.4f}")
```

This function computes micro, macro, and weighted F1 scores using scikit-learn, offering a detailed picture of model performance. The `zero_division=0` parameter is important here as well to handle edge cases as in example 2.

For more in-depth understanding, I highly recommend checking out "Speech and Language Processing" by Daniel Jurafsky and James H. Martin. Also, delve into the scikit-learn documentation on classification metrics, specifically concerning precision, recall, and F1 score. Additionally, the seminal paper "Attention is All You Need" by Vaswani et al. (2017) will help solidify the foundations of transformer-based models like Roberta. Reading these resources will provide you with a robust theoretical foundation and practical approaches to accurately calculate and interpret the F1 scores.

In conclusion, calculating the F1 score with one-hot encoded Roberta outputs involves understanding how to transition from probability distributions to discrete predictions while accurately assessing the model's classification capability. The examples I’ve provided should give you a solid starting point. Be sure to handle cases where predicted positives for certain classes may be absent, and always be deliberate with your averaging strategies.
