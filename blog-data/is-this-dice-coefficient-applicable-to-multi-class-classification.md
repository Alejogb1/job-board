---
title: "Is this Dice coefficient applicable to multi-class classification?"
date: "2024-12-23"
id: "is-this-dice-coefficient-applicable-to-multi-class-classification"
---

Alright, let’s talk about the Dice coefficient and its applicability to multi-class classification. It’s a topic that, frankly, I’ve spent more hours than I care to count both researching and debugging in various image segmentation projects, particularly with satellite imagery data which often requires multi-class handling. So, let me break this down from a practical perspective, based on lessons learned, rather than just reciting textbook definitions.

The core concept of the Dice coefficient, also known as the F1 score when derived from the confusion matrix, is centered around measuring the overlap between two sets, typically the predicted segmentation and the ground truth segmentation. Mathematically, it’s calculated as `2 * |A ∩ B| / (|A| + |B|)`. Essentially, we’re doubling the size of the intersection and dividing it by the sum of the sizes of both sets. This works exceptionally well for binary classification problems, where you have a single class of interest (like identifying “tumors” vs. “not tumors”), because you're directly comparing the overlap of predicted and actual instances of that single class.

However, the story gets more nuanced when you move to multi-class scenarios – like differentiating between ‘urban areas,’ ‘forest,’ and ‘agricultural land’ in satellite images. Now, we have multiple classes to consider. The standard, straightforward Dice coefficient calculation won’t suffice, and applying it directly could lead to misleading metrics. We need to think about how to generalize this for more than two classes.

Here's the essential adaptation: we typically calculate a *per-class* Dice coefficient. This means that for each class, we treat it as the "positive" class and all the others as the "negative" class. We compute the Dice score for that class, and then we might average them to get a single global metric. Now, this average can be an arithmetic mean, weighted mean, or something else depending on your project's specific needs, and each choice will reveal different aspects of your model's performance. Let me elaborate on these practical nuances.

Firstly, it's critical to understand how class imbalances affect the Dice coefficient in multi-class scenarios. If, say, you have way more instances of “forest” than “urban” or “agricultural land,” the Dice score for “forest” might be high simply because it has a larger number of potential correctly predicted pixels. This can mask significant underperformance in the other classes. In my experience, neglecting class imbalances here has often led me down the wrong path in tuning models, and led me to put more emphasis on overall numbers rather than the nuances of specific class performance. The unweighted arithmetic mean of per-class Dice scores can easily mask this issue.

To counter this, we often use weighted averages or metrics that are more robust against class imbalance. The *macro average* (arithmetic mean of per class scores) is suitable if you value each class equally. However, if some classes are more critical, you may opt for a *weighted average*, where each class’s score contributes proportionally to its support in your dataset (number of pixels belonging to that class).

Here are a few illustrative examples implemented in Python using `numpy`, because often, this is how one deals with these things at a low level:

**Example 1: Calculating per-class Dice and Macro Average:**

```python
import numpy as np

def calculate_dice_per_class(y_true, y_pred, num_classes):
    """Calculates per-class Dice coefficients.
    Args:
        y_true (np.ndarray): True labels as one-hot encoded array (num_samples, height, width, num_classes)
        y_pred (np.ndarray): Predicted labels as one-hot encoded array. (num_samples, height, width, num_classes)
        num_classes (int): Number of classes.
    Returns:
        np.ndarray: Dice coefficients for each class.
    """
    dice_scores = []
    for class_idx in range(num_classes):
        true_class = y_true[:,:,:,class_idx].flatten()
        pred_class = y_pred[:,:,:,class_idx].flatten()

        intersection = np.sum(true_class * pred_class)
        sum_sets = np.sum(true_class) + np.sum(pred_class)
        if sum_sets == 0:  # Handle cases where a class is absent in both true and pred
            dice = 1.0
        else:
           dice = 2.0 * intersection / sum_sets
        dice_scores.append(dice)
    return np.array(dice_scores)

def macro_dice(dice_per_class):
   return np.mean(dice_per_class)


# Example Usage
y_true_example = np.array([[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], [[[0, 1, 0], [1, 0, 0], [0, 0, 1]]]]) # Example ground truth
y_pred_example = np.array([[[[0, 1, 0], [0, 1, 0], [0, 0, 1]]], [[[0, 1, 0], [0, 1, 0], [1, 0, 0]]]]) # Example prediction

num_classes_example = 3

dice_scores = calculate_dice_per_class(y_true_example, y_pred_example, num_classes_example)
macro_avg = macro_dice(dice_scores)

print("Per-class Dice:", dice_scores)
print("Macro-average Dice:", macro_avg)


```

**Example 2: Calculating Weighted Average Dice**

```python
import numpy as np

def calculate_weighted_dice(y_true, y_pred, num_classes):
    """Calculates weighted average Dice coefficients.
    Args:
        y_true (np.ndarray): True labels as one-hot encoded array (num_samples, height, width, num_classes)
        y_pred (np.ndarray): Predicted labels as one-hot encoded array. (num_samples, height, width, num_classes)
        num_classes (int): Number of classes.
    Returns:
        float: Weighted Dice coefficient.
    """
    dice_scores = []
    weights = []
    for class_idx in range(num_classes):
      true_class = y_true[:,:,:,class_idx].flatten()
      pred_class = y_pred[:,:,:,class_idx].flatten()

      intersection = np.sum(true_class * pred_class)
      sum_sets = np.sum(true_class) + np.sum(pred_class)
      if sum_sets == 0:
          dice = 1.0 # handling when a class is absent
      else:
        dice = 2.0 * intersection / sum_sets
      dice_scores.append(dice)
      weights.append(np.sum(true_class))

    weights = np.array(weights)
    if np.sum(weights)==0: # To avoid division by zero
       return 0.0
    weighted_avg = np.sum(np.array(dice_scores) * weights) / np.sum(weights)
    return weighted_avg

# Example Usage
y_true_example_2 = np.array([[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], [[[0, 1, 0], [1, 0, 0], [0, 0, 1]]]]) # Example ground truth
y_pred_example_2 = np.array([[[[0, 1, 0], [0, 1, 0], [0, 0, 1]]], [[[0, 1, 0], [0, 1, 0], [1, 0, 0]]]]) # Example prediction

num_classes_example_2 = 3


weighted_avg = calculate_weighted_dice(y_true_example_2, y_pred_example_2, num_classes_example_2)


print("Weighted Average Dice:", weighted_avg)

```
**Example 3: Handling Edge Cases with Absent Classes**
```python
import numpy as np

def calculate_dice_per_class_edge_case(y_true, y_pred, num_classes):
  dice_scores = []
  for class_idx in range(num_classes):
      true_class = y_true[:,:,:,class_idx].flatten()
      pred_class = y_pred[:,:,:,class_idx].flatten()
      intersection = np.sum(true_class * pred_class)
      sum_sets = np.sum(true_class) + np.sum(pred_class)
      if sum_sets == 0: # This is the key - handing edge case where a class is absent.
          dice = 1.0 # Assuming no error, best dice, since both are absent
      else:
          dice = 2.0 * intersection / sum_sets
      dice_scores.append(dice)
  return np.array(dice_scores)



# Example Usage

# Case where class 0 is completely absent from both true and predicted
y_true_example_3a = np.array([[[[0, 1, 0], [0, 1, 0]]], [[[0, 1, 0], [0, 1, 0]]]])
y_pred_example_3a = np.array([[[[0, 1, 0], [0, 1, 0]]], [[[0, 1, 0], [0, 1, 0]]]])
num_classes_example_3a = 3

dice_scores_3a = calculate_dice_per_class_edge_case(y_true_example_3a, y_pred_example_3a, num_classes_example_3a)
print("Per-class Dice (case 3a - absent class 0):", dice_scores_3a) # Observe class 0 is 1.0

#Case where class 2 is absent only in true
y_true_example_3b = np.array([[[[0, 1, 0], [0, 1, 0]]], [[[0, 1, 0], [0, 1, 0]]]])
y_pred_example_3b = np.array([[[[0, 1, 1], [0, 1, 1]]], [[[0, 1, 1], [0, 1, 1]]]])
num_classes_example_3b = 3

dice_scores_3b = calculate_dice_per_class_edge_case(y_true_example_3b, y_pred_example_3b, num_classes_example_3b)
print("Per-class Dice (case 3b - absent class 2 in true):", dice_scores_3b)
```

In essence, using the Dice coefficient in multi-class classification requires careful consideration. You can't simply apply the binary version; you *must* adapt it to a per-class calculation and then consider how to aggregate them. Understanding the impact of class imbalance is crucial, and the choice between macro and weighted averaging depends on the specific objectives of your project. I have seen teams stumble upon inaccurate conclusions using the Dice scores because they glossed over these key details.

If you want to delve deeper into the theoretical aspects, I’d highly recommend exploring the paper “A review on evaluation metrics for multi-class classification focusing on imbalanced data” by Tharwat (2018) in Applied Computing and Informatics. For a solid foundation on classification metrics in general, "Pattern Classification" by Duda, Hart, and Stork is a classic. And for the practical implementation side, looking into the Scikit-learn documentation on metric calculation is always a worthwhile endeavor. Keep your mind on the nuances of each situation and remember there are no one-size fits all answers.
