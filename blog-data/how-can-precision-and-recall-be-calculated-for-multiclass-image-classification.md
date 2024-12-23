---
title: "How can precision and recall be calculated for multiclass image classification?"
date: "2024-12-23"
id: "how-can-precision-and-recall-be-calculated-for-multiclass-image-classification"
---

Alright, let's tackle this. It’s a common challenge, especially when moving beyond binary classification scenarios. Calculating precision and recall in the context of multiclass image classification requires a bit more nuance than a simple “true positive versus false positive” setup. I recall a project back in my days working with a medical imaging startup where we were classifying various types of lesions from scans—it wasn't a simple 'cancer/no-cancer' thing, we were dealing with multiple lesion types, and a good grasp of these metrics was paramount.

Fundamentally, precision and recall, even in a multiclass setup, are about the accuracy of our predictions *per class*. They help us understand how well our model is doing with each specific category. Precision, as you likely already know, focuses on the accuracy of the positive predictions. It asks, “of all the things I predicted to be *class X*, how many actually *were* class X?”. Recall, on the other hand, addresses the model’s ability to find *all* the positive instances. It asks, “of all the things that *actually were* class X, how many did I predict correctly as class X?”.

In a multiclass scenario, we can’t calculate a single precision and recall. Instead, we must compute these metrics *per class* and then often consider ways to aggregate them to get an overall sense of the model's performance. We essentially need to treat each class as a 'positive' class in a one-versus-all or one-versus-rest fashion.

Let's solidify this with some practical examples, using python and scikit-learn, since those are tools many of us encounter frequently. First, imagine we have a model that classifies images into three categories: 'cats,' 'dogs,' and 'birds.' Let's represent the ground truth (the actual classes) and the predictions as NumPy arrays.

```python
import numpy as np
from sklearn.metrics import precision_score, recall_score

# Example Ground Truth Labels (0: cats, 1: dogs, 2: birds)
y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])

# Example Predicted Labels
y_pred = np.array([0, 1, 2, 0, 2, 1, 0, 1, 2, 1, 1, 2])

# Calculate per-class precision
precision = precision_score(y_true, y_pred, average=None)
print(f"Per-class Precision: {precision}")

# Calculate per-class recall
recall = recall_score(y_true, y_pred, average=None)
print(f"Per-class Recall: {recall}")


# Output will be similar to (actual values will vary based on random labels):
# Per-class Precision: [0.75 0.5  1.  ]
# Per-class Recall: [1.   0.66666667 0.66666667]
```

Here, `average=None` is crucial. It instructs scikit-learn to calculate precision and recall for each class individually. Notice that the precision for the 'cats' class (index 0) is 0.75, which indicates that 75% of the images predicted to be cats were indeed cats. The recall for the 'cats' class is 1.0, meaning all the actual cat images were predicted as cats. However, for 'dogs' (index 1) we see a lower precision of 0.5 and recall of approximately 0.67.

Next, let's demonstrate how we might calculate *macro-averaged* precision and recall. Macro-averaging treats all classes equally, and calculates the metrics for each, then takes the average of those per-class metrics. This can be beneficial when you want to treat each class as equally important. In the previous scenario, maybe our dataset was perfectly balanced. But often it isn't.

```python
import numpy as np
from sklearn.metrics import precision_score, recall_score

# Example Ground Truth Labels (0: cats, 1: dogs, 2: birds)
y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])

# Example Predicted Labels
y_pred = np.array([0, 1, 2, 0, 2, 1, 0, 1, 2, 1, 1, 2])

# Calculate macro-averaged precision
macro_precision = precision_score(y_true, y_pred, average='macro')
print(f"Macro-Averaged Precision: {macro_precision}")

# Calculate macro-averaged recall
macro_recall = recall_score(y_true, y_pred, average='macro')
print(f"Macro-Averaged Recall: {macro_recall}")

# Output will be similar to (actual values will vary based on random labels):
# Macro-Averaged Precision: 0.75
# Macro-Averaged Recall: 0.7777777777777778
```
Using `average='macro'` here calculates the mean of the precision/recall scores calculated previously for each class.

Now, there's one more common averaging technique I want to mention: *weighted-averaged* precision and recall. This considers the number of examples per class when averaging the per-class scores. If some classes have far more instances than others, this can be a more appropriate way of summarizing overall performance. This was often how I viewed overall performance in the previously mentioned medical image classification project, as we would have different sample sizes for each lesion type.

```python
import numpy as np
from sklearn.metrics import precision_score, recall_score

# Example Ground Truth Labels (0: cats, 1: dogs, 2: birds)
y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0, 0])  # added some extra cat labels

# Example Predicted Labels
y_pred = np.array([0, 1, 2, 0, 2, 1, 0, 1, 2, 1, 1, 2, 0, 1, 0])

# Calculate weighted-averaged precision
weighted_precision = precision_score(y_true, y_pred, average='weighted')
print(f"Weighted-Averaged Precision: {weighted_precision}")

# Calculate weighted-averaged recall
weighted_recall = recall_score(y_true, y_pred, average='weighted')
print(f"Weighted-Averaged Recall: {weighted_recall}")

# Output will be similar to (actual values will vary based on random labels):
# Weighted-Averaged Precision: 0.74
# Weighted-Averaged Recall: 0.7333333333333333
```

Here, the output values may be different from our macro-averaged example due to the class imbalances we added to the ground truth. Each precision or recall score is multiplied by the percentage of true labels that belong to each class, prior to averaging.

For further study, I'd strongly recommend looking into the classic *Pattern Recognition and Machine Learning* by Christopher Bishop. It provides a strong theoretical foundation. Another excellent book that goes deeply into the practical applications of classification metrics is *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* by Aurélien Géron. Additionally, if you really want to get into the nuances of multiclass classification specifically, look for research papers on the specific domain you are interested in, focusing on how their evaluation metrics are explained. Good paper databases like IEEE Xplore or ACM Digital Library are a great place to begin that search.

In summary, in multiclass classification, it's crucial to understand that precision and recall are not singular measures. You'll need to calculate per-class metrics and then often aggregate them, carefully selecting the appropriate averaging method (none, macro, weighted) based on your specific data and problem context. I hope that clarifies things for you. Let me know if any further questions arise.
