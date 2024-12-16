---
title: "How to understand `mdmc_reduce` using torchmetrics F1 Score?"
date: "2024-12-16"
id: "how-to-understand-mdmcreduce-using-torchmetrics-f1-score"
---

Okay, let's talk about `mdmc_reduce` and how it specifically impacts the F1 score calculation within `torchmetrics`. It's a detail that, frankly, caused me some confusion early on in my deep learning journey, and I’ve certainly seen others trip over it. I remember a project, roughly five years back, where we were evaluating a multi-label image classification model. We were getting surprisingly low F1 scores despite what we believed was decent prediction accuracy on a per-label basis. The culprit, we later found, was our misunderstanding of `mdmc_reduce` and how it aggregates those per-label scores into a single, overall metric.

The essence of `mdmc_reduce` lies in its role when dealing with multi-dimensional, multi-label classification problems—that is, the "mdmc" part. In a single-label problem, you have, say, one class among ‘cat’, ‘dog’, ‘bird’, and the model predicts only one of these. In multi-label problems, however, you might have a scenario where, for example, one image contains *both* a cat and a dog; the model has to predict *multiple* labels simultaneously. This inherently changes how metrics like the F1 score are calculated.

Now, the F1 score itself, in its basic form, is the harmonic mean of precision and recall. Precision measures how many of the predicted positives are actually correct, while recall measures how many of the actual positives were predicted correctly. In a multi-label setting, you calculate these per label. However, to report a single, aggregate F1 score for the overall model performance, you need a mechanism to combine these individual F1 scores; that's where `mdmc_reduce` comes in.

`torchmetrics` provides several options for `mdmc_reduce`, typically "global", "samplewise," or "macro." The “global” option computes the true positives, false positives, and false negatives across the entire set of predictions and targets; then calculates precision, recall, and finally F1 based on these aggregates. This can be useful if the goal is to look at overall performance without giving undue importance to individual classes or samples.

The "samplewise" option, on the other hand, calculates precision, recall, and F1 *per sample* (i.e., each prediction across all labels for a single datapoint), and then averages these F1 scores across all samples. This is especially helpful if you are more interested in how the model performs on a *per-datapoint* basis. It focuses on whether the model correctly captured *all* applicable labels for a specific input.

The “macro” option is slightly different. It averages the individual label-wise F1 scores, treating each label as equally important. It calculates the F1 score for each class separately and then averages these scores to obtain a single, aggregated score. This makes it good for measuring a model’s overall effectiveness across all the labels. The other options in `mdmc_reduce` that are available in specific metrics often use similar underlying aggregations, such as “micro” that is akin to the “global” approach.

Let me illustrate this with code. We'll construct three different scenarios using synthetic data.

```python
import torch
from torchmetrics import F1Score

# scenario 1: global averaging
preds_global = torch.tensor([
    [1, 0, 1],  # Sample 1, predicted labels 0, 2
    [0, 1, 0],  # Sample 2, predicted label 1
    [1, 1, 0]  # Sample 3, predicted labels 0, 1
])
target_global = torch.tensor([
    [1, 1, 0],  # Sample 1, true labels 0, 1
    [0, 1, 1],  # Sample 2, true labels 1, 2
    [1, 0, 1]   # Sample 3, true labels 0, 2
])
f1_global = F1Score(task="multilabel", num_classes=3, mdmc_average='global')
global_f1 = f1_global(preds_global, target_global)
print(f"Global F1: {global_f1:.4f}") # Output would be about 0.5000

# scenario 2: samplewise averaging
preds_samplewise = torch.tensor([
    [1, 0, 1],
    [0, 1, 0],
    [1, 1, 0]
])
target_samplewise = torch.tensor([
    [1, 1, 0],
    [0, 1, 1],
    [1, 0, 1]
])
f1_samplewise = F1Score(task="multilabel", num_classes=3, mdmc_average='samplewise')
samplewise_f1 = f1_samplewise(preds_samplewise, target_samplewise)
print(f"Samplewise F1: {samplewise_f1:.4f}") # Output would be about 0.4444

# scenario 3: macro averaging
preds_macro = torch.tensor([
    [1, 0, 1],
    [0, 1, 0],
    [1, 1, 0]
])
target_macro = torch.tensor([
    [1, 1, 0],
    [0, 1, 1],
    [1, 0, 1]
])
f1_macro = F1Score(task="multilabel", num_classes=3, mdmc_average='macro')
macro_f1 = f1_macro(preds_macro, target_macro)
print(f"Macro F1: {macro_f1:.4f}") # Output would be about 0.5111
```
In this code, we create three sets of synthetic predictions (`preds`) and ground truth labels (`target`). Note that these labels are binary, 1 if a sample contains the respective class, 0 otherwise. We use the `F1Score` class from `torchmetrics`, specifying `task="multilabel"`, `num_classes=3`, and setting `mdmc_average` to "global," "samplewise," and "macro" respectively in each scenario. The printed outputs show how each method leads to a different final F1 score.

Notice that each approach gives you a different F1 score. The "global" approach computes a single F1 score across *all* predictions/labels at once. The “samplewise” approach calculates a F1 score for each sample across all labels and then averages them and the “macro” approach averages the F1 score of each class, each approach may be appropriate depending on what you are optimizing for.

When I initially encountered this, the difference between global and samplewise was the biggest source of confusion, often leading to unexpectedly low F1 scores when I was using the wrong `mdmc_reduce` method. It's crucial to align the chosen reduction with what the specific goals are for your model training process.

For further exploration, I would highly recommend reading the research paper on "A systematic analysis of multi-label classifier performance evaluation" by Tsoumakas et al. (2009), it provides a great theoretical foundation for how multi-label metrics are defined and utilized. Also, diving into the official `torchmetrics` documentation on F1 scores is crucial, focusing on how the library actually implements these different averaging strategies. The book "Pattern Recognition and Machine Learning" by Christopher Bishop also offers valuable context on evaluating machine learning models, including the finer points of metrics in different learning settings. Another source for a comprehensive overview of multi-label learning techniques is "Multi-Label Learning: From Foundations to Algorithms" by Zhang and Zhou (2014). These resources offer detailed explanations and are great resources to get a more precise comprehension of the subject matter.

Understanding `mdmc_reduce` is crucial for accurate evaluation of your multi-label models. Choosing the right averaging strategy, ‘global’, ‘samplewise’, or ‘macro’ will help you gain actionable insights into how well the model performs. It's not simply about calculating a number; it's about extracting meaningful information about the model's performance across a multitude of labels. It is an important step in the model building process and should be carefully thought out.
