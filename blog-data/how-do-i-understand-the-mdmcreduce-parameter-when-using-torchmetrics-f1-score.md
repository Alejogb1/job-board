---
title: "How do I understand the 'mdmc_reduce' parameter when using torchmetrics F1 Score?"
date: "2024-12-23"
id: "how-do-i-understand-the-mdmcreduce-parameter-when-using-torchmetrics-f1-score"
---

Alright, let's tackle this. I remember a project a few years back involving a particularly complex multi-label classification problem. I was banging my head against the wall trying to get the f1 score to behave as expected, and it turned out the `mdmc_reduce` parameter in torchmetrics' `F1Score` was the culprit—or rather, my lack of understanding of it. It's a crucial detail, especially when working with multi-dimensional data, so let's break it down in a way that hopefully makes sense.

The `mdmc_reduce` parameter, short for "multi-dimensional multi-class reduce," determines how the f1 score is computed when you have a situation where each sample can belong to multiple classes simultaneously (multi-label) and possibly across different dimensions (multi-dimensional). Think of it like this: you're not just classifying an image as either a cat or a dog; you could be classifying parts of an image (regions) and each of those regions can have multiple tags. This introduces complexity that the standard binary or even simple multi-class f1 scores can’t adequately address.

The crux lies in what happens when torchmetrics encounters multiple labels for a single sample, and possibly in different dimensions of the output tensor. The `mdmc_reduce` parameter defines how those individual class-level f1 scores are aggregated to produce a single overall score. Ignoring it or using the wrong reduction strategy can lead to wildly misleading results. There are several options: `'global'`, `'samplewise'`, and `'macro'`, each suited to a specific scenario.

First, `'global'` computes the f1 score by considering all samples and all labels across all dimensions as one giant aggregate. This means that every true positive, false positive, and false negative across your entire dataset and all classes is counted in a single calculation. This can be a good choice if you are primarily interested in the overall performance and don’t care about class or per-sample performance.

Then, we have `'samplewise'`. This calculates the f1 score *per sample* and then averages those sample-specific scores. In a multi-label scenario, for each sample, it determines the true positives, false positives, and false negatives for all its labels combined, computes the sample's f1 score, and finally averages these sample scores across the dataset. This is useful when performance varies significantly from sample to sample and you want to understand this diversity.

Finally, `'macro'` computes the f1 score for each class across all samples and dimensions first, then averages these class-specific f1 scores. This means you are effectively treating each class independently and then looking at the average. This can be really beneficial when your classes are imbalanced, as the macro average doesn't give greater importance to common classes compared to rare ones.

To clarify with code, let’s look at a series of examples. Let's begin with a basic example using the `'global'` reduction, where everything is bundled into one computation. This is my default approach for preliminary performance analysis.

```python
import torch
from torchmetrics import F1Score

# Assume two samples, three possible classes
preds = torch.tensor([[1, 0, 1], [0, 1, 1]])
target = torch.tensor([[1, 0, 0], [0, 1, 1]])

f1 = F1Score(task="multilabel", num_classes=3, mdmc_reduce='global')
result = f1(preds, target)
print(f"Global F1 Score: {result}")
```

In this case, everything is flattened. You are not looking at performance per sample or per class but as an aggregate. If you’re interested in getting more granularity, the other reduction options are the way to go.

Now, let's consider the `samplewise` option. This focuses on how well each *sample* performs.

```python
import torch
from torchmetrics import F1Score

# Assume two samples, three possible classes
preds = torch.tensor([[1, 0, 1], [0, 1, 1]])
target = torch.tensor([[1, 0, 0], [0, 1, 1]])


f1 = F1Score(task="multilabel", num_classes=3, mdmc_reduce='samplewise')
result = f1(preds, target)
print(f"Samplewise F1 Score: {result}")
```

Here, the average per-sample f1 score is returned. We’re not looking at classes specifically, but rather the aggregate of labels in each sample independently and then averaged.

Finally, let’s examine the `'macro'` option, which helps us understand the performance of each class irrespective of imbalance.

```python
import torch
from torchmetrics import F1Score

# Assume two samples, three possible classes
preds = torch.tensor([[1, 0, 1], [0, 1, 1]])
target = torch.tensor([[1, 0, 0], [0, 1, 1]])

f1 = F1Score(task="multilabel", num_classes=3, mdmc_reduce='macro')
result = f1(preds, target)
print(f"Macro F1 Score: {result}")
```

Here, you get an average of all class-specific f1 scores. This can be the most informative when dealing with imbalanced datasets. Choosing the wrong `mdmc_reduce` setting can severely skew your perception of model performance.

Now, regarding learning resources, I highly recommend diving into the original paper by Manning, Raghavan, and Schütze, "Introduction to Information Retrieval." While not directly about torchmetrics, it covers the foundational concepts of precision, recall, and the f1 score (and also goes through different kinds of averaging techniques) in substantial detail, giving you the crucial context needed to understand why these reduction strategies exist. Also, read the documentation for `torchmetrics` carefully, particularly the section pertaining to `F1Score`, which provides precise explanations and examples for all the parameters. In addition, explore the scikit-learn documentation, specifically focusing on the multi-label evaluation metrics. Often, the explanations there are very helpful in making the concepts clearer.

In conclusion, the `mdmc_reduce` parameter within the torchmetrics `F1Score` is not some esoteric detail; it is essential for correct evaluation of multi-label and multi-dimensional classification models. By understanding the specific implications of each setting—`'global'`, `'samplewise'`, and `'macro'`—you can ensure you are accurately measuring and understanding the performance of your models. Choosing the right reduction method depends heavily on your problem and goals. It is not a one-size-fits-all scenario. This is a skill you build through practice and careful analysis of the results from different reduction options. Spend time experimenting and make sure to properly evaluate your choices. It's definitely worth the effort.
