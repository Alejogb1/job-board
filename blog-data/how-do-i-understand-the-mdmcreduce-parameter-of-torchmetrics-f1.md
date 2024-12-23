---
title: "How do I understand the `mdmc_reduce` parameter of torchmetrics F1?"
date: "2024-12-23"
id: "how-do-i-understand-the-mdmcreduce-parameter-of-torchmetrics-f1"
---

Let’s tackle this one; it’s a question that's tripped up quite a few folks, including myself when I first started using `torchmetrics`. It looks straightforward at first glance, but `mdmc_reduce` in the context of `torchmetrics.F1` (and other metrics, for that matter) actually packs a bit of complexity that's worth unraveling. It's essentially about handling multi-dimensional outputs, specifically in a multi-label classification scenario.

See, when we talk about binary or even multi-class classification, the shape of our predictions and targets often aligns quite nicely—usually as single vectors or matrices. But when you introduce multi-label classification, where an instance can belong to multiple classes *simultaneously*, things get a little more intricate. That’s where `mdmc_reduce` comes in, dictating how we aggregate the F1 scores calculated across different dimensions of our output.

My first real encounter with this issue arose during a project involving the analysis of satellite imagery. We were trying to classify different land cover types, and any given patch could easily contain more than one category—say, both "forest" and "grassland". The initial output of our model was, therefore, something like a one-hot encoded tensor for each pixel, where each dimension corresponded to a distinct land cover category. This was far from the simpler single-label scenario I was initially accustomed to.

The challenge is that `torchmetrics.F1`, by default, computes F1 scores individually for each class (or label). So, with our multi-label satellite imagery project, each patch had *multiple* F1 scores – one for each land cover category we were tracking. This meant we’d need a strategy to distill it down to a single overall score to track our model's progress. That's precisely what `mdmc_reduce` controls.

`mdmc_reduce` offers us three primary options, and each of them influences how we aggregate those per-label scores into a single representative metric. These options are `"global"`, `"samplewise"`, and `"macro"` (sometimes you'll see "none" too, though that's usually for diagnostic purposes, and not as the ultimate aggregation). The key difference lies in the scope at which the aggregation happens.

*   **`"global"`**: This option treats all instances and all labels as one giant pool of data. Precision, recall, and hence F1 are calculated based on the *sum* of all the true positives, false positives, and false negatives aggregated across all samples and all labels. It effectively considers the performance of your model overall, across all labels and instances. You get one singular F1 score that gives an idea of the overall performance. This is useful when you're most concerned about the overall system performance.

*   **`"samplewise"`**: The samplewise strategy first computes the F1 scores *for each sample individually*, across *all labels* associated with that sample. It then averages these per-sample F1 scores to give a single aggregated F1 for the entire batch (or evaluation set). This option is beneficial when you want to assess how well your model performs on *average* at classifying each instance, regardless of the distribution of labels. If some samples have many labels and some have only a few, samplewise gives each individual sample equal weight, in terms of the contribution it makes to the final score.

*   **`"macro"`**: This is the default value for `torchmetrics.F1` and likely what you’ll see when reading the documentation. Macro averaging calculates the F1 score *for each label individually*, and then averages these per-label F1 scores across all labels. This method gives equal weight to each class (label) regardless of class frequency. This can be especially beneficial if you're concerned about under-represented classes that might be easily overlooked if using the "global" aggregation method.

To make this more concrete, let me share some code snippets. Consider a multi-label classification problem with three classes and a batch of four samples.

**Example 1: `mdmc_reduce="global"`**

```python
import torch
import torchmetrics

# Dummy data - using floats so we can showcase all cases
preds = torch.tensor([[0.8, 0.2, 0.6],
                     [0.1, 0.9, 0.3],
                     [0.7, 0.4, 0.5],
                     [0.2, 0.3, 0.8]])
target = torch.tensor([[1, 0, 1],
                      [0, 1, 0],
                      [1, 1, 0],
                      [0, 0, 1]])

f1 = torchmetrics.F1(mdmc_reduce='global', num_classes=3, average='none', threshold=0.5)
f1_score = f1(preds, target)
print(f"Global F1 score: {f1_score}")  # Output: a single value like 0.7142
```

In this first example, `mdmc_reduce` is set to "global". Notice that `average` is set to `'none'` within the F1 constructor. This will output the precision and recall at a per-class level initially, but `mdmc_reduce='global'` will proceed to compute true positives, false positives and false negatives *across all classes* in the batch when the metric is called by giving us an overall F1. The `threshold` parameter dictates the cut-off for considering a prediction as 'positive' or 'negative', essential in the multi-label scenario. This threshold is typically 0.5 for probabilities, but it may be adjusted depending on the nature of the task.

**Example 2: `mdmc_reduce="samplewise"`**

```python
import torch
import torchmetrics

preds = torch.tensor([[0.8, 0.2, 0.6],
                     [0.1, 0.9, 0.3],
                     [0.7, 0.4, 0.5],
                     [0.2, 0.3, 0.8]])
target = torch.tensor([[1, 0, 1],
                      [0, 1, 0],
                      [1, 1, 0],
                      [0, 0, 1]])

f1 = torchmetrics.F1(mdmc_reduce='samplewise', num_classes=3, threshold=0.5)
f1_score = f1(preds, target)
print(f"Samplewise F1 score: {f1_score}")  # Output: a single value like 0.70
```

Here, the `mdmc_reduce` is set to “samplewise”. The output will be an average of F1 scores, calculated individually for each of the four samples, across the multiple labels. So each sample has its F1 computed and then these F1 scores are averaged to give us the overall metric.

**Example 3: `mdmc_reduce="macro"`**

```python
import torch
import torchmetrics

preds = torch.tensor([[0.8, 0.2, 0.6],
                     [0.1, 0.9, 0.3],
                     [0.7, 0.4, 0.5],
                     [0.2, 0.3, 0.8]])
target = torch.tensor([[1, 0, 1],
                      [0, 1, 0],
                      [1, 1, 0],
                      [0, 0, 1]])

f1 = torchmetrics.F1(mdmc_reduce='macro', num_classes=3, threshold=0.5)
f1_score = f1(preds, target)
print(f"Macro F1 score: {f1_score}") # Output: a single value like 0.70
```

In this final example, `mdmc_reduce` is set to “macro”, and we'll find that the final result is a single, scalar F1 score that represents the *average* F1 across each individual class label.

When choosing the appropriate `mdmc_reduce` method for your multi-label classification problem, several considerations must be made. If you are interested in seeing the overall performance of the model, `global` would be suitable; if you are concerned with the average performance per-sample, then `samplewise` is recommended. Alternatively, if you would like to weight classes equally regardless of their sample frequency, `macro` is the way to go.

For anyone aiming to further deepen their understanding of multi-label classification and evaluation metrics, I recommend the following sources:

*   **"Pattern Recognition and Machine Learning" by Christopher Bishop**: This textbook provides a solid foundation in machine learning fundamentals, including a discussion of evaluation metrics and different classification scenarios.

*   **The Scikit-learn documentation:** While it's not about pytorch, the sklearn documentation section on `sklearn.metrics` provides a nice overview on different averaging methods and its effect on overall performance scores, this information translates quite well to `torchmetrics`.

*   **Research papers on multi-label classification:** Specifically look for papers that address evaluation metrics in multi-label scenarios and the considerations that come with them. This can be quickly uncovered on Google Scholar via the search “Multi-label classification metrics”.

Understanding `mdmc_reduce` is essential when working with `torchmetrics` in multi-label problems. It dictates how you aggregate per-class F1 scores into a meaningful representation of your model's performance. Choosing the right aggregation method will be crucial to drawing accurate conclusions about your model's success, and this will be dependent on your specific needs and aims for the project at hand.
