---
title: "How can I plot confusion matrices using Wandb with Pytorch?"
date: "2024-12-16"
id: "how-can-i-plot-confusion-matrices-using-wandb-with-pytorch"
---

, let’s tackle this. I've plotted my fair share of confusion matrices, and integrating them with Wandb (Weights & Biases) for Pytorch workflows is something I’ve refined over time. Let me walk you through it, drawing from my experiences and best practices.

Plotting confusion matrices is crucial for understanding the performance of a classification model, going beyond basic accuracy metrics. Wandb offers excellent tools to log and visualize these matrices, allowing for better error analysis and model improvement. The trick is in formatting your data correctly and then using Wandb's APIs effectively. Let me illustrate with code snippets and explanations, using a scenario based on my time working on a medical imaging project.

Before jumping into specifics, remember the basic structure of a confusion matrix. It's a square matrix where rows typically represent the actual classes, and columns represent the predicted classes. The diagonal elements indicate correctly classified instances, while off-diagonal elements highlight misclassifications. This is key for identifying if our model is struggling with specific class combinations. Now, let's move to the practical implementation.

The first thing we need is the prediction data and ground truth labels from our Pytorch model. Typically, during validation or test, this involves collecting all predictions and labels, then accumulating them across mini-batches. Here's a basic function I've frequently used to prepare that data during my model's evaluation loop:

```python
import torch
import numpy as np

def collect_predictions_and_labels(model, dataloader, device):
    model.eval()  # Set model to evaluation mode
    all_predictions = []
    all_labels = []

    with torch.no_grad(): # disable gradient calculations
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1) # get class indices
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_predictions, all_labels

# Example Usage (assuming 'model', 'test_loader', and 'device' are already defined)
# predictions, labels = collect_predictions_and_labels(model, test_loader, device)
```

In this function, we iterate through our dataloader, making sure to move everything to the correct device. I’ve added `torch.no_grad()` since we don't need gradients during inference. The crucial part is how we extract the predictions.  `torch.max(outputs, 1)` returns the maximum value in the `outputs` tensor along the dimension specified by the second argument (dimension 1 in this case), and the corresponding indices of where these max values occur, which represent the predicted class. Finally, we convert these to numpy arrays and extend our accumulating lists.

After collecting this data, we need to use Wandb to create a confusion matrix object. Wandb expects two specific formats: either `(true_labels, predicted_labels)` arrays, or a fully pre-computed matrix. In our case, I've found that creating a matrix directly from true and predicted labels provides better flexibility and control, and allows us to also visualize a normalized matrix.

Here's a snippet I use frequently that demonstrates creating a matrix using `sklearn` and then logging it to Wandb:

```python
import wandb
from sklearn.metrics import confusion_matrix
import numpy as np
def log_confusion_matrix_to_wandb(predictions, labels, class_names, step, wandb_run):

    conf_mat = confusion_matrix(labels, predictions)
    normalized_conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    wandb_run.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=None,
                                                                   y_true=labels,
                                                                   preds=predictions,
                                                                   class_names=class_names,
                                                                   title="Confusion Matrix"),
                      "normalized_confusion_matrix": wandb.plot.confusion_matrix(probs=None,
                                                                             y_true=labels,
                                                                             preds=predictions,
                                                                             class_names=class_names,
                                                                             title="Normalized Confusion Matrix",
                                                                             normalize=True)
                    }, step=step)

# Example usage (assuming predictions, labels, class_names, current_step are defined)
# wandb_run = wandb.init(project="your-project")
# log_confusion_matrix_to_wandb(predictions, labels, class_names, current_step, wandb_run)
# wandb_run.finish()
```

This function uses `confusion_matrix` from `sklearn.metrics` to produce the raw counts. Then I normalize it row wise, which is useful for understanding recall across classes. We then use the `wandb.plot.confusion_matrix` functionality, passing it our true labels (`y_true`), predicted labels (`preds`), and class names (`class_names`). The `title` helps with clarity in the Wandb UI. I also log the normalized version separately. The `step` argument ensures that the matrix is logged at the correct step of the training.

Sometimes you might be dealing with very large datasets, which can make logging individual predictions and labels cumbersome, especially if you're logging them at every step. In such cases, you can pre-compute the entire matrix locally, and just log the matrix itself to Wandb. Here is an example function I created for a particularly large pathology dataset, it avoids storing individual predictions entirely, instead aggregating them directly into the confusion matrix:

```python
import torch
import numpy as np
import wandb
from sklearn.metrics import confusion_matrix

def collect_confusion_matrix_direct(model, dataloader, device, num_classes):
    model.eval()
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            batch_conf_mat = confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy(), labels=np.arange(num_classes))
            conf_matrix += batch_conf_mat
    return conf_matrix


def log_precomputed_confusion_matrix_to_wandb(matrix, class_names, step, wandb_run):
    normalized_conf_mat = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    wandb_run.log({"precomputed_confusion_matrix": wandb.plot.confusion_matrix(matrix=matrix,
                                                                             class_names=class_names,
                                                                             title="Precomputed Confusion Matrix"),
                    "normalized_precomputed_confusion_matrix": wandb.plot.confusion_matrix(matrix=normalized_conf_mat,
                                                                                       class_names=class_names,
                                                                                       title="Normalized Precomputed Confusion Matrix")
                   }, step=step)


#Example usage(assuming model, test_loader, device, num_classes, class_names, current_step are defined)
#conf_matrix = collect_confusion_matrix_direct(model, test_loader, device, num_classes)
#wandb_run = wandb.init(project="your-project")
#log_precomputed_confusion_matrix_to_wandb(conf_matrix, class_names, current_step, wandb_run)
#wandb_run.finish()
```
This function, `collect_confusion_matrix_direct`, calculates the confusion matrix iteratively within the data loop, avoiding storage of all predictions and ground truths. We then pass the `matrix` directly into the plotting function.

For further reading on the theory and application of confusion matrices, I highly recommend consulting the book *Pattern Recognition and Machine Learning* by Christopher M. Bishop. It’s a rigorous and comprehensive treatment of these concepts. Additionally, for a deeper dive into error analysis and model evaluation techniques, explore papers published by the NIPS (NeurIPS) and ICML conferences, as these venues often present the latest advancements in this domain. Wandb’s own documentation also has helpful examples for plotting different kinds of matrices and is regularly updated.

In conclusion, while the process of plotting confusion matrices using Wandb and Pytorch might initially seem like a complex task, it’s fairly straightforward once you break it down. By accumulating your prediction data properly, using the appropriate Wandb functions, and perhaps incorporating sklearn’s `confusion_matrix` for flexibility, you can seamlessly integrate these powerful visualizations into your machine-learning workflow, which is essential for robust model development and debugging. I hope these insights and code snippets have been helpful; they reflect a few of the approaches I have used across numerous projects.
