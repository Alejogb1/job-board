---
title: "How can I plot a confusion matrix into wandb using pytorch?"
date: "2024-12-23"
id: "how-can-i-plot-a-confusion-matrix-into-wandb-using-pytorch"
---

,  I’ve been down this road quite a few times, specifically when fine-tuning complex models and needing to understand where exactly they're making mistakes. Getting a confusion matrix into weights & biases (wandb) using pytorch is definitely achievable and, frankly, a critical step in robust model analysis. I remember one particularly gnarly image classification project where visualising the confusion was the only way we discovered a subtle overlap in label categories—without the wandb integration, we’d have just been spinning our wheels. So, let me break down how I typically approach this.

The core idea is to calculate the confusion matrix after your model has made predictions on your validation set, or test set, and then format this matrix in a way that wandb can ingest it as a custom table or heatmap. The calculation involves iterating over the true labels and the predicted labels, incrementing the appropriate cell of the matrix. Wandb doesn't provide a direct 'confusion matrix logging' function—instead, you're creating the matrix and then either logging it as a table object or visualising it as an image or heatmap. I prefer the table object approach for its flexibility in later analysis.

Let’s walk through it. First, you'll need your predictions and ground truth labels. Typically, during evaluation, you're capturing these. Here’s the foundational python code that illustrates creating the confusion matrix with numpy:

```python
import torch
import numpy as np

def compute_confusion_matrix(predictions, targets, num_classes):
    """
    Computes the confusion matrix.

    Args:
        predictions (torch.Tensor): Predicted class indices.
        targets (torch.Tensor): Ground truth class indices.
        num_classes (int): The total number of classes.

    Returns:
        numpy.ndarray: The confusion matrix.
    """
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(targets)):
      matrix[targets[i], predictions[i]] += 1
    return matrix

# Example Usage
num_classes = 5  # replace with your number of classes.
predictions = torch.randint(0, num_classes, (100,)) #dummy data
targets = torch.randint(0, num_classes, (100,)) #dummy data
confusion_mat = compute_confusion_matrix(predictions, targets, num_classes)
print(confusion_mat)
```

This function takes the predicted class indices and the true class indices as torch tensors, moves them to the cpu and then converts them to numpy arrays before building the confusion matrix. A simple loop then increments the counter in the matrix at the index that corresponds to the true label and the predicted label. You'll note that numpy arrays are used; this is because it makes creating the zero-filled matrix and incrementing the values easier than in native python lists.

Now, to push this into wandb, you'll need to structure your confusion matrix as a table object. Wandb offers a very handy `wandb.Table` object. The next code snippet demonstrates how to do just that:

```python
import wandb
import pandas as pd

def wandb_confusion_matrix(matrix, class_names, step, log_prefix=""):
    """
    Logs a confusion matrix to wandb as a table.

    Args:
        matrix (numpy.ndarray): The confusion matrix.
        class_names (list): List of class names.
        step (int): The current training step (for logging).
        log_prefix (str): Prefix for the wandb table name.
    """

    df = pd.DataFrame(matrix, index=class_names, columns=class_names)

    # Convert dataframe to list of lists
    table_data = [df.columns.to_list()]
    table_data.extend(df.values.tolist())

    wandb_table = wandb.Table(columns=table_data[0], data = table_data[1:])

    wandb.log({f"{log_prefix}confusion_matrix_table": wandb_table, "global_step": step})

# Example Usage (assuming wandb.init() and your 'confusion_mat' is available):

class_names = ["class_a", "class_b", "class_c", "class_d", "class_e"] #Replace with actual class names
step_number = 100 #Replace with actual step number
wandb_confusion_matrix(confusion_mat, class_names, step_number)
```
Here we're converting the numpy matrix to a pandas dataframe to facilitate adding the labels as headers to the rows and columns before conversion back to a list of lists that wandb can understand. The crucial part is creating `wandb_table` with headers and the data, and logging it. The `log_prefix` allows you to log multiple confusion matrices (e.g., for train vs. validation) using different table names. The `step` is important to keep track of which training epoch or evaluation run this matrix belongs to when using wandb.

Finally, while logging the confusion matrix as a table is excellent for granular inspection, sometimes a more intuitive visual representation as a heatmap can be useful. You can do that with wandb directly, or you can make it as an image via matplotlib. Here is an example of doing it via matplotlib.

```python
import matplotlib.pyplot as plt
import seaborn as sns
import io

def wandb_confusion_matrix_image(matrix, class_names, step, log_prefix=""):
    """
    Logs a confusion matrix to wandb as an image.

    Args:
        matrix (numpy.ndarray): The confusion matrix.
        class_names (list): List of class names.
        step (int): The current training step (for logging).
        log_prefix (str): Prefix for the wandb table name.
    """

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    # Convert plot to image
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    plt.close()

    wandb.log({f"{log_prefix}confusion_matrix_image": wandb.Image(img_buffer), "global_step": step})

# Example Usage (assuming wandb.init() and your 'confusion_mat' is available):
wandb_confusion_matrix_image(confusion_mat, class_names, step_number)
```

This snippet first creates a heatmap using seaborn and matplotlib. It labels the axes and adds a title, then saves the entire plot into an image buffer in memory, allowing it to be logged into wandb via `wandb.Image()`. The advantage here is the visually compelling format of the data.

Regarding resources, I'd recommend the following: First and foremost, the official weights & biases documentation. They continuously update with new features. Second, for a deeper understanding of confusion matrices, the 'Pattern Recognition and Machine Learning' book by Christopher Bishop provides a very solid and rigorous theoretical base. Lastly, while not directly about logging in wandb, 'Hands-On Machine Learning with Scikit-Learn, Keras & Tensorflow' by Aurélien Géron will help you understand model evaluations better, and this will improve your choices regarding data preparation. This book includes practical steps for model performance evaluation, which directly influences the quality of the matrix.

In practice, I usually log both the table and the image. I find that the table is superb for quick inspection and programmatical extraction of stats, while the heatmap can immediately highlight problematic areas where the model needs improvement. This provides both granular numerical insight and overall visual clarity. It’s this combined perspective that’s really helped me iterate effectively on my models over the years.
