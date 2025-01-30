---
title: "How can I plot a confusion matrix in wandb using PyTorch?"
date: "2025-01-30"
id: "how-can-i-plot-a-confusion-matrix-in"
---
Plotting a confusion matrix within Weights & Biases (wandb) using PyTorch requires a nuanced approach, exceeding a simple `wandb.log`.  My experience integrating sophisticated model evaluation metrics into wandb, particularly during extensive hyperparameter sweeps on large-scale datasets, highlights the need for careful data structuring and leveraging wandb's capabilities beyond basic logging.  The core issue lies in properly formatting the confusion matrix data for wandb's visualization engine.  Directly feeding a NumPy array doesn't suffice;  wandb expects structured data, often in a dictionary format suitable for its table visualization.

**1. Clear Explanation**

The process involves three distinct steps:  (a) generating the confusion matrix from your PyTorch model's predictions and ground truth labels; (b) transforming the confusion matrix into a format wandb understands; and (c) logging this formatted data to wandb.

Step (a) necessitates careful handling of prediction outputs.  Ensure your model outputs probabilities (e.g., using a `softmax` activation) if your loss function utilizes cross-entropy. These probabilities need converting into class predictions via `argmax`.  Then, use functions like `sklearn.metrics.confusion_matrix` to compute the matrix.

Step (b) involves creating a structured representation of the confusion matrix suitable for wandb.  Wandb's table visualization is well-suited for this.  Each row in the table will represent a true class, and each column a predicted class.  The cell values are the counts of instances belonging to each (true, predicted) class pair.

Step (c) leverages wandb's `log` function with the structured data from step (b).  This allows wandb to render the data as an interactive confusion matrix within your project's run history.  Crucially,  including relevant metadata within your log call, such as the epoch number or hyperparameter settings, enables effective comparison across different runs.

**2. Code Examples with Commentary**

**Example 1: Basic Confusion Matrix Logging**

```python
import wandb
import torch
import numpy as np
from sklearn.metrics import confusion_matrix

# ... your PyTorch model and data loading ...

predictions = model(test_loader.dataset.tensors[0]) # Assuming your data loader structure
_, predicted_classes = torch.max(predictions, 1)
true_classes = test_loader.dataset.tensors[1] # Assuming your targets are in the second tensor of your dataset

cm = confusion_matrix(true_classes.cpu().numpy(), predicted_classes.cpu().numpy())

cm_data = {}
for i, row in enumerate(cm):
    cm_data[f"Class {i}"] = {f"Predicted Class {j}": count for j, count in enumerate(row)}

wandb.log({"confusion_matrix": wandb.Table(data=cm_data)})

# ... rest of your training loop ...
```

This example demonstrates a basic logging approach. The core idea is transforming the NumPy array into a dictionary suitable for `wandb.Table`.  Each key in the dictionary represents a true class, and the value is another dictionary where keys are predicted classes and values are counts.  This structure enables wandb to interpret the data correctly.

**Example 2:  Confusion Matrix with Class Labels**

```python
import wandb
import torch
import numpy as np
from sklearn.metrics import confusion_matrix

# ... your PyTorch model and data loading ...

# Assuming you have class labels defined
class_labels = ["cat", "dog", "bird"]

# ... prediction and true class extraction (same as Example 1) ...

cm = confusion_matrix(true_classes.cpu().numpy(), predicted_classes.cpu().numpy())

cm_data = []
for i, row in enumerate(cm):
    row_data = {"True Class": class_labels[i]}
    for j, count in enumerate(row):
        row_data[class_labels[j]] = count
    cm_data.append(row_data)

wandb.log({"confusion_matrix": wandb.Table(columns=["True Class"] + class_labels, data=cm_data)})

# ... rest of your training loop ...
```

This builds upon the first example by incorporating meaningful class labels instead of numerical indices. This significantly enhances readability and interpretability within the wandb interface.  The `columns` argument in `wandb.Table` ensures proper column naming.

**Example 3:  Confusion Matrix with Additional Metadata**

```python
import wandb
import torch
import numpy as np
from sklearn.metrics import confusion_matrix

# ... your PyTorch model and data loading ...

# ... prediction and true class extraction ...

cm = confusion_matrix(true_classes.cpu().numpy(), predicted_classes.cpu().numpy())

# ... formatting cm into cm_data (as in Example 2) ...

wandb.log({
    "epoch": epoch,
    "learning_rate": learning_rate,
    "confusion_matrix": wandb.Table(columns=["True Class"] + class_labels, data=cm_data)
})

# ... rest of your training loop ...
```

This example demonstrates adding crucial metadata – epoch number and learning rate – to the logged data.  This allows for comprehensive analysis of the confusion matrix evolution during training and across different hyperparameter settings within the wandb interface.


**3. Resource Recommendations**

The official Weights & Biases documentation.  The scikit-learn documentation for confusion matrix generation. A thorough understanding of PyTorch's data handling mechanisms and tensor manipulation is essential.   Consult relevant PyTorch tutorials focusing on model evaluation and metrics calculation. Familiarize yourself with data structuring techniques for effective visualization tools like wandb.  Exploring interactive data visualization best practices will further enhance your analysis.


In my experience, meticulously structuring your data for wandb is crucial for achieving effective visualization.  The examples provided offer a starting point;  adaptations might be needed based on your specific model output and dataset structure.  Remember to handle potential errors, such as mismatched dimensions between predictions and true labels, to ensure robustness.  Furthermore, effective interpretation of the confusion matrix requires understanding of class imbalances and the implications for model performance.  These considerations are paramount in building a comprehensive and insightful model evaluation pipeline.
