---
title: "How to visualize PyTorch predictions with TensorBoard?"
date: "2025-01-30"
id: "how-to-visualize-pytorch-predictions-with-tensorboard"
---
Visualizing PyTorch model predictions within TensorBoard requires a structured approach leveraging TensorBoard's logging capabilities.  My experience developing a multi-class image classification model for satellite imagery highlighted the crucial role of effective visualization in identifying model strengths and weaknesses; misclassifications, in particular, were easier to pinpoint after implementing robust TensorBoard logging.  The key lies in formatting your prediction data appropriately before writing it to TensorBoard logs.  This necessitates careful consideration of your data structure and the desired visualization type.

**1. Clear Explanation:**

TensorBoard offers several visualization tools, but for prediction visualization, the most suitable are the `add_image` and `add_scalar` functions.  `add_image` allows for direct visualization of predictions, particularly useful for tasks involving image data, while `add_scalar` facilitates the tracking of scalar metrics derived from predictions, like accuracy or precision.  The process begins within your PyTorch training loop.  After making predictions on a batch of data, you need to transform these predictions into a format compatible with TensorBoard's image or scalar logging functions.  This usually involves converting tensors to NumPy arrays and then potentially reshaping or normalizing them depending on the visualization method. The crucial step is to maintain a clear correspondence between your logged predictions and the input data.  For instance, if you're visualizing image predictions, you must ensure that each logged image is paired with its correct ground truth label.  This allows for side-by-side comparisons and a comprehensive understanding of the model's performance.  Finally, you need to define appropriate tagging schemes to organize the logged data within TensorBoard effectively.   Meaningful tags are crucial for navigation and interpretation.


**2. Code Examples with Commentary:**

**Example 1: Visualizing Image Classification Predictions with `add_image`**

This example focuses on visualizing image predictions from a multi-class image classification model. We assume you have a pre-trained model, a test dataset, and necessary preprocessing functions.

```python
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

# ... (Pre-trained model loading and data loading code omitted for brevity) ...

writer = SummaryWriter()
transform = transforms.ToTensor()

# Assume 'test_loader' is your test data loader, providing (image, label) pairs

for i, (images, labels) in enumerate(test_loader):
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)  # Get model predictions
    _, predicted = torch.max(outputs, 1) # Get predicted class indices

    for j in range(images.size(0)):  # Iterate through batch
        image_np = images[j].cpu().numpy().transpose((1, 2, 0)) # Convert to numpy array
        image_np = (image_np * 255).astype(np.uint8) #Unnormalize
        image_pil = Image.fromarray(image_np) #Convert back to PIL image
        writer.add_image(f'Prediction/{labels[j].item()}/Image_{i}_{j}', transform(image_pil), global_step=i*len(images)+j) #log image

writer.close()
```

This code iterates through the test data, generates predictions, converts tensor representations into PIL Images suitable for TensorBoard, and logs them using `add_image`. The tagging structure (`Prediction/{labels[j].item()}/Image_{i}_{j}`) allows for easy filtering and comparison based on ground truth labels.


**Example 2: Visualizing Scalar Metrics of Predictions with `add_scalar`**

Here, we focus on tracking scalar metrics derived from predictions, such as accuracy and precision.  This example assumes that you have a function to compute these metrics.


```python
import torch
from torch.utils.tensorboard import SummaryWriter

# ... (Model loading and data loading code omitted) ...

writer = SummaryWriter()

for epoch in range(num_epochs):
    # ... (Training loop code omitted) ...

    # Evaluation on the validation set
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        writer.add_scalar('Validation/Accuracy', accuracy, epoch)
        # ... (Additional metrics like precision, recall could be added similarly) ...

writer.close()
```

This demonstrates how to log scalar metrics such as validation accuracy to TensorBoard after each epoch.  This allows for monitoring the model's performance during training.


**Example 3:  Visualizing Class Probabilities as Histograms using `add_histogram`**

For a more in-depth analysis of predictions, visualizing class probabilities can be beneficial. This uses `add_histogram` which is especially useful for observing the distribution of prediction confidences.

```python
import torch
from torch.utils.tensorboard import SummaryWriter

# ... (Model loading and data loading code omitted) ...

writer = SummaryWriter()

for i, (images, labels) in enumerate(test_loader):
    # ...(Move data to device and perform prediction) ...
    probabilities = torch.nn.functional.softmax(outputs, dim=1)  #Obtain class probabilities
    for j in range(probabilities.size(0)):
        writer.add_histogram(f"Prediction_Probabilities/Class_{labels[j].item()}", probabilities[j], global_step=i*len(images)+j)

writer.close()
```
This code snippet calculates class probabilities using the softmax function and then logs them to TensorBoard as histograms, offering insights into the confidence levels associated with different predictions.


**3. Resource Recommendations:**

*   The official PyTorch documentation.  This provides comprehensive details on PyTorch's functionalities, including TensorBoard integration.
*   The official TensorBoard documentation.  This covers all aspects of TensorBoard, including visualization options and usage.
*   A thorough understanding of data structures and tensor manipulations within PyTorch.   This is essential for preparing data for TensorBoard logging.


By carefully structuring your prediction data and utilizing TensorBoard's logging functions, you can create informative visualizations that significantly aid in the evaluation and improvement of your PyTorch models.  Remember consistent and well-organized tagging is paramount for efficient navigation and interpretation of the generated visualizations.
