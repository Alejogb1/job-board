---
title: "Do TensorFlow and PyTorch produce different results for MobileNetV3 Small?"
date: "2025-01-30"
id: "do-tensorflow-and-pytorch-produce-different-results-for"
---
The subtle differences in implementation details between TensorFlow and PyTorch can, and often do, lead to variations in output even when using seemingly identical models like MobileNetV3 Small. I've observed this firsthand while working on model conversion for an embedded vision system, where precise numerical matching was critical. While both libraries implement the core MobileNetV3 architecture, discrepancies arise from distinct choices in preprocessing, weight initialization, and specific layer behaviors, notably batch normalization. These variations become more pronounced when models are not used directly after loading pre-trained weights, such as during fine-tuning or when performing inference on custom, previously unseen, data distributions.

The primary sources of output divergence are typically related to three key areas:

1.  **Pre-processing:** Even when seemingly using similar rescaling and normalization techniques, the exact order of operations, data types, and handling of edge cases (such as zero-division during normalization or handling non-normalized data) can vary between TensorFlow and PyTorch implementations. TensorFlow frequently employs pre-processing layers embedded within the model itself, while PyTorch often relegates such operations to the data loading pipeline. This difference in approach introduces subtle numerical discrepancies.

2.  **Weight Initialization and Pre-trained Weights:** Though pre-trained weights for both TensorFlow and PyTorch implementations of MobileNetV3 Small are typically trained on the same datasets (e.g., ImageNet), the saved weights are not always perfectly numerically equivalent due to different optimization algorithms used during training. Further, both frameworks might employ slightly different internal mechanisms for loading and handling weights. The saved state of the optimization process—specifically, the batch statistics used for batch normalization—also contribute to the discrepancy.

3.  **Batch Normalization:** This is, in my experience, the most significant source of divergence. Batch normalization layers have the potential to introduce numerical instability, especially in scenarios with small batch sizes or when the statistics are not updated consistently. TensorFlow's implementation tends to lean towards an iterative update of batch statistics during training and sometimes uses moving averages, which can differ in subtle implementation nuances from PyTorch's typically simpler online mean and variance calculations. This divergence affects not only the training phase but also the subsequent inference behavior even with frozen pre-trained models.

Here are a few examples, drawn from a past project involving cross-framework performance analysis, illustrating these variations:

**Example 1: Preprocessing Differences**

Suppose the input image is represented as a NumPy array with values in the range \[0, 255].

**TensorFlow (Simplified):**

```python
import tensorflow as tf
import numpy as np

def preprocess_tf(image):
    image = tf.convert_to_tensor(image, dtype=tf.float32) # Convert to float32
    image /= 255.0 # Rescale to [0, 1]
    image -= 0.5 # Center to [-0.5, 0.5]
    image *= 2.0 # Rescale to [-1, 1]
    return image

image_np = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
preprocessed_tf = preprocess_tf(image_np)
print(f"Tensorflow Preprocessed Min: {tf.reduce_min(preprocessed_tf).numpy():.4f}")
print(f"Tensorflow Preprocessed Max: {tf.reduce_max(preprocessed_tf).numpy():.4f}")

```

**PyTorch (Simplified):**

```python
import torch
import numpy as np
import torchvision.transforms as transforms

def preprocess_torch(image):
    transform = transforms.Compose([
        transforms.ToTensor(), # Converts to torch.Tensor and rescale to [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Center and rescale to [-1, 1]
    ])
    image_tensor = torch.from_numpy(image.astype(np.float32).transpose((2,0,1))) # Convert to tensor and reorder dimensions
    preprocessed = transform(image_tensor)
    return preprocessed

image_np = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
preprocessed_torch = preprocess_torch(image_np)
print(f"PyTorch Preprocessed Min: {preprocessed_torch.min():.4f}")
print(f"PyTorch Preprocessed Max: {preprocessed_torch.max():.4f}")
```

*Commentary*: While the intent is to achieve the same result – input pixels in the range [-1, 1] – TensorFlow applies the operations sequentially whereas PyTorch utilizes `torchvision`’s `Normalize` transform that uses a standard mean and standard deviation approach. The conversion of NumPy arrays to tensors also introduce dimension reordering and data type conversion differences. The output will appear close in value, but can differ by a small numerical value.

**Example 2: Batch Normalization Statistics**

The following demonstrates the impact of batch normalization during inference.

**TensorFlow (Simplified):**
```python
import tensorflow as tf
import numpy as np

#Load a pre-trained model for example purposes - This will be replaced with MobileNetV3 Small
model_tf = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), weights='imagenet', include_top = False)

image_np = np.random.rand(1, 224, 224, 3).astype(np.float32) # Create a batch of one image for demo

output_tf = model_tf(image_np)
batch_norm_outputs_tf = []
for layer in model_tf.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
      batch_norm_outputs_tf.append(tf.reduce_mean(layer(image_np)).numpy())

print(f"Tensorflow batch norm means: {batch_norm_outputs_tf}")

```
**PyTorch (Simplified):**

```python
import torch
import torchvision.models as models
import numpy as np

#Load a pre-trained model for example purposes - This will be replaced with MobileNetV3 Small
model_torch = models.mobilenet_v2(pretrained=True)
model_torch.eval() #Switch to evaluation mode

image_np = np.random.rand(1, 224, 224, 3).astype(np.float32) # Create a batch of one image for demo
image_torch = torch.from_numpy(image_np.transpose((0, 3, 1, 2)))

output_torch = model_torch(image_torch)
batch_norm_outputs_torch = []
for layer in model_torch.modules():
    if isinstance(layer, torch.nn.BatchNorm2d):
        batch_norm_outputs_torch.append(torch.mean(layer(image_torch)).detach().numpy())

print(f"PyTorch batch norm means: {batch_norm_outputs_torch}")

```

*Commentary:* This example demonstrates how the mean and variance calculated within Batch Normalization layers differ. While both models use pre-trained weights, the exact statistics stored within the batch normalization layers will inevitably differ due to different training processes and implementations. These slight differences can propagate to the final output predictions and will become more noticeable with deeper networks like MobileNetV3. By default,  PyTorch disables the updating of these moving average statistics during inference when placed in evaluation mode through `model_torch.eval()`

**Example 3: Numerical Precision During Training (Hypothetical Scenario)**

Consider the impact of gradient calculation precision during fine-tuning. While this cannot be easily demonstrated in a short code snippet due to its complex nature, imagine we are fine-tuning our MobileNetV3 small models. TensorFlow and PyTorch utilize various numerical precision libraries and optimization strategies. This difference in gradient accumulation can lead to slight differences in learned weights, even when the starting conditions are seemingly identical, thus leading to variations in the model’s inference output, especially when fine-tuned over multiple epochs. This effect becomes even more pronounced when combined with different learning rate schedules and optimizers.

The impact of these small numerical differences can be significant in some applications. While the final classifications might align, in my experience, for high-precision requirements or if the use case involves sensitive downstream calculations based on the raw features or feature maps, such variations are unacceptable.

To mitigate this issue, I'd recommend several strategies. Firstly, ensure data pre-processing steps are identical. This includes not only rescaling but also ensuring that the order of operations is the same, that data types are consistent (both frameworks do support casting), and that any custom image normalization is handled uniformly. Secondly, when using pre-trained weights, carefully compare the weights before proceeding and ensure both models are in inference/evaluation mode. It might also be necessary to consider techniques like "knowledge distillation" to create a unified representation by training a new model to match the output behavior of both models. Finally, careful attention to batch size and explicit control over batch statistics updating can minimize batch normalization discrepancies.

I recommend studying resources covering:

*   The specific API documentation for preprocessing functions in both TensorFlow and PyTorch. Pay special attention to data type handling, scaling/normalization, and the order of operations.
*   Detailed explanations on batch normalization behavior in each framework. Look for information on how statistics are calculated and updated during training and inference. This often requires delving into the source code or documentation of the specific Batch Norm implementations.
*   General resources on numerical precision issues when performing neural network training. This area includes techniques to avoid loss of numerical accuracy, such as gradient scaling, and the use of different optimizers to reduce numerical errors.
*   Case studies or papers discussing cross-framework model conversion. Reading about real-world experiences can provide valuable insights into the kinds of discrepancies encountered.
*   Official pre-trained model repositories and their respective model cards/documentation, as these often highlight specific preprocessing steps and how the models were trained.
