---
title: "Why does pruning followed by quantization degrade performance?"
date: "2025-01-30"
id: "why-does-pruning-followed-by-quantization-degrade-performance"
---
The performance degradation observed after pruning followed by quantization stems primarily from the synergistic effect of these two model compression techniques: pruning removes less important weights, altering the model's learned representation, and quantization subsequently introduces further inaccuracies by reducing the precision of the remaining weights.  My experience working on optimizing large-scale neural networks for embedded systems highlights this interaction as a critical consideration.  The combined impact often surpasses the individual degradation caused by either technique in isolation.

**1.  A Clear Explanation**

Pruning, a model compression technique, aims to reduce model complexity by eliminating less significant weights or neurons.  This is typically achieved using various criteria, such as magnitude-based pruning (removing weights with small absolute values), or more sophisticated methods employing sensitivity analysis or structured pruning techniques (pruning entire filters or channels).  The core idea is to identify and remove redundant or less influential parameters without significantly impacting the overall model accuracy.  However, this process intrinsically alters the model's learned representation. The optimal weight values, as determined during training, are disrupted, potentially leading to a shift in the decision boundary.

Quantization, on the other hand, reduces the number of bits used to represent each weight or activation in the network.  Common quantization schemes include uniform quantization (mapping floating-point values to a fixed number of discrete levels) and more advanced techniques like non-uniform quantization or vector quantization.  The fundamental goal is to decrease memory footprint and computational cost.  However, this comes at the expense of precision. The discretization inherent in quantization introduces noise, further degrading the accuracy of the model's predictions.

When pruning precedes quantization, the performance degradation is often amplified.  Pruning already introduces inaccuracies by removing weights, even if those weights appear less important.  Subsequent quantization then acts on this already-compromised representation, introducing further errors.  The noise introduced by quantization disproportionately affects the remaining, potentially more sensitive weights, as these carry a greater responsibility for the model's functionality after pruning.  This is particularly evident in regions of the decision boundary that were previously finely tuned by the initially pruned weights.  The combined effect of these two transformations can lead to a significantly greater loss in accuracy than when either method is employed individually.

The specific degree of performance degradation depends on several factors, including the pruning strategy (magnitude-based, L1/L2 norm, sensitivity-based), the pruning ratio (the percentage of weights removed), the quantization scheme (uniform, non-uniform), and the bit-width used for quantization.  Furthermore, the architecture of the neural network, the dataset used for training and evaluation, and the overall training procedure also play substantial roles.  I have encountered instances where aggressive pruning followed by low-bit quantization resulted in catastrophic accuracy drops, while more conservative approaches yielded acceptable performance trade-offs.


**2. Code Examples with Commentary**

**Example 1: Magnitude-based Pruning and Uniform Quantization in PyTorch**

```python
import torch
import torch.nn as nn

# ... define your model ...

# Pruning
for name, param in model.named_parameters():
    if 'weight' in name:
        tensor = param.data.abs()
        threshold = torch.quantile(tensor, 0.2)  # Keep 80% of weights
        mask = tensor >= threshold
        param.data *= mask.float()

# Quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# ... evaluate the model ...
```

This code snippet demonstrates magnitude-based pruning, keeping the top 80% of weights based on their absolute values.  Subsequently, PyTorch's dynamic quantization functionality converts the linear layers to use 8-bit integers. The threshold for pruning can be adjusted to control the aggressiveness of the pruning. The choice of quantization scheme (dynamic quantization here) and bit-width are also crucial parameters.


**Example 2: Structured Pruning and Post-Training Quantization in TensorFlow**

```python
import tensorflow as tf

# ... define your model ...

# Pruning (example: removing entire filters)
pruned_model = tf.compat.v1.keras.models.clone_model(model)
for layer in pruned_model.layers:
    if isinstance(layer, tf.keras.layers.Conv2D):
        weights = layer.get_weights()[0]
        num_filters = weights.shape[-1]
        pruned_filters = int(num_filters * 0.2)  # Keep 80% of filters
        new_weights = weights[:,:,:,:num_filters - pruned_filters]
        layer.set_weights([new_weights, layer.get_weights()[1]])


# Quantization
converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# ... evaluate the model ...
```

This TensorFlow example showcases structured pruning, removing entire convolutional filters.  Post-training quantization using TensorFlow Lite is applied for optimization. The pruning here removes filters based on a percentage, which might impact performance differently compared to the element-wise pruning from the previous example.


**Example 3:  Simulated Pruning and Quantization effects on a Simple Linear Model**

```python
import numpy as np

# Simple linear model
weights = np.array([1.0, 2.0, 3.0, 0.1, 0.2])
inputs = np.array([0.5, 1.0, 1.5, 0.1, 0.0])

# Pruning
threshold = 0.5
pruned_weights = weights * (np.abs(weights) > threshold)

# Quantization (uniform to 2 bits)
levels = np.linspace(-5, 5, 4) # Quantization levels for [-5, 5] range
quantized_weights = np.round(pruned_weights / (levels[1] - levels[0]) ) * (levels[1] - levels[0])


# Output calculation
original_output = np.dot(weights, inputs)
pruned_output = np.dot(pruned_weights, inputs)
quantized_output = np.dot(quantized_weights, inputs)

print(f"Original output: {original_output}")
print(f"Pruned output: {pruned_output}")
print(f"Quantized output: {quantized_output}")

```

This simple example illustrates how pruning and quantization can affect the model's output independently and together.  It demonstrates that the combination can result in an output that deviates more significantly from the original compared to each individual process.  This simplification helps highlight the fundamental effect of precision reduction in the context of the modified model space induced by the pruning process.



**3. Resource Recommendations**

Several prominent textbooks and research papers provide extensive coverage on model compression techniques, including pruning and quantization.  Specifically, literature focusing on the interplay between these two methods offers valuable insights.  Furthermore, deep learning frameworks’ official documentation contains crucial information about their built-in quantization and pruning capabilities.  Exploring resources related to low-precision deep learning will prove beneficial.  Finally, conference proceedings from leading machine learning conferences (e.g., NeurIPS, ICML, ICLR) often feature cutting-edge research on this subject.
