---
title: "How can neural network regression models be quantized post-training using full integer values?"
date: "2025-01-30"
id: "how-can-neural-network-regression-models-be-quantized"
---
Quantization of neural network regression models to full integer values, specifically post-training, represents a significant opportunity for reducing model size and increasing inference speed, particularly on resource-constrained devices. This process, while offering substantial benefits, requires a careful understanding of numerical representation and the potential trade-offs in accuracy. I’ve personally implemented this process numerous times across embedded systems and server environments, and the key lies in understanding the scaling factors and how to maintain precision after integer conversion.

**Explanation of Post-Training Quantization to Integer Values**

The core idea behind post-training quantization involves converting the floating-point weights and activations of a trained neural network into integer representations. This contrasts with quantization-aware training, where the model is trained with quantization in mind. The benefit of post-training quantization is its simplicity; it can be applied to pre-existing, fully trained models. This significantly speeds up deployment pipelines and simplifies the workflow.

The typical floating-point representation in deep learning uses 32-bit (FP32) or sometimes 16-bit (FP16) precision, which allows for a very wide dynamic range and high precision. However, these formats are inefficient in terms of storage and computation. Integer representation, usually 8-bit (INT8), reduces memory footprint and computation costs by reducing the number of bits per value. This reduction comes at a cost: a reduced dynamic range.

The critical element in integer quantization is the introduction of scaling parameters that map the floating-point values to the desired integer range and back. For each tensor (weights or activations), a scale factor ('s') and sometimes a zero-point ('z') are computed. The quantization equation is:

*   **Quantization:** `q = round(s * (r - z))` where 'r' is the original floating-point value, and 'q' is the quantized integer value.
*   **Dequantization:** `r' = (q / s) + z` where 'r'' is the reconstructed floating point value.

The `round()` operation maps to the nearest integer, creating the discrete representation. For full integer quantization, we avoid fractional bits, opting for whole integer numbers, which is essential for dedicated integer arithmetic units.

Post-training quantization usually determines these scale and zero-point parameters through calibration, using a representative subset of the training data. This calibration data is passed through the model to observe the dynamic range of activations at each layer. Min and max values of the observed tensors are collected and then used to calculate the scaling parameters, ensuring the range is mapped to the integer range appropriately (e.g., -128 to 127 for INT8).

The zero point ('z') is particularly important for signed integer quantization. Without a zero-point, a floating-point zero would map to an integer value that does not represent zero, leading to systematic errors.

**Code Examples and Commentary**

I'll provide three Python code examples using `numpy` to illustrate the concepts involved in post-training integer quantization. Note that a deep learning framework such as TensorFlow or PyTorch would be the actual tooling used in practice, which handles the details of the mapping. Here, we'll focus on the conceptual implementation for understanding the mechanics.

**Example 1: Basic Symmetric Quantization (Scale Only)**

This example showcases symmetric quantization, where we assume a zero-point of 0. This approach is appropriate when the data distribution is centered around zero.

```python
import numpy as np

def symmetric_quantize(tensor, num_bits):
    max_value = np.max(np.abs(tensor))
    scale = (2 ** (num_bits - 1) - 1) / max_value  # Scale for symmetric range
    quantized = np.round(tensor * scale)
    quantized = np.clip(quantized, -(2 ** (num_bits - 1)), (2 ** (num_bits - 1) - 1))
    return quantized.astype(np.int32), scale

def symmetric_dequantize(quantized, scale):
    return quantized / scale

# Sample data
fp_tensor = np.array([-2.5, -1.0, 0.0, 1.5, 3.0])
bits = 8

# Quantize
int_tensor, scale_factor = symmetric_quantize(fp_tensor, bits)
print(f"Quantized tensor: {int_tensor}")
print(f"Scale factor: {scale_factor}")

# Dequantize
reconstructed_tensor = symmetric_dequantize(int_tensor, scale_factor)
print(f"Reconstructed tensor: {reconstructed_tensor}")
```

In this example, `symmetric_quantize` computes the scaling factor such that the maximum absolute value in the floating-point tensor maps to the maximum value of the signed integer range. The `clip()` operation ensures that the integer values stay within the valid range. Dequantization is a direct inverse operation. The critical feature here is the mapping of the FP32 values to the quantized range using only the scale.

**Example 2: Asymmetric Quantization with Zero-Point**

Asymmetric quantization is more flexible since we introduce a zero-point to account for cases where the data isn't centered at zero.

```python
import numpy as np

def asymmetric_quantize(tensor, num_bits):
    min_val = np.min(tensor)
    max_val = np.max(tensor)
    scale = (2 ** num_bits - 1) / (max_val - min_val)
    zero_point = np.round((min_val * scale) * -1)
    quantized = np.round((tensor * scale) - zero_point)
    quantized = np.clip(quantized, 0, (2**num_bits - 1)) #Unsigned range for simplicity
    return quantized.astype(np.int32), scale, zero_point

def asymmetric_dequantize(quantized, scale, zero_point):
    return (quantized + zero_point) / scale

# Sample data
fp_tensor = np.array([0.5, 2.0, 4.0, 6.0, 8.0])
bits = 8

# Quantize
int_tensor, scale_factor, zero_point_val  = asymmetric_quantize(fp_tensor, bits)
print(f"Quantized tensor: {int_tensor}")
print(f"Scale factor: {scale_factor}")
print(f"Zero point: {zero_point_val}")


# Dequantize
reconstructed_tensor = asymmetric_dequantize(int_tensor, scale_factor, zero_point_val)
print(f"Reconstructed tensor: {reconstructed_tensor}")
```

Here, we calculate the scale based on the data range, and zero point as an offset. By incorporating the zero point, we can represent data that's biased toward positive or negative values more accurately. The int output is within unsigned range (0 - 255), but the principles are similar to the signed range.

**Example 3: Per-Tensor versus Per-Channel Quantization**

The examples above show per-tensor quantization, where scale/zero-point values are calculated across the entire tensor. When dealing with convolutional layers, it can be more effective to calculate the scale/zero-point per channel, because each channel could have very different ranges.

```python
import numpy as np

def per_channel_quantize(tensor, num_bits, axis=0):
    quantized_tensor = np.zeros_like(tensor, dtype=np.int32)
    scales = []
    zero_points = []

    for i in range(tensor.shape[axis]):
        if axis == 0:
           channel = tensor[i]
        elif axis == 1:
           channel = tensor[:,i]
        elif axis == 2:
           channel = tensor[:,:,i]
        min_val = np.min(channel)
        max_val = np.max(channel)
        scale = (2 ** num_bits - 1) / (max_val - min_val)
        zero_point = np.round((min_val * scale) * -1)

        if axis == 0:
           quantized_channel = np.round((channel * scale) - zero_point)
           quantized_channel = np.clip(quantized_channel, 0, (2**num_bits -1)).astype(np.int32)
           quantized_tensor[i] = quantized_channel
        elif axis == 1:
            quantized_channel = np.round((channel * scale) - zero_point)
            quantized_channel = np.clip(quantized_channel, 0, (2**num_bits -1)).astype(np.int32)
            quantized_tensor[:, i] = quantized_channel
        elif axis == 2:
            quantized_channel = np.round((channel * scale) - zero_point)
            quantized_channel = np.clip(quantized_channel, 0, (2**num_bits -1)).astype(np.int32)
            quantized_tensor[:,:, i] = quantized_channel

        scales.append(scale)
        zero_points.append(zero_point)
    return quantized_tensor, np.array(scales), np.array(zero_points)

def per_channel_dequantize(quantized, scales, zero_points, axis=0):
  dequantized_tensor = np.zeros_like(quantized, dtype = np.float32)

  for i in range(quantized.shape[axis]):
    if axis == 0:
      dequantized_channel = (quantized[i].astype(np.float32) + zero_points[i]) / scales[i]
      dequantized_tensor[i] = dequantized_channel
    elif axis == 1:
      dequantized_channel = (quantized[:,i].astype(np.float32) + zero_points[i]) / scales[i]
      dequantized_tensor[:,i] = dequantized_channel
    elif axis == 2:
      dequantized_channel = (quantized[:,:,i].astype(np.float32) + zero_points[i]) / scales[i]
      dequantized_tensor[:,:,i] = dequantized_channel

  return dequantized_tensor


# Sample tensor (e.g., conv layer weights)
fp_tensor_3d = np.array([
    [[1.0, 2.0], [3.0, 4.0]],
    [[5.0, 6.0], [7.0, 8.0]],
    [[9.0,10.0], [11.0, 12.0]]
]).astype(np.float32)

bits = 8

# Quantize
quantized_3d, scales_3d, zero_points_3d = per_channel_quantize(fp_tensor_3d, bits, axis=2)
print(f"Quantized tensor shape: {quantized_3d.shape}")
print(f"Scales per channel:{scales_3d}")
print(f"Zero points per channel:{zero_points_3d}")

# Dequantize
dequantized_3d = per_channel_dequantize(quantized_3d, scales_3d, zero_points_3d, axis=2)
print(f"Dequantized tensor: {dequantized_3d}")
```

The `per_channel_quantize` function iterates through the chosen axis of the tensor (e.g., the channel dimension in a convolutional layer), computing separate scale and zero-point parameters for each channel. This results in more accurate and less noisy quantizations for more complicated data tensors and thus a better mapping of data range. The same applies to the dequantization function `per_channel_dequantize` as it applies the specific scales and zero-points per channel.

**Resource Recommendations**

For further exploration, I’d recommend consulting the documentation of deep learning frameworks regarding their quantization tools. Look into the quantization sections for TensorFlow Lite and PyTorch, which provide comprehensive resources and practical examples. Investigate research papers and tutorials focused on post-training quantization techniques for a more theoretical and rigorous understanding. Also, understanding the intricacies of fixed-point arithmetic and numerical error analysis will further cement your knowledge and enable proper parameter selection when doing this work. Finally, studying hardware specific quantization operations on target inference devices are also crucial for successful deployments.

By mastering the concepts of scaling, zero-points, and per-channel quantization, you'll be well equipped to deploy quantized models to resource-constrained environments without compromising too much on model accuracy. These practical examples alongside theoretical research, will prepare you for the challenge of optimizing models for real world deployment.
