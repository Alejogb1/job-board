---
title: "Why does network quantization require a zero-point, while symmetric quantization does not?"
date: "2024-12-23"
id: "why-does-network-quantization-require-a-zero-point-while-symmetric-quantization-does-not"
---

Okay, let's unpack this. I remember tackling this very issue back when we were optimizing our convolutional neural network for deployment on a low-power embedded system. The need for a zero-point in network quantization versus its absence in symmetric quantization initially seemed like an arbitrary detail. It wasn't, of course, and getting a solid understanding of the underlying mechanics made all the difference in achieving acceptable performance on our target device.

The core distinction arises from how data is mapped from its original, typically floating-point range to a lower-bit integer range. Let's start by defining some terms. *Quantization* in the context of neural networks refers to reducing the numerical precision of weights and activations. *Symmetric quantization* maps values such that the zero point in the original range maps to the zero point in the quantized range, effectively mirroring positive and negative values around zero. This is relatively straightforward. The common formula, in its simplified form, is *`quantized_value = round(original_value / scale)`* where `scale` is a single, positive value.

However, neural network activations often do not have a symmetric distribution around zero. A common example is ReLU activations, which by definition clip negative values to zero. After multiple layers, the data distribution tends to drift away from a symmetric arrangement. If we use only a scale factor as in symmetric quantization to map this data to integers, it can lead to a significant loss of information, particularly if there are a limited number of integer levels (think 8-bit or less). If the values of activations are primarily positive, mapping the original zero to the quantized zero doesn't fully utilize the range and pushes more values towards zero. A significant portion of the available integer range will be underutilized, and the resolution of our values are reduced.

Here's where the *zero-point* comes in. A zero-point, sometimes referred to as an offset or bias, provides an ability to map a given value in the original range to an arbitrary location in the integer range. This enables *asymmetric quantization*. So, we have two things happening; we’re scaling the input, and then adding the zero point to move the range.

The formula now becomes *`quantized_value = round(original_value / scale + zero_point)`*. In this form, the original zero in floating point no longer has to map to integer zero.

Let’s say we have a floating point activation range of [0, 10], and an 8-bit integer range [0, 255]. With symmetric quantization, the integer 0 would represent floating point 0. The remaining 255 would represent 10, leaving most of the range pushed towards 0. However, with a zero-point, we could calculate a zero-point that maps 0 to integer 0 and 10 to integer 255. The resulting values would be more accurate than symmetric quantization with the same integer range.

Let's illustrate this with a simple Python snippet. First, let's look at symmetric quantization:

```python
import numpy as np

def symmetric_quantization(data, scale):
    """Quantizes data symmetrically."""
    return np.round(data / scale).astype(np.int8)

# Example
data = np.array([-2.5, -1.0, 0.0, 1.5, 3.0])
scale = 2.0  # Chosen such that it encompasses the range
quantized_data = symmetric_quantization(data, scale)
print(f"Original data: {data}")
print(f"Symmetrically quantized data: {quantized_data}")
```

Here, we can see the symmetric nature. The range of the original data is centered around zero, and the quantized range also maps zero to zero. No problem. Now let's examine asymmetric quantization using a zero-point:

```python
import numpy as np

def asymmetric_quantization(data, scale, zero_point):
    """Quantizes data asymmetrically using a zero-point."""
    return np.round(data / scale + zero_point).astype(np.int8)

# Example with a skewed data range
data = np.array([0.0, 2.0, 4.0, 6.0, 8.0])  # Note how the data starts at zero
scale = 2.0  # Scale the range
zero_point = 0  # Initially start with no offset
quantized_data = asymmetric_quantization(data, scale, zero_point)
print(f"Original data: {data}")
print(f"Asymmetrically quantized data without a zero_point: {quantized_data}")

min_val = data.min()
max_val = data.max()
target_min = 0
target_max = 127 # 8-bit signed integer range
scale = (max_val-min_val) / (target_max - target_min)
zero_point = - (min_val / scale)
quantized_data = asymmetric_quantization(data, scale, zero_point)

print(f"Asymmetrically quantized data with calculated zero_point: {quantized_data}")

```

Here, we can see the advantage of the zero point. The initial run of this function with the zero point set to zero results in a less than optimal usage of the 8-bit integer range. The second run with a calculated zero point effectively maps the activation range to the maximum usable integer range.

Now, let's take another example that demonstrates how we would incorporate the de-quantization step with a zero point:

```python
import numpy as np

def asymmetric_quantization_and_dequantization(data, scale, zero_point):
  """Quantizes data asymmetrically and then de-quantizes."""
  quantized_data = np.round(data / scale + zero_point).astype(np.int8)
  dequantized_data = (quantized_data - zero_point) * scale

  return quantized_data, dequantized_data


data = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])  # Skewed range
min_val = data.min()
max_val = data.max()
target_min = 0
target_max = 127 # 8-bit signed integer range
scale = (max_val-min_val) / (target_max - target_min)
zero_point = - (min_val / scale)

quantized, dequantized = asymmetric_quantization_and_dequantization(data, scale, zero_point)

print(f"Original data: {data}")
print(f"Quantized data: {quantized}")
print(f"Dequantized data: {dequantized}")

```

Notice how in the last example, we show both the quantization *and* the de-quantization process. This is crucial because the hardware that runs the quantized network needs to reverse the quantization to get the approximate values.

You can observe in this case that we have more fine-grained control of the mapping using both scale *and* zero_point, and thus, a more accurate integer representation. The de-quantized values are close to the original values with small deviations that are a byproduct of the quantization process.

In practice, calculating `scale` and `zero_point` is an optimization problem in itself. Methods such as min-max or percentile based strategies, which use samples from a training or validation set, are common. But the idea remains the same: we tailor the mapping to the data distribution to minimize the loss of information and increase inference accuracy on the quantized model.

In short, symmetric quantization works well when the data distribution is roughly centered around zero, while asymmetric quantization with a zero-point is necessary when this assumption does not hold—which is frequently the case with intermediate activations in neural networks.

For anyone looking to go deeper, I highly recommend diving into the original Google paper on integer quantization for neural network inference, it’s typically cited under ‘quantization and training of neural networks for efficient integer-arithmetic-only inference’. Furthermore, the ‘Handbook of Floating-Point Arithmetic’ by Muller et al. is a phenomenal resource for understanding number representation and precision implications of different operations, this provides essential background knowledge for understanding quantization. You could also refer to a classic book on embedded systems, such as ‘Embedded Computing: A System Design Approach’ by Wayne Wolf, for a broader context on applying optimization techniques to resource constrained systems. I hope this helps clear things up.
