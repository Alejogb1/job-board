---
title: "What causes errors when quantizing a TensorFlow model using vitis_quantize?"
date: "2025-01-26"
id: "what-causes-errors-when-quantizing-a-tensorflow-model-using-vitisquantize"
---

Quantization, while beneficial for model deployment on resource-constrained devices, can introduce errors if not performed meticulously.  Specifically, when utilizing the `vitis_quantize` tool for TensorFlow model quantization within Xilinx's Vitis AI framework, discrepancies between the floating-point precision of the original model and the reduced precision of the quantized version often cause performance degradation or incorrect results. This stems from the inherent challenge of representing real numbers with limited bit widths.

The primary error source arises from information loss during the conversion of floating-point values to lower-bit integers (typically 8-bit integers in `vitis_quantize`). This process introduces several complexities. First, the dynamic range of floating-point numbers is far greater than the representable range of integers. To map floating-point values to an integer range, a scaling and zero-point adjustment are applied. The process, however, is not lossless. Values that are either too small or too large will saturate during this conversion, leading to significant inaccuracies. This occurs both in the weights of the neural network and in the activations computed during inference.

Secondly, quantization errors can be particularly pronounced if the distribution of weights or activations is not well-suited to the chosen integer range. If a large proportion of values is clustered in a small portion of the dynamic range, much of the range may be wasted. Similarly, outliers significantly impact the quantization parameters, forcing the bulk of the data to be represented with insufficient precision. In my experience working with a custom convolutional neural network for image segmentation, I encountered severe segmentation artifacts due to a non-uniform distribution of activation values.

Thirdly, and crucial for the accuracy of the quantized model, is the way the scaling and zero-point are determined. This involves calibration using a representative dataset. Calibration data not accurately reflecting the statistical distribution of real-world inputs during inference will result in sub-optimal quantization parameters and, consequently, errors. Specifically, this causes the quantized values to not correspond to the floating-point values in the areas used during real inference and can also lead to significant activation saturation if a calibration step has not properly captured the entire input distribution.

Furthermore, post-quantization, rounding errors and approximation of computations in integer arithmetic introduce cumulative errors as the data propagates through the layers of the model. For example, the calculation of matrix multiplications can introduce rounding errors at each calculation in the chain. These are significantly more substantial when dealing with lower bit widths.

Finally, implementation details of the specific target hardware can also introduce inaccuracies. While Vitis AI and `vitis_quantize` attempt to account for the specific properties of the target FPGA, there may still be discrepancies between the software model simulation and the actual hardware execution.

Let us illustrate this with code examples.

**Example 1: Basic Quantization and Saturation**

The following code demonstrates a basic quantization process using a simple scaling approach, simulating the behavior inside `vitis_quantize`.

```python
import numpy as np

def quantize(float_value, scale, zero_point, num_bits=8):
    """Quantizes a floating point value to an integer."""
    quantized_value = int(round(float_value / scale) + zero_point)
    min_val = -(2**(num_bits-1))
    max_val = (2**(num_bits-1)) - 1
    quantized_value = np.clip(quantized_value, min_val, max_val)
    return quantized_value

# Example Data:
float_data = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0]

# Example Scaling and Zero Point
scale = 0.5
zero_point = 0

quantized_data = [quantize(val, scale, zero_point) for val in float_data]
print(f"Floating-point data: {float_data}")
print(f"Quantized data: {quantized_data}")

float_data_large_scale = [val * 100 for val in float_data]
quantized_data_large_scale = [quantize(val,scale, zero_point) for val in float_data_large_scale]
print(f"Large floating point data: {float_data_large_scale}")
print(f"Quantized data for large scale float values: {quantized_data_large_scale}")

```

Here, several key issues are highlighted. The `quantize` function demonstrates the clipping process, where floating point values are converted to integer values and saturated when outside the representable range. The second case demonstrates that if the original floating-point values fall outside of the range that can be represented by the integer range after being scaled, the data will saturate. This is especially relevant when a poorly selected or unrepresentative calibration dataset leads to a smaller scale value than ideal. These saturated values can propagate and have significant downstream effects on accuracy.

**Example 2: Activation Clipping and Data Distribution**

This example shows the impact of non-uniform activation distributions.

```python
import numpy as np
import matplotlib.pyplot as plt

def calibrate_and_quantize(activations, num_bits=8):
    """Calibrates and quantizes activation data."""
    min_val = np.min(activations)
    max_val = np.max(activations)
    scale = (max_val - min_val) / ((2**(num_bits-1)) - 1)
    zero_point = -round(min_val/scale)
    quantized_activations = []
    for val in activations:
         quantized_activations.append(int(round(val / scale) + zero_point))
    return quantized_activations, scale, zero_point


# Simulate a uniform distribution
uniform_dist = np.random.normal(0, 1, 1000)

# Simulate a non-uniform distribution (with outliers)
non_uniform_dist = np.concatenate((np.random.normal(0, 0.1, 900), np.random.uniform(2, 5, 100)))


# Quantize both distributions
quantized_uniform, uniform_scale, uniform_zero = calibrate_and_quantize(uniform_dist)
quantized_non_uniform, non_uniform_scale, non_uniform_zero = calibrate_and_quantize(non_uniform_dist)


print(f"Uniform Distribution: Scale = {uniform_scale:.3f}, Zero Point = {uniform_zero}")
print(f"Non-Uniform Distribution: Scale = {non_uniform_scale:.3f}, Zero Point = {non_uniform_zero}")
print("Note: the scale for the uniform distribution is higher")

#Histogram of both distributions.
plt.hist(uniform_dist, bins = 50, alpha = 0.5, label = 'Uniform')
plt.hist(non_uniform_dist, bins = 50, alpha = 0.5, label = 'Non-uniform')
plt.legend(loc = 'upper right')
plt.title("Distribution")
plt.show()

```

The above code creates both uniform and non-uniform data distributions. The quantization parameters calculated using the non-uniform data are skewed by the outlier data points.  This results in a smaller scale and a different zero point. This demonstrates that the outlier data is impacting the way the values are quantized even though those outliers are small portion of the data. If during inference there is data within the bulk of the uniform data that is not captured during calibration, then the performance will be poor. This demonstrates the need for an accurate and representative calibration dataset.

**Example 3: The Cumulative Effects of Quantization**

This example demonstrates the accumulation of rounding errors in a simple linear operation. This effect is exaggerated in larger, more complex computations.

```python
import numpy as np

def float_multiply(a, b):
    return a * b

def quantize_multiply(a, b, scale_a, scale_b, zero_a, zero_b, num_bits = 8):

    def quantize(float_value, scale, zero_point, num_bits=8):
            quantized_value = int(round(float_value / scale) + zero_point)
            min_val = -(2**(num_bits-1))
            max_val = (2**(num_bits-1)) - 1
            quantized_value = np.clip(quantized_value, min_val, max_val)
            return quantized_value


    quantized_a = quantize(a, scale_a, zero_a)
    quantized_b = quantize(b, scale_b, zero_b)

    quantized_result = quantized_a * quantized_b

    result_scale = scale_a * scale_b
    result_zero = 0 #Simplified assumption

    float_result = (quantized_result - result_zero) * result_scale
    return float_result

a_float = 0.6
b_float = 0.7

scale_a = 0.1
zero_a = 0

scale_b = 0.1
zero_b = 0

float_result = float_multiply(a_float, b_float)
quantized_result = quantize_multiply(a_float, b_float, scale_a, scale_b, zero_a, zero_b)


print(f"Floating-point result: {float_result:.3f}")
print(f"Quantized result: {quantized_result:.3f}")


a_list = [0.1, 0.2, 0.3, 0.4, 0.5]
b_list = [0.6, 0.7, 0.8, 0.9, 1.0]

float_results = [float_multiply(a,b) for a,b in zip(a_list, b_list)]
quant_results = [quantize_multiply(a,b,scale_a, scale_b, zero_a, zero_b) for a,b in zip(a_list, b_list)]

print(f"Floating-point results: {float_results}")
print(f"Quantized Results:{quant_results}")

```
The code here shows that a multiplication operation, when performed using quantized integer values, can introduce errors compared to the floating-point equivalent.  This effect is also present in the list of multiplications performed at the end of the code and can compound in more complex operations, like matrix multiplications within the neural network. Although, these errors can seem small individually, their cumulative impact on deeper networks can be more significant.

For further understanding, I recommend reviewing resources on digital signal processing (DSP), particularly topics concerning quantization and its effects on numerical accuracy. Exploring the Xilinx Vitis AI documentation (though it should not be directly linked here), is also useful as it details the specific settings and considerations for model quantization.  Finally, texts on numerical methods offer extensive analyses of error accumulation and propagation in numerical computations, which is directly applicable to this topic. Thoroughly understanding these concepts will provide a basis for effective model quantization with minimal error.
