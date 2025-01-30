---
title: "How can post-training quantization be performed outside of TensorFlow Lite?"
date: "2025-01-30"
id: "how-can-post-training-quantization-be-performed-outside-of"
---
Post-training quantization, specifically its more prevalent form of dynamic range quantization, can be effectively achieved outside of the TensorFlow Lite ecosystem. The key lies in replicating the core mathematical operations and data manipulations that TensorFlow Lite's quantization process performs. I've personally implemented this in custom embedded systems and found it offers considerable performance and footprint optimization when TFLite is not an option, or its overhead is deemed too high for very constrained devices.

The fundamental idea behind dynamic range quantization, as used by TFLite, involves converting floating-point weights and activations to integers, typically 8-bit integers (int8). This reduces memory usage and accelerates computation on hardware that is optimized for integer arithmetic. This conversion is done by determining the dynamic range of each tensor individually and then mapping the floating-point values to an integer range. The crucial parameters are a 'scale' factor and a 'zero-point'. The scale determines the linear mapping from the floating-point domain to the integer domain. The zero-point is the integer value that represents the floating-point 0. This allows us to represent positive and negative floating-point values around the 0 point with the int8 range.

The process, generally, has these primary steps:

1. **Calibration:**  This involves feeding a representative dataset (calibration set) through the model, typically using floating-point operations. For each activation tensor, we track the minimum and maximum floating-point values encountered during this forward pass. We do not, here, alter the actual model weights but instead collect activation-based data.
2. **Scale and Zero-Point Calculation:** From the collected min and max floating-point values for each activation tensor (and weights, when applicable) we derive the scale and zero-point using these equations:
    * `scale = (max_float - min_float) / (int_max - int_min)` where `int_max` is 127 and `int_min` is -128 for 8-bit signed integer representation.
    * `zero_point = round((int_min - (min_float / scale))`. 
3. **Quantization:** This process converts the floating point tensors to integer tensors using the calculated scale and zero-point:
     * `quantized_value = round(float_value / scale) + zero_point`
4. **Dequantization:** For the final model inference, we dequantize the result for the final output:
    * `float_value = (quantized_value - zero_point) * scale`
Note that the model is never fully in integer math throughout the full forward pass. Instead, weights are stored and used in their integer representation after the quantization step, and activations are converted from float to int8 for calculations, and then back to float when necessary, in each layer. This is primarily to enable the integer math to be applied on optimized hardware accelerators like DSPs or NPUs that operate in integer mode.

Here’s how you can implement this in a standard Python environment, using NumPy, without reliance on TensorFlow Lite or TensorFlow, emphasizing the steps mentioned:

**Example 1: Simple Activation Quantization**

```python
import numpy as np

def quantize_activation(activation, min_val, max_val):
  """Quantizes an activation tensor to int8.

  Args:
      activation: A numpy array representing the floating-point activation.
      min_val: The minimum observed floating-point value.
      max_val: The maximum observed floating-point value.

  Returns:
      A tuple containing the quantized int8 activation tensor, the scale,
      and the zero point.
  """
  int_min = -128
  int_max = 127

  scale = (max_val - min_val) / (int_max - int_min)
  zero_point = round(int_min - (min_val / scale))

  quantized = np.round(activation / scale) + zero_point
  quantized = np.clip(quantized, int_min, int_max).astype(np.int8)

  return quantized, scale, zero_point


def dequantize_activation(quantized, scale, zero_point):
    """Dequantizes an int8 activation tensor to float32.
    Args:
      quantized: A numpy array representing the int8 quantized activation.
      scale: The float scale.
      zero_point: The integer zero point.
    Returns:
      The dequantized float32 activation tensor
    """
    return (quantized - zero_point) * scale

# Demonstration
activation = np.array([-2.5, 0.0, 1.2, 3.8, 5.1], dtype=np.float32)
min_val = -2.5
max_val = 5.1

quantized, scale, zero_point = quantize_activation(activation, min_val, max_val)
print("Quantized Activation:", quantized)
print("Scale:", scale)
print("Zero Point:", zero_point)
dequantized = dequantize_activation(quantized, scale, zero_point)
print ("Dequantized Activation:", dequantized)

```

In this example, we explicitly compute the scale and zero-point based on the provided min and max values. The `quantize_activation` function performs the quantization operation and clips the values to stay within the int8 bounds. The `dequantize_activation` converts the tensor back to a float type.

**Example 2: Weight Quantization and Layer Calculation**

Now, we look at weight quantization applied to a linear layer implementation:

```python
import numpy as np

def quantize_weight(weight):
    """Quantizes a weight tensor to int8, using its min and max as the observed data.

    Args:
      weight: A numpy array representing the floating-point weight tensor.

    Returns:
      A tuple containing the quantized int8 weight tensor, scale, and zero point.
    """

    min_val = np.min(weight)
    max_val = np.max(weight)
    return quantize_activation(weight, min_val, max_val)


def linear_layer_quantized(input_act, quantized_weight, scale_weight, zero_point_weight, scale_in, zero_point_in, scale_bias, bias):
  """Performs a linear layer operation with quantized weights and activations.

  Args:
      input_act: A numpy array representing the quantized int8 input activation tensor.
      quantized_weight: A numpy array representing the quantized int8 weight tensor.
      scale_weight: The scale for the weight tensor.
      zero_point_weight: The zero point for the weight tensor.
      scale_in: The scale for the input activations
      zero_point_in: The zero point for the input activations
      scale_bias: The scale for the bias
      bias: The floating point bias

  Returns:
       A numpy array representing the floating point result of the linear layer.
  """

  input_float = dequantize_activation(input_act, scale_in, zero_point_in)
  weight_float = dequantize_activation(quantized_weight, scale_weight, zero_point_weight)

  #Perform the calculation in the floating point domain
  result_float = np.dot(input_float, weight_float.T) + bias
  return result_float


# Sample linear layer (matrix multiplication)
input_activation = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
weights = np.array([[0.5, -0.2, 0.1], [-0.3, 0.4, 0.2]], dtype=np.float32)
bias = np.array([0.1, -0.2, 0.3], dtype = np.float32)

# Perform calibration (here we use dummy values for min/max, in reality we would run data through the forward pass)
min_act = 0.0
max_act = 5.0

quantized_act, scale_act, zero_point_act = quantize_activation(input_activation, min_act, max_act)

quantized_weights, scale_weights, zero_point_weights = quantize_weight(weights)
#Perform the quantized calculation
result_float = linear_layer_quantized(quantized_act, quantized_weights, scale_weights, zero_point_weights, scale_act, zero_point_act, 1.0, bias)
print ("Result Float: ", result_float)
```

In this expanded example, the `quantize_weight` function encapsulates the weight quantization logic, similarly to `quantize_activation`. It automatically calculates min and max values from the weights. The `linear_layer_quantized` now expects both quantized activations and weights and performs the calculation by first dequantizing to float values, before performing the calculation in floating point domain (which is essential since the output of the matrix multiplication is not a quantized value).

**Example 3: Handling Multiple Layers (Concept)**

A realistic model consists of several layers. This concept is demonstrated with a dummy class:

```python
import numpy as np

class QuantizedModel:
    def __init__(self):
        # In a real scenario, these would be weights from a pre-trained model
        self.weights1 = np.array([[0.5, -0.2], [-0.3, 0.4]], dtype=np.float32)
        self.weights2 = np.array([[0.8, -0.1], [0.3, 0.6]], dtype=np.float32)

        self.bias1 = np.array([0.1, -0.2], dtype = np.float32)
        self.bias2 = np.array([0.05, 0.05], dtype = np.float32)
        
        self.scale_act1 = 1.0
        self.zero_point_act1 = 0
    
        self.quantized_weights1, self.scale_weights1, self.zero_point_weights1 = quantize_weight(self.weights1)
        self.quantized_weights2, self.scale_weights2, self.zero_point_weights2 = quantize_weight(self.weights2)


    def forward(self, input_act):
        #Dummy calibration of the first activation
        if isinstance(input_act, np.ndarray):
          min_act = np.min(input_act)
          max_act = np.max(input_act)
        else:
            min_act = 0
            max_act = 10
        
        quantized_act1, self.scale_act1, self.zero_point_act1 = quantize_activation(input_act, min_act, max_act)
        
        # First quantized layer
        output1 = linear_layer_quantized(quantized_act1, self.quantized_weights1, self.scale_weights1, self.zero_point_weights1, self.scale_act1, self.zero_point_act1, 1.0, self.bias1)
        # Dummy calibration of the second activation.
        if isinstance(output1, np.ndarray):
          min_act2 = np.min(output1)
          max_act2 = np.max(output1)
        else:
          min_act2 = 0
          max_act2 = 10

        quantized_output1, scale_act2, zero_point_act2 = quantize_activation(output1, min_act2, max_act2)

        # Second quantized layer
        output2 = linear_layer_quantized(quantized_output1, self.quantized_weights2, self.scale_weights2, self.zero_point_weights2, scale_act2, zero_point_act2, 1.0, self.bias2)

        return output2


# Usage
model = QuantizedModel()
input_data = np.array([[2.0, 3.0], [1.0, 4.0]], dtype=np.float32)
output = model.forward(input_data)
print("Final Output:", output)
```

This example illustrates how you would structure a simple model using multiple quantized layers. It demonstrates the propagation of activations through the network, including the dequantization steps and re-quantization steps at each level, demonstrating the concept.

**Resource Recommendations:**

For gaining a deeper understanding of quantization techniques in general and for specific implementation details, consider the following resources:

1. **General Machine Learning Optimization Literature**: Textbooks and research papers on model compression provide theoretical background and insights into various quantization approaches beyond the basic dynamic range method. Pay particular attention to sections on integer arithmetic and fixed-point representations.
2. **Hardware Architecture Documentation**: If you are targeting specific hardware (DSPs, NPUs), delve into the processor architecture manuals. These materials will describe the nuances of their integer arithmetic capabilities and optimization techniques which are critical to efficient implementation. Understanding the target architecture is essential for gaining the most performance.
3. **Numerical Analysis References:** Resources covering numerical analysis will help you understand the errors and trade-offs introduced by the conversion to fixed-point integers. Pay particular attention to error analysis related to numerical computations and rounding effects.

In summary, post-training quantization outside of TensorFlow Lite, although requiring more manual effort, offers a viable alternative that provides customization and can lead to better optimization for specific hardware.  Understanding the mathematics behind scale and zero-point calculations and the careful tracking of activation ranges is essential. Through this process, you can achieve model size reduction and inference speedup for environments where the TFLite runtime would be unsuitable.
