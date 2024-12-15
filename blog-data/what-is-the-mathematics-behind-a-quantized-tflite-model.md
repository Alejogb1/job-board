---
title: "What is the Mathematics behind a quantized Tflite model?"
date: "2024-12-15"
id: "what-is-the-mathematics-behind-a-quantized-tflite-model"
---

alright, so you're asking about the math behind quantized tflite models, huh? it's a pretty interesting area, and i've spent more time than i care to remember elbow deep in this stuff. i've debugged my share of models that went sideways due to quantization weirdness. let me break it down from my perspective, someone who's actually been there, seen the wonky outputs, and had to fix it.

basically, quantization is about representing numbers with less precision than you would with the standard 32-bit floating-point numbers (float32). these float32 values, they're great for training, but they're computationally expensive when you're running things on resource-constrained devices like phones or embedded systems. so, we downgrade to lower-bit integers to get the performance gains.

the core idea is to map the floating-point range to a smaller integer range. the most common case is 8-bit integers (int8), since it provides a good balance between accuracy and speed, although other bit depths exist, like 16 or even 4 bits in some extreme cases.

it's not a magic shrinking process, it relies on a few key mathematical operations to make this translation work. fundamentally, it's about finding a linear transformation that takes a floating-point number and maps it to a whole number representation.

first, you figure out the *scale* and *zero point*. the scale, is the multiplier needed to convert the quantized value back to an approximation of the original float value. the zero-point is like the offset that aligns the zero value in the float domain with the zero of the integer representation. this part is crucial.

to make it clearer, suppose we have a range of floating-point numbers `[min_float, max_float]` that we want to map to int8 values ranging from `-128 to 127`.

the formulas go something like this:

*   `scale = (max_float - min_float) / (max_int - min_int)`

*   `zero_point = round(min_int - min_float / scale)`

here, `max_int` would be `127` and `min_int` would be `-128` for int8.

the quantization process then goes like:

*   `quantized_value = round(float_value / scale + zero_point)`

and to get back to approximate floating-point:

*   `float_approx = (quantized_value - zero_point) * scale`

so now you are starting to see the basic formula. note that the approximation is usually not perfect and loss of information occurs because of quantization and truncation from floats to ints. these formulas do have minor variants with subtle differences in the way rounding is performed, but the underlying idea remains the same.

you might wonder, how are `min_float` and `max_float` decided? well, in practice it’s not some guessing game. usually we determine these values from the weights and activations of the model during the training phase or afterwards, through some techniques that calculate the range. it depends on the quantization technique used (post-training or quantization aware training).

now, lets put this in practice with a quick python code snippet:

```python
import numpy as np

def quantize_linear(float_value, min_float, max_float, min_int, max_int):
    scale = (max_float - min_float) / (max_int - min_int)
    zero_point = int(round(min_int - min_float / scale))
    quantized_value = int(round(float_value / scale + zero_point))
    return quantized_value

def dequantize_linear(quantized_value, min_float, max_float, min_int, max_int):
    scale = (max_float - min_float) / (max_int - min_int)
    zero_point = int(round(min_int - min_float / scale))
    float_approx = (quantized_value - zero_point) * scale
    return float_approx

# example usage
min_float_val = -1.0
max_float_val = 1.0
min_int_val = -128
max_int_val = 127

test_float = 0.5
quantized = quantize_linear(test_float, min_float_val, max_float_val, min_int_val, max_int_val)
dequantized = dequantize_linear(quantized, min_float_val, max_float_val, min_int_val, max_int_val)

print(f"original float: {test_float}")
print(f"quantized value: {quantized}")
print(f"dequantized value: {dequantized}")
```

this piece of code should give you a basic idea. you will see the `dequantized` output will be close to the original value, depending on range and rounding, there might be some degree of error. this is loss of information when switching to integers.

tflite often uses per-tensor and per-axis quantization. per-tensor means a single scale and zero-point are used for an entire tensor. per-axis quantization, particularly in convolution layers, is more nuanced where different channels can have its own scale and zero-point. it's a bit more math, but you're just applying the linear quantization idea separately to the channels of the weights or activations. this improves the accuracy of quantized model. i spent a lot of time tracking down errors due to wrong quantization in a CNN for image recognition once; it was just a matter of misplacing per-tensor and per-axis quantization. that one was a headache.

the mathematics also includes the specifics of how these integer operations are optimized for different architectures. for instance, special vector instructions like *simd* (single instruction, multiple data) are used on processors to multiply and add int8 numbers very efficiently. this is where all the performance gains appear. when a tflite model is executing, the underlying engine does the heavy lifting of using the quantized representations effectively.

also, it's worth looking into details related to *fused operations* which is another performance optimization technique where a series of operations, like convolution and batch normalization, are computed within a single step as a single operation. this process typically requires recalculating scales and zero-points carefully. if these operations are not well fused you get inaccurate models and slower computations as well. these optimizations aren't always simple, and it can lead to strange issues if you are not careful. i remember once my fused operation calculations were off by a factor of 2. the model outputs looked like they were generated in another universe.

let's add an example about per-channel quantization.

```python
import numpy as np

def quantize_per_channel(float_tensor, min_float_vals, max_float_vals, min_int, max_int, axis):
    num_channels = float_tensor.shape[axis]
    quantized_tensor = np.zeros_like(float_tensor, dtype=np.int8)
    for i in range(num_channels):
        if axis == 0:
            current_channel = float_tensor[i, ...]
        elif axis == 1:
             current_channel = float_tensor[:, i, ...]
        elif axis == 2:
             current_channel = float_tensor[:, :, i, ...]
        else:
            raise ValueError("unsupported axis")

        min_float = min_float_vals[i]
        max_float = max_float_vals[i]
        scale = (max_float - min_float) / (max_int - min_int)
        zero_point = int(round(min_int - min_float / scale))
        quantized_channel = np.round(current_channel / scale + zero_point).astype(np.int8)

        if axis == 0:
            quantized_tensor[i, ...] = quantized_channel
        elif axis == 1:
             quantized_tensor[:, i, ...] = quantized_channel
        elif axis == 2:
            quantized_tensor[:, :, i, ...] = quantized_channel
        
    return quantized_tensor
    

def dequantize_per_channel(quantized_tensor, min_float_vals, max_float_vals, min_int, max_int, axis):
    num_channels = quantized_tensor.shape[axis]
    dequantized_tensor = np.zeros_like(quantized_tensor, dtype=np.float32)
    for i in range(num_channels):
        if axis == 0:
            current_channel = quantized_tensor[i, ...]
        elif axis == 1:
             current_channel = quantized_tensor[:, i, ...]
        elif axis == 2:
             current_channel = quantized_tensor[:, :, i, ...]
        else:
            raise ValueError("unsupported axis")
        
        min_float = min_float_vals[i]
        max_float = max_float_vals[i]
        scale = (max_float - min_float) / (max_int - min_int)
        zero_point = int(round(min_int - min_float / scale))
        dequantized_channel = (current_channel - zero_point) * scale

        if axis == 0:
            dequantized_tensor[i, ...] = dequantized_channel
        elif axis == 1:
             dequantized_tensor[:, i, ...] = dequantized_channel
        elif axis == 2:
            dequantized_tensor[:, :, i, ...] = dequantized_channel
        
    return dequantized_tensor
    
# example usage with 3D tensor (e.g. weights for conv layer)
min_int_val = -128
max_int_val = 127

test_tensor = np.array([
        [[0.1, 0.2, -0.1], [0.3, 0.4, -0.2], [0.5, 0.6, -0.3]],
        [[0.7, 0.8, -0.4], [0.9, 1.0, -0.5], [1.1, 1.2, -0.6]],
        [[1.3, 1.4, -0.7], [1.5, 1.6, -0.8], [1.7, 1.8, -0.9]]
    ], dtype=np.float32)

min_float_per_channel = np.array([-1.0, -1.0, -1.0]) #for each channel
max_float_per_channel = np.array([1.0, 1.0, 1.0]) #for each channel
axis_to_quantize = 2 # the third dimension (channel dimension)

quantized_tensor = quantize_per_channel(test_tensor, min_float_per_channel, max_float_per_channel, min_int_val, max_int_val, axis_to_quantize)
dequantized_tensor = dequantize_per_channel(quantized_tensor, min_float_per_channel, max_float_per_channel, min_int_val, max_int_val, axis_to_quantize)

print(f"original tensor:\n{test_tensor}")
print(f"quantized tensor:\n{quantized_tensor}")
print(f"dequantized tensor:\n{dequantized_tensor}")
```

again, check the code for usage, it should give you a practical idea. you'll see, in this example, that the quantization and dequantization occur "per channel".

one last thing: sometimes, a technique called *symmetric quantization* is used where the zero-point is forced to zero, with both positive and negative values, so the range is always symmetrical around 0. then the scale is just calculated based on the maximum value in the tensor. this simplification is often used in areas where you can get away with some accuracy loss to further optimize for speed and resources.

and, just for fun, why did the quantized model cross the road? to get to the other side, *faster*! (i know, i know, i'll see myself out).

if you really want to go deeper into the math, i highly recommend looking at papers by benoit jacob and skoglund (2018) on quantization, and the tflite official documentation itself, that includes a lot of details on the mathematical aspects. they include the nitty-gritty details that i've skipped here. also the books "deep learning with python" by francois chollet and "hands-on machine learning with scikit-learn, keras & tensorflow" by aurélien géron contain some very important details on quantization and optimizations.

i've been burned many times by ignoring some detail about the mathematics of quantization. understanding this math isn't just a theoretical exercise, it's essential for building performant and reliable models, particularly if you are deploying them on the edge.
