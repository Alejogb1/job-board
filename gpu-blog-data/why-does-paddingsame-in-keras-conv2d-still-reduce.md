---
title: "Why does padding=SAME in Keras Conv2D still reduce output size?"
date: "2025-01-30"
id: "why-does-paddingsame-in-keras-conv2d-still-reduce"
---
The assertion that `padding='same'` in Keras' `Conv2D` *always* preserves the input spatial dimensions is incorrect.  While its intention is to maintain input and output shape consistency, the actual outcome is dependent on the interplay between the kernel size, strides, and the input dimensions themselves.  My experience debugging complex convolutional neural networks, particularly those employing dilated convolutions and non-square kernels, has highlighted this subtlety repeatedly. The crucial factor overlooked is the implicit handling of boundary conditions when calculating padding for non-divisible input dimensions by the kernel size plus stride.

**1. Explanation:**

The `padding='same'` argument in Keras' `Conv2D` aims to pad the input tensor such that the output tensor has the same spatial dimensions as the input.  However, the padding calculation isn't a simple addition of padding on both sides to match the kernel size. Keras employs a formula that ensures the output dimensions are *approximately* equal to the input dimensions.  This approximation becomes evident when the input dimensions are not perfectly divisible by the stride plus kernel size minus one.

The calculation implemented by Keras implicitly uses floor division (//) rather than ceiling division.  Let's dissect this.  Consider the formula used to compute the output height/width:

`Output_dimension = ceil(Input_dimension / Stride)`

where `ceil` denotes the ceiling function (rounding up to the nearest integer).  However, Keras’ implementation effectively replaces this with:

`Output_dimension = floor((Input_dimension + Stride - 1) / Stride)`

This effectively means that if the input dimension isn't perfectly divisible by the stride, the output will be slightly smaller. The difference in this implementation compared to `ceil` appears only when the input dimensions aren't evenly divisible by the stride.  Further, `padding='same'` adds padding to both sides; therefore, the amount of padding on each side is further influenced by the even or odd nature of the padding required.  The padding amount is determined to maintain the approximate spatial dimensions rather than achieving an exact match.

Let's analyze how the padding is computed:

1. **Stride:** Determines the sampling step of the convolution.
2. **Kernel Size:** The size of the convolutional filter.
3. **Input Shape:** The height and width of the input tensor.

The calculation of the padding involves ensuring that the output dimensions are integers. This leads to situations where the padding isn't perfectly symmetrical and, as a consequence, the output dimensions are slightly smaller than the input dimensions, particularly when the input dimensions and strides are not optimally paired.

**2. Code Examples:**

The following examples illustrate the behavior of `padding='same'` under varying conditions.  I've encountered these precise scenarios during my work on real-world image segmentation models.

**Example 1:  Even Input Dimensions and Stride 1**

```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same', input_shape=(64,64,3))
])

input_tensor = tf.random.normal((1, 64, 64, 3))
output_tensor = model(input_tensor)
print(output_tensor.shape) # Output: (1, 64, 64, 32)
```

Here, because the input dimensions (64x64) are perfectly divisible by the stride (1) and the kernel size (3) creates a scenario where `padding='same'` results in the expected output shape matching the input spatial dimensions.

**Example 2: Odd Input Dimensions and Stride 1**

```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same', input_shape=(65,65,3))
])

input_tensor = tf.random.normal((1, 65, 65, 3))
output_tensor = model(input_tensor)
print(output_tensor.shape) # Output: (1, 65, 65, 32)
```

In this case, with odd input dimensions, despite the stride of 1, `padding='same'` still manages to maintain the input spatial dimensions.  This is because the padding calculation is implicitly adjusting for the non-divisibility resulting in additional padding on one side compared to the other.

**Example 3: Non-Divisible Input Dimensions and Stride > 1**

```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), padding='same', input_shape=(65,65,3))
])

input_tensor = tf.random.normal((1, 65, 65, 3))
output_tensor = model(input_tensor)
print(output_tensor.shape) # Output: (1, 33, 33, 32)
```

Here, with a stride of 2 and an odd input dimension, the output dimensions are smaller than the input. The floor division in the implicit calculation reduces the output size, despite the `padding='same'` setting. The output dimensions are `ceil(65/2) = 33`.  This exemplifies the subtle differences from the mathematically intuitive `ceil(Input_dimension / Stride)` formula.


**3. Resource Recommendations:**

The Keras documentation, particularly the section detailing the `Conv2D` layer and its arguments, offers comprehensive information. Examining the source code of Keras’ convolutional layer implementation provides an in-depth understanding of the underlying padding calculations.  Finally, a solid grasp of discrete mathematics, specifically focusing on floor and ceiling functions and integer division, is essential for a full comprehension of this behavior.  Understanding how these functions impact calculation of padding is crucial.
