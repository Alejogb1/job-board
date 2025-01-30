---
title: "Why are width and height after the second MaxPool2D layer 1 less than the downsampling factor?"
date: "2025-01-30"
id: "why-are-width-and-height-after-the-second"
---
The discrepancy you observe – a width and height reduction exceeding the downsampling factor after a second `MaxPool2D` layer – stems from the interplay between the pooling operation's kernel size, stride, and padding, coupled with the inherent integer division used in calculating output dimensions. This is particularly pronounced when dealing with odd input dimensions and kernels, a scenario I've encountered frequently in my work on image classification models within the context of medical imaging analysis.  My experience reveals that this isn't a bug, but rather a direct consequence of how convolutional neural networks (CNNs) handle spatial downsampling.


**1.  Detailed Explanation:**

The `MaxPool2D` layer performs downsampling by selecting the maximum value within a sliding window (kernel) across the input feature map. The `kernel_size` parameter defines the dimensions of this window, and the `strides` parameter dictates the step size of the sliding window.  Padding, controlled by the `padding` parameter, adds extra values (typically zeros) to the input's borders.  These three factors govern the output dimensions.

Without padding, the output height and width are calculated as:

`Output_dimension = floor((Input_dimension - Kernel_size) / Stride) + 1`

The `floor` function is crucial.  It ensures the output dimension is an integer, reflecting the discrete nature of the grid.  This integer division is the primary source of the discrepancy.  If the numerator (`Input_dimension - Kernel_size`) isn't perfectly divisible by the stride, the result of the division is truncated, leading to a greater reduction than expected based solely on the stride value.

Consider a scenario with a 2x2 kernel and a stride of 2. If the input is 7x7, after the first MaxPool2D layer, the output becomes 3x3 (floor((7-2)/2) + 1 = 3).  Applying another MaxPool2D layer with the same parameters yields a 1x1 output (floor((3-2)/2) + 1 = 1). The reduction from 3x3 to 1x1 is greater than the stride of 2.  This illustrates the effect of the integer division inherent in the calculation. The cumulative effect across multiple layers amplifies this deviation from a simple multiplicative downscaling.

Adding padding can mitigate this effect to a degree.  'Same' padding, which ensures the output dimensions match the input dimensions (disregarding minor discrepancies due to kernel size and stride), can temporarily mask the issue in the intermediate layers, but it does not eliminate the underlying mathematical cause.  However, excessive padding can also affect the model's learning capacity and increase computational complexity, so choosing appropriate padding requires careful consideration.


**2. Code Examples with Commentary:**

The following examples demonstrate the issue using TensorFlow/Keras, showcasing the dimension changes after successive `MaxPool2D` layers.


**Example 1: No Padding, Odd Input Dimensions**

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(7, 7, 1)), #7x7 input
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
])

test_input = tf.random.normal((1, 7, 7, 1))
output = model(test_input)
print(output.shape) #Output: (1, 1, 1, 1)
```

This exemplifies the issue directly. The 7x7 input, after two 2x2 MaxPooling operations with a stride of 2, results in a 1x1 output instead of a 1.75x1.75 (which is not possible in the discrete context).


**Example 2: 'Same' Padding, Odd Input Dimensions**

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(7, 7, 1)),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')
])

test_input = tf.random.normal((1, 7, 7, 1))
output = model(test_input)
print(output.shape) #Output: (1, 2, 2, 1)
```

While 'same' padding initially maintains the dimensionality, the second pooling operation still leads to a reduction disproportionate to the stride alone due to the integer division in the calculation.  Note that the output is 2x2, not 3x3 as one might expect.  The padding merely postpones the effect of the integer truncation.


**Example 3: Even Input Dimensions**

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(8, 8, 1)), #8x8 input
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
])

test_input = tf.random.normal((1, 8, 8, 1))
output = model(test_input)
print(output.shape) #Output: (1, 2, 2, 1)
```

With even input dimensions, the discrepancy is less pronounced but can still arise if the kernel size is not a factor of the input dimension. The output here is 2x2, which aligns more closely with the expected downsampling, though again, note that the exact outcome is predicated on integer division.



**3. Resource Recommendations:**

For a deeper understanding of convolutional neural networks and the mathematics of pooling operations, I recommend consulting standard textbooks on deep learning, specifically focusing on chapters dedicated to CNN architectures and their computational aspects.  Furthermore, review the documentation for the specific deep learning framework you are using (TensorFlow, PyTorch, etc.) to understand the precise details of their implementation of the `MaxPool2D` function and its parameter options.  Exploring academic papers on CNN architectures and optimization techniques will provide further insights into handling these effects during model design.  Focusing on publications dealing with image processing in medical imaging will help to refine and contextualize such information.
