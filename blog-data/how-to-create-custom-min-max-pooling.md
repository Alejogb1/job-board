---
title: "How to create custom min max pooling?"
date: "2024-12-16"
id: "how-to-create-custom-min-max-pooling"
---

Alright, let's talk custom min-max pooling. I've seen my fair share of this kind of operation, especially back when I was deep in image processing and custom deep learning architectures. It's a powerful technique when standard pooling doesn't quite fit the bill, and getting it implemented efficiently is paramount. The usual max and average pooling operations are straightforward, but sometimes we need something different, perhaps to emphasize specific features or to maintain information lost during max-only pooling.

Custom min-max pooling, as the name suggests, allows us to not just extract the maximum values from our input regions, but also the minimum values, and then either concatenate or combine these in some user-defined way. There are many reasons for this. Sometimes you might want to extract features that are both very active (high values) and very inactive (low values). other times you might be combining feature maps derived from different processes that might benefit from such an operation. In my experience, it’s been useful when developing algorithms for anomaly detection in medical imaging, for instance, where both unusually high and unusually low signals in an area can indicate a problem, rather than just the highest one.

The core idea is that instead of a single output per region, you generate *two* outputs - the maximum and the minimum. Then, you decide what to do with this paired output. This could be as simple as stacking them or as intricate as doing a linear combination or applying a custom function. The flexibility is where the real advantage lies.

Now, let's delve into how we achieve this in a practical way. I'll be using Python with NumPy for the sake of demonstration. These examples should be easy to translate into other numerical environments or deep learning frameworks. The trick is to handle the indexing and operations carefully to maintain clarity and efficiency.

**Example 1: Basic Min-Max Pooling with Stacking**

This first example demonstrates the simplest approach: calculating the minimum and maximum within a pooling window and concatenating them along a new axis. Let’s assume a 2x2 pooling window and no padding and no stride. The pooling will run over non-overlapping regions.

```python
import numpy as np

def min_max_pooling_stack(input_array, pool_size=(2, 2)):
  """
  Performs min-max pooling and stacks results along a new axis.

  Args:
      input_array (np.ndarray): The input array (height, width, channels).
      pool_size (tuple): Size of the pooling window.

  Returns:
      np.ndarray: The pooled output (pooled_height, pooled_width, 2*channels).
  """
  h, w, c = input_array.shape
  ph, pw = pool_size

  pooled_h = h // ph
  pooled_w = w // pw

  pooled_output = np.zeros((pooled_h, pooled_w, 2*c))

  for i in range(pooled_h):
    for j in range(pooled_w):
        start_h = i * ph
        end_h = start_h + ph
        start_w = j * pw
        end_w = start_w + pw

        region = input_array[start_h:end_h, start_w:end_w, :]
        region_min = np.min(region, axis=(0, 1))
        region_max = np.max(region, axis=(0, 1))

        pooled_output[i, j, :c] = region_min #put min values first, into c first channels
        pooled_output[i, j, c:] = region_max #put max values into following c channels
  return pooled_output

# Example Usage
input_data = np.random.rand(4, 4, 3)  # A 4x4 image with 3 channels
pooled_data = min_max_pooling_stack(input_data)
print("Original shape:", input_data.shape)
print("Pooled shape:", pooled_data.shape)
```

This first snippet initializes an output array based on the dimensions and calculates the minimum and maximum along the channel dimension for each pooled region. I then stack the minimum and maximum results along the depth dimension. It illustrates the fundamental mechanics of calculating these quantities across regions.

**Example 2: Min-Max Pooling with Custom Combination Function**

Now, let's spice it up. We will introduce a custom combination function instead of simply stacking the min and max results. For this example, I’ll average them, though you could easily pass in any custom function you needed.

```python
import numpy as np

def custom_min_max_pooling(input_array, pool_size=(2, 2), combination_function=np.mean):
    """
    Performs min-max pooling and applies a custom function.

    Args:
        input_array (np.ndarray): The input array (height, width, channels).
        pool_size (tuple): Size of the pooling window.
        combination_function (callable): Function to combine min and max.

    Returns:
        np.ndarray: The pooled output (pooled_height, pooled_width, channels).
    """
    h, w, c = input_array.shape
    ph, pw = pool_size
    pooled_h = h // ph
    pooled_w = w // pw

    pooled_output = np.zeros((pooled_h, pooled_w, c))

    for i in range(pooled_h):
        for j in range(pooled_w):
            start_h = i * ph
            end_h = start_h + ph
            start_w = j * pw
            end_w = start_w + pw

            region = input_array[start_h:end_h, start_w:end_w, :]
            region_min = np.min(region, axis=(0, 1))
            region_max = np.max(region, axis=(0, 1))

            pooled_output[i, j, :] = combination_function(np.stack((region_min, region_max)), axis=0)

    return pooled_output

# Example usage
input_data = np.random.rand(4, 4, 3)
pooled_data_avg = custom_min_max_pooling(input_data)
print("Original shape:", input_data.shape)
print("Averaged pooled shape:", pooled_data_avg.shape)

def weighted_sum(min_max_stack, weights=[0.25, 0.75]):
    return (min_max_stack[0] * weights[0]) + (min_max_stack[1]*weights[1])

pooled_data_weighted = custom_min_max_pooling(input_data, combination_function=weighted_sum)
print("Weighted pooled shape:", pooled_data_weighted.shape)
```

This example highlights the versatility of custom pooling by utilizing a `combination_function`. By taking the average or a weighted sum of the min and max values, we are getting something different than just taking the highest values or just the lowest ones. You can pass any arbitrary function, which opens up numerous possibilities for feature extraction.

**Example 3: Min-Max Pooling with Stride and Padding**

Finally, let’s incorporate strides and padding which I encountered countless times when working with convolutional layers. This introduces a more complex version that requires careful indexing. In practice, padding is required to avoid shrinking dimensions too quickly when using stride, so this is vital.

```python
import numpy as np

def min_max_pooling_stride_pad(input_array, pool_size=(2, 2), stride=(1, 1), padding=(0, 0), combination_function = np.stack):
    """
    Performs min-max pooling with stride and padding.

    Args:
        input_array (np.ndarray): The input array (height, width, channels).
        pool_size (tuple): Size of the pooling window.
        stride (tuple): Stride of the pooling window.
        padding (tuple): Padding to be added to the input array
        combination_function (callable): Function to combine min and max.

    Returns:
        np.ndarray: The pooled output.
    """
    h, w, c = input_array.shape
    ph, pw = pool_size
    sh, sw = stride
    pad_h, pad_w = padding

    padded_input = np.pad(input_array, ((pad_h, pad_h), (pad_w, pad_w), (0,0)), mode='constant')
    h, w, _ = padded_input.shape

    pooled_h = (h - ph) // sh + 1
    pooled_w = (w - pw) // sw + 1

    pooled_output_shape = (pooled_h, pooled_w, 2 * c if combination_function is np.stack else c)
    pooled_output = np.zeros(pooled_output_shape)

    for i in range(pooled_h):
        for j in range(pooled_w):
            start_h = i * sh
            end_h = start_h + ph
            start_w = j * sw
            end_w = start_w + pw

            region = padded_input[start_h:end_h, start_w:end_w, :]
            region_min = np.min(region, axis=(0, 1))
            region_max = np.max(region, axis=(0, 1))

            pooled_output[i, j, :] = combination_function(np.stack((region_min, region_max)), axis=0) if combination_function is np.stack else combination_function(np.stack((region_min, region_max)), axis=0)

    return pooled_output

# Example usage with stride and padding
input_data = np.random.rand(4, 4, 3)
pooled_data_stride_pad = min_max_pooling_stride_pad(input_data, pool_size=(2, 2), stride=(2, 2), padding=(1,1))
print("Original shape:", input_data.shape)
print("Stride/pad pooled shape:", pooled_data_stride_pad.shape)
```

This final example shows the complexities that arise when applying strides and padding. You now need to calculate the size of the padded input, and the size of the final pooled result. This is often where I’ve seen mistakes creep in. It requires careful calculation of your indices to ensure correctness, but it offers the most power by controlling feature down-sampling. Also, this last example allows for more generic combination functions, which is useful.

For anyone looking to really understand the theoretical underpinnings and best practices in this area, I'd recommend delving into *Deep Learning* by Goodfellow, Bengio, and Courville. This book provides an excellent foundation for understanding pooling and neural network architectures. Additionally, I've found *Convolutional Neural Networks for Visual Recognition* by Li Fei-Fei et al., to be very insightful, particularly when dealing with computer vision specific applications of pooling. Also, look into research papers dealing with "attentional pooling" and "multi-scale feature aggregation" as that's where more advanced and relevant ideas are generally found. The “ImageNet Classification with Deep Convolutional Neural Networks” paper is a good starting point for understanding how different types of pooling operations can be incorporated into a deep learning architecture.

In summary, creating custom min-max pooling involves computing both minimum and maximum values across local regions of an input tensor, and then combining these results in a desired way. This approach can enable finer-grained feature extraction, especially when combined with user defined combination functions and parameters like stride and padding. While simple in concept, it requires thoughtful implementation to avoid indexing errors and to achieve efficient performance.
