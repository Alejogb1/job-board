---
title: "What is the TensorFlow equivalent of PyTorch's `unfold` operation?"
date: "2025-01-30"
id: "what-is-the-tensorflow-equivalent-of-pytorchs-unfold"
---
The core functionality of PyTorch's `unfold` operation, extracting sliding local blocks from an input tensor, finds its equivalent in TensorFlow through a combination of `tf.image.extract_patches` and, in some cases, careful reshaping. While no single TensorFlow function mirrors `unfold` directly in its signature and output shape, understanding how to replicate its behavior is crucial for porting models or algorithms between the two frameworks.

The fundamental purpose of `unfold` is to create a view of the input tensor that presents overlapping patches. Given an input of shape `(N, C, H, W)` where `N` represents batch size, `C` channels, and `H` and `W` the height and width, and specified kernel size, stride, and padding, `unfold` outputs a tensor of shape `(N, C * kernel_height * kernel_width, L)`, where `L` is the number of patches calculated based on input dimensions, kernel size, stride, and padding. In essence, it flattens each kernel window into a vector and arranges these vectors along the last dimension. TensorFlow’s `tf.image.extract_patches` offers the basis for this, but additional steps might be necessary for achieving equivalent output tensor structure.

Unlike `unfold`, `tf.image.extract_patches` outputs a tensor where the patch vectors are not flattened as the second dimension. Instead, the extracted patches retain their spatial structure. The output has dimensions `(N, out_height, out_width, kernel_height * kernel_width * channels)`. Therefore, for direct comparison with the standard `unfold` output, we usually need to apply a `tf.reshape` operation after using `tf.image.extract_patches`. Further, the output shape of `tf.image.extract_patches` is inherently dependent on the way that padding is handled; the default is typically ‘VALID’ padding, which means that the output dimensions will not necessarily reflect the same output spatial size as PyTorch's `unfold`, particularly when using ‘SAME’ padding. This difference in defaults often leads to unexpected shape mismatches when translating models.

Let me illustrate this with examples, drawing from scenarios I've encountered during the conversion of models for distributed inference.

**Example 1: Basic Patch Extraction**

Imagine an image tensor with shape `(1, 3, 10, 10)`, a single 10x10 RGB image. We wish to extract 3x3 patches using a stride of 1, and without padding. PyTorch's `unfold` can easily accomplish this task. Here's how to achieve the equivalent with TensorFlow:

```python
import tensorflow as tf
import numpy as np

# Example Image (N, C, H, W) format
image_np = np.random.rand(1, 3, 10, 10).astype(np.float32)
image_tf = tf.convert_to_tensor(np.transpose(image_np, (0, 2, 3, 1))) # Convert and transpose to (N, H, W, C)
# PyTorch equivalent: image_unfolded = torch.from_numpy(image_np).unfold(2, 3, 1).unfold(3, 3, 1)

# Configuration
kernel_size = [1, 3, 3, 1]  # [batch, height, width, channels]
strides = [1, 1, 1, 1]
rates = [1, 1, 1, 1]
padding = 'VALID' # PyTorch 'VALID' is implicitly no padding for unfold

# Extract Patches
patches = tf.image.extract_patches(
    images=image_tf,
    sizes=kernel_size,
    strides=strides,
    rates=rates,
    padding=padding
)

# Reshape to match unfold's output dimension (N, flattened_patch_size, L)
n, out_h, out_w, patch_depth = patches.shape
reshaped_patches = tf.reshape(patches, (n, out_h*out_w, patch_depth))
final_output = tf.transpose(reshaped_patches, (0, 2, 1))

print("Original Image Shape:", image_tf.shape) # Output: (1, 10, 10, 3)
print("Extracted Patches Shape:", patches.shape) # Output: (1, 8, 8, 27)
print("Reshaped Patches Shape:", reshaped_patches.shape) # Output: (1, 64, 27)
print("Final Output Shape:", final_output.shape) # Output: (1, 27, 64)

```

In this example, we have converted the input numpy array into TensorFlow's expected `(N, H, W, C)` format. We extract the 3x3 patches using `tf.image.extract_patches`. Then, we reshape the result to flatten the spatial dimensions to create the tensor of size `(N, L, C * kernel_height * kernel_width)`, which we then transpose to `(N, C * kernel_height * kernel_width, L)` to align the dimensions with a PyTorch `unfold`. While in this simple example the stride is set to 1 to simplify the code, more general stride parameters for the same spatial input can also be configured.

**Example 2: Handling Different Stride and Padding**

Now let's explore a situation where we need to simulate a specific stride and padding behavior. Suppose we want to use a kernel of size `(5, 5)` with a stride of 2 and utilize 'SAME' padding, ensuring the output spatial dimensions remain consistent with the input dimensions given the stride of 2.

```python
import tensorflow as tf
import numpy as np

# Example Image (N, C, H, W) format
image_np = np.random.rand(1, 3, 10, 10).astype(np.float32)
image_tf = tf.convert_to_tensor(np.transpose(image_np, (0, 2, 3, 1)))

# Configuration
kernel_size = [1, 5, 5, 1]  # [batch, height, width, channels]
strides = [1, 2, 2, 1]
rates = [1, 1, 1, 1]
padding = 'SAME' # To match common use case

# Extract Patches
patches = tf.image.extract_patches(
    images=image_tf,
    sizes=kernel_size,
    strides=strides,
    rates=rates,
    padding=padding
)
# Reshape to match unfold's output dimension (N, flattened_patch_size, L)
n, out_h, out_w, patch_depth = patches.shape
reshaped_patches = tf.reshape(patches, (n, out_h*out_w, patch_depth))
final_output = tf.transpose(reshaped_patches, (0, 2, 1))

print("Original Image Shape:", image_tf.shape)
print("Extracted Patches Shape:", patches.shape) # output (1, 5, 5, 75)
print("Reshaped Patches Shape:", reshaped_patches.shape) # (1, 25, 75)
print("Final Output Shape:", final_output.shape) # (1, 75, 25)
```

Here we maintain the same approach as above. However, by setting the padding to 'SAME' and using a stride of 2, the `tf.image.extract_patches` function automatically adds padding as necessary so that, with an input of 10x10, and 5x5 kernels, with stride 2, results in the output patch spatial dimensions being roughly `ceil(10 / 2) = 5`. This is a common use case where you expect spatial dimensions to be maintained when using a strided kernel, where 'SAME' padding can match the PyTorch behavior. As before, the final reshape and transpose operations yield the output shape we'd expect from `unfold`. Note the `patch_depth` is now 75 due to the 5 x 5 kernel across 3 channels.

**Example 3: Application within a Layer**

In most machine learning applications, such functionality is not implemented in isolation; but rather, it often appears within a network layer. This example demonstrates how you might abstract this functionality as a custom layer:

```python
import tensorflow as tf

class UnfoldLayer(tf.keras.layers.Layer):
    def __init__(self, kernel_size, strides, padding, rates = [1,1,1,1], **kwargs):
        super(UnfoldLayer, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.strides = strides
        self.rates = rates
        self.padding = padding

    def call(self, inputs):
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=self.kernel_size,
            strides=self.strides,
            rates = self.rates,
            padding=self.padding
        )
        n, out_h, out_w, patch_depth = patches.shape
        reshaped_patches = tf.reshape(patches, (n, out_h * out_w, patch_depth))
        final_output = tf.transpose(reshaped_patches, (0, 2, 1))
        return final_output

# Example usage
image_np = np.random.rand(1, 10, 10, 3).astype(np.float32)
image_tf = tf.convert_to_tensor(image_np)

unfold_layer = UnfoldLayer(kernel_size=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID')
output = unfold_layer(image_tf)

print("Original Image Shape:", image_tf.shape)
print("Output Shape:", output.shape) #output (1, 27, 64)

unfold_layer_with_padding = UnfoldLayer(kernel_size=[1, 5, 5, 1], strides=[1, 2, 2, 1], padding='SAME')
output_padding = unfold_layer_with_padding(image_tf)

print("Output shape same padding:", output_padding.shape) #(1, 75, 25)

```
This custom layer encapsulates the patch extraction and reshaping process, making it reusable within larger TensorFlow models. The layer can accept varying sizes of kernels, strides, rates and padding options just like the PyTorch `unfold`. Using this abstraction reduces complexity and errors when integrating the functionality into more complicated networks.

**Resource Recommendations**

To further understand TensorFlow image operations and tensor manipulations, consulting the official TensorFlow documentation is highly recommended. Particularly useful are sections on `tf.image`, especially `tf.image.extract_patches`, as well as `tf.reshape`, `tf.transpose`, and the documentation on tensor broadcasting and manipulation. Studying practical examples from image processing and computer vision applications using TensorFlow can also help to solidify the understanding of such functionality. I have also found that understanding tensor layouts by reading papers involving tensor algebra is often helpful for these types of operations. Finally, experimenting with the different available padding options for the various functions can prove useful in matching desired behaviors.
