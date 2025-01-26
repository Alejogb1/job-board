---
title: "Why is TensorFlow's `decode_png` function repeatedly printing 'Cleanup called...'?"
date: "2025-01-26"
id: "why-is-tensorflows-decodepng-function-repeatedly-printing-cleanup-called"
---

TensorFlow’s `tf.io.decode_png` function exhibits repeated "Cleanup called..." messages, typically not indicative of an actual error, but rather a consequence of its internal resource management, specifically related to the underlying libpng library. In my experience debugging large-scale image processing pipelines involving complex custom data loading, this behavior manifested itself frequently, often leading to initial concern that I had introduced a memory leak.

The root cause stems from how TensorFlow interacts with external C libraries like libpng for decoding image formats. When `tf.io.decode_png` is called, it allocates temporary memory buffers within the libpng library to perform the decoding operation. These buffers are crucial, as they hold the pixel data and other metadata extracted from the PNG file during the decoding process. After the decoding completes and the TensorFlow tensor containing the image data is created, the libpng library, through its internal cleanup mechanisms, deallocates this memory. This deallocation is accompanied by the "Cleanup called..." print statement, triggered within a TensorFlow context whenever the `tf.io.decode_png` operation executes and completes. This pattern is deliberately designed to avoid memory leaks and is a sign of responsible resource management. It is essentially a confirmation that libpng's cleanup routine has been executed successfully for a given `decode_png` call.

The frequency of this message is proportional to the number of times `tf.io.decode_png` is called. If the code includes multiple calls within a data loading pipeline or within a loop that processes multiple images, the "Cleanup called..." message will appear repeatedly, matching the rate at which the decoding operation is invoked. Furthermore, since `tf.io.decode_png` often resides within the TensorFlow graph execution, these messages will usually occur during the actual runtime, rather than during graph construction.

The message itself, although somewhat verbose and initially alarming, is not problematic. It does not represent a memory leak, nor does it indicate a performance issue in most cases. It’s simply a verbose notification from the underlying library about its internal state. There are no parameters that can be tweaked within `tf.io.decode_png` to suppress this specific message. Attempting to mitigate the print statements programmatically is generally unnecessary, and in most cases, would be indicative of an incomplete understanding of the function's inner workings. Instead, I recommend focusing on validating that the overall image processing workflow is correct and efficient, rather than concerning oneself with this innocuous message.

To illustrate these principles, consider these examples:

**Example 1: Basic Image Decoding**

This showcases a minimal example illustrating the "Cleanup called..." message with a single image:

```python
import tensorflow as tf
import numpy as np

# Assume we have some PNG data stored in a bytes object
png_data = tf.io.encode_png(tf.constant(np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8))).numpy()

# Decode the PNG data
decoded_image = tf.io.decode_png(png_data, channels=3)

# Print the decoded image shape to verify output
print(decoded_image.shape)
```

Running this code will produce the "Cleanup called..." message once. The `encode_png` function is used to create a sample bytes array, which simulates the reading of PNG file content from disk. Note the need to explicitly convert it to a `numpy` array before feeding it to the `decode_png` function. This single invocation of `decode_png` results in a single "Cleanup called..." output. The output also confirms the resulting tensor shape.

**Example 2: Loop-Based Decoding**

This example demonstrates how multiple calls will generate multiple messages.

```python
import tensorflow as tf
import numpy as np

# Assume we have a list of PNG data bytes
num_images = 5
png_data_list = [tf.io.encode_png(tf.constant(np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8))).numpy() for _ in range(num_images)]

# Loop through the list and decode each image
decoded_images = []
for png_data in png_data_list:
  decoded_image = tf.io.decode_png(png_data, channels=3)
  decoded_images.append(decoded_image)

# Print the first decoded image shape
print(decoded_images[0].shape)
```

Executing this snippet will produce the "Cleanup called..." message five times, corresponding to each iteration in the loop. This example effectively illustrates how the message rate scales with the number of decoding operations performed within a script.

**Example 3: Decoding within tf.data Pipeline**

This demonstrates the message within a more practical data pipeline scenario using `tf.data.Dataset`:

```python
import tensorflow as tf
import numpy as np

# Assume we have a list of PNG data bytes (as in Example 2)
num_images = 3
png_data_list = [tf.io.encode_png(tf.constant(np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8))).numpy() for _ in range(num_images)]

# Create a tf.data.Dataset from the list of image data
dataset = tf.data.Dataset.from_tensor_slices(png_data_list)

# Function to decode each image
def decode_image(image_data):
  return tf.io.decode_png(image_data, channels=3)

# Apply the decoding function to the dataset
dataset = dataset.map(decode_image)

# Iterate through the dataset
for image in dataset:
  print(image.shape) #Trigger the ops
```

When the `for` loop iterates through the dataset, the `decode_image` function will be executed for each element, leading to the “Cleanup called...” being printed as the decoding operations are triggered by the loop's requirement to access each tensor in the dataset. This highlights how the message is also integrated when working with common preprocessing workflows.

While this message is benign, it can clutter the terminal output when running experiments or training models. A common practice I have adopted is to manage console verbosity effectively when logging significant events, using Python's `logging` module to control the detail level of messages, allowing the libpng output to be handled in the background without impacting debugging or experimentation.

For further information on image processing within TensorFlow, I highly suggest consulting the official TensorFlow documentation on `tf.image` and `tf.io`, which contains exhaustive details about all related operations. Additionally, reviewing example code available on the TensorFlow GitHub repository or in community tutorials related to image dataset loading and processing can prove immensely useful. Lastly, exploring online resources on image formats, such as PNG, will provide deeper understanding of the complexities related to image decoding in general. Such resources, though not specific to TensorFlow’s implementation, can help build foundational knowledge for comprehending the behavior of the underlying libraries.
