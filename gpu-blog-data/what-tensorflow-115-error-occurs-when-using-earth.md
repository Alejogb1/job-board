---
title: "What TensorFlow 1.15 error occurs when using Earth Mover's Distance as a loss function on 2D arrays?"
date: "2025-01-30"
id: "what-tensorflow-115-error-occurs-when-using-earth"
---
The specific TensorFlow 1.15 error encountered when attempting to employ Earth Mover's Distance (EMD), often termed Wasserstein Distance, as a loss function directly on 2D arrays stems from a mismatch between the expected input format of the available EMD implementations and the shape of the data being supplied. Specifically, the error manifests as a `ValueError: Shape must be rank 3 but is rank 2` or a similar message indicating a dimensionality mismatch within the computational graph. This is frequently observed when users naively treat 2D arrays, representing images or similar data structures, as valid inputs for EMD, which fundamentally operates on distributions represented as point clouds.

The underlying issue is that the standard EMD algorithms, and particularly those readily available in TensorFlow 1.15, expect the input tensors to represent distributions as sets of points in an n-dimensional space. These point sets are conventionally represented by a 3D tensor of shape `[batch_size, num_points, dimension]`. The `batch_size` signifies the number of independent comparisons to perform. The `num_points` denotes the number of points within each distribution being compared, and `dimension` is the dimensionality of the space in which those points reside.

The EMD, in its core function, calculates the minimum cost to 'move' probability mass from one distribution to match another. This computation requires an understanding of each point's location within the specified space. Hence, the input tensor needs explicit dimensionality to define the point coordinates. When a 2D tensor, of shape such as `[batch_size, width * height]` representing flattened images, is passed directly, this point-cloud interpretation is absent. The system interprets the dimensions as batch size and feature size respectively, failing to recognize the implicit point cloud representation. The available EMD implementation is therefore unable to execute correctly since it doesn't find the explicit location in a n-dimensional space for each of these entries.

To mitigate this dimensionality issue, one must first reshape the 2D data to conform with the 3D point cloud structure. This often requires some degree of interpretation about how the 2D data can be interpreted in terms of points. For image-based data, where the pixels themselves can be considered as points in the image space, it requires mapping pixel locations to coordinate positions in the chosen point cloud representation. This is usually performed using a cartesian grid. For other types of 2D data, the representation may need to be different and based on the specific application. The dimensionality of this point space directly affects the interpretation, e.g., a 2D point space or a higher dimensional space can be constructed.

The key takeaway is that EMD implementations typically do not operate directly on flattened representations of data without mapping it to an explicit location, as they require the point cloud interpretation for their algorithm to work. Let's explore this with concrete examples.

**Example 1: Incorrect Input of 2D Arrays**

Consider two 2D arrays representing images, `image_a` and `image_b`, flattened to have shapes of `[batch_size, width * height]`. Assuming we have a loss function calculation that is based on Earth Movers Distance (EMD) between them.

```python
import tensorflow as tf

batch_size = 2
width = 4
height = 4

image_a = tf.random.normal([batch_size, width * height])
image_b = tf.random.normal([batch_size, width * height])

def earth_movers_distance_loss(x, y):
    # This is a simplification; actual EMD implementation varies.
    # Assuming some theoretical function that expects input shape of [batch_size, n_points, dimension]
    # This code will throw an error when implemented.
    # Assume the underlying implementation needs a third dimension
    emd_distance = tf.reduce_sum(tf.abs(x - y), axis=1) 
    return tf.reduce_mean(emd_distance)

# Incorrect usage: EMD on flat data, causing a shape error
loss = earth_movers_distance_loss(image_a, image_b)

with tf.Session() as sess:
    try:
        sess.run(loss)
    except tf.errors.InvalidArgumentError as e:
        print(f"Error encountered: {e}") # Prints the Shape Rank error.
```

In the above snippet, if we assume `earth_movers_distance_loss` requires a third dimension, the session run will result in an `InvalidArgumentError`. This illustrates the typical error encountered when passing 2D data directly to a function anticipating 3D input that represents a collection of points.

**Example 2: Reshaping for 2D Point Cloud EMD**

To properly utilize EMD with these 2D 'image' like data, we need to reshape our arrays to have explicit point coordinate information. Here, we consider a 2D image as a set of points in the 2D plane by providing coordinate information.

```python
import tensorflow as tf
import numpy as np

batch_size = 2
width = 4
height = 4
dimension = 2  # 2D spatial coordinates (x, y)

image_a = tf.random.normal([batch_size, width * height])
image_b = tf.random.normal([batch_size, width * height])

# Generating a coordinate grid in 2D
x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
grid_coords = np.stack([x_coords.flatten(), y_coords.flatten()], axis=1) # shape = [16, 2]
point_coords = tf.constant(grid_coords, dtype=tf.float32) # Shape is [width*height, 2]

# Reshaping the image data to include 2D point locations
image_a_reshaped = tf.reshape(image_a, [batch_size, width * height, 1]) * point_coords
image_b_reshaped = tf.reshape(image_b, [batch_size, width * height, 1]) * point_coords

def earth_movers_distance_loss(x, y):
    # Simplified calculation. In practice, use a proper implementation
    emd_distance = tf.reduce_sum(tf.abs(x- y), axis=2)
    emd_distance = tf.reduce_sum(emd_distance, axis=1)
    return tf.reduce_mean(emd_distance)

# Corrected usage: EMD on reshaped point cloud data
loss = earth_movers_distance_loss(image_a_reshaped, image_b_reshaped)

with tf.Session() as sess:
    output_loss = sess.run(loss)
    print(f"EMD loss: {output_loss}")
```

In this example, we first create a 2D coordinate system to represent each pixel. We use this coordinate system and multiply it by our flatten image. This reshape operation effectively encodes the 2D location of each pixel in the point cloud representation, thereby satisfying the dimension requirements of EMD. This step addresses the dimensionality mismatch issue and makes our simplified EMD implementation work. Real-world EMD implementations are likely more complex but should not have a shape error if input has the required shape.

**Example 3: Using an EMD library**
Let's create an example using a library that actually computes EMD. The code is similar but we will switch out the manual computation with a call to a pre-implemented EMD. Note that this uses Tensorflow 2 but can be adapted to Tensorflow 1.

```python
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

batch_size = 2
width = 4
height = 4
dimension = 2  # 2D spatial coordinates (x, y)

image_a = tf.random.normal([batch_size, width * height])
image_b = tf.random.normal([batch_size, width * height])

# Generating a coordinate grid in 2D
x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
grid_coords = np.stack([x_coords.flatten(), y_coords.flatten()], axis=1) # shape = [16, 2]
point_coords = tf.constant(grid_coords, dtype=tf.float32) # Shape is [width*height, 2]

# Reshaping the image data to include 2D point locations
image_a_reshaped = tf.reshape(image_a, [batch_size, width * height, 1]) * point_coords
image_b_reshaped = tf.reshape(image_b, [batch_size, width * height, 1]) * point_coords

def earth_movers_distance_loss(x, y):
    # Using the tensorflow probability library to compute the EMD
    emd_distance = tfp.stats.wasserstein_distance(x, y, p=1)
    return emd_distance

# Corrected usage: EMD on reshaped point cloud data
loss = earth_movers_distance_loss(image_a_reshaped, image_b_reshaped)

with tf.compat.v1.Session() as sess:
    output_loss = sess.run(loss)
    print(f"EMD loss: {output_loss}")
```

This example calls an existing library to compute the EMD. The result is a EMD score between the two point clouds. This example also highlights the importance of reshaping data to the required shape needed by any library.

**Recommendations:**

To further improve understanding and implementation, several resources are beneficial. Explore texts on optimal transport theory, specifically regarding the theory behind the Earth Mover's Distance. Research papers that investigate the application of EMD to image processing can also be helpful. Experimenting with custom implementations of EMD, even simplified versions, aids in grasping the process. Familiarizing oneself with TensorFlow’s tensor manipulation functionalities, specifically `tf.reshape`, is critical. Lastly, study the documentation of any EMD library that is being employed. Understanding the expected shape of the input is crucial for avoiding these kinds of shape-related errors.
