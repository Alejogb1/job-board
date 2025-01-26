---
title: "How can I extract individual class labels from Keras image segmentation output?"
date: "2025-01-26"
id: "how-can-i-extract-individual-class-labels-from-keras-image-segmentation-output"
---

Image segmentation using Keras, particularly with models like U-Net, often produces a multi-channel output where each channel corresponds to a different class probability map. Therefore, directly interpreting the raw output requires processing to extract usable class labels.

My past work on medical image analysis extensively used this architecture. I found that the most direct method involves identifying the channel with the highest probability for each pixel, effectively assigning a class label to that pixel. The Keras model itself doesn't directly produce these labels; it provides probability scores per pixel per class.

The output tensor from a segmentation model, when processing a single image, typically has a shape of `(height, width, num_classes)`. Each element `(h, w, c)` represents the probability that pixel at location `(h, w)` belongs to class `c`. To obtain discrete class labels, the `argmax` operation is crucial. The `argmax` function, when applied along the class dimension (the last axis), returns the index of the channel with the maximum probability, corresponding to the predicted class. This creates a `(height, width)` tensor containing the integer class labels.

This process essentially converts a set of continuous probability maps into a single-channel image representing class assignments. Consequently, further operations, such as visualization or calculation of metrics, can be performed on this label map. This process should not be confused with one-hot encoding; it is instead the inverse operation following the model's prediction. The one-hot encoding occurred during data preparation before training.

Here are three examples illustrating different ways to achieve this, along with explanations:

**Example 1: Using NumPy**

This example relies solely on NumPy for processing the Keras output, making it the most portable. Consider `output_tensor`, the tensor from our segmentation model, obtained after `model.predict()`. The assumed output shape is `(1, height, width, num_classes)` in case of prediction on a single image, and we will drop the first dimension after.

```python
import numpy as np

# Assume output_tensor is a NumPy array from model.predict() with shape (1, height, width, num_classes)
# Example output for demonstration (replace with your actual tensor)
output_tensor = np.random.rand(1, 256, 256, 3) #Example 256x256 image with 3 classes

# Remove the batch dimension, if present
output_tensor = output_tensor.squeeze(axis=0)

# Apply argmax along the channel (last) axis
predicted_labels = np.argmax(output_tensor, axis=-1)

# predicted_labels is now a 2D array (height, width) containing integer class labels

print(f"Shape of predicted labels: {predicted_labels.shape}")
print(f"Data type of predicted labels: {predicted_labels.dtype}")

#To verify a slice of the prediction
print(f"Example predicted labels: {predicted_labels[100:105, 100:105]}")
```

In this code snippet, `np.squeeze(axis=0)` removes the batch dimension, and `np.argmax(output_tensor, axis=-1)` performs the core operation. The parameter `axis=-1` instructs NumPy to find the index of the maximum value along the last axis, i.e., the class probabilities. The resulting `predicted_labels` is now a 2D array representing the class segmentation. The data type is integer, suitable for representing class indices. This array is amenable to further image analysis techniques and can be visualized using libraries like `matplotlib`.

**Example 2: Utilizing TensorFlow Operations**

TensorFlow offers similar functionality directly within its graph, which is often more efficient if you intend to perform further computations within a TensorFlow workflow. This method avoids unnecessary data transfer between NumPy and TensorFlow.

```python
import tensorflow as tf

# Assume output_tensor is a tensor from model(input_image) with shape (1, height, width, num_classes)
# Example placeholder for demo, replace with actual model output
output_tensor = tf.random.normal((1, 256, 256, 3))

# Remove batch dimension, if present, to be able to work with the tensor.
output_tensor = tf.squeeze(output_tensor, axis=0)

# Use tf.argmax to get the class labels
predicted_labels = tf.argmax(output_tensor, axis=-1)

# To evaluate the tensor and view the predicted labels in Eager mode or with a session in graph mode.
with tf.Session() as sess:
   predicted_labels_eval = sess.run(predicted_labels)


print(f"Shape of predicted labels: {predicted_labels_eval.shape}")
print(f"Data type of predicted labels: {predicted_labels_eval.dtype}")
print(f"Example predicted labels: {predicted_labels_eval[100:105, 100:105]}")


```

Here, `tf.argmax` fulfills the same role as `np.argmax`. Importantly, if using TensorFlow within a graph, the resulting `predicted_labels` tensor needs to be evaluated in a session in graph mode or can be observed directly if running Eager mode by converting to numpy array by calling `predicted_labels.numpy()`. This example demonstrates a more computationally integrated approach. The data type of `predicted_labels` using TensorFlow is `tf.int64`.

**Example 3: Keras Backend Implementation**

This method provides direct access to Keras' backend functions. It's helpful when working within custom loss functions or callbacks where you need to process the model's output using the Keras backend.

```python
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf

# Assume output_tensor is a tensor from model(input_image) with shape (1, height, width, num_classes)
# Example placeholder for demo, replace with actual model output
output_tensor = tf.random.normal((1, 256, 256, 3))

# Remove the batch dimension, if present.
output_tensor = K.squeeze(output_tensor, axis=0)


# Utilize K.argmax to extract the class labels
predicted_labels = K.argmax(output_tensor, axis=-1)


#To evaluate in Eager mode or using session in graph mode
with tf.Session() as sess:
    predicted_labels_eval = sess.run(predicted_labels)

print(f"Shape of predicted labels: {predicted_labels_eval.shape}")
print(f"Data type of predicted labels: {predicted_labels_eval.dtype}")
print(f"Example predicted labels: {predicted_labels_eval[100:105, 100:105]}")
```

This example parallels the TensorFlow implementation using backend functions provided by `tensorflow.keras.backend`. Like in the previous example, this approach is useful for integrating label extraction directly within the Keras model building workflow using Keras functions and backend methods.

In all three examples, the underlying process is the same: extracting the class indices with the highest probability. The resulting `predicted_labels` array, in all instances, holds a class ID at every pixel location.

For additional study, consider exploring resources covering deep learning concepts specifically focused on semantic segmentation. Look for material that covers U-Net architectures and their outputs, along with practical code examples using libraries like TensorFlow and Keras. Additionally, studying documentation related to NumPy's array manipulation functions and TensorFlow's tensor operations would be beneficial. A deeper understanding of data formats used in computer vision tasks can improve your ability to efficiently process the results of complex machine learning models.
