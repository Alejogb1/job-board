---
title: "How to resolve TensorFlow error 'ValueError: Cannot feed value of shape (100, 10) for Tensor 'Placeholder_1:0', which has shape '(?, 5)''?"
date: "2025-01-30"
id: "how-to-resolve-tensorflow-error-valueerror-cannot-feed"
---
The root cause of the "ValueError: Cannot feed value of shape (100, 10) for Tensor 'Placeholder_1:0', which has shape '(?, 5)'" error in TensorFlow stems from a fundamental mismatch between the shape of the data you're attempting to provide as input and the shape of the placeholder you've defined within your TensorFlow graph. I’ve encountered this often while developing various deep learning models, especially when dealing with flexible batch sizes. This issue arises because placeholders are symbolic variables requiring explicit shape declaration when the graph is built, and subsequently these shapes must be compatible with data provided at runtime.

Let's break down the error. The message indicates that you tried to "feed" a value – likely a NumPy array or a TensorFlow tensor – that has dimensions of (100, 10) into a placeholder named 'Placeholder_1:0'. However, that placeholder has been configured to accept values with a shape of '(?, 5)'. The `?` symbol signifies that the first dimension (typically the batch size) is variable and can be adapted to the input, while the second dimension, denoted as 5, is fixed. Therefore, the mismatch lies in the second dimension: the placeholder expects a data array with 5 columns, while your data has 10 columns.

This scenario commonly occurs when you inadvertently misconfigure the input data's shape during pre-processing, make assumptions about feature vector lengths that don’t hold in the pipeline, or fail to properly align placeholder definitions with the expected characteristics of your input data. This is a debugging situation often faced when data pipelines become complicated. I've found a structured approach to be crucial when tackling this.

First, one should double-check the definition of the placeholder. The following snippet demonstrates a typical TensorFlow 1.x placeholder declaration:

```python
import tensorflow as tf
import numpy as np

# Correct placeholder with second dimension of 5.
input_placeholder = tf.placeholder(tf.float32, shape=(None, 5), name="Placeholder_1")

# Example data that matches the shape.
input_data = np.random.rand(100, 5).astype(np.float32)

# Attempting to feed the data into the placeholder
with tf.Session() as sess:
    output = sess.run(tf.reduce_sum(input_placeholder), feed_dict={input_placeholder: input_data})
    print(output)
```

Here, I declare a placeholder using `tf.placeholder()` with a `shape` parameter, specifying `(None, 5)`. This means the placeholder can accept any number of batches (represented by `None`), provided that each data point consists of 5 numerical values (e.g., 5 features). Then, I generate example input data with shape (100, 5), where 100 represents the batch size. The `feed_dict` within the `sess.run` method is used to link data values to placeholders. This code executes successfully as the data shape (100, 5) matches the placeholder’s expected shape.

Now let’s consider a case that produces the error:

```python
import tensorflow as tf
import numpy as np

# Placeholder with second dimension of 5.
input_placeholder = tf.placeholder(tf.float32, shape=(None, 5), name="Placeholder_1")

# Incorrect data with second dimension of 10.
incorrect_input_data = np.random.rand(100, 10).astype(np.float32)

# Attempting to feed incorrect data into the placeholder.
with tf.Session() as sess:
    try:
        output = sess.run(tf.reduce_sum(input_placeholder), feed_dict={input_placeholder: incorrect_input_data})
    except ValueError as e:
         print(f"Error: {e}")
```

In this snippet, `input_placeholder` is still defined to have a shape of `(None, 5)`. However, I've generated `incorrect_input_data` with dimensions (100, 10). Therefore, the `feed_dict` now attempts to push data with incompatible dimensionality into the placeholder, triggering the "ValueError" described previously. The error message clearly indicates the discrepancy between the expected and provided shapes. Note that the error happens during `sess.run` when TensorFlow tries to evaluate the graph with the provided data.

To correct this, I would need to either change the definition of the placeholder, alter the data pre-processing, or modify the way my model is consuming the input. In my work, I’ve often found that the data is being improperly parsed or transformed during feature engineering, resulting in the wrong shape at feeding time.

Another important scenario is the implicit assumption of batch size within a placeholder.  While `None` allows flexibility in training, a placeholder can explicitly define a batch size. Consider this:

```python
import tensorflow as tf
import numpy as np

# Placeholder defined with fixed batch size of 25.
input_placeholder = tf.placeholder(tf.float32, shape=(25, 5), name="Placeholder_1")

# Data with correct batch size.
correct_input_data = np.random.rand(25, 5).astype(np.float32)

# Attempting to feed correct data
with tf.Session() as sess:
    output = sess.run(tf.reduce_sum(input_placeholder), feed_dict={input_placeholder: correct_input_data})
    print(output)
    
# Incorrect Data with different batch size
incorrect_input_data = np.random.rand(100, 5).astype(np.float32)
with tf.Session() as sess:
  try:
    output = sess.run(tf.reduce_sum(input_placeholder), feed_dict={input_placeholder: incorrect_input_data})
  except ValueError as e:
         print(f"Error: {e}")
```

In this example, I've defined the `input_placeholder` with a fixed batch size of 25 using `shape=(25, 5)`. Therefore, feeding data with a batch size different from 25 will raise the error even when the second dimension (number of features) is correct.  The first call using `correct_input_data`, having dimensions (25, 5), works successfully because the dimensions align. However, feeding `incorrect_input_data` with dimensions (100, 5) will fail since the batch size is different.

To avoid these issues, I always perform rigorous shape checks at multiple stages in my training pipeline: after data loading, after each transformation step, and before the feed into TensorFlow graph. Visualizing the data shapes through print statements or using debuggers is helpful, especially for complex pipelines.  I found that a systematic check of each shape is always better than assuming, because those assumptions can sometimes lead to very subtle errors that are hard to trace back.

Furthermore, I would recommend reading the TensorFlow documentation on `tf.placeholder` and related input pipeline mechanisms. Review materials about data pre-processing techniques, ensuring that your feature engineering steps consistently produce data with the expected dimensions for your model. In the context of this error, focusing on aspects of shape consistency during training is most beneficial. Textbooks on deep learning offer comprehensive explanations on how the models need to be fed, and offer best practice advice on this aspect. Also, consulting tutorials on building custom data pipelines for TensorFlow can help understand data feeding nuances better. These resources can provide a thorough understanding of how data flows and how to enforce the right input format to your models.
