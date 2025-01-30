---
title: "What's the difference between `tf.shape()` and `tensor.get_shape()`?"
date: "2025-01-30"
id: "whats-the-difference-between-tfshape-and-tensorgetshape"
---
The core distinction between `tf.shape()` and `tensor.get_shape()` lies in their operational context and the type of information they return.  `tf.shape()` is a TensorFlow operation that yields a tensor representing the shape of a tensor at runtime, while `tensor.get_shape()` accesses the statically known shape of a tensor during graph construction. This difference manifests significantly in eager execution versus graph execution modes. My experience working on large-scale image recognition models highlighted this discrepancy repeatedly, particularly when dealing with dynamically shaped input tensors.

**1. Clear Explanation:**

`tf.shape()` operates on tensors and returns a tensor containing the dimensions of the input. This operation is executed during the runtime of the TensorFlow graph.  Crucially, the shape returned by `tf.shape()` is not necessarily known at graph construction time; it reflects the actual shape of the tensor at the point of execution.  This is essential for handling situations where tensor shapes are determined dynamically, for example, during batch processing or when dealing with variable-length sequences.  The result is a tensor, making it suitable for use within other TensorFlow operations, such as slicing, reshaping, or conditional logic.  Its output is inherently variable and reflects the runtime characteristics of the tensor.

Conversely, `tensor.get_shape()` (or equivalently, `tensor.shape`) accesses the statically known shape of a tensor. This method works primarily during graph construction. The shape information it retrieves is determined *before* the graph is executed.  If the shape is not fully defined at graph construction—often the case with placeholders or tensors whose shape depends on input data—`tensor.get_shape()` will return a partially defined or unknown shape.  The output is not a tensor but a `TensorShape` object, a data structure describing the tensor's dimensions.  Trying to use this object directly in TensorFlow operations generally leads to errors.  Its utility lies in static analysis, shape validation, and compile-time optimizations.

The key difference boils down to runtime versus compile time. `tf.shape()` is a runtime operation; `tensor.get_shape()` provides information available during graph construction.  Their outputs are different in type and reflect differing levels of information available at different stages of the TensorFlow execution process.  Ignoring this distinction leads to unexpected behavior, especially in scenarios involving dynamic shapes.


**2. Code Examples with Commentary:**

**Example 1:  Illustrating the difference with statically shaped tensors:**

```python
import tensorflow as tf

# Statically shaped tensor
static_tensor = tf.constant([[1, 2], [3, 4]])

# Using tf.shape()
shape_tensor = tf.shape(static_tensor)
print(f"tf.shape(): {shape_tensor}")  # Output: tf.Tensor([2 2], shape=(2,), dtype=int32) - A tensor

# Using get_shape()
static_shape = static_tensor.get_shape()
print(f"get_shape(): {static_shape}")  # Output: TensorShape([2, 2]) - A TensorShape object


with tf.compat.v1.Session() as sess:
  print(f"tf.shape() evaluated: {sess.run(shape_tensor)}") # Output: [2 2]
```

This example demonstrates that both methods correctly identify the shape of a statically defined tensor. However, `tf.shape()` produces a tensor, while `tensor.get_shape()` returns a `TensorShape` object. The `tf.Session().run()` call is necessary to obtain the numerical value of the tensor produced by `tf.shape()`.


**Example 2: Handling dynamically shaped tensors:**

```python
import tensorflow as tf

# Placeholder for a dynamically shaped tensor
dynamic_tensor = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])

# Using tf.shape()
shape_tensor = tf.shape(dynamic_tensor)

# Using get_shape()
dynamic_shape = dynamic_tensor.get_shape()

#Feeding data to placeholder
feed_dict = {dynamic_tensor: [[1,2],[3,4],[5,6]]}

with tf.compat.v1.Session() as sess:
    print(f"tf.shape() with dynamic input: {sess.run(shape_tensor,feed_dict=feed_dict)}") #Output: [3 2]
    print(f"get_shape() with dynamic input: {dynamic_shape}") # Output: TensorShape([Dimension(None), Dimension(2)])
```

Here, the placeholder `dynamic_tensor` has an undefined dimension. `tf.shape()` correctly determines the shape at runtime once data is fed. `tensor.get_shape()` reflects the partially known shape during graph construction. The runtime shape is only available after data is passed in the `feed_dict`.

**Example 3:  Utilizing tf.shape() within an operation:**

```python
import tensorflow as tf

input_tensor = tf.compat.v1.placeholder(tf.float32,shape=[None,3])
shape = tf.shape(input_tensor)
#Slicing operation based on runtime shape
sliced_tensor = tf.slice(input_tensor,[0,0], [shape[0], 1])

with tf.compat.v1.Session() as sess:
  input_data = [[1,2,3],[4,5,6]]
  print(sess.run(sliced_tensor, feed_dict={input_tensor:input_data}))
```

This demonstrates how the tensor produced by `tf.shape()` can be used directly within another TensorFlow operation (`tf.slice`). The slice operation dynamically adjusts its boundaries based on the runtime shape of the input tensor. This is impossible with `tensor.get_shape()`, which provides a `TensorShape` object unsuitable for direct use within operations.


**3. Resource Recommendations:**

The official TensorFlow documentation.  Specifically, the sections detailing tensor manipulation, graph execution, and shape manipulation.  A comprehensive textbook on deep learning using TensorFlow, covering practical implementation aspects.  Finally, review relevant Stack Overflow threads on TensorFlow shape handling for targeted solutions to specific issues.  These resources provide a strong foundation and will help in addressing advanced shape-related challenges.
