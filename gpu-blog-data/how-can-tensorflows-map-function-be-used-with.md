---
title: "How can TensorFlow's `map` function be used with `if/else` statements?"
date: "2025-01-30"
id: "how-can-tensorflows-map-function-be-used-with"
---
TensorFlow’s `tf.data.Dataset.map` function, while powerful for element-wise transformations, presents a challenge when incorporating conditional logic because it operates within a graph execution context. Direct Pythonic `if/else` statements, as typically used outside of graph contexts, are not directly compatible. Instead, we leverage TensorFlow's control flow operations, notably `tf.cond` or `tf.case`, to achieve conditional mapping. My experience building custom data pipelines for image recognition tasks often required conditional preprocessing steps based on label information, highlighting the practical necessity for mastering this technique.

The core issue stems from the nature of TensorFlow’s graph execution. When using `tf.data`, transformations specified within the `map` function are compiled into a TensorFlow graph representing the data pipeline. Standard Python control flow statements are not directly incorporated into this graph. They would be evaluated eagerly during graph construction, which isn't what we desire. Instead, `tf.cond` and `tf.case` create control flow nodes within the graph itself, allowing the execution to branch based on tensor values. `tf.cond` is suitable for a single `if/else` condition, while `tf.case` offers a more general framework for multiple conditional branches.

Let's illustrate this with a few scenarios. Imagine we're working with a dataset of numerical data where we need to apply different scaling factors based on whether the number is positive or negative.

**Code Example 1: Conditional Scaling using `tf.cond`**

```python
import tensorflow as tf

def conditional_scaling(number):
  """Applies different scaling factors based on the sign of the number."""
  def true_fn():
    return number * 2.0  # Scale positive numbers by 2
  def false_fn():
    return number * 0.5  # Scale negative numbers by 0.5
  return tf.cond(tf.greater_equal(number, 0), true_fn, false_fn)


# Creating a sample dataset
dataset = tf.data.Dataset.from_tensor_slices([-2, -1, 0, 1, 2, 3])
mapped_dataset = dataset.map(conditional_scaling)

# Example usage to view the scaled values
for scaled_number in mapped_dataset:
    print(scaled_number.numpy())
```

In this example, the `conditional_scaling` function accepts a tensor `number`. Inside, we define two functions, `true_fn` and `false_fn`, that encapsulate the respective scaling operations. The `tf.cond` operation takes a predicate, `tf.greater_equal(number, 0)`, which evaluates to a boolean tensor. Based on this predicate, either `true_fn` or `false_fn` is selected and its returned value passed as output. This approach avoids eager Python evaluation and correctly places the conditional execution within the TensorFlow graph.

This pattern is more powerful than it initially appears. For instance, one might use label information contained within a dataset to enable augmentations or other transformations. As part of my past projects, it was common to see the data augmentation strategy for an image change depending on the label, such as increased rotation for a subset of categories.

Now, let’s consider a more complex scenario, one involving multiple conditions based on the label itself. Suppose we have a dataset consisting of (image, label) pairs where labels are numerical class identifiers, and we intend to apply different preprocessing to each class. Here `tf.case` comes into play.

**Code Example 2: Multiple Conditional Transformations with `tf.case`**

```python
import tensorflow as tf

def conditional_preprocessing(image, label):
  """Applies different preprocessing operations based on the label."""

  def process_class0():
    return tf.image.rgb_to_grayscale(image)
  def process_class1():
    return tf.image.flip_left_right(image)
  def process_class2():
    return tf.image.adjust_brightness(image, delta=0.2)
  def default_process():
    return image

  # Specify the cases: label 0, 1, 2; otherwise, use default_process
  case_fns = [
      (tf.equal(label, 0), process_class0),
      (tf.equal(label, 1), process_class1),
      (tf.equal(label, 2), process_class2)
  ]
  return tf.case(case_fns, default_process, exclusive=True)

# Creating a sample dataset
dummy_images = tf.random.normal(shape=[5, 28, 28, 3])
dummy_labels = tf.constant([0, 1, 2, 1, 0])
dataset = tf.data.Dataset.from_tensor_slices((dummy_images, dummy_labels))
mapped_dataset = dataset.map(conditional_preprocessing)

#Example usage to view tensor shape changes
for processed_image in mapped_dataset:
    print(processed_image.shape)
```

In this code, `conditional_preprocessing` takes an image tensor and its corresponding label. A list of tuples, `case_fns`, defines each condition, where each tuple contains a boolean tensor and corresponding transformation function. The `tf.case` operation, given `case_fns` and a default function, will evaluate predicates within `case_fns` sequentially. If a predicate evaluates to `True`, the corresponding function is executed and that output is returned. The `exclusive=True` argument guarantees that only the first predicate which is `True` will cause a branch.  Otherwise, the default function `default_process` executes. The dataset is mapped using this conditional function, modifying the images based on class-specific preprocessing. I've found such an approach vital in managing datasets with different types of data needing various preprocessing routes.

Another aspect, often encountered in practical machine learning, relates to conditionally filtering out certain samples from a dataset based on a complex condition related to a sample's feature values.

**Code Example 3: Filtering data based on sample features using `tf.cond`**

```python
import tensorflow as tf

def conditional_filtering(data_point):
  """Filters data based on conditional logic applied to its features"""
  features = data_point['features']
  label = data_point['label']

  def true_fn():
    return True #Keep this sample
  def false_fn():
    return False #Discard this sample

  #Example: Keep samples where feature at index 0 is greater than 0 and label is 1.
  predicate = tf.logical_and(tf.greater(features[0], 0), tf.equal(label, 1))

  return tf.cond(predicate, true_fn, false_fn)


#Create Sample Data
dummy_features = tf.random.normal(shape=[10, 3])
dummy_labels = tf.constant([0, 1, 2, 1, 0, 1, 2, 1, 0, 1])
dummy_dataset = tf.data.Dataset.from_tensor_slices(
    ({'features': dummy_features, 'label': dummy_labels})
)

filtered_dataset = dummy_dataset.filter(conditional_filtering)


#Example Usage
for sample in filtered_dataset:
    print(sample)
```

Here, the `conditional_filtering` function takes a data_point, which is a dictionary, and extracts its feature vector and label.  The goal is to conditionally keep or discard samples based on an applied rule. `tf.logical_and` demonstrates complex rule combination. This shows how `tf.cond` is integrated with dataset `filter`. This technique is vital when a dataset contains unusable samples or when it needs to be limited to certain values. This is useful for complex cases which can't be easily handled by a simple lambda expression passed to the `filter` method.

To deepen your understanding, refer to the official TensorFlow documentation on `tf.cond`, `tf.case`, and data input pipelines using `tf.data`. Books on machine learning with TensorFlow will also offer valuable insights into designing custom pipelines. Seek examples and tutorials focusing on practical applications within image recognition, natural language processing, or time series analysis, areas that often require robust, conditional data manipulation.
