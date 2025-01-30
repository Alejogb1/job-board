---
title: "How can Momentum variables be excluded from slim.assign_from_checkpoint?"
date: "2025-01-30"
id: "how-can-momentum-variables-be-excluded-from-slimassignfromcheckpoint"
---
TensorFlow’s `slim.assign_from_checkpoint` function, while powerful for initializing model parameters from pre-trained checkpoints, presents a challenge when you need to exclude specific variables, most commonly momentum accumulators, during the assignment process. These accumulators, integral to optimization algorithms like Adam and Momentum, are typically not intended to be initialized from a pre-trained model's checkpoint, as they are tied to the training process. Trying to do so can lead to erratic training behavior. Addressing this requires careful manipulation of the variable scopes and names passed to `slim.assign_from_checkpoint`, and an understanding of how TensorFlow manages variable collections.

I've encountered this specific issue multiple times during my work on large-scale image classification models. In one instance, I was fine-tuning a pre-trained ResNet-50 on a novel dataset and inadvertently included momentum variables, resulting in training that diverged quickly. I had to thoroughly review my code to understand what was happening and implement the correct filtering. The crucial understanding is that `slim.assign_from_checkpoint` works by mapping variable names in the checkpoint to those in the current computational graph. To exclude variables, you need to provide it with either a subset of variable names or variable scopes, effectively filtering out the unwanted ones.

The key lies in how TensorFlow organizes variables within scopes and the naming conventions used for momentum variables. For example, if you're using the Adam optimizer, momentum variables are frequently named with the scope of the target variable followed by the `m` or `v` suffix. For a convolution layer weight named `conv1/weights`, the corresponding momentum variables often are named `conv1/weights/Adam/m` and `conv1/weights/Adam/v`. When using optimizers with momentum built-in to their `apply_gradients`, variables for the momentum terms are frequently defined in the same scope as the variables they accumulate gradients with and suffixed with a similar pattern (`Momentum` rather than `Adam`). To exclude these, you must selectively load weights from the checkpoint based on these name and scope characteristics, which is achievable through judicious use of scopes and exclusion logic.

Here are a few common methods and examples I've found effective:

**Example 1: Excluding variables using a variable filter function:**

This approach dynamically filters variable names as they are being loaded from the checkpoint. You define a lambda function or a regular function that takes a variable name as input and returns `True` if the variable should be loaded and `False` otherwise. This provides fine-grained control over which variables are included.

```python
import tensorflow as tf
import tf_slim as slim

def create_model():
  with tf.variable_scope('model'):
    weights = tf.get_variable("weights", shape=[10, 10], initializer=tf.random_normal_initializer())
    bias = tf.get_variable("bias", shape=[10], initializer=tf.zeros_initializer())
    tf.get_variable("momentum_m", shape=[10, 10], trainable=False)
    tf.get_variable("momentum_v", shape=[10, 10], trainable=False)
    return weights, bias

weights, bias = create_model()

def variable_filter(variable_name):
    return not (variable_name.endswith('momentum_m') or variable_name.endswith('momentum_v'))

checkpoint_path = 'path/to/your/checkpoint.ckpt'
variables_to_restore = tf.contrib.framework.filter_variables(tf.global_variables(), include_patterns=None, exclude_patterns=None, filter_fn=variable_filter)

init_assign_op, init_feed_dict = slim.assign_from_checkpoint(checkpoint_path, variables_to_restore)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(init_assign_op, init_feed_dict)
  print(sess.run(weights))
  print(sess.run(bias))
```

In this example, I create a dummy model with `weights`, `bias`, and momentum variables. The `variable_filter` function checks if a given variable name ends with `momentum_m` or `momentum_v` and excludes those variables. The `filter_variables` utility function applies this filter, effectively passing only the weights and bias to the checkpoint loader, ensuring that momentum variables remain initialized from scratch. This approach offers a high level of control and is beneficial for complex name structures.

**Example 2: Excluding scopes with a manual variable list:**

Instead of filtering by name, you can explicitly define the variables you want to include, using their scope or variable object. This requires you to have a good understanding of the model's scope hierarchy. I've found this approach most effective when working with predefined model architectures where I can confidently predict the naming of relevant variables.

```python
import tensorflow as tf
import tf_slim as slim

def create_model():
  with tf.variable_scope('conv1'):
      weights_conv1 = tf.get_variable("weights", shape=[3, 3, 3, 16], initializer=tf.random_normal_initializer())
      bias_conv1 = tf.get_variable("bias", shape=[16], initializer=tf.zeros_initializer())
      tf.get_variable("momentum_m", shape=[3, 3, 3, 16], trainable=False)
      tf.get_variable("momentum_v", shape=[3, 3, 3, 16], trainable=False)
  with tf.variable_scope('fc'):
      weights_fc = tf.get_variable("weights", shape=[100, 10], initializer=tf.random_normal_initializer())
      bias_fc = tf.get_variable("bias", shape=[10], initializer=tf.zeros_initializer())
      tf.get_variable("momentum_m", shape=[100, 10], trainable=False)
      tf.get_variable("momentum_v", shape=[100, 10], trainable=False)
  return weights_conv1, bias_conv1, weights_fc, bias_fc

weights_conv1, bias_conv1, weights_fc, bias_fc = create_model()

checkpoint_path = 'path/to/your/checkpoint.ckpt'
variables_to_restore = [var for var in tf.global_variables() if "momentum" not in var.name]


init_assign_op, init_feed_dict = slim.assign_from_checkpoint(checkpoint_path, variables_to_restore)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(init_assign_op, init_feed_dict)
  print(sess.run(weights_conv1))
  print(sess.run(bias_conv1))
  print(sess.run(weights_fc))
  print(sess.run(bias_fc))
```

In this variant, I build a model with convolution and fully connected layers, each containing weights, biases, and momentum variables defined in their respective scopes. Before initializing from the checkpoint, I build a list of variables to be initialized using list comprehension; filtering any variable with `momentum` in its name by filtering variable objects. While this requires more direct knowledge about model structure, it offers better precision compared to blanket scope exclusions, particularly where scopes may not follow naming conventions.

**Example 3: Using regular expressions within a filter function:**

For situations with variable naming conventions that are more intricate, regular expressions provide a robust way to filter out specific patterns within the variable names.  This is particularly useful when dealing with optimizers whose naming conventions vary by version. I used regular expressions extensively when porting training code from older TF versions, especially when dealing with legacy optimizers.

```python
import tensorflow as tf
import tf_slim as slim
import re

def create_model():
  with tf.variable_scope('block1'):
    weights_block1 = tf.get_variable("weights", shape=[5, 5, 1, 32], initializer=tf.random_normal_initializer())
    bias_block1 = tf.get_variable("bias", shape=[32], initializer=tf.zeros_initializer())
    tf.get_variable("block1/weights/Momentum", shape=[5, 5, 1, 32], trainable=False) #old TF style momentum
    tf.get_variable("block1/weights/Momentum_1", shape=[5, 5, 1, 32], trainable=False) #old TF style momentum

  with tf.variable_scope('block2'):
    weights_block2 = tf.get_variable("weights", shape=[5, 5, 32, 64], initializer=tf.random_normal_initializer())
    bias_block2 = tf.get_variable("bias", shape=[64], initializer=tf.zeros_initializer())
    tf.get_variable("block2/weights/Adam/m", shape=[5, 5, 32, 64], trainable=False) # new style momentum
    tf.get_variable("block2/weights/Adam/v", shape=[5, 5, 32, 64], trainable=False)

  return weights_block1, bias_block1, weights_block2, bias_block2

weights_block1, bias_block1, weights_block2, bias_block2 = create_model()

checkpoint_path = 'path/to/your/checkpoint.ckpt'

def regex_filter(variable_name):
    pattern = re.compile(r'.*Momentum(_\d*)?$|.*Adam/(m|v)$') #matches both old and new styles.
    return not bool(pattern.match(variable_name))

variables_to_restore = tf.contrib.framework.filter_variables(tf.global_variables(), include_patterns=None, exclude_patterns=None, filter_fn=regex_filter)

init_assign_op, init_feed_dict = slim.assign_from_checkpoint(checkpoint_path, variables_to_restore)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(init_assign_op, init_feed_dict)
  print(sess.run(weights_block1))
  print(sess.run(bias_block1))
  print(sess.run(weights_block2))
  print(sess.run(bias_block2))
```

In this example, both older `Momentum` and newer `Adam` style momentum variable names are excluded.  The regex pattern `.*Momentum(_\d*)?$|.*Adam/(m|v)$` checks for names ending in "Momentum", "Momentum" followed by an optional underscore and digits, or names containing "Adam" followed by `m` or `v`. This pattern can be modified further to adapt to any naming scheme, ensuring consistent and reliable variable exclusion when dealing with complex or inconsistent naming conventions.

**Resource Recommendations:**

For a deeper understanding of TensorFlow variables, I recommend reviewing the official TensorFlow documentation, focusing on variable scopes and collection management. The TF-Slim documentation, specifically the `assign_from_checkpoint` function and related utilities in `tf.contrib.framework`, is also invaluable. I have found that a thorough study of the source code of `slim.assign_from_checkpoint` often provides the most accurate understanding of the function's underlying mechanics. Finally, experimentation with these approaches in a contained development environment provides valuable practical experience, aiding in the ability to adapt these methods to various model architectures.
