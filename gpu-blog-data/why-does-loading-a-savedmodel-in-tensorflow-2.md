---
title: "Why does loading a SavedModel in TensorFlow 2 result in a 'signature_wrapper takes 0 positional arguments but 1 was given' error?"
date: "2025-01-30"
id: "why-does-loading-a-savedmodel-in-tensorflow-2"
---
The core issue behind the "signature_wrapper takes 0 positional arguments but 1 was given" error when loading a SavedModel in TensorFlow 2 stems from a discrepancy between the function signatures defined during model construction and how those functions are later called during inference after model reloading. Specifically, TensorFlow's SavedModel format preserves the *concrete functions*—specific graph instances with fixed argument structures—alongside the high-level Python function definitions. Problems arise when the input structure at the time of loading and subsequent calls doesn't perfectly match the expected input structure baked into the concrete function.

When a model is saved using `tf.saved_model.save`, TensorFlow traces the Python functions associated with the model’s call graph, such as those within a custom Keras layer or subclassed model. During this tracing, specific graph operations are generated based on the exact positional and keyword arguments provided to the Python functions. These argument configurations, including the types, shapes, and even whether they’re positional or keyword, become part of the concrete function's signature, effectively a 'compiled' version of your Python function tailored to those specific arguments.

When the SavedModel is loaded via `tf.saved_model.load`, TensorFlow reconstructs the saved graph operations and provides a mechanism to call into them. However, the call mechanism directly references the previously created concrete functions. Therefore, if you attempt to call a loaded model’s function with an input structure that doesn't exactly mirror what it expects from its concrete signature, the error occurs. The message "signature_wrapper takes 0 positional arguments" is misleading, implying a function definition problem. What it *actually* means is that the concrete function being invoked doesn't accept a positional argument in the way you're providing it, because it was traced without it during the `save` operation. This issue frequently arises in dynamically shaped tensor situations or where data pipelining has altered the input structure during load/inference compared to what was present during the model save operation.

Here's a breakdown using some hypothetical scenarios and code examples from projects I've personally encountered:

**Example 1: Positional vs. Keyword Argument Mismatch**

Consider a simple custom Keras layer designed for element-wise addition:

```python
import tensorflow as tf

class AddLayer(tf.keras.layers.Layer):
    def call(self, x, y):
        return x + y

add_layer = AddLayer()
input_tensor_x = tf.constant([1.0, 2.0, 3.0])
input_tensor_y = tf.constant([4.0, 5.0, 6.0])

result = add_layer(input_tensor_x, input_tensor_y)  # Positional args in training
tf.saved_model.save(add_layer, 'add_layer_model')

loaded_add_layer = tf.saved_model.load('add_layer_model')
```

In the training stage, I call `add_layer` using positional arguments for `x` and `y`. Now, if I were to try calling the loaded model using keyword arguments, it would generate the mentioned error:

```python
try:
    loaded_result = loaded_add_layer(x=input_tensor_x, y=input_tensor_y) # Error
except Exception as e:
  print(f"Error encountered: {e}")

loaded_result = loaded_add_layer(input_tensor_x, input_tensor_y)  # Correct call
print(loaded_result)
```

*Commentary:* During model saving, the positional arguments were recorded within the concrete function's signature. Consequently, calling the loaded model with keyword arguments fails because the saved concrete function does not have that specific keyword-argument structure. The second correct call, using positional arguments, directly satisfies the saved function's input expectations.

**Example 2: Input Shape Mismatch from Data Preprocessing**

Let's take a more complicated example of a text classification model with a simple preprocessing layer:

```python
import tensorflow as tf
import numpy as np

class PreprocessingLayer(tf.keras.layers.Layer):
    def call(self, text_input):
        # Simple example: convert to lowercase and encode using a static embedding
        text_input = tf.strings.lower(text_input)
        word_ids = tf.strings.split(text_input).to_tensor() # padding
        return word_ids

class ClassificationModel(tf.keras.Model):
  def __init__(self, vocab_size):
    super(ClassificationModel, self).__init__()
    self.preprocessing = PreprocessingLayer()
    self.embedding = tf.keras.layers.Embedding(vocab_size, 64)
    self.dense = tf.keras.layers.Dense(2, activation='softmax')

  def call(self, text_input):
    preprocessed_text = self.preprocessing(text_input)
    embedded_text = self.embedding(preprocessed_text)
    mean_embedding = tf.reduce_mean(embedded_text, axis=1)
    output = self.dense(mean_embedding)
    return output

vocab_size = 1000
model = ClassificationModel(vocab_size)
input_text = tf.constant(["This is a test sentence", "Another one here"])

_ = model(input_text)  # Run once to define graph
tf.saved_model.save(model, 'text_classification_model')
```

Here, the `PreprocessingLayer` performs operations that can lead to different shapes. If we reload this model and attempt to feed different shapes than used when saving, we risk this error:

```python
loaded_model = tf.saved_model.load('text_classification_model')

new_input_text = tf.constant(["One shorter sent"])

try:
   loaded_output = loaded_model(new_input_text)  # Possible error due to string tokenizing changing output
except Exception as e:
   print(f"Error during inference: {e}")

padded_input_text = tf.constant(["This is a test sentence", "Another one here", "       "])
loaded_output = loaded_model(padded_input_text) # Correct shapes will be accepted.
print(loaded_output)

```

*Commentary:* The preprocessing layer, when used during the initial model invocation prior to saving, produces padded tensors based on the max length in the batch provided. During loading and subsequent inference with a different input batch containing a shorter string, the pre-processing results in a different shape than the original, causing the error because the embedding layer and downstream operations are expecting the original padded shape based on save. Padding the test set will result in a consistent shape for model loading.

**Example 3: Inconsistent Batching or Data Formats**

In scenarios involving batched input data, inconsistencies during saving and loading are a potential pitfall.

```python
import tensorflow as tf
import numpy as np

class SimpleDense(tf.keras.Model):
    def __init__(self, num_units):
      super(SimpleDense, self).__init__()
      self.dense = tf.keras.layers.Dense(num_units)
    def call(self, inputs):
      return self.dense(inputs)

model = SimpleDense(10)
example_batch_input = tf.random.normal((32, 5))
example_output = model(example_batch_input)
tf.saved_model.save(model, 'dense_model')

loaded_model = tf.saved_model.load('dense_model')

new_input_single = tf.random.normal((1, 5))
try:
    loaded_output = loaded_model(new_input_single) # Error, requires batch size
except Exception as e:
    print(f"Error: {e}")

new_input_batch = tf.random.normal((10, 5))
loaded_output = loaded_model(new_input_batch) # Correct call
print(loaded_output)
```

*Commentary:* When the model was saved, it was traced based on the shape of `example_batch_input`, which has a batch dimension of 32. Trying to provide a single sample during inference, having only a single sample in the batch dimension, causes an error because of the concrete function having expectations for batch size. The second inference, providing a batch, satisfies the concrete function.

**Recommendations**

To mitigate the "signature_wrapper" error:

1.  **Consistent Input Structure:** Ensure the input structure during loading and inference exactly matches the structure used during model construction and saving. Pay meticulous attention to both the number of positional/keyword arguments, data type, and shape.
2.  **Data Preprocessing within the Model:** Encapsulate any preprocessing steps within the model itself (as demonstrated in Example 2) to guarantee consistency. If preprocessing is kept outside, ensure all data is processed in the same way.
3.  **Explicit Signature Definition (Advanced):** While typically unnecessary, explore `tf.function`'s signature specification if you need fine-grained control over the saved concrete function's argument types. This can be advantageous in specific scenarios requiring complex argument structures.
4.  **Thorough Testing:** Rigorously test your model loading and inference pipeline with a variety of input shapes, sizes and data types that you might encounter in your production environment.

Understanding that the “signature\_wrapper” error isn’t about the Python definition but the generated concrete function, and maintaining identical data pipelines during training and inference are critical steps for working with TensorFlow's SavedModel format effectively.
