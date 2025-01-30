---
title: "How can I predict with a TensorFlow SavedModel using string tensor input locally?"
date: "2025-01-30"
id: "how-can-i-predict-with-a-tensorflow-savedmodel"
---
A crucial aspect often overlooked when deploying TensorFlow models involves handling string inputs, particularly when dealing with pre-processing steps embedded within the SavedModel. I've encountered this frequently in my work developing NLP pipelines for sentiment analysis where text is the primary input. Directly passing raw strings to the inference function of a SavedModel can be problematic if the model expects numerical representations following a specific tokenization or vectorization process. The key lies in ensuring the input signature of the SavedModel accommodates string tensors and in managing any necessary pre-processing.

The problem arises because a SavedModel, by default, expects tensors with specific shapes and data types defined during training. Often, these will be numeric, such as `tf.float32` or `tf.int32`, representing numerical encodings derived from text. If the SavedModel was trained with a pre-processing layer directly within the model architecture, like a TextVectorization layer, the model's input signature will likely expect a tensor of that vectorization outcome, rather than raw strings. Therefore, to predict directly with a string tensor, you must ensure the SavedModel was explicitly created to accept such a type or manually handle the pre-processing before inference.

Here’s a breakdown of how this process works, illustrated through several scenarios, including ways to overcome common issues:

**Scenario 1: SavedModel Accepts String Tensors Directly**

In some cases, you might have trained your model with a preprocessing step outside of the graph and deliberately configured the input signature of the SavedModel to accept raw strings, typically a tensor of `tf.string`. In this situation, prediction is straightforward:

```python
import tensorflow as tf

# Load the SavedModel
model = tf.saved_model.load("path/to/your/saved_model")

# Access the inference function
infer = model.signatures["serving_default"]

# Prepare the input as a string tensor
string_input = tf.constant(["This is a test sentence.", "Another example."])

# Run inference
output = infer(input_1=string_input)

# Process the output (model dependent)
print(output)
```

Here, `path/to/your/saved_model` would be replaced with the actual path to your exported SavedModel directory. The crucial part here is the `input_1` parameter within `infer()`. This assumes the SavedModel’s input signature is configured to expect a tensor with a key named `input_1`.  You can determine the exact input signature by inspecting the SavedModel's metadata using `tf.saved_model.load("path/to/your/saved_model").signatures["serving_default"].structured_input_signature` or using the `saved_model_cli` command-line tool. The output will vary depending on the specific model and its final layers, but generally you should expect a dictionary containing tensors of predicted values. This approach assumes that the model itself has the ability to process raw string inputs. This is less common but achievable if the original model was explicitly designed this way.

**Scenario 2: Pre-processing within the SavedModel**

Often, the text pre-processing steps are part of the model. For instance, a TextVectorization layer. If the SavedModel contains such layers as part of the computational graph, then you are not required to manually preprocess the strings before prediction. Here's how this might look:

```python
import tensorflow as tf

# Load the SavedModel
model = tf.saved_model.load("path/to/your/saved_model")

# Access the inference function
infer = model.signatures["serving_default"]

# Prepare the input as a string tensor
string_input = tf.constant(["This is another test.", "A second sentence."])

# Run inference, passing in the string tensor directly.
output = infer(input_1=string_input)

# Process the output.
print(output)
```

In this case, the model automatically handles the preprocessing of the string inputs, performing all the vectorization logic within its computational graph, which was saved with the model. This avoids the need for a separate pre-processing step before calling the model's predict function.

**Scenario 3: Manual Pre-processing Required**

If the SavedModel expects a numerical tensor resulting from preprocessing, you will need to manually preprocess the string before calling the model. For example, assume that you used `tf.keras.layers.TextVectorization` with a fixed vocabulary during training, and then, for some reason (not ideal for deployment) didn't include the layer in the model's graph. You need to manually perform the same vectorization.

```python
import tensorflow as tf

# Load the SavedModel
model = tf.saved_model.load("path/to/your/saved_model")

# Access the inference function
infer = model.signatures["serving_default"]

# Example string inputs
string_input = ["This is a third test.", "And a final one."]

# Define vocabulary and TextVectorization setup to mimic the training settings
VOCAB = ["this", "is", "a", "test", "and", "final", "one", "third"]
vectorizer = tf.keras.layers.TextVectorization(vocabulary=VOCAB, output_mode='int')
vectorizer.adapt(VOCAB) # Needed to make sure the layer is ready

# Manually apply TextVectorization
vectorized_input = vectorizer(tf.constant(string_input))

# Run Inference
output = infer(input_1=vectorized_input)

# Process output.
print(output)
```

Here, the code explicitly sets up a TextVectorization layer using the original vocabulary, and then vectorizes the input strings to the model. In practice, extracting this information from the original training setup might be challenging and require meticulous recording of configurations during model creation, or accessing the information if the TextVectorization layer was originally part of the model and you can load and inspect it.

**Resource Recommendations:**

To solidify your understanding of SavedModels and string input handling in TensorFlow, I recommend exploring the following resources:

1.  **Official TensorFlow documentation on SavedModel:** The official documentation provides the most authoritative source on the intricacies of saving and loading models, including how to inspect the input signatures.
2.  **TensorFlow tutorials on text classification:** These tutorials often showcase examples of text preprocessing using layers like TextVectorization and provide insights into how to build models that handle text inputs.
3.  **TensorFlow discussion forums:** There are valuable discussions and FAQs in the TensorFlow discussion forum that address common challenges encountered when working with models and specific input types, especially string. These forums often contain practical suggestions from users who have faced and overcome similar issues.

These resources will provide a solid foundation for understanding how to work with SavedModels, and particularly the subtleties of feeding string tensors and handling pre-processing steps either inside or outside of the saved model. Always confirm your model's specific input structure using the provided tools and techniques to correctly perform inference. The key to avoiding issues is a careful understanding of how your model expects its inputs and a precise reproduction of these expectations during deployment.
