---
title: "How can I convert a .model file to TensorFlow Lite without the error 'str' object has no attribute 'call''?"
date: "2025-01-30"
id: "how-can-i-convert-a-model-file-to"
---
The "str' object has no attribute 'call'" error during TensorFlow Lite conversion typically arises when the TensorFlow model being converted is not a fully defined Keras model object, but rather a serialized path (string) to a model file. This can happen when a saved model or custom model structure is inadvertently treated as the model itself. I’ve encountered this numerous times while optimizing models for embedded deployment, and it stems from a misunderstanding of the expected input types for the TensorFlow Lite converter. The crucial distinction is that `tf.lite.TFLiteConverter.from_keras_model()` expects a *model object*, not a file path.

The core problem lies in how the TensorFlow Lite converter interprets its input. When a string representing a file path is provided, the converter assumes it’s a model definition and subsequently attempts to invoke the `call` method on this string, which of course does not exist. This contrasts with a legitimate Keras model object, which possesses the `call` method for performing inference. Therefore, the remedy is to load the model properly using Keras' loading functions before passing it to the converter.

My initial experience involved a pre-trained object detection model saved as a `.pb` file. I had naively provided the file path to `tf.lite.TFLiteConverter.from_keras_model()`, encountering this error repeatedly. Through trial and error, I realized the necessity of using the correct loader before the conversion stage. The error is not about the model's internal architecture, but how TensorFlow itself handles model loading.

Here are a few code examples, illustrating both the incorrect approach leading to the error and the correct solutions:

**Example 1: Incorrect Usage (Leads to the error)**

```python
import tensorflow as tf

# Assume 'my_model.pb' is a saved model file (e.g. SavedModel format).
model_path = "my_model.pb"

# Incorrectly attempting to convert directly from path.
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model_path)
    tflite_model = converter.convert()
except Exception as e:
    print(f"Error encountered: {e}")
```

**Commentary:**
This code snippet directly passes the `model_path` string to `from_keras_model()`. Since the input is a string, the converter mistakenly attempts to call the method on the file path. This causes the `AttributeError: 'str' object has no attribute 'call'` exception, which is printed to the console. This approach is fundamentally flawed.

**Example 2: Correct Usage for SavedModel Format (.pb files)**

```python
import tensorflow as tf

# Assume 'my_model.pb' is a SavedModel format model.
model_path = "my_model.pb"

# Load the model from the SavedModel format.
model = tf.saved_model.load(model_path)

# Correctly convert the model object.
try:
  converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
  tflite_model = converter.convert()
  with open("model.tflite", "wb") as f:
    f.write(tflite_model)
  print("Conversion to TFLite completed.")
except Exception as e:
    print(f"Error encountered: {e}")
```

**Commentary:**
Here, the crucial change is using `tf.saved_model.load(model_path)` to first load the model.  `tf.saved_model.load()` returns a TensorFlow model object. The `tf.lite.TFLiteConverter.from_saved_model()` function then accepts the model's file path as it implicitly knows how to load the model and use the resulting model object internally. Note that in the previous example, `from_keras_model()` was used. That approach is best suited for models defined using the Keras API or already loaded into memory as a Keras object. This correction resolves the error and properly converts the model to TensorFlow Lite. The converted model is then saved as "model.tflite".

**Example 3: Correct Usage for H5 Format (.h5 files)**

```python
import tensorflow as tf

# Assume 'my_model.h5' is a Keras model saved in H5 format.
model_path = "my_model.h5"

# Load the model from the H5 file.
model = tf.keras.models.load_model(model_path)

# Convert the Keras model object.
try:
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()
  with open("model.tflite", "wb") as f:
    f.write(tflite_model)
  print("Conversion to TFLite completed.")
except Exception as e:
    print(f"Error encountered: {e}")
```

**Commentary:**
This example demonstrates the procedure for converting a model saved as an H5 file. The key step is employing `tf.keras.models.load_model(model_path)` to obtain a Keras model object. This object is then correctly passed to `tf.lite.TFLiteConverter.from_keras_model()`, allowing the conversion to proceed without errors. This method is appropriate if the model was created directly in Keras using sequential or functional APIs. After successful conversion, the TFLite model is saved.

In all cases, the solution involves loading the model using the appropriate TensorFlow function before initiating the conversion process. Each example provides the necessary context for different scenarios. Understanding the distinction between a file path and a loaded model object is fundamental for resolving the “str’ object has no attribute ‘call’” error.

For further understanding and detailed guides, I recommend consulting the TensorFlow documentation on model saving and loading, specifically sections covering SavedModel and H5 formats. Additionally, explore the TensorFlow Lite converter documentation for specifics on conversion. Information on preparing models for edge devices is also beneficial, found within materials for mobile or embedded TensorFlow. For a broader understanding, the TensorFlow guide on working with Keras is recommended, especially in terms of loading previously saved models.
