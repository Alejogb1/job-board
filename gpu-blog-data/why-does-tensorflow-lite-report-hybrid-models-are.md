---
title: "Why does TensorFlow Lite report 'Hybrid models are not supported' after quantization?"
date: "2025-01-30"
id: "why-does-tensorflow-lite-report-hybrid-models-are"
---
TensorFlow Lite’s “Hybrid models are not supported” error following quantization, specifically post-training quantization (PTQ), arises primarily because quantization introduces operations that are not fully supported across all target hardware. My experience deploying models on embedded systems has repeatedly highlighted this issue.

Quantization transforms floating-point operations (FP32) to lower-precision representations, typically int8. This process reduces model size and potentially accelerates inference on devices with specialized integer-based computation units. However, this transformation often introduces operations that are conceptually "hybrid," mixing both floating-point and integer calculations within a single model execution graph. These hybrids result from the quantization process and can manifest due to several key factors.

The TensorFlow Lite converter, when applying PTQ, inserts "fake quantization" nodes during the conversion. These nodes, while representing the quantization effect during training, don't actually perform quantization. Instead, they simulate the quantization process. This is done for simulating and training with quantized values. However, during conversion to a `.tflite` file these fake quantization nodes can be used or replaced by real quantization operations.

The converted `.tflite` model will contain either `dequantize`, `quantize`, and integer operations. In some situations, the conversion can lead to mixed or hybrid operators, such as a `dequantize` operation directly followed by a floating-point operation in the computational graph. The interpreter on the target device receives these hybrid operators which it then needs to perform. Not all interpreter versions and hardware implementations support those mixed operations.

This lack of support is primarily caused by the underlying hardware accelerators or the software libraries that enable inference on the target device. Many hardware accelerators are optimized for pure integer operations. A fully quantized model offers the best opportunity for optimization. When presented with a hybrid model, the acceleration might be bypassed, or worse, the interpreter may fail completely. This inconsistency of support is what leads to the reported error "Hybrid models are not supported".

The problem can become even more prevalent when using specific quantization parameters. If during quantization the conversion process cannot completely eliminate the floating-point operations due to limitations in the precision, or due to lack of optimization for a certain operation, the hybrid model will occur. When this occurs, the interpreter reports the error and refuses to use the model.

Here are three examples that demonstrate this concept:

**Example 1: Simple convolution model**

Consider a basic convolutional neural network model for image classification. The original model uses standard floating-point operations:

```python
import tensorflow as tf

# Define the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(10, activation='softmax')
])

# Dummy input data
img = tf.random.normal(shape=(1, 28, 28, 1))

# Run inference to ensure it works
result = model(img)

# Save the model
tf.saved_model.save(model, "my_model")
```

The above code creates a simple keras model, saves it as a saved model in the local directory, and then shows it performs inference.

Now we convert this to a `.tflite` model using post-training quantization:

```python
converter = tf.lite.TFLiteConverter.from_saved_model("my_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()
with open("converted_model.tflite", "wb") as f:
    f.write(tflite_model)
```

The above code loads the saved model, sets the optimization strategy, converts the model into a `tflite` model, and then saves the file. In many cases, this conversion will work as expected. However, if the conversion generates a hybrid model due to the layers not supporting pure integer execution, then when trying to perform inference on the tflite interpreter, the “Hybrid models are not supported” error can occur.

**Example 2: A model with unsupported operations**

Assume a scenario where the model incorporates a custom operation not directly mapped to an integer implementation. I encountered this once when using a specific activation function not natively supported.

```python
import tensorflow as tf
from tensorflow.keras import layers

class CustomActivation(layers.Layer):
    def __init__(self, **kwargs):
        super(CustomActivation, self).__init__(**kwargs)

    def call(self, inputs):
        #Some complex floating point operation
        return tf.math.sin(inputs * 0.5)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, input_shape=(10,)),
    CustomActivation(),
    tf.keras.layers.Dense(1)
])

# Dummy input data
input_data = tf.random.normal(shape=(1, 10))

# Run inference to ensure it works
result = model(input_data)

tf.saved_model.save(model, "my_model_custom")
```

The above code defines a custom layer that computes a non-linear operation. This model then uses it and performs inference to ensure it works. This custom operation can cause the converter to generate a hybrid model. The conversion is similar to the first example:

```python
converter = tf.lite.TFLiteConverter.from_saved_model("my_model_custom")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open("custom_model.tflite", "wb") as f:
    f.write(tflite_model)
```

The custom `CustomActivation` layer, during quantization, might not be fully converted to an integer-based version. Consequently, the resulting `.tflite` model likely contains floating-point operations, alongside quantized integer operations. During inference, if the interpreter doesn't support these mixed operations, it will report "Hybrid models are not supported."

**Example 3: Quantization with incomplete operator support**

A final example I encountered was related to certain pooling operations. Consider the following model that makes use of `average pooling`.

```python
import tensorflow as tf

# Define the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.AveragePooling2D((2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(10, activation='softmax')
])

# Dummy input data
img = tf.random.normal(shape=(1, 28, 28, 1))

# Run inference to ensure it works
result = model(img)

# Save the model
tf.saved_model.save(model, "my_model_avg")
```

This above model again saves a simple model to the local directory. Now, using the same conversion logic as before:

```python
converter = tf.lite.TFLiteConverter.from_saved_model("my_model_avg")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open("avg_model.tflite", "wb") as f:
    f.write(tflite_model)
```

The conversion might result in a hybrid operation involving an `average_pooling` node which might not have native integer implementations. While TensorFlow Lite is continuously extending the supported operator list, some edge cases can result in these hybrid structures. If the device lacks optimized kernels for quantized average pooling, the resulting model will generate the "Hybrid models are not supported" error.

To mitigate this, I recommend a few key strategies. First, carefully examine the model’s architecture to minimize reliance on operations without integer-based counterparts. Pre-quantization analysis can help identify these operations and enable targeted modification to fully support integer operations.

Secondly, always keep your TensorFlow Lite libraries, both on the conversion side and the interpreter side, up to date. New versions regularly add support for more operations and fix bugs that can lead to hybrid models.

Lastly, during conversion ensure that the quantization process is using either full integer or float fallback models. The `converter.target_spec.supported_ops` parameter can be modified to either explicitly allow float fallback or disallow it. This ensures the models are of consistent types.

Debugging hybrid model issues requires iterative experimentation, using the converter's diagnostics to identify the root cause. This can sometimes require a deeper analysis using a visualization tool for the model graph. There are also methods such as selective quantization to remove float operations in sub-graphs which can assist in the process.

The "Hybrid models are not supported" error is an indicator of operational incompatibility, requiring careful consideration of model architecture, quantization settings, and target device constraints. Careful planning and debugging will usually resolve the issue.
