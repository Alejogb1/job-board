---
title: "How can I resolve 'ValueError: Shapes (None, 2) and (None, 1) are incompatible' when compiling a model with multiple metrics?"
date: "2025-01-30"
id: "how-can-i-resolve-valueerror-shapes-none-2"
---
The `ValueError: Shapes (None, 2) and (None, 1) are incompatible` error during model compilation with multiple metrics typically arises from a mismatch in the output shapes of your model and the expected input shapes of your metrics.  This often occurs when a metric expects a single-value output (shape `(None, 1)`) while your model produces multiple outputs (shape `(None, 2)`), perhaps due to multiple loss functions or separate prediction heads.  I've encountered this numerous times during my work on large-scale NLP tasks and time-series forecasting projects,  requiring careful attention to both model architecture and metric definition.

The core issue lies in understanding how Keras (or TensorFlow/PyTorch) handles multiple outputs and metric calculations. Each metric needs a corresponding output from your model to operate on. If there's a dimensionality mismatch—the metric's input shape doesn't align with the model's output shape for that specific metric—this error surfaces.  The `(None, 2)` signifies a batch size of `None` (dynamic) and two output features.  The `(None, 1)` indicates a single output feature per sample in the batch.

**1. Clear Explanation**

The solution requires either aligning your model's output with the metrics' expected input shapes, or selecting/adapting metrics suitable for your multi-output model.  This involves a careful review of your model's architecture and the specific metrics you are employing.  For instance, if you're using binary cross-entropy for two classification outputs, you'll need to ensure that your model provides two separate outputs—one for each classification task—and then specify these outputs explicitly when compiling the model with `metrics`.  Failing to do so leads to the shape mismatch error.  Incorrectly applying a metric designed for a single output (like Mean Absolute Error) to a multi-output model (e.g., predicting both price and volume) is a common cause.


**2. Code Examples with Commentary**

**Example 1: Correct Handling of Multiple Metrics with Multiple Outputs**

```python
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import MeanSquaredError, BinaryAccuracy

# Define a multi-output model
input_layer = tf.keras.Input(shape=(10,))
dense1 = Dense(64, activation='relu')(input_layer)
output1 = Dense(1, name='price_output')(dense1) #Single output for price
output2 = Dense(1, activation='sigmoid', name='binary_output')(dense1) #Single output for binary classification

model = Model(inputs=input_layer, outputs=[output1, output2])

# Compile the model with appropriate metrics for each output
model.compile(optimizer='adam',
              loss={'price_output': 'mse', 'binary_output': 'binary_crossentropy'},
              metrics={'price_output': MeanSquaredError(), 'binary_output': BinaryAccuracy()})

#Example training data, replace with your actual data
x_train = tf.random.normal((100,10))
y_train_price = tf.random.normal((100,1))
y_train_binary = tf.random.uniform((100,1), minval=0, maxval=2, dtype=tf.int32)
y_train = [y_train_price, y_train_binary]

model.fit(x_train, y_train, epochs=10)
```

This example demonstrates the correct way to handle multiple metrics with a model possessing two distinct output layers. Each output has its own loss function and metric.  Note the specific naming of outputs and the corresponding metrics. This ensures that the model outputs are correctly mapped to the designated metrics.  If you only provided `metrics = [MeanSquaredError(), BinaryAccuracy()]`, the error would have occurred because the metric functions wouldn't know which output to operate on.


**Example 2: Adapting a Single-Output Metric for Multi-Output**

```python
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import MeanAbsoluteError

#Multi-output model (simplified for brevity)
input_layer = tf.keras.Input(shape=(10,))
dense1 = Dense(64, activation='relu')(input_layer)
output = Dense(2)(dense1) #Two outputs from this layer

model = Model(inputs=input_layer, outputs=output)

#Using Mean Absolute Error (MAE) which expects a single output.
# we need to calculate MAE separately for each output.

def custom_mae(y_true, y_pred):
    mae1 = MeanAbsoluteError()(y_true[:, 0:1], y_pred[:, 0:1])  #MAE for the first output
    mae2 = MeanAbsoluteError()(y_true[:, 1:2], y_pred[:, 1:2])  #MAE for the second output
    return (mae1 + mae2) / 2  #Average MAE across both outputs

model.compile(optimizer='adam', loss='mse', metrics=[custom_mae])

# Example training data. Adapt this to your specific needs.
x_train = tf.random.normal((100, 10))
y_train = tf.random.normal((100, 2))

model.fit(x_train, y_train, epochs=10)

```

Here, I address the problem by creating a custom metric function.  This custom metric explicitly calculates the Mean Absolute Error for each of the two outputs individually and then averages them. This approach works around the incompatibility by preprocessing the outputs to meet the metric's expectations, instead of changing the model.


**Example 3:  Restructuring the Model for Single Output Metric Compatibility**

```python
import tensorflow as tf
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import MeanSquaredError

#Model producing a single combined output
input_layer = tf.keras.Input(shape=(10,))
dense1 = Dense(64, activation='relu')(input_layer)
output1 = Dense(1, name='output1')(dense1)
output2 = Dense(1, name='output2')(dense1)
combined_output = Concatenate()([output1, output2]) # Combine outputs

model = Model(inputs=input_layer, outputs=combined_output)

model.compile(optimizer='adam', loss='mse', metrics=[MeanSquaredError()])

# Example training data. Adapt this to your specific needs.
x_train = tf.random.normal((100, 10))
y_train = tf.random.normal((100, 2))

model.fit(x_train, y_train, epochs=10)

```
In this final example, I demonstrate how restructuring the model can eliminate the error.  Instead of two distinct outputs, the model now produces a single combined output by concatenating the individual predictions.  This allows the use of a single metric (Mean Squared Error) which now matches the model's output shape.  However, remember this might not always be the desirable approach, especially if the two outputs represent distinct concepts that should be evaluated separately.


**3. Resource Recommendations**

The official TensorFlow and Keras documentation provide comprehensive guides on model compilation, multi-output models, and custom metrics.  Thoroughly studying these documents is crucial.  Additionally, exploring advanced concepts like custom training loops in Keras can offer deeper control over the training process and metric calculation, resolving complex shape incompatibilities.  Finally, reviewing introductory and advanced materials on deep learning architecture will help in building models whose structure aligns perfectly with the intended metrics.
