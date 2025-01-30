---
title: "How do I convert a Keras 'merge' layer to the new equivalent?"
date: "2025-01-30"
id: "how-do-i-convert-a-keras-merge-layer"
---
The deprecation of the `merge` layer in Keras, specifically within versions 2.x and prior to its complete removal in 3.x, necessitates a refactoring of models relying on it. I've encountered this challenge repeatedly in my work, particularly when transitioning legacy projects to more contemporary Keras implementations. The core issue stems from `merge`'s overly flexible, and therefore potentially confusing, nature, which the Keras team addressed through a more explicit and modular approach. The newer, equivalent methods rely primarily on the `tf.keras.layers.concatenate`, `tf.keras.layers.add`, `tf.keras.layers.multiply`, and custom layers to achieve similar functionality. The key shift is moving from an all-in-one layer to composition through distinct layers.

The `merge` layer offered a single interface, accepting a `mode` argument that controlled the merging behavior, encompassing summation, concatenation, multiplication, and more. This flexibility, while convenient, masked the underlying operations. In contrast, the current approach dictates a more explicit selection of the intended operation, directly reflecting the underlying mathematical or logical function being executed. This reduces ambiguity and often leads to more maintainable code.

The conversion process begins with identifying the `mode` argument used in your `merge` layer. Common `mode` values like 'sum', 'concat', 'mul', and 'ave' (average) each require a distinct translation. For ‘sum,’ we directly employ the `tf.keras.layers.Add` layer. ‘Concat’ utilizes `tf.keras.layers.Concatenate`. ‘Mul’ corresponds to `tf.keras.layers.Multiply`. ‘Ave’ is achievable through the use of `tf.keras.layers.Average`. Other modes, like ‘dot,’ often require custom layer implementation using the backend's tensor operations directly. The crucial point is that each mode has a more defined and specific replacement, and no single layer will substitute all of the `merge` layer's original functionality. This requires analyzing the specific use of `merge` within a given model architecture.

Here are three illustrative code examples, demonstrating the conversion process:

**Example 1: 'sum' mode replacement**

```python
# Old Keras merge layer (Keras 2.x)
from keras.layers import Input, Dense, merge
from keras.models import Model

input_1 = Input(shape=(10,))
input_2 = Input(shape=(10,))

dense_1 = Dense(32)(input_1)
dense_2 = Dense(32)(input_2)

# Old sum merge
merged_layer = merge([dense_1, dense_2], mode='sum')

output_layer = Dense(1)(merged_layer)

old_model = Model([input_1, input_2], output_layer)
```

```python
# New Keras replacement (Keras 3.x and later)
import tensorflow as tf

input_1 = tf.keras.Input(shape=(10,))
input_2 = tf.keras.Input(shape=(10,))

dense_1 = tf.keras.layers.Dense(32)(input_1)
dense_2 = tf.keras.layers.Dense(32)(input_2)

# New sum layer
merged_layer = tf.keras.layers.Add()([dense_1, dense_2])

output_layer = tf.keras.layers.Dense(1)(merged_layer)

new_model = tf.keras.Model([input_1, input_2], output_layer)
```

*Commentary:* In this instance, the `merge(..., mode='sum')` is replaced directly by `tf.keras.layers.Add()`. The inputs to `Add` are provided as a list during the call, which is consistent with how the older `merge` layer accepted inputs. No additional modifications are required, and the model logic remains the same, only with more explicit addition happening. The underlying tensor operation is now readily visible in the model definition.

**Example 2: 'concat' mode replacement**

```python
# Old Keras merge layer (Keras 2.x)
from keras.layers import Input, Dense, merge
from keras.models import Model

input_1 = Input(shape=(10,))
input_2 = Input(shape=(20,))

dense_1 = Dense(32)(input_1)
dense_2 = Dense(32)(input_2)

# Old concat merge
merged_layer = merge([dense_1, dense_2], mode='concat', concat_axis=1)

output_layer = Dense(1)(merged_layer)

old_model = Model([input_1, input_2], output_layer)
```

```python
# New Keras replacement (Keras 3.x and later)
import tensorflow as tf

input_1 = tf.keras.Input(shape=(10,))
input_2 = tf.keras.Input(shape=(20,))

dense_1 = tf.keras.layers.Dense(32)(input_1)
dense_2 = tf.keras.layers.Dense(32)(input_2)

# New concat layer
merged_layer = tf.keras.layers.Concatenate(axis=1)([dense_1, dense_2])

output_layer = tf.keras.layers.Dense(1)(merged_layer)

new_model = tf.keras.Model([input_1, input_2], output_layer)
```

*Commentary:* The core of the conversion remains the same, but in this case, the `tf.keras.layers.Concatenate` layer is used. Notice that `concat_axis` in the original `merge` call is translated directly to the `axis` argument in `Concatenate`. This ensures that the concatenation occurs along the expected dimension. The explicit indication of concatenation replaces the older implicit configuration.

**Example 3: 'dot' mode replacement using a custom layer**

```python
# Old Keras merge layer (Keras 2.x)
from keras.layers import Input, Dense, merge
from keras.models import Model
import keras.backend as K

input_1 = Input(shape=(10,))
input_2 = Input(shape=(10,))

dense_1 = Dense(32)(input_1)
dense_2 = Dense(32)(input_2)

# Old dot merge
merged_layer = merge([dense_1, dense_2], mode='dot', dot_axes=1)

output_layer = Dense(1)(merged_layer)

old_model = Model([input_1, input_2], output_layer)
```

```python
# New Keras replacement (Keras 3.x and later)
import tensorflow as tf

class DotProductLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DotProductLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.keras.backend.batch_dot(inputs[0], inputs[1], axes=1)

input_1 = tf.keras.Input(shape=(10,))
input_2 = tf.keras.Input(shape=(10,))

dense_1 = tf.keras.layers.Dense(32)(input_1)
dense_2 = tf.keras.layers.Dense(32)(input_2)

# New custom dot layer
merged_layer = DotProductLayer()([dense_1, dense_2])

output_layer = tf.keras.layers.Dense(1)(merged_layer)

new_model = tf.keras.Model([input_1, input_2], output_layer)
```

*Commentary:* The 'dot' mode necessitates a custom layer as there isn't a direct replacement. The `DotProductLayer` encapsulates the required `tf.keras.backend.batch_dot` operation, and accepts the list of tensors as inputs. The `axes` argument mirrors the `dot_axes` parameter from the original `merge` layer. This demonstrates the more explicit and granular approach encouraged by the newer Keras API, where specific operations require targeted layer implementation. For common operations, the built-in layers provide sufficient functionality, but more complex cases necessitate creating custom layers.

For more in-depth understanding, consulting the official Keras documentation is highly recommended. The documentation provides clear and comprehensive explanations of each layer and its expected inputs and outputs. Also helpful are code examples directly demonstrating the usage of these layers in various scenarios. I would suggest studying the core Tensorflow guides related to custom layers and the Keras API directly. Finally, exploring source code of commonly used implementations of different complex models, where the conversion from `merge` has been adopted, can be beneficial to understanding these patterns in practical use. This has been my experience converting and adapting models.
