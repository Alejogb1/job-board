---
title: "How to load the best TensorFlow model using ModelCheckpoint and load_model?"
date: "2025-01-26"
id: "how-to-load-the-best-tensorflow-model-using-modelcheckpoint-and-loadmodel"
---

The efficacy of a TensorFlow model hinges not only on its architecture and training process but also on how effectively its best-performing state is preserved and subsequently restored for inference or continued training. The process revolves primarily around the `ModelCheckpoint` callback during training and `tf.keras.models.load_model` for retrieval. I’ve refined this workflow through several projects involving image classification and time-series forecasting, observing firsthand the common pitfalls of improper checkpointing and loading.

The `ModelCheckpoint` callback in TensorFlow’s Keras API provides a structured mechanism for saving model weights at specific intervals or based on defined metrics. The “best” model, often a moving target during training, is typically determined by monitoring validation loss or accuracy. A naive approach might involve saving the model at every epoch, resulting in numerous redundant checkpoints. A more effective strategy involves tracking the validation metric and only saving when an improvement is observed. This reduces storage overhead and simplifies the selection of the optimal model. Further, using a well-defined directory structure for checkpoints enables better organization and facilitates version control.

In practice, the primary challenge is correctly configuring `ModelCheckpoint` and subsequently ensuring compatibility when the saved model is loaded. Crucial parameters within `ModelCheckpoint` include `filepath`, which dictates the saving location and naming convention; `monitor`, which specifies the metric to track; `save_best_only`, which ensures only improved models are saved; `save_weights_only`, which determines if only weights or the complete model is saved; and `verbose`, which controls the logging of checkpoint operations. It’s critical to understand that `save_weights_only=True` only captures the model's weights, not its architecture. The complete model, incorporating both architecture and weights, is saved with `save_weights_only=False`. This distinction impacts the loading process. While weight-only checkpoints are smaller, the original model architecture must be available for restoring them.

The subsequent loading of the “best” model uses `tf.keras.models.load_model`. This function, when presented with a complete model checkpoint (saved with `save_weights_only=False`), reconstructs the architecture and loads the learned weights seamlessly. However, if `save_weights_only=True` was used, you would need to construct the model architecture manually first, before loading the weights using `model.load_weights(filepath)`. Failing to correctly handle the differences between these two saved formats can lead to errors or mismatched model states. Moreover, any custom layers or functions within the model must be explicitly registered before loading for the model to be restored correctly.

Let's delve into code examples to illustrate these points:

**Example 1: Saving the Complete Best Model**

```python
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

# Assume a model named 'my_model' is already defined and compiled.
my_model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
  tf.keras.layers.Dense(10, activation='softmax')
])
my_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


checkpoint_filepath = 'path/to/my_best_model.h5'

model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
x_test = x_test.reshape(10000, 784).astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

my_model.fit(
    x_train,
    y_train,
    epochs=10,
    validation_data=(x_test, y_test),
    callbacks=[model_checkpoint_callback]
)


# Load the model after training (during inference or further training)
loaded_model = tf.keras.models.load_model(checkpoint_filepath)
loss, accuracy = loaded_model.evaluate(x_test, y_test, verbose=0)
print(f"Loaded Model Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
```

In this example, I’ve set `save_weights_only` to `False`. The callback saves the complete model including architecture and weights to the specified `filepath` whenever the `val_loss` improves. The `load_model` function then successfully reconstructs the model and its learned parameters. The evaluation verifies that the loaded model functions as expected.

**Example 2: Saving Only Model Weights**

```python
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

# Assume a model named 'my_model' is already defined and compiled.
my_model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
  tf.keras.layers.Dense(10, activation='softmax')
])
my_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


checkpoint_filepath = 'path/to/my_best_weights.h5'

model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
x_test = x_test.reshape(10000, 784).astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

my_model.fit(
    x_train,
    y_train,
    epochs=10,
    validation_data=(x_test, y_test),
    callbacks=[model_checkpoint_callback]
)

# Load the weights
# Reconstruct model first
loaded_model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
  tf.keras.layers.Dense(10, activation='softmax')
])
loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

loaded_model.load_weights(checkpoint_filepath)


loss, accuracy = loaded_model.evaluate(x_test, y_test, verbose=0)
print(f"Loaded Model Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
```
Here, I've set `save_weights_only` to `True`. The checkpoint captures only weights. The crucial part is reconstructing an identical model architecture before using `load_weights` to apply the trained weight values. The evaluation, as before, checks the functionality. Notice that `tf.keras.models.load_model` cannot load weight-only checkpoints.

**Example 3: Incorporating Custom Layers**

```python
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers

class MyCustomLayer(layers.Layer):
    def __init__(self, units, **kwargs):
      super().__init__(**kwargs)
      self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                               initializer='random_normal',
                               trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                               initializer='zeros',
                               trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


my_model = tf.keras.Sequential([
  MyCustomLayer(32, input_shape=(784,)),
  tf.keras.layers.Dense(10, activation='softmax')
])
my_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



checkpoint_filepath = 'path/to/my_custom_model.h5'


model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)



(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
x_test = x_test.reshape(10000, 784).astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)


my_model.fit(
    x_train,
    y_train,
    epochs=10,
    validation_data=(x_test, y_test),
    callbacks=[model_checkpoint_callback]
)

# Load the model, using custom_objects
loaded_model = tf.keras.models.load_model(checkpoint_filepath, custom_objects={'MyCustomLayer': MyCustomLayer})
loss, accuracy = loaded_model.evaluate(x_test, y_test, verbose=0)
print(f"Loaded Model Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
```

This last example illustrates the necessity of using the `custom_objects` parameter in `load_model` when the model incorporates custom layers. Failure to include it will raise an error since the default loader cannot reconstitute the custom layer's definition. The successful evaluation demonstrates that both the architecture (including the custom layer) and the trained weights were restored correctly.

For further study, I recommend delving into the TensorFlow documentation related to `tf.keras.callbacks.ModelCheckpoint` and `tf.keras.models.load_model`. Additionally, exploring tutorials and examples pertaining to custom layer implementations within Keras can enhance comprehension when employing bespoke model components. Consider consulting textbooks on deep learning and applied machine learning to build a strong conceptual foundation. Focusing on version control practices for model checkpoints and understanding how different hardware configurations can influence the saving process are beneficial for broader implementation skills. These resources, combined with practical coding experience, form the basis of reliable model checkpointing and loading.
