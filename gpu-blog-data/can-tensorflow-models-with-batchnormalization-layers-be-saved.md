---
title: "Can TensorFlow models with BatchNormalization layers be saved?"
date: "2025-01-30"
id: "can-tensorflow-models-with-batchnormalization-layers-be-saved"
---
Yes, TensorFlow models containing `BatchNormalization` layers can be saved and subsequently loaded without losing their trained state; however, several nuances require careful consideration, particularly when dealing with inference after loading a saved model. These nuances stem from how `BatchNormalization` operates and how TensorFlow manages model saving and restoration. In my experience developing and deploying large-scale image recognition models, I've encountered and resolved several challenges related to this process, which I'll detail below.

The critical aspect of `BatchNormalization` is its maintenance of moving averages and variances for each layer. These statistics are not trainable weights like the kernel of a `Dense` or `Conv2D` layer; they are computed and updated during the training phase across mini-batches and then used during inference. Consequently, when a TensorFlow model is saved using `tf.keras.models.save_model` or the lower-level `tf.train.Checkpoint`, it's crucial that these moving averages and variances are also included in the saved artifacts. If they are not properly captured, the loaded model will perform significantly worse during inference than when it was initially trained, essentially because it will lack the statistical understanding gleaned from training data.

The saving mechanism in TensorFlow (whether using the high-level Keras API or the lower-level save/restore tools) is designed to correctly serialize these non-trainable variables associated with BatchNormalization. When you use the high-level `tf.keras` API to construct a model with `BatchNormalization`, the `moving_mean` and `moving_variance` variables are automatically managed as part of the layer’s state. Therefore, using `model.save(path)` or `tf.keras.models.save_model(model, path)` typically ensures these are captured during the saving process. However, it’s important to be aware that loading a model requires careful handling. When you load the saved model, you're not just re-instantiating the model architecture, you're also restoring the numerical values associated with the trainable weights *and* the non-trainable variables, like the moving statistics. In my experience, model degradation after loading most often traces back to these improperly restored moving averages.

Furthermore, the mode of the `BatchNormalization` layer, i.e. training or inference mode, influences behavior of moving statistics and requires attention. During training, `BatchNormalization` computes the mean and variance of the current batch and then updates its moving averages. In inference, the layer uses those stored, moving averages and variances directly, without further updates. The TensorFlow framework automatically switches the mode based on its current understanding of the operation context. However, when using lower-level TensorFlow operations or deploying the model in environments that do not inherit a framework-specific training context, it is essential to explicitly set the training parameter of the `BatchNormalization` layer to `False` when performing inference. Failure to do so can lead to incorrect predictions because the layer will then update its moving statistics during the inference phase which is undesirable.

I'll illustrate these considerations with some code examples. The first example demonstrates a standard case of saving and loading a Keras model with `BatchNormalization`, showing how the moving statistics are inherently handled:

```python
import tensorflow as tf
import numpy as np

# Define a simple sequential model
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(10, activation='relu', input_shape=(20,)),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Dense(5, activation='softmax')
])

# Generate some dummy data for training
x_train = np.random.rand(100, 20)
y_train = np.random.randint(0, 5, size=(100,))
y_train = tf.keras.utils.to_categorical(y_train, num_classes=5)


# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, verbose=0)

# Save the entire model
model.save('my_model_with_bn')

# Load the model
loaded_model = tf.keras.models.load_model('my_model_with_bn')

# Verify that the loaded model produces the same results as the original model
predictions_original = model(x_train[:5])
predictions_loaded = loaded_model(x_train[:5])

# Using allclose for numerical comparisons due to floating-point calculations
print("Predictions identical:", np.allclose(predictions_original, predictions_loaded))
```

This example showcases the straightforward process. The saved model inherently captures the `moving_mean` and `moving_variance`, and the loaded model successfully restores them. This is the most frequent use case and generally provides an expected behavior.

Now, consider a slightly modified scenario where we might need to explicitly control the `training` parameter of BatchNormalization in an unconventional environment:

```python
import tensorflow as tf
import numpy as np

# Define a simple model with a custom training loop
class MyModel(tf.keras.Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.dense1 = tf.keras.layers.Dense(10, activation='relu', input_shape=(20,))
    self.bn = tf.keras.layers.BatchNormalization()
    self.dense2 = tf.keras.layers.Dense(5, activation='softmax')

  def call(self, x, training=False):
    x = self.dense1(x)
    x = self.bn(x, training=training)  #Explicit training flag
    x = self.dense2(x)
    return x

# Initialize and compile
model = MyModel()
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.CategoricalCrossentropy()

# Generate dummy data
x_train = np.random.rand(100, 20).astype(np.float32)
y_train = np.random.randint(0, 5, size=(100,)).astype(np.int64)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=5).astype(np.float32)

# Train step with explicit training=True
@tf.function
def train_step(x, y):
  with tf.GradientTape() as tape:
    y_pred = model(x, training=True)
    loss = loss_fn(y, y_pred)
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  return loss

# Training loop with explicit training mode
for epoch in range(5):
  for i in range(10):
    loss = train_step(x_train[i*10:(i+1)*10], y_train[i*10:(i+1)*10])
  print(f"Epoch {epoch}, loss {loss}")

# Save using checkpoint approach
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.save('./custom_model_checkpoint')

# Load model and its parameters
loaded_checkpoint = tf.train.Checkpoint(model=MyModel())
loaded_checkpoint.restore(tf.train.latest_checkpoint('./custom_model_checkpoint'))

# Verify that inference mode works with proper Batchnorm behavior
predictions_original = model(x_train[:5], training=False)
predictions_loaded = loaded_checkpoint.model(x_train[:5], training=False)
print("Inference predictions identical:", np.allclose(predictions_original, predictions_loaded))

```

This example demonstrates a more flexible and also a more error-prone scenario. The `training=False` flag is crucial in the `call` method for inference after training and loading the model. If omitted, `BatchNormalization` would behave as in training, causing significant performance issues.

Finally, it's relevant to note that while the above two examples utilize Keras save and restore capabilities, a model can also be saved using `tf.train.Checkpoint`, which saves only model weights (trainable and non-trainable variables). When saving via `tf.train.Checkpoint`, it's crucial to include all required variables and when restoring, it’s important that the loaded model correctly references those variables in its layers. For instance:

```python
import tensorflow as tf
import numpy as np


# Define a model similar to the first example
class MyCheckpointModel(tf.keras.Model):
  def __init__(self):
    super(MyCheckpointModel, self).__init__()
    self.dense1 = tf.keras.layers.Dense(10, activation='relu', input_shape=(20,))
    self.bn = tf.keras.layers.BatchNormalization()
    self.dense2 = tf.keras.layers.Dense(5, activation='softmax')

  def call(self, x, training=False):
    x = self.dense1(x)
    x = self.bn(x, training=training)
    x = self.dense2(x)
    return x

# Initialize model and some dummy data
model = MyCheckpointModel()
x_train = np.random.rand(100, 20).astype(np.float32)
y_train = np.random.randint(0, 5, size=(100,)).astype(np.int64)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=5).astype(np.float32)

# Compile and train (basic fitting)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, verbose=0)

# Save using the checkpoint method and ensure non trainable variables are captured
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.save('./my_checkpoint')

# Restore the model from checkpoint
restored_model = MyCheckpointModel()
restored_checkpoint = tf.train.Checkpoint(model=restored_model)
restored_checkpoint.restore(tf.train.latest_checkpoint('./my_checkpoint'))


# Check if predictions match (inference mode)
original_predictions = model(x_train[:5], training=False)
restored_predictions = restored_model(x_train[:5], training=False)
print("Checkpoint predictions identical:", np.allclose(original_predictions, restored_predictions))
```

Here, the key is to initialize a new model with the same architecture before loading the checkpoint. The `restored_checkpoint` is then restored by using its `restore` method. Note again the explicit `training=False` during inference to enforce using accumulated statistics.

In summary, the key considerations for saving and loading TensorFlow models containing `BatchNormalization` layers are: ensuring the correct capture of moving averages and variances which is usually automatic with `model.save` or `tf.keras.models.save_model`; setting the `training=False` flag during inference when appropriate to activate inference behavior; and to use `tf.train.Checkpoint` correctly by ensuring all necessary variables are saved and restored, and that the layer's inference setting is applied.

For additional information, I recommend consulting the TensorFlow documentation pertaining to `tf.keras.layers.BatchNormalization`, `tf.keras.models.save_model`, `tf.train.Checkpoint` and relevant examples on TensorFlow's official website. These resources offer in-depth explanations of the underlying mechanisms and best practices for saving and restoring TensorFlow models, including those employing `BatchNormalization`.
