---
title: "Is periodically saving Keras models as SavedModels in TensorFlow 2 the optimal approach?"
date: "2025-01-30"
id: "is-periodically-saving-keras-models-as-savedmodels-in"
---
The assertion that periodically saving Keras models as SavedModels in TensorFlow 2 constitutes the universally optimal approach requires careful examination. While this practice is robust and broadly applicable, its effectiveness is contingent upon specific application demands and deployment environments. From my experience building and maintaining several large-scale machine learning systems, I've found that a one-size-fits-all solution rarely exists; rather, the ideal checkpointing strategy often involves a balance between saving frequency, storage overhead, and inference latency.

The SavedModel format in TensorFlow 2 provides a language-agnostic, self-contained representation of a trained model, encompassing the computational graph, weights, and metadata required for execution. This format facilitates model deployment across diverse platforms, from cloud-based servers to edge devices, and also ensures compatibility with TensorFlow Serving, a scalable infrastructure for model inference. The ability to rapidly reload a model from disk, without requiring Python execution, is a key strength of this approach.

However, the term “optimal” is highly contextual. Periodically saving SavedModels comes with costs. The primary consideration is disk I/O and storage space. Serializing a complete model and writing it to disk is a computationally and time-consuming operation. A large model with numerous parameters can lead to substantial write times, potentially delaying training progression. Moreover, repeatedly writing the entire model can occupy considerable storage, particularly if high-frequency checkpointing is adopted. This can become problematic with cloud-based training environments where storage resources carry financial implications.

The frequency of saving models must also be evaluated against the risk of losing intermediate progress. If training is interrupted by hardware failure or software issues, the last saved model serves as the recovery point. Insufficient checkpointing frequency means more training iterations are lost. Conversely, excessively frequent saves can hinder training performance, consume disk space rapidly, and not yield significant benefits beyond a certain point. For example, if training loss plateaus over long intervals, the frequent save will be mostly redundant.

In addition to the time taken by writes, the model’s ability to be loaded back up also needs to be factored in. While SavedModel format loads efficiently for inference once stored, the load operation still takes a measurable amount of time. If the model is re-loaded on every single use, the overhead would be quite considerable, defeating the purpose of optimizing inference speed. Therefore the model loading needs to be done judiciously.

Beyond the inherent overhead of SavedModel creation, the specific needs of the machine learning workflow often dictate a more nuanced checkpointing strategy. For instance, in a scenario where continual learning is required, checkpointing only the weights and not the entire computational graph, would be a more effective approach. Here the full model architecture can be preserved in memory, and only parameters are updated and saved for efficiency. This would entail the use of mechanisms besides standard SavedModels.

Here are three code examples demonstrating different saving approaches with commentary:

**Example 1: Periodically Saving the Full SavedModel**

```python
import tensorflow as tf
import os

def train_model(model, dataset, epochs, save_dir, save_freq):
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    metric = tf.keras.metrics.CategoricalAccuracy()
    step = tf.Variable(0, dtype=tf.int64)
    for epoch in range(epochs):
        for batch_x, batch_y in dataset:
            with tf.GradientTape() as tape:
                predictions = model(batch_x)
                loss = loss_fn(batch_y, predictions)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            metric.update_state(batch_y, predictions)
            step.assign_add(1)
            if step % save_freq == 0:
                 checkpoint_path = os.path.join(save_dir, f'model_step_{step}')
                 model.save(checkpoint_path)
                 print(f'Saved model at step: {step}')
        print(f'Epoch {epoch + 1}, Accuracy {metric.result().numpy()}')
        metric.reset_states()

# Example Usage:
if __name__ == '__main__':
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(10, activation='relu', input_shape=(100,)),
      tf.keras.layers.Dense(5, activation='softmax')
  ])

  dataset = tf.data.Dataset.from_tensor_slices((tf.random.normal((1000, 100)), tf.one_hot(tf.random.uniform((1000,), minval=0, maxval=5, dtype=tf.int32), 5))).batch(32)
  save_dir = 'saved_models'
  os.makedirs(save_dir, exist_ok=True)
  train_model(model, dataset, epochs=5, save_dir=save_dir, save_freq=50)
```

*Commentary:* This example demonstrates the basic periodic saving of the entire model as a SavedModel. After every `save_freq` training steps, the current model state is persisted to disk. The advantage here is the ease of deployment and recovery in production environments. The disadvantage is the potentially high disk I/O and storage overhead particularly for large models. The `step` variable keeps track of the training step, which is included in the filename. It also uses a `step` based saving as opposed to epochs.

**Example 2: Saving Only Model Weights**

```python
import tensorflow as tf
import os

def train_model(model, dataset, epochs, save_dir, save_freq):
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    metric = tf.keras.metrics.CategoricalAccuracy()
    step = tf.Variable(0, dtype=tf.int64)
    for epoch in range(epochs):
        for batch_x, batch_y in dataset:
            with tf.GradientTape() as tape:
                predictions = model(batch_x)
                loss = loss_fn(batch_y, predictions)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            metric.update_state(batch_y, predictions)
            step.assign_add(1)
            if step % save_freq == 0:
              checkpoint_path = os.path.join(save_dir, f'weights_step_{step}.ckpt')
              model.save_weights(checkpoint_path)
              print(f'Saved weights at step: {step}')
        print(f'Epoch {epoch + 1}, Accuracy {metric.result().numpy()}')
        metric.reset_states()


# Example Usage:
if __name__ == '__main__':
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(100,)),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    dataset = tf.data.Dataset.from_tensor_slices((tf.random.normal((1000, 100)), tf.one_hot(tf.random.uniform((1000,), minval=0, maxval=5, dtype=tf.int32), 5))).batch(32)

    save_dir = 'saved_weights'
    os.makedirs(save_dir, exist_ok=True)
    train_model(model, dataset, epochs=5, save_dir=save_dir, save_freq=50)

```

*Commentary:*  This variant saves only the model’s weights, excluding the architecture itself. This approach significantly reduces disk space and write time when compared to saving the entire model. To reload, one needs to re-instantiate the Keras model, and then load the saved weights. This is particularly useful when the model’s architecture remains constant, such as during fine-tuning, or when using the same model for continuous learning.

**Example 3: Utilizing Callbacks for Saving**

```python
import tensorflow as tf
import os

class ModelCheckpointWithStep(tf.keras.callbacks.Callback):
    def __init__(self, filepath, save_freq=1, verbose=0):
        super().__init__()
        self.filepath = filepath
        self.save_freq = save_freq
        self.verbose = verbose
        self.step = 0

    def on_train_batch_end(self, batch, logs=None):
         self.step+=1
         if self.step % self.save_freq == 0:
            filepath_with_step = self.filepath.format(step=self.step)
            self.model.save(filepath_with_step)
            if self.verbose > 0:
                print(f"\nSaving model at step {self.step} to {filepath_with_step}")


# Example Usage
if __name__ == '__main__':
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(10, activation='relu', input_shape=(100,)),
      tf.keras.layers.Dense(5, activation='softmax')
  ])

  dataset = tf.data.Dataset.from_tensor_slices((tf.random.normal((1000, 100)), tf.one_hot(tf.random.uniform((1000,), minval=0, maxval=5, dtype=tf.int32), 5))).batch(32)
  save_dir = 'saved_models_callbacks'
  os.makedirs(save_dir, exist_ok=True)
  callback = ModelCheckpointWithStep(filepath = os.path.join(save_dir, "model_step_{step}"), save_freq=50, verbose=1)
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  model.fit(dataset, epochs=5, callbacks=[callback], verbose=0)
```
*Commentary:* This example leverages Keras callbacks to achieve the periodic saving of SavedModels.  This allows modularity and integration directly into `model.fit()` workflow. Using a custom callback with step counter allows us to save at a specific number of steps, instead of the epoch-based saving that the standard checkpoint callback implements. This can offer better control over save frequency.

In conclusion, while periodically saving Keras models as SavedModels is a reliable and general practice, its “optimality” is not guaranteed across diverse situations. The choice between full model saves, weight-only saves, or the use of techniques like custom callbacks, depends upon the specific requirements of the task, the model architecture, available resources and training pipeline limitations. A thorough understanding of these tradeoffs is crucial for constructing an efficient and robust training procedure.

For further exploration of model saving techniques and related performance optimizations, I would suggest consulting the TensorFlow documentation focusing on `tf.saved_model`, `tf.train.Checkpoint` objects and Keras callbacks. Furthermore, research focused on specific deployment scenarios will enable better adaptation and application of checkpointing techniques.
