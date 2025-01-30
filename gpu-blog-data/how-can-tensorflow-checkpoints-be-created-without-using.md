---
title: "How can TensorFlow checkpoints be created without using a session?"
date: "2025-01-30"
id: "how-can-tensorflow-checkpoints-be-created-without-using"
---
TensorFlow checkpoints, while historically tied to `tf.compat.v1.Session` objects, can be effectively created and managed using the `tf.train.Checkpoint` API introduced in TensorFlow 2.x. This shift decouples checkpoint creation from the explicit session management that was prevalent in earlier versions. My experience migrating legacy model training code has highlighted the benefits of this approach in terms of flexibility and clarity.

The fundamental concept underpinning sessionless checkpointing revolves around capturing and restoring the state of trainable variables and other TensorFlow objects using a `tf.train.Checkpoint` instance. Instead of running operations within a session, this API directly tracks the assigned values of variables. This enables a more declarative style of coding.

Let’s break this down into practical components. First, to enable checkpointing, trainable variables, models, and optimizers (anything that holds state we wish to persist) must be assigned to a `tf.train.Checkpoint` object as named attributes. When creating this object, I usually start by adding the most important components first, such as the trainable variables. During training, the state of these attributes, i.e. their numeric values, is automatically tracked as it evolves.

Then, when we wish to save the checkpoint, we simply invoke the `save()` method of the `tf.train.Checkpoint` instance. This saves the current values of all the tracked variables, among other things. Later, we can use `restore()` to load the previously saved values. The restore operation does not require a session and is efficient to use. It finds the appropriate variables and assigns the stored values into them. The key point is that these actions are independent of any explicit session.

Now, let’s illustrate this with concrete examples. In the first example, we'll build a basic single-layer linear model.

```python
import tensorflow as tf

# Define the linear model variables
w = tf.Variable(tf.random.normal(shape=[1, 1]), name="weight")
b = tf.Variable(tf.zeros(shape=[1]), name="bias")

# Define the model operation
def linear_model(x):
    return tf.matmul(x, w) + b

# Define the optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# Create a tf.train.Checkpoint object
checkpoint = tf.train.Checkpoint(step=tf.Variable(0), model_variables=(w, b), optimizer=optimizer)

# Training loop (simplified example)
inputs = tf.constant([[1.0], [2.0], [3.0]])
targets = tf.constant([[2.0], [4.0], [6.0]])
epochs = 20

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        predictions = linear_model(inputs)
        loss = tf.reduce_mean(tf.square(predictions - targets))
    
    gradients = tape.gradient(loss, [w,b])
    optimizer.apply_gradients(zip(gradients, [w,b]))
    checkpoint.step.assign_add(1)

    if epoch % 10 == 0:
        checkpoint.save("./checkpoints/model-checkpoint")
        print(f"Epoch: {epoch}, Loss: {loss.numpy()}")


# Restore from checkpoint to demonstrate loading process
checkpoint.restore(tf.train.latest_checkpoint("./checkpoints"))
print("Model weights after restoration:", w.numpy(), b.numpy())

```

In this example, I created `w` and `b` as `tf.Variable` instances and encapsulated them within a `tf.train.Checkpoint` under the name 'model_variables'. This includes an explicit `optimizer` and a step counter. The training loop demonstrates a typical forward pass and optimization step. Crucially, the checkpoint's `save` method is called periodically, and the `restore` method is used to load the model from the latest checkpoint. This entire workflow is independent of a `tf.compat.v1.Session`.

Now, let’s expand the scope with a more complex model, using `tf.keras`. This also demonstrates the hierarchical naming of checkpoints.

```python
import tensorflow as tf

# Define a Keras model
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(16, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return x

model = MyModel()

# Define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Create a tf.train.Checkpoint instance
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, step=tf.Variable(0))


# Training loop
inputs = tf.random.normal((32, 32))
targets = tf.random.normal((32, 1))
epochs = 10

for epoch in range(epochs):
    with tf.GradientTape() as tape:
      predictions = model(inputs)
      loss = tf.reduce_mean(tf.square(predictions - targets))

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    checkpoint.step.assign_add(1)

    if epoch % 5 == 0:
      checkpoint.save("./checkpoints/keras-model-checkpoint")
      print(f"Epoch {epoch}, loss: {loss}")

# Restore
checkpoint.restore(tf.train.latest_checkpoint("./checkpoints"))
print("Model restored from checkpoint")

```

In the second example, we're utilizing a `tf.keras.Model`. The model itself is a trainable object, and all of its `trainable_variables` are inherently part of the checkpoint via the assignment `checkpoint= tf.train.Checkpoint(model=model, optimizer=optimizer, step=tf.Variable(0))`. This illustrates how checkpointing smoothly extends to more complex structures within the TensorFlow ecosystem. Once again, the training loop is session-less and operates directly on TensorFlow tensors using eager execution. The structure of the code here is significantly cleaner than the comparable code in TensorFlow 1.x utilizing sessions. This is a key advantage in development speed and maintainability.

Finally, let's consider a scenario involving multiple checkpoints across different stages of training. This allows us to save and restore specific model configurations and other parameters as training progresses.

```python
import tensorflow as tf

# Model (using the previous MyModel class)
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(16, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return x

model = MyModel()

# Define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Create two Checkpoint instances
checkpoint_pretrain = tf.train.Checkpoint(model=model, optimizer=optimizer, step=tf.Variable(0))
checkpoint_finetune = tf.train.Checkpoint(model=model, optimizer=optimizer, step=tf.Variable(0))


# Pre-training loop (simplified)
inputs_pre = tf.random.normal((32, 32))
targets_pre = tf.random.normal((32, 1))
epochs_pre = 5

for epoch in range(epochs_pre):
  with tf.GradientTape() as tape:
    predictions = model(inputs_pre)
    loss = tf.reduce_mean(tf.square(predictions-targets_pre))
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  checkpoint_pretrain.step.assign_add(1)
  if epoch % 2 == 0:
    checkpoint_pretrain.save("./checkpoints/pretrain-checkpoint")
    print(f"Pre-training epoch {epoch}, loss: {loss}")

# Fine-tuning loop
checkpoint_pretrain.restore(tf.train.latest_checkpoint("./checkpoints"))
inputs_fine = tf.random.normal((64, 32))
targets_fine = tf.random.normal((64, 1))
epochs_fine = 5

for epoch in range(epochs_fine):
  with tf.GradientTape() as tape:
    predictions = model(inputs_fine)
    loss = tf.reduce_mean(tf.square(predictions - targets_fine))
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  checkpoint_finetune.step.assign_add(1)
  if epoch % 2 == 0:
    checkpoint_finetune.save("./checkpoints/finetune-checkpoint")
    print(f"Finetuning epoch: {epoch}, loss: {loss}")

checkpoint_finetune.restore(tf.train.latest_checkpoint("./checkpoints"))
print("Fine-tuned model restored from checkpoint")
```

In this scenario, I use two `tf.train.Checkpoint` instances, `checkpoint_pretrain` and `checkpoint_finetune`. This allows saving the model at different stages, representing, for example, a pre-trained and fine-tuned state. The crucial part here is that each checkpoint is self-contained, holding its specific values and tracking step progress. This example highlights a more advanced use of `tf.train.Checkpoint`.

For further exploration, I recommend consulting the official TensorFlow documentation on `tf.train.Checkpoint` and `tf.train.CheckpointManager`. The guides on saving and restoring models within TensorFlow Core are also invaluable resources. The "Training and saving" section on the TensorFlow website provides solid, practical examples. Additionally, reviewing TensorFlow’s source code, specifically the implementation of `tf.train.Checkpoint` within `tensorflow/python/training/checkpointable.py`, provides deeper insights. These resources allow one to gain a complete understanding of the checkpoint mechanism.
