---
title: "How do I save a TensorFlow model after training?"
date: "2025-01-30"
id: "how-do-i-save-a-tensorflow-model-after"
---
Saving a TensorFlow model after training is a critical step for deploying and reusing machine learning models, and neglecting this process renders the training effort largely futile. From my experience, a common pitfall for beginners is focusing solely on the training loop and not implementing a robust saving mechanism, leading to lost progress and the need for repetitive training. TensorFlow offers various formats for model persistence, each with specific advantages and use cases. I will outline the primary methods I employ and why they are effective, along with illustrative examples.

Fundamentally, TensorFlow allows models to be saved as either a SavedModel or as individual checkpoint files. The SavedModel format, which I typically favor, is a comprehensive directory structure that encapsulates the entire model, including the graph structure, trained weights, and signatures for serving. This format is highly recommended for its portability and suitability for deployment across various platforms. The checkpoint format, conversely, stores only the weights of the model variables, and thus needs the model architecture to be defined again before restoring these weights. This can be useful for intermediate saves or when fine-grained control over restoration is required, but it isn't the best choice for general model persistence.

**1. Saving a Model Using the SavedModel Format**

The `tf.saved_model.save` function is the core method for saving models in the SavedModel format. It requires a model object (either an instance of `tf.keras.Model` or a lower-level object that fulfills the saved model interface) and the desired save directory as arguments. I always structure my saving logic to incorporate timestamps or version numbers within the directory name to facilitate model tracking.

```python
import tensorflow as tf
import datetime

# Assume model is a compiled tf.keras.Model called 'my_model'
my_model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

my_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Generate sample training data
import numpy as np
x_train = np.random.rand(100, 784)
y_train = np.random.randint(0, 10, (100,))
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
my_model.fit(x_train, y_train, epochs=5)


# Generate save directory with timestamp
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
save_path = f"my_saved_model_{current_time}"

# Save the model in SavedModel format
tf.saved_model.save(my_model, save_path)

print(f"Model saved to: {save_path}")
```

In this example, I first train a simple Keras sequential model for demonstration. Then, instead of hard-coding the save directory, I generate a timestamped directory name using Python's `datetime` module. This timestamped directory aids in keeping track of model versions. The `tf.saved_model.save` function then serializes and saves the model into the specified directory. After running this script, a new directory (e.g., "my_saved_model_20231027-143542") will appear containing a collection of files, including a `saved_model.pb` file and an `assets` folder, encompassing all data necessary to reconstitute the model for deployment.

**2. Saving a Model Using Checkpoints with Custom Training Loops**

While SavedModel is the more encompassing choice, there are scenarios, particularly with custom training loops, where saving individual checkpoints becomes valuable. These checkpoints store only the model's trainable variables, not the full architecture. This requires careful attention, as the architecture must be redefined before loading the weights. The `tf.train.Checkpoint` API is used for managing checkpoint saves.

```python
import tensorflow as tf
import os

# Define a custom model (not a tf.keras.Model)
class MyCustomModel(tf.Module):
    def __init__(self, num_units):
        super().__init__()
        self.dense1 = tf.Variable(tf.random.normal([784, num_units]), name='dense1_weight')
        self.dense2 = tf.Variable(tf.random.normal([num_units, 10]), name='dense2_weight')
        self.bias1 = tf.Variable(tf.zeros([num_units]), name='bias1')
        self.bias2 = tf.Variable(tf.zeros([10]), name='bias2')
    def __call__(self, x):
        x = tf.nn.relu(tf.matmul(x, self.dense1) + self.bias1)
        return tf.nn.softmax(tf.matmul(x, self.dense2) + self.bias2)

# Create a custom model and optimizer
my_custom_model = MyCustomModel(100)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Define a checkpoint
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=my_custom_model)

# Assume custom training loop exists. Here's a toy version.
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        logits = my_custom_model(images)
        loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, logits))
    gradients = tape.gradient(loss, my_custom_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, my_custom_model.trainable_variables))
    return loss

num_epochs = 5
num_samples = 100
x_train = tf.random.normal([num_samples, 784])
y_train = tf.one_hot(tf.random.uniform([num_samples], minval=0, maxval=10, dtype=tf.int32), 10)

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for i in range(num_samples):
        current_loss = train_step(x_train[i:i+1], y_train[i:i+1])
        epoch_loss += current_loss

    epoch_loss /= num_samples
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

    # Save checkpoint every 2 epochs
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)
        print("Checkpoint saved.")
```

In this second example, I defined a simple custom model using `tf.Module` instead of Keras' `Model`. The training loop was simplified for clarity, but the concept is similar: compute loss, calculate gradients, and apply optimization. Crucially, I created a `tf.train.Checkpoint` object, associating it with the optimizer and model to be saved.  After training, I save the checkpoint periodically using the `checkpoint.save` method. The created directory contains data files such as `checkpoint` and `ckpt-1.index`, along with `ckpt-1.data-00000-of-00001`, that store the model's parameters, allowing us to restore the model's weights.

**3. Loading a SavedModel**

Loading a SavedModel is straightforward using the `tf.saved_model.load` function. The function requires the path to the saved model directory. Once loaded, the model's signatures (functions exposed for serving) can be accessed through the model object.

```python
import tensorflow as tf
import numpy as np

# Assume save_path is the directory from the first example

save_path = "my_saved_model_20231027-143542"

# Load the saved model
loaded_model = tf.saved_model.load(save_path)

# Define an input tensor
input_tensor = np.random.rand(1, 784).astype(np.float32)

# Make a prediction using the model's serving signature
infer = loaded_model.signatures['serving_default']
output = infer(tf.constant(input_tensor))

print("Prediction from loaded model:", output)
```

This code segment demonstrates how to load a previously saved model using `tf.saved_model.load` specifying the path to the SavedModel. By accessing the signatures via `loaded_model.signatures['serving_default']` and passing in a sample input, I can execute predictions utilizing the trained network. This confirms the integrity of the loaded model.

**Resource Recommendations**

For a deeper understanding of model persistence in TensorFlow, I would recommend exploring the official TensorFlow documentation covering `tf.saved_model` and `tf.train.Checkpoint`. Furthermore, examining tutorials and examples related to model deployment will provide valuable insights into the practical implications of choosing the right format. The TensorFlow guide on custom training loops provides details that are pertinent when dealing with Checkpoints, as well. Additionally, the Keras API documentation can be useful, especially when using the Keras layers and high-level model building abstractions.
