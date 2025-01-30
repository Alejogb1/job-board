---
title: "Does a TensorFlow SavedModel (.pb) file contain only the model's graph or also its weights?"
date: "2025-01-30"
id: "does-a-tensorflow-savedmodel-pb-file-contain-only"
---
The TensorFlow SavedModel format (.pb) definitively contains both the model's computational graph and its trained weights. Having spent considerable time deploying TensorFlow models, I can attest that the .pb file itself encapsulates the entire model definition necessary for execution, barring external dependencies like custom operations. This single file facilitates model portability by packaging the structure (graph) and its parameterized state (weights) together.

A common misconception is that the .pb file contains only the structural information, requiring a separate checkpoint file for the weights. This arises from TensorFlow's older checkpointing mechanisms, which are distinct from SavedModel. SavedModel, however, intentionally consolidates these two elements into a serialized Protocol Buffer format. The graph specifies the operations that the model performs (matrix multiplication, convolutions, etc.), while the weights are the floating-point numbers representing the learned parameters within that graph. Without the weights, the graph would be effectively useless, functioning as a template without any specific knowledge.

The process of creating a SavedModel using `tf.saved_model.save()` serializes the entire model including:

1.  **The Graph Definition:** This is the abstract representation of the operations and their interconnections. It's essentially a computational blueprint that defines how input data is transformed into output.
2.  **The Trained Weights:** These are the specific floating-point values associated with each trainable parameter in the graph (e.g., filter kernels in a convolutional layer, weights in a fully-connected layer). These numbers have been determined by the training process.
3.  **Signatures:** SavedModel also includes information about the inputs and outputs of the model, allowing easy specification of what type of data to expect for inference.
4. **Asset Information:** Sometimes assets like vocabularies can also be stored within the SavedModel structure, though not directly within the .pb file itself. These are typically found in an 'assets' folder associated with the SavedModel directory.

The .pb file is technically a Protocol Buffer (protobuf) file, a platform-neutral, extensible mechanism for serializing structured data. It's important to recognize that the .pb file itself is a binary file not meant for manual inspection; the information within is organized in a manner that TensorFlow can understand when restoring a model.

To illustrate how weights are included, let's examine simplified examples using TensorFlow.

**Code Example 1: Simple Linear Model**

```python
import tensorflow as tf

# Create a simple linear model
class LinearModel(tf.Module):
  def __init__(self):
    self.w = tf.Variable(tf.random.normal([1, 1]), name="weight")
    self.b = tf.Variable(tf.zeros([1]), name="bias")

  @tf.function(input_signature=[tf.TensorSpec(shape=[1, 1], dtype=tf.float32)])
  def __call__(self, x):
    return tf.matmul(x, self.w) + self.b

# Instantiate the model, and save it
model = LinearModel()
input_data = tf.constant([[1.0]], dtype=tf.float32)
_ = model(input_data)  # Run once to ensure variables are created
tf.saved_model.save(model, "linear_model")

#Verify the SavedModel has variables
loaded = tf.saved_model.load("linear_model")
print("Model variables :",[v.name for v in loaded.variables]) #prints the names of the variables.


# You can also check directly if the `weight` and `bias` attributes are there, and their values.
print("Weight value: ",loaded.w)
print("Bias value: ", loaded.b)

```

In this example, I define a simple linear model with a single weight (`w`) and bias (`b`). I initialize these variables with random values. After running the model once to initiate variables, I use `tf.saved_model.save` to store the model in the "linear_model" directory.  The `saved_model` command creates a subfolder with model artifacts including a .pb file. Importantly, the values of `w` and `b`, not just the graph structure, are also saved within the SavedModel. When loaded back, it confirms the presence of saved variables, showing that both the structure and weights are included within. The loaded model has direct access to `w` and `b` through object access, proving that the weights are part of the model loaded from the saved format.

**Code Example 2: Using Keras Model API**

```python
import tensorflow as tf

# Keras sequential model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1)
])

# Random input
input_data = tf.random.normal(shape=(1, 10))

# Initialize the model with an input to create the variables.
_ = model(input_data)

# Save the model
tf.saved_model.save(model, 'keras_model')

# Load and check variables
loaded_model = tf.saved_model.load('keras_model')
print("Variables of keras model :",[v.name for v in loaded_model.variables])
```

This example demonstrates saving a Keras sequential model. The process is the same as the first example, I first create an input to let the model initiate the variables, and save it as a `SavedModel` in 'keras_model' directory using `tf.saved_model.save` command.  Once loaded using `tf.saved_model.load`, I inspect the variables to ensure that the parameters of the `Dense` layers are persisted within the saved model. These variables encapsulate the learned weights of each layer, showing that they’re stored together with the graph in the SavedModel structure. Note that variables created by Keras models may not be accessible by accessing `model.variable_name` attribute, but it's still stored as part of the saved model.

**Code Example 3: Custom Training Loop**

```python
import tensorflow as tf

class CustomModel(tf.Module):
  def __init__(self, num_units):
    self.dense = tf.keras.layers.Dense(num_units, activation='relu')
    self.out = tf.keras.layers.Dense(1)

  @tf.function(input_signature=[tf.TensorSpec(shape=[None, 2], dtype=tf.float32)])
  def __call__(self, x):
    x = self.dense(x)
    return self.out(x)

model = CustomModel(num_units=4)
input_data = tf.random.normal(shape=[1,2])
_ = model(input_data) # initialize the weights.

optimizer = tf.keras.optimizers.Adam(0.01)
loss_fn = tf.keras.losses.MeanSquaredError()
x_train = tf.random.normal(shape=[10,2])
y_train = tf.random.normal(shape=[10,1])

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = loss_fn(y, y_pred)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# Training loop to modify weights
epochs = 3
for i in range(epochs):
  loss = train_step(x_train, y_train)
  print(f"Epoch:{i+1}, Loss: {loss.numpy():.4f}")


tf.saved_model.save(model, 'custom_model')

loaded_model = tf.saved_model.load("custom_model")
print("Variables of custom model :",[v.name for v in loaded_model.variables])

```

This example goes a step further by training a custom model within a training loop.  The model is trained for a few epochs, modifying the weights during the training process, and then saved. This confirms that the model's *trained* weights are saved within the SavedModel. It demonstrates that the trained parameter values, resulting from the optimization process, are captured along with the computational graph. When this model is loaded, the variables will reflect those *trained* values.

These code examples highlight that `tf.saved_model.save` does not simply store a blueprint, but rather it saves the model with its current state including both structure *and* trained parameters. When restoring this model, it has all the required parameters for execution.

Regarding resources, I recommend exploring the official TensorFlow documentation on SavedModel. In particular, the sections detailing the saving and loading procedures, as well as the internal structure, are exceptionally helpful. Reading research papers focused on model serialization techniques will provide more advanced insight. Finally, the source code of TensorFlow itself is also a valuable, though advanced, resource.  Examining the implementation of `tf.saved_model.save` is a very technical way to confirm the details of how weights are stored.
